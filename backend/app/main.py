import logging
import os
import sys
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymilvus import Collection, connections, list_collections, utility
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.chunking import chunk_document_record, deduplicate_by_parent
from app.rag_service import generate_rag_summary
from app.reranker import rerank_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-search-api")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "documents")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

app = FastAPI(
    title="RAG Research Search & Ingestion API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str = Field(..., min_length=1)
    doc_type: Optional[str] = "both"
    date_range: Optional[List[str]] = None
    citation_min: Optional[int] = Field(default=0, ge=0)
    field_of_research: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)


class DocumentResult(BaseModel):
    id: str
    title: str
    abstract: str
    doc_type: str
    pub_date: str
    citation_count: int
    field_of_research: str
    sub_topic: str


class SearchResponse(BaseModel):
    documents: List[DocumentResult]
    trends: Dict[str, Dict[str, int]]
    velocity: Dict[str, Any]
    ai_summary: str


class IngestRequest(BaseModel):
    id: Optional[str] = None
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    doc_type: str = Field(default="paper", max_length=20)
    pub_date: str = Field(default="2026-01-01", max_length=20)
    citation_count: int = Field(default=0, ge=0)
    field_of_research: str = Field(default="Computer Science", max_length=100)
    sub_topic: str = Field(default="General", max_length=120)
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)


class IngestResponse(BaseModel):
    status: str
    parent_id: str
    chunks_indexed: int
    chunk_ids: List[str]


class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    collection_loaded: bool
    collection: str


try:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("MILVUS CONNECTED")
except Exception as exc:
    logger.exception("Failed to connect to Milvus: %s", exc)
    sys.exit(1)

if not utility.has_collection(COLLECTION_NAME):
    logger.error("%s collection missing", COLLECTION_NAME)
    sys.exit(1)

collection = Collection(COLLECTION_NAME)
collection.load()
logger.info("Collection loaded")

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_subtopic_names(documents, labels, n_clusters):
    tfidf = TfidfVectorizer(max_features=3, stop_words="english")
    clusters = [[] for _ in range(n_clusters)]

    for doc, label in zip(documents, labels):
        clusters[label].append(doc.get("abstract", ""))

    names = []
    for cluster in clusters:
        if not cluster:
            names.append("Miscellaneous")
            continue
        try:
            tfidf.fit_transform(cluster)
            names.append(" ".join(tfidf.get_feature_names_out()[:2]))
        except Exception:
            names.append("Miscellaneous")
    return names


def _escape_milvus_str(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _build_filter_expr(query: Query) -> Optional[str]:
    expr_parts: List[str] = []

    if query.doc_type and query.doc_type.strip() and query.doc_type != "both":
        expr_parts.append(f"doc_type == '{_escape_milvus_str(query.doc_type.strip())}'")

    if query.citation_min and query.citation_min > 0:
        expr_parts.append(f"citation_count >= {int(query.citation_min)}")

    if query.field_of_research and query.field_of_research.strip():
        safe_field = _escape_milvus_str(query.field_of_research.strip())
        expr_parts.append(f"field_of_research like '%{safe_field}%'")

    return " and ".join(expr_parts) if expr_parts else None


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Backend running"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        connected = connections.has_connection("default")
        exists = utility.has_collection(COLLECTION_NAME)
        return HealthResponse(
            status="ok" if (connected and exists) else "degraded",
            milvus_connected=connected,
            collection_loaded=exists,
            collection=COLLECTION_NAME,
        )
    except Exception as exc:
        logger.exception("Health check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store health check failed",
        ) from exc


@app.get("/ping")
def ping() -> Dict[str, str]:
    return {"status": "alive"}


@app.get("/list_collections")
def collections_api() -> Dict[str, List[str]]:
    try:
        return {"collections": list_collections()}
    except Exception as exc:
        logger.exception("Failed to list collections: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to list Milvus collections",
        ) from exc


@app.post("/test")
def test() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest_document(payload: IngestRequest) -> IngestResponse:
    parent_id = payload.id.strip() if payload.id and payload.id.strip() else f"doc_{uuid.uuid4().hex[:10]}"

    try:
        chunks = chunk_document_record(
            doc={
                "id": parent_id,
                "title": payload.title,
                "content": payload.content,
                "doc_type": payload.doc_type,
                "pub_date": payload.pub_date,
                "citation_count": payload.citation_count,
                "field_of_research": payload.field_of_research,
                "sub_topic": payload.sub_topic,
            },
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No non-empty text chunks produced from input",
        )

    try:
        texts_to_embed = [f"{c['title']}. {c['abstract']}" for c in chunks]
        vectors = model.encode(texts_to_embed)

        rows = [
            {
                "id": chunk["id"],
                "vector": vectors[idx].tolist(),
                "title": chunk["title"],
                "abstract": chunk["abstract"],
                "doc_type": chunk["doc_type"],
                "pub_date": chunk["pub_date"],
                "citation_count": chunk["citation_count"],
                "field_of_research": chunk["field_of_research"],
                "sub_topic": chunk["sub_topic"],
            }
            for idx, chunk in enumerate(chunks)
        ]

        collection.insert(rows)
        collection.flush()
    except Exception as exc:
        logger.exception("Document ingestion failed for parent_id=%s: %s", parent_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to embed or store document chunks",
        ) from exc

    return IngestResponse(
        status="ingested",
        parent_id=parent_id,
        chunks_indexed=len(rows),
        chunk_ids=[r["id"] for r in rows],
    )


@app.post("/search", response_model=SearchResponse)
def search(query: Query) -> SearchResponse:
    cleaned_text = query.text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty",
        )

    try:
        query_vector = model.encode([cleaned_text])[0]
        expr = _build_filter_expr(query)
        search_limit = min(max(query.top_k * 2, query.top_k), 50)

        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=search_limit,
            expr=expr,
            output_fields=[
                "title",
                "abstract",
                "doc_type",
                "pub_date",
                "citation_count",
                "field_of_research",
                "sub_topic",
            ],
        )

        candidates: List[Dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.entity
            candidates.append(
                {
                    "id": str(hit.id),
                    "title": entity.get("title") or "Untitled",
                    "abstract": entity.get("abstract") or "",
                    "doc_type": entity.get("doc_type") or "unknown",
                    "pub_date": entity.get("pub_date") or "Unknown",
                    "citation_count": int(entity.get("citation_count") or 0),
                    "field_of_research": entity.get("field_of_research") or "",
                    "sub_topic": entity.get("sub_topic") or "General",
                }
            )

        reranked = rerank_documents(cleaned_text, candidates) if candidates else []
        final_docs = deduplicate_by_parent(reranked, limit=query.top_k)

        summary = ""
        if final_docs:
            try:
                summary = generate_rag_summary(cleaned_text, final_docs)
            except Exception as llm_exc:
                logger.warning("RAG summary generation failed: %s", llm_exc)
                summary = ""

        trends_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for doc in final_docs:
            topic = doc.get("sub_topic", "General")
            year = str(doc.get("pub_date", ""))[:4]
            if year.isdigit():
                trends_map[topic][year] += 1

        trends = {topic: dict(years) for topic, years in trends_map.items()}

        return SearchResponse(
            documents=[DocumentResult(**doc) for doc in final_docs],
            trends=trends,
            velocity={},
            ai_summary=summary,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search pipeline failed",
        ) from exc