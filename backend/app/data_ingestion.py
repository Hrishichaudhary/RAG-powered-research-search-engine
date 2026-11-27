import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_document_record

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data" / "raw"))
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "documents")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

FIELD_STANDARDIZATION = {
    "machine learning": "Computer Science",
    "neural network": "Computer Science",
    "artificial intelligence": "Computer Science",
    "data classification": "Computer Science",
    "deep learning": "Computer Science",
    "advanced neural network": "Computer Science",
    "natural language processing": "Computer Science",
    "computational physics": "Physics",
    "bioinformatics": "Biology",
    "genetics": "Biology",
    "scheduling": "Operations Research",
    "optimization": "Operations Research",
}

SUB_TOPIC_MAP = {
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "neural network": "Neural Networks",
    "natural language processing": "NLP",
    "transformer": "Transformers",
    "optimization": "Optimization",
    "classification": "Classification",
    "genetics": "Genetics",
    "bioinformatics": "Bioinformatics",
    "physics": "Physics",
}


def generate_subtopic(title: Any, abstract: Any) -> str:
    text = f"{title or ''} {abstract or ''}".lower()
    for key, value in SUB_TOPIC_MAP.items():
        if key in text:
            return value
    return "General"


def standardize_field(field: Any) -> str:
    if not isinstance(field, str) or not field.strip():
        return "Unknown"
    field_lower = field.lower()
    for key, value in FIELD_STANDARDIZATION.items():
        if key in field_lower:
            return value
    return field.strip()


def convert_date(date_str: Any) -> str:
    raw = str(date_str or "").strip()
    if not raw:
        return "1970-01-01"

    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    if len(raw) == 4 and raw.isdigit():
        return f"{raw}-01-01"

    return raw[:20]


def normalize_patents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    if "patent_id" in df.columns and "id" not in df.columns:
        rename_map["patent_id"] = "id"
    if "patent_title" in df.columns and "title" not in df.columns:
        rename_map["patent_title"] = "title"
    if "patent_abstract" in df.columns and "abstract" not in df.columns:
        rename_map["patent_abstract"] = "abstract"
    if "patent_date" in df.columns and "pub_date" not in df.columns:
        rename_map["patent_date"] = "pub_date"
    elif "publication_date" in df.columns and "pub_date" not in df.columns:
        rename_map["publication_date"] = "pub_date"

    df = df.rename(columns=rename_map)

    for idx, col in enumerate(["id", "title", "abstract", "pub_date", "field_of_research"]):
        if col not in df.columns:
            df[col] = f"patent_{idx}" if col == "id" else ""

    if "citation_count" not in df.columns:
        df["citation_count"] = 0

    df["id"] = [
        str(val).strip() if str(val).strip() else f"patent_{i}"
        for i, val in enumerate(df["id"])
    ]
    df["doc_type"] = "patent"
    df["pub_date"] = df["pub_date"].apply(convert_date)
    return df[["id", "title", "abstract", "doc_type", "pub_date", "citation_count", "field_of_research"]]


def normalize_papers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "publication_date" in df.columns and "pub_date" not in df.columns:
        df = df.rename(columns={"publication_date": "pub_date"})

    for col in ["title", "abstract", "pub_date", "citation_count", "field_of_research"]:
        if col not in df.columns:
            df[col] = "" if col != "citation_count" else 0

    df["id"] = [f"paper_{i}" for i in range(len(df))]
    df["doc_type"] = "paper"
    df["pub_date"] = df["pub_date"].apply(convert_date)
    return df[["id", "title", "abstract", "doc_type", "pub_date", "citation_count", "field_of_research"]]


def build_collection(collection_name: str) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="pub_date", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="citation_count", dtype=DataType.INT32),
        FieldSchema(name="field_of_research", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="sub_topic", dtype=DataType.VARCHAR, max_length=120),
    ]
    schema = CollectionSchema(fields=fields, description="Chunked research & patent search")

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped old collection '{collection_name}'")

    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 100},
        },
    )
    print("Created collection and IVF_FLAT vector index")
    return collection


def main() -> None:
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)

    patents_path = RAW_DIR / "patents.csv"
    papers_path = RAW_DIR / "papers.csv"

    try:
        patents_raw = pd.read_csv(patents_path, nrows=10000)
        papers_raw = pd.read_csv(papers_path, nrows=10000)
    except Exception as exc:
        print(f"CSV load failed ({patents_path}, {papers_path}): {exc}")
        sys.exit(1)

    patents_df = normalize_patents(patents_raw)
    papers_df = normalize_papers(papers_raw)
    data = pd.concat([patents_df, papers_df], ignore_index=True)

    data.fillna(
        {
            "id": "",
            "title": "",
            "abstract": "",
            "citation_count": 0,
            "field_of_research": "Unknown",
        },
        inplace=True,
    )
    data["citation_count"] = pd.to_numeric(data["citation_count"], errors="coerce").fillna(0).astype(int)
    data["field_of_research"] = data["field_of_research"].apply(standardize_field)
    data["sub_topic"] = [
        generate_subtopic(t, a) for t, a in zip(data["title"], data["abstract"])
    ]

    chunk_records: List[Dict[str, Any]] = []
    for _, row in data.iterrows():
        doc_chunks = chunk_document_record(
            doc={
                "id": row["id"],
                "title": row["title"],
                "content": row["abstract"],
                "doc_type": row["doc_type"],
                "pub_date": row["pub_date"],
                "citation_count": int(row["citation_count"]),
                "field_of_research": row["field_of_research"],
                "sub_topic": row["sub_topic"],
            },
            chunk_size=500,
            chunk_overlap=80,
        )
        chunk_records.extend(doc_chunks)

    if not chunk_records:
        print("No chunks generated from input CSVs.")
        sys.exit(1)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Generating embeddings for {len(chunk_records)} chunks...")
    texts_to_embed = [f"{rec['title']}. {rec['abstract']}" for rec in chunk_records]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    milvus_rows: List[Dict[str, Any]] = []
    for idx, rec in enumerate(chunk_records):
        milvus_rows.append(
            {
                "id": rec["id"],
                "parent_id": rec["parent_id"],
                "chunk_index": int(rec["chunk_index"]),
                "vector": embeddings[idx].tolist(),
                "title": rec["title"],
                "abstract": rec["abstract"],
                "doc_type": rec["doc_type"],
                "pub_date": rec["pub_date"],
                "citation_count": int(rec["citation_count"]),
                "field_of_research": rec["field_of_research"],
                "sub_topic": rec["sub_topic"],
            }
        )

    collection = build_collection(COLLECTION_NAME)

    batch_size = 1000
    for start in range(0, len(milvus_rows), batch_size):
        batch = milvus_rows[start : start + batch_size]
        collection.insert(batch)

    collection.flush()
    collection.load()
    print(f"Inserted and loaded {len(milvus_rows)} chunks into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()