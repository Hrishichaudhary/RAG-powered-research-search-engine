import re
from typing import Any, Dict, List, Optional


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[str]:
    """
    Split text into overlapping chunks using natural boundaries
    (paragraphs -> lines -> sentences -> words -> characters).
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    cleaned = re.sub(r"\r\n", "\n", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)
    separators = ["\n\n", "\n", ". ", " "]

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            best_split = -1
            window = cleaned[start:end]
            for sep in separators:
                idx = window.rfind(sep)
                if idx != -1 and idx > int(chunk_size * 0.4):
                    best_split = start + idx + (len(sep) if sep == ". " else 0)
                    break
            if best_split > start:
                end = best_split

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def chunk_document_record(
    doc: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Convert a single document dict into chunk records while preserving metadata.
    """
    parent_id = str(doc.get("id") or doc.get("parent_id") or "doc_0")
    title = str(doc.get("title") or "Untitled")
    content = str(
        doc.get("content")
        or doc.get("text")
        or doc.get("abstract")
        or ""
    ).strip()

    combined_text = f"{title}. {content}".strip() if content else title
    text_chunks = chunk_text(
        combined_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_records: List[Dict[str, Any]] = []
    for idx, part in enumerate(text_chunks):
        chunked_records.append(
            {
                "id": f"{parent_id}_c{idx}"[:50],
                "parent_id": parent_id,
                "chunk_index": idx,
                "title": title[:512],
                "abstract": part[:65535],
                "doc_type": str(doc.get("doc_type") or "paper")[:20],
                "pub_date": str(doc.get("pub_date") or "1970-01-01")[:20],
                "citation_count": int(doc.get("citation_count") or 0),
                "field_of_research": str(doc.get("field_of_research") or "Unknown")[:100],
                "sub_topic": str(doc.get("sub_topic") or "General")[:120],
            }
        )

    return chunked_records


def deduplicate_by_parent(
    documents: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Keep the highest-ranked chunk per parent document (based on existing order).
    Falls back to chunk 'id' if 'parent_id' is absent.
    """
    seen = set()
    deduped: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = str(doc.get("id", ""))
        parent_key = doc.get("parent_id")
        if not parent_key and "_c" in doc_id:
            parent_key = doc_id.rsplit("_c", 1)[0]
        parent_key = str(parent_key or doc_id)

        if parent_key in seen:
            continue
        seen.add(parent_key)
        deduped.append(doc)

        if limit is not None and len(deduped) >= limit:
            break

    return deduped