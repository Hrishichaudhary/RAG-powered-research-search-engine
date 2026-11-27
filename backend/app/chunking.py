import re
from typing import Any, Dict, List


def _find_split_index(text: str, start: int, target_end: int) -> int:
    text_len = len(text)
    if target_end >= text_len:
        return text_len

    window_start = max(start + 1, target_end - 80)
    window = text[window_start:target_end]

    # Prefer sentence boundaries near target_end
    sentence_matches = list(re.finditer(r"[.!?](?:\s+|$)", window))
    if sentence_matches:
        return window_start + sentence_matches[-1].end()

    # Otherwise snap back to the last whitespace to avoid splitting words
    last_space = text.rfind(" ", start + 1, target_end)
    if last_space > start:
        return last_space

    return target_end


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        raw_end = min(start + chunk_size, text_len)
        end = _find_split_index(cleaned, start, raw_end)
        if end <= start:
            end = raw_end

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = max(0, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _build_chunk_id(parent_id: str, idx: int, max_len: int = 50) -> str:
    suffix = f"_c{idx}"
    allowed_parent_len = max(1, max_len - len(suffix))
    return f"{parent_id[:allowed_parent_len]}{suffix}"[:max_len]


def chunk_document_record(
    doc: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Dict[str, Any]]:
    parent_id = str(doc.get("id") or "").strip()
    if not parent_id:
        raise ValueError("Document id is required")

    raw_text = str(doc.get("content") if "content" in doc else doc.get("abstract", "")).strip()
    text_chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    records: List[Dict[str, Any]] = []
    for idx, chunk_str in enumerate(text_chunks):
        records.append(
            {
                "id": _build_chunk_id(parent_id, idx, max_len=50),
                "parent_id": parent_id[:50],
                "chunk_index": idx,
                "title": str(doc.get("title") or "Untitled")[:512],
                "abstract": chunk_str[:65535],
                "doc_type": str(doc.get("doc_type") or "unknown")[:20],
                "pub_date": str(doc.get("pub_date") or "Unknown")[:20],
                "citation_count": int(doc.get("citation_count") or 0),
                "field_of_research": str(doc.get("field_of_research") or "Unknown")[:100],
                "sub_topic": str(doc.get("sub_topic") or "General")[:120],
            }
        )

    return records


def _extract_parent_id(doc: Dict[str, Any]) -> str:
    if doc.get("parent_id"):
        return str(doc["parent_id"])
    doc_id = str(doc.get("id") or "")
    if "_c" in doc_id:
        return doc_id.rsplit("_c", 1)[0]
    return doc_id


def deduplicate_by_parent(documents: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    seen_parents = set()
    deduped: List[Dict[str, Any]] = []

    for doc in documents:
        parent_key = _extract_parent_id(doc)
        if not parent_key or parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        deduped.append(doc)
        if len(deduped) >= limit:
            break

    return deduped