import os
from typing import Any, Dict, List

import requests

DEFAULT_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{DEFAULT_OLLAMA_BASE}/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))


def generate_rag_summary(query: str, documents: List[Dict[str, Any]]) -> str:
    """Generates AI insight summary from retrieved documents."""
    context_docs = []
    for doc in documents[:5]:
        text = f"Title: {doc['title']}\nAbstract: {doc['abstract']}"
        context_docs.append(text)

    context = "\n\n".join(context_docs)

    prompt = f"""
You are an AI research assistant.

User question:
{query}

Relevant research documents:
{context}

Tasks:
1. Summarize the key insights
2. Identify emerging research themes
3. Highlight interesting innovations

Answer clearly using bullet points.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    return str(result.get("response") or "").strip()


def generate_document_summary(
    title: str,
    abstract: str,
    doc_type: str = "paper",
    pub_date: str = "",
) -> str:
    """Generates a concise summary for a single paper or patent."""
    clean_abstract = (abstract or "").strip()
    if not clean_abstract:
        return "No abstract available to summarize."

    prompt = f"""
Summarize this {doc_type} clearly and concisely using only the provided text. Do not add outside facts.

Title: {title or "Untitled"}
Date: {pub_date or "Unknown"}
Abstract: {clean_abstract}

Provide:
- Core problem / objective
- Key method or finding
- Main takeaway
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    return str(result.get("response") or "").strip()