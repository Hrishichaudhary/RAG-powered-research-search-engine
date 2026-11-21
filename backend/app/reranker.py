from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, documents):
    if not documents:
        return []

    pairs = [
        (query, f"{doc.get('title', '')} {doc.get('abstract', '')}".strip())
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, score in ranked_docs]