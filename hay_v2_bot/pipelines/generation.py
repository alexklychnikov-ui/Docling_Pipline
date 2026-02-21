from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever


def get_context_for_user(retriever: PineconeEmbeddingRetriever, user_id: str, query_embedding: list, top_k: int = 15, logger=None):
    """Достаёт релевантный контекст (диалог + чанки документов) по user_id и эмбеддингу запроса."""
    if logger:
        logger(f"[retrieve] user_id={user_id} query_embedding present: {query_embedding is not None}")
    if query_embedding is None:
        return ""
    filters = {"field": "user_id", "operator": "==", "value": str(user_id)}
    docs = retriever.run(query_embedding=query_embedding, filters=filters, top_k=top_k)
    documents = docs.get("documents") or []
    if logger:
        logger(f"[retrieve] user_id={user_id} found {len(documents)} docs (top_k={top_k})")
    if not documents:
        return ""
    return "\n".join(d.content for d in documents)
