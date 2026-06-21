"""
retrieval/
----------
Three-signal hybrid retrieval layer for EKGA.

    HybridRetriever  — fuses BM25 + vector + graph via RRF
    GraphQuery       — Cypher traversal patterns (entity lookup,
                       path traversal, co-entity, author-centric)
    QueryRewriter    — self-reflection rewrite (acronym expansion,
                       entity restatement, LLM fallback)

Typical usage from agents:
    from src.retrieval import HybridRetriever

    retriever = HybridRetriever()
    result = retriever.retrieve_with_reflection(
        query="Why was auth migrated off Postgres?",
        query_type="relational",
        top_k=10,
    )
"""

__all__ = [
    "HybridRetriever",
    "RetrievalResult",
    "RetrievedChunk",
    "GraphQuery",
    "QueryRewriter",
]


def __getattr__(name: str):
    if name in {"HybridRetriever", "RetrievalResult", "RetrievedChunk"}:
        from .hybrid_retriever import HybridRetriever, RetrievalResult, RetrievedChunk

        values = {
            "HybridRetriever": HybridRetriever,
            "RetrievalResult": RetrievalResult,
            "RetrievedChunk": RetrievedChunk,
        }
        return values[name]

    if name == "GraphQuery":
        from .graph_query import GraphQuery

        return GraphQuery

    if name == "QueryRewriter":
        from .query_rewriter import QueryRewriter

        return QueryRewriter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")