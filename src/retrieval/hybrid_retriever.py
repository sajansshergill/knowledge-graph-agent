"""
hybrid_retriever.py
-------------------
Fuses three retrieval signals into a single ranked result list:

    1. BM25 (keyword)    — AlloyDB full-text search via tsvector
    2. Dense vector      — AlloyDB pgvector cosine similarity
    3. Graph traversal   — Neo4j Cypher path queries

Signal fusion strategy: Reciprocal Rank Fusion (RRF)
    score(d) = Σ  1 / (k + rank_i(d))
    where k=60 (standard constant), rank_i is position in signal i.

RRF is used instead of raw score combination because:
  - BM25 scores (unbounded float) and cosine scores (0–1) are on
    different scales — normalization is lossy and dataset-dependent
  - RRF is robust to missing signals (graph may return 0 results
    for factual queries; BM25 may miss semantic matches)

Retrieval flow:
    query_text
        ├── embed_query()       → query_embedding
        ├── bm25_search()       → BM25 hits   (AlloyDB)
        ├── vector_search()     → vector hits  (AlloyDB)
        └── graph_search()      → graph hits   (Neo4j)
                ↓
            rrf_fuse()
                ↓
            RetrievalResult (top-k chunks with provenance)

Env vars:
    RETRIEVAL_TOP_K         default 10
    RETRIEVAL_CANDIDATE_K   candidates per signal before fusion (default 30)
    RRF_K                   RRF constant (default 60)
    RETRIEVAL_BM25_WEIGHT   weight for BM25 signal in logging (default 0.3)
    RETRIEVAL_VECTOR_WEIGHT weight for vector signal in logging (default 0.5)
    RETRIEVAL_GRAPH_WEIGHT  weight for graph signal in logging (default 0.2)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ..storage.alloydb_loader import AlloyDBLoader
from .graph_query import GraphQuery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    source_type: str
    source_id: str
    title: str
    text: str
    author: str
    url: str
    labels: list[str]
    updated_at: str

    # Signal scores (None if signal didn't return this chunk)
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    rrf_score: float = 0.0

    # Which signals contributed
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "source_type": self.source_type,
            "source_id":   self.source_id,
            "title":       self.title,
            "text":        self.text,
            "author":      self.author,
            "url":         self.url,
            "labels":      self.labels,
            "updated_at":  str(self.updated_at),
            "rrf_score":   round(self.rrf_score, 6),
            "signals":     self.signals,
            "bm25_score":  self.bm25_score,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
        }


@dataclass
class RetrievalResult:
    query: str
    query_type: str                     # factual | relational | procedural
    chunks: list[RetrievedChunk]
    retrieval_path: str                 # bm25 | vector | graph | hybrid
    latency_ms: int
    candidate_counts: dict[str, int]    # {signal: n_candidates}

    def to_dict(self) -> dict:
        return {
            "query":            self.query,
            "query_type":       self.query_type,
            "chunk_count":      len(self.chunks),
            "retrieval_path":   self.retrieval_path,
            "latency_ms":       self.latency_ms,
            "candidate_counts": self.candidate_counts,
            "chunks":           [c.to_dict() for c in self.chunks],
        }


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Usage:
        retriever = HybridRetriever(alloydb=loader, graph=graph_query)
        result = retriever.retrieve(
            query="Why was the auth service migrated off Postgres?",
            query_type="relational",
            top_k=10,
        )
        for chunk in result.chunks:
            print(chunk.rrf_score, chunk.title)
    """

    _TOP_K      = int(os.environ.get("RETRIEVAL_TOP_K", 10))
    _CAND_K     = int(os.environ.get("RETRIEVAL_CANDIDATE_K", 30))
    _RRF_K      = int(os.environ.get("RRF_K", 60))

    def __init__(
        self,
        alloydb: Optional[AlloyDBLoader] = None,
        graph: Optional[GraphQuery] = None,
    ) -> None:
        self._db    = alloydb or AlloyDBLoader()
        self._graph = graph   or GraphQuery()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        query_type: str = "factual",
        top_k: int = _TOP_K,
        source_filter: Optional[str] = None,
        disable_graph: bool = False,
    ) -> RetrievalResult:
        """
        Main retrieval entry point called by the Retrieval Agent.

        Args:
            query:         natural language query string
            query_type:    "factual" | "relational" | "procedural"
            top_k:         number of chunks to return
            source_filter: restrict to one source type (optional)
            disable_graph: skip graph traversal (e.g. for simple factual queries)
        """
        t0 = time.time()

        # 1. Embed query once — shared across vector + graph signals
        query_embedding = self._db.embed_query(query)

        # 2. Run all three signals in parallel (sequential for simplicity;
        #    swap to asyncio.gather or ThreadPoolExecutor for production)
        bm25_hits   = self._bm25_search(query, source_filter)
        vector_hits = self._vector_search(query_embedding, source_filter)
        graph_hits  = (
            self._graph_search(query, query_type, query_embedding)
            if not disable_graph and query_type in ("relational", "procedural")
            else []
        )

        candidate_counts = {
            "bm25":   len(bm25_hits),
            "vector": len(vector_hits),
            "graph":  len(graph_hits),
        }

        # 3. Fuse via RRF
        fused = self._rrf_fuse(bm25_hits, vector_hits, graph_hits)

        # 4. Apply source filter (if not already applied at DB level)
        if source_filter:
            fused = [c for c in fused if c.source_type == source_filter]

        top_chunks = fused[:top_k]

        # 5. Determine retrieval path label
        active = [k for k, v in candidate_counts.items() if v > 0]
        path = "hybrid" if len(active) > 1 else (active[0] if active else "none")

        latency_ms = int((time.time() - t0) * 1000)

        logger.info(
            "HybridRetriever: query_type=%s path=%s candidates=%s latency=%dms top_k=%d",
            query_type, path, candidate_counts, latency_ms, len(top_chunks),
        )

        return RetrievalResult(
            query=query,
            query_type=query_type,
            chunks=top_chunks,
            retrieval_path=path,
            latency_ms=latency_ms,
            candidate_counts=candidate_counts,
        )

    def retrieve_with_reflection(
        self,
        query: str,
        query_type: str = "factual",
        top_k: int = _TOP_K,
        min_chunks: int = 3,
        confidence_threshold: float = 0.05,
    ) -> RetrievalResult:
        """
        Self-reflection wrapper: if initial retrieval returns fewer than
        min_chunks above confidence_threshold, rewrites the query and retries.
        Used by the Retrieval Agent's self-reflection pattern.
        """
        result = self.retrieve(query, query_type, top_k)

        high_conf = [c for c in result.chunks if c.rrf_score >= confidence_threshold]

        if len(high_conf) < min_chunks:
            logger.info(
                "HybridRetriever: low confidence (found %d/%d) — triggering rewrite",
                len(high_conf), min_chunks,
            )
            from .query_rewriter import QueryRewriter
            rewriter = QueryRewriter()
            rewritten = rewriter.rewrite(query, query_type, result)

            if rewritten and rewritten != query:
                logger.info("HybridRetriever: retrying with rewritten query: %r", rewritten)
                result = self.retrieve(rewritten, query_type, top_k)
                result.query = f"{query} [rewritten: {rewritten}]"

        return result

    # ------------------------------------------------------------------
    # Private: signal runners
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        source_filter: Optional[str],
    ) -> list[dict]:
        """Full-text BM25 via AlloyDB tsvector."""
        try:
            with self._db._get_connection().cursor(
                cursor_factory=__import__("psycopg2.extras", fromlist=["RealDictCursor"]).RealDictCursor
            ) as cur:
                filter_clause = "AND source_type = %s" if source_filter else ""
                params = [query, self._CAND_K]
                if source_filter:
                    params = [query, source_filter, self._CAND_K]

                sql = f"""
                    SELECT
                        chunk_id, source_type, source_id, title, text,
                        author, url, labels, updated_at,
                        ts_rank_cd(ts_content, plainto_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE ts_content @@ plainto_tsquery('english', %s)
                    {filter_clause}
                    ORDER BY score DESC
                    LIMIT %s
                """
                # Duplicate query param for WHERE clause
                if source_filter:
                    cur.execute(sql, [query, query, source_filter, self._CAND_K])
                else:
                    cur.execute(sql, [query, query, self._CAND_K])

                return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            logger.warning("HybridRetriever: BM25 failed — %s", exc)
            return []

    def _vector_search(
        self,
        query_embedding: list[float],
        source_filter: Optional[str],
    ) -> list[dict]:
        """Dense vector search via pgvector HNSW."""
        try:
            return self._db.similarity_search(
                query_embedding,
                k=self._CAND_K,
                source_type=source_filter,
            )
        except Exception as exc:
            logger.warning("HybridRetriever: vector search failed — %s", exc)
            return []

    def _graph_search(
        self,
        query: str,
        query_type: str,
        query_embedding: list[float],
    ) -> list[dict]:
        """
        Neo4j graph traversal — extracts entity hints from query,
        finds related chunks via graph paths.
        Returns chunks in dict form compatible with RRF fuser.
        """
        try:
            return self._graph.search(query, query_type)
        except Exception as exc:
            logger.warning("HybridRetriever: graph search failed — %s", exc)
            return []

    # ------------------------------------------------------------------
    # Private: RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        bm25_hits: list[dict],
        vector_hits: list[dict],
        graph_hits: list[dict],
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion across three ranked lists.
        score(d) = Σ 1 / (k + rank_i(d))

        Deduplicates by chunk_id; merges score provenance.
        """
        k = self._RRF_K
        chunks: dict[str, RetrievedChunk] = {}

        signal_lists = [
            ("bm25",   bm25_hits),
            ("vector", vector_hits),
            ("graph",  graph_hits),
        ]

        for signal_name, hits in signal_lists:
            for rank, hit in enumerate(hits, start=1):
                cid = hit.get("chunk_id", "")
                if not cid:
                    continue

                rrf_contribution = 1.0 / (k + rank)

                if cid not in chunks:
                    chunks[cid] = RetrievedChunk(
                        chunk_id=cid,
                        source_type=hit.get("source_type", ""),
                        source_id=hit.get("source_id", ""),
                        title=hit.get("title", ""),
                        text=hit.get("text", ""),
                        author=hit.get("author", ""),
                        url=hit.get("url", ""),
                        labels=hit.get("labels") or [],
                        updated_at=hit.get("updated_at", ""),
                        rrf_score=rrf_contribution,
                        signals=[signal_name],
                    )
                else:
                    chunks[cid].rrf_score += rrf_contribution
                    if signal_name not in chunks[cid].signals:
                        chunks[cid].signals.append(signal_name)

                # Attach per-signal scores
                chunk = chunks[cid]
                raw_score = hit.get("score") or hit.get("combined_score", 0.0)
                if signal_name == "bm25":
                    chunk.bm25_score = float(raw_score)
                elif signal_name == "vector":
                    chunk.vector_score = float(raw_score)
                elif signal_name == "graph":
                    chunk.graph_score = float(raw_score)

        return sorted(chunks.values(), key=lambda c: c.rrf_score, reverse=True)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    retriever = HybridRetriever()

    queries = [
        ("Why was the auth service migrated off PostgreSQL?",       "relational"),
        ("What does onboarding for a backend engineer look like?",  "procedural"),
        ("Which team owns the rate-limiting module?",               "relational"),
        ("What is the JWT token lifetime?",                         "factual"),
    ]

    for query, qtype in queries:
        print(f"\n{'='*60}")
        print(f"Query ({qtype}): {query}")
        result = retriever.retrieve(query, query_type=qtype, top_k=5)
        print(f"Path: {result.retrieval_path} | Latency: {result.latency_ms}ms")
        print(f"Candidates: {result.candidate_counts}")
        for i, chunk in enumerate(result.chunks, 1):
            print(f"  {i}. [{chunk.source_type}] {chunk.title[:60]} "
                  f"(rrf={chunk.rrf_score:.4f}, signals={chunk.signals})")