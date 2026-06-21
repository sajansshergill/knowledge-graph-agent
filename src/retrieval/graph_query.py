"""
graph_query.py
--------------
Neo4j graph traversal signal for the hybrid retriever.

The module returns dictionaries shaped like AlloyDB retrieval hits so the
HybridRetriever can combine graph results with BM25/vector results via RRF.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "our", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with",
}

_DOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("JIRA_KEY", re.compile(r"\b([A-Z]{2,8}-\d{1,6})\b")),
    ("ADR_REF", re.compile(r"\b((?:ADR|RFC)[-\s]?\d{1,4})\b", re.I)),
    ("GIT_SHA", re.compile(r"\b(?:commit|sha|ref)[:\s]+([0-9a-f]{7,40})\b", re.I)),
    ("SVC_NAME", re.compile(r"\b(?:service|svc|api)[:\s]+([a-z][a-z0-9-]{2,30})\b", re.I)),
    ("TEAM", re.compile(r"\b(?:([A-Z][A-Za-z\s]{1,20})\s+team|team\s+([A-Z][A-Za-z\s]{1,20}))\b")),
    ("PR_REF", re.compile(r"\bPR[:\s#]+(\d{1,6})\b", re.I)),
    ("ENV_NAME", re.compile(r"\b(prod(?:uction)?|staging|dev(?:elopment)?|canary|sandbox)\b", re.I)),
    ("EMAIL", re.compile(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)),
]


@dataclass
class GraphSearchTrace:
    """Small explanation payload for debugging graph retrieval decisions."""

    query: str
    query_type: str
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "entities": self.entities,
            "keywords": self.keywords,
            "strategies": self.strategies,
        }


class GraphQuery:
    """
    Execute graph retrieval patterns over Neo4j.

    Args:
        neo4j: Optional object exposing ``cypher_query(query, params)``.
               Tests can inject a fake loader; production lazily creates
               ``Neo4jLoader`` only when a graph query actually runs.
    """

    _LIMIT = int(os.environ.get("GRAPH_QUERY_LIMIT", 30))
    _MAX_HOPS = int(os.environ.get("GRAPH_MAX_HOPS", 2))

    def __init__(
        self,
        neo4j: Optional[Any] = None,
        limit: Optional[int] = None,
        max_hops: Optional[int] = None,
    ) -> None:
        self._neo4j = neo4j
        self._limit = limit or self._LIMIT
        self._max_hops = max(1, min(max_hops or self._MAX_HOPS, 4))
        self._last_trace: Optional[GraphSearchTrace] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_type: str = "relational",
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Return graph-ranked chunk dictionaries compatible with RRF fusion.
        """
        result_limit = limit or self._limit
        entities = self._extract_query_entities(query)
        keywords = self._keyword_terms(query)
        trace = GraphSearchTrace(
            query=query,
            query_type=query_type,
            entities=entities,
            keywords=keywords,
        )

        params = {
            "entities": entities,
            "jira_keys": [e.upper() for e in entities if re.fullmatch(r"[a-zA-Z]{2,8}-\d{1,6}", e)],
            "keywords": keywords,
            "limit": result_limit,
        }

        hits: list[dict[str, Any]] = []
        if entities:
            hits.extend(self._run_strategy("entity_lookup", self._entity_lookup_cypher(), params, trace))
            hits.extend(self._run_strategy("co_entity", self._co_entity_cypher(), params, trace))

            if query_type in ("relational", "procedural"):
                hits.extend(
                    self._run_strategy(
                        "document_relationship",
                        self._document_relationship_cypher(),
                        params,
                        trace,
                    )
                )

            if params["jira_keys"]:
                hits.extend(self._run_strategy("jira_chain", self._jira_chain_cypher(), params, trace))

        if query_type == "procedural":
            hits.extend(self._run_strategy("author_docs", self._author_docs_cypher(), params, trace))

        if not hits and keywords:
            hits.extend(self._run_strategy("keyword_fallback", self._keyword_fallback_cypher(), params, trace))

        self._last_trace = trace
        return self._merge_dedup(hits, result_limit)

    def explain(self) -> dict[str, Any]:
        """Return the most recent graph search plan and extracted hints."""
        return self._last_trace.to_dict() if self._last_trace else {}

    def close(self) -> None:
        if self._neo4j and hasattr(self._neo4j, "close"):
            self._neo4j.close()

    # ------------------------------------------------------------------
    # Cypher strategy runners
    # ------------------------------------------------------------------

    def _run_strategy(
        self,
        name: str,
        cypher: str,
        params: dict[str, Any],
        trace: GraphSearchTrace,
    ) -> list[dict[str, Any]]:
        try:
            rows = self._run(cypher, params)
        except Exception as exc:
            logger.warning("GraphQuery: %s failed: %s", name, exc)
            return []

        if rows:
            trace.strategies.append(name)
        return [self._normalize_hit(row, name, rank) for rank, row in enumerate(rows, start=1)]

    def _run(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        loader = self._get_loader()
        return loader.cypher_query(cypher, params)

    def _get_loader(self) -> Any:
        if self._neo4j is None:
            from ..storage.neo4j_loader import Neo4jLoader

            self._neo4j = Neo4jLoader()
        return self._neo4j

    # ------------------------------------------------------------------
    # Cypher templates
    # ------------------------------------------------------------------

    def _entity_lookup_cypher(self) -> str:
        return """
        MATCH (e:Entity)
        WHERE e.normalized IN $entities OR toLower(e.text) IN $entities
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
        OPTIONAL MATCH (d)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (d)-[:TAGGED_WITH]->(l:Label)
        WITH c, d, a, collect(DISTINCT l.name) AS labels, count(DISTINCT e) AS entity_matches
        RETURN
            c.chunk_id AS chunk_id,
            coalesce(c.source_type, d.source_type, '') AS source_type,
            coalesce(c.source_id, d.doc_id, '') AS source_id,
            coalesce(c.title, d.title, '') AS title,
            coalesce(c.text_preview, '') AS text,
            coalesce(a.email, d.author, '') AS author,
            coalesce(d.url, '') AS url,
            labels AS labels,
            coalesce(d.updated_at, '') AS updated_at,
            1.0 + (0.1 * entity_matches) AS score
        ORDER BY score DESC, updated_at DESC
        LIMIT $limit
        """

    def _co_entity_cypher(self) -> str:
        return """
        MATCH (e:Entity)
        WHERE e.normalized IN $entities OR toLower(e.text) IN $entities
        MATCH (:Chunk)-[:MENTIONS]->(e)<-[:MENTIONS]-(c:Chunk)
        OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
        OPTIONAL MATCH (d)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (d)-[:TAGGED_WITH]->(l:Label)
        WITH c, d, a, collect(DISTINCT l.name) AS labels, count(DISTINCT e) AS shared_entities
        RETURN
            c.chunk_id AS chunk_id,
            coalesce(c.source_type, d.source_type, '') AS source_type,
            coalesce(c.source_id, d.doc_id, '') AS source_id,
            coalesce(c.title, d.title, '') AS title,
            coalesce(c.text_preview, '') AS text,
            coalesce(a.email, d.author, '') AS author,
            coalesce(d.url, '') AS url,
            labels AS labels,
            coalesce(d.updated_at, '') AS updated_at,
            0.85 + (0.05 * shared_entities) AS score
        ORDER BY score DESC, updated_at DESC
        LIMIT $limit
        """

    def _document_relationship_cypher(self) -> str:
        max_hops = self._max_hops
        return f"""
        MATCH (e:Entity)
        WHERE e.normalized IN $entities OR toLower(e.text) IN $entities
        MATCH (seed:Chunk)-[:MENTIONS]->(e)
        MATCH (seed)-[:PART_OF]->(seedDoc:Document)
        MATCH path = (seedDoc)-[*1..{max_hops}]-(relatedDoc:Document)
        WHERE relatedDoc <> seedDoc
        MATCH (c:Chunk)-[:PART_OF]->(relatedDoc)
        OPTIONAL MATCH (relatedDoc)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (relatedDoc)-[:TAGGED_WITH]->(l:Label)
        WITH c, relatedDoc AS d, a, collect(DISTINCT l.name) AS labels, min(length(path)) AS distance
        RETURN
            c.chunk_id AS chunk_id,
            coalesce(c.source_type, d.source_type, '') AS source_type,
            coalesce(c.source_id, d.doc_id, '') AS source_id,
            coalesce(c.title, d.title, '') AS title,
            coalesce(c.text_preview, '') AS text,
            coalesce(a.email, d.author, '') AS author,
            coalesce(d.url, '') AS url,
            labels AS labels,
            coalesce(d.updated_at, '') AS updated_at,
            1.0 / (1 + distance) AS score
        ORDER BY score DESC, updated_at DESC
        LIMIT $limit
        """

    def _jira_chain_cypher(self) -> str:
        max_hops = self._max_hops
        return f"""
        MATCH (start:Document)
        WHERE start.doc_id IN $jira_keys OR toUpper(start.doc_id) IN $jira_keys
        MATCH path = (start)-[*0..{max_hops}]-(d:Document)
        MATCH (c:Chunk)-[:PART_OF]->(d)
        OPTIONAL MATCH (d)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (d)-[:TAGGED_WITH]->(l:Label)
        WITH c, d, a, collect(DISTINCT l.name) AS labels, min(length(path)) AS distance
        RETURN
            c.chunk_id AS chunk_id,
            coalesce(c.source_type, d.source_type, '') AS source_type,
            coalesce(c.source_id, d.doc_id, '') AS source_id,
            coalesce(c.title, d.title, '') AS title,
            coalesce(c.text_preview, '') AS text,
            coalesce(a.email, d.author, '') AS author,
            coalesce(d.url, '') AS url,
            labels AS labels,
            coalesce(d.updated_at, '') AS updated_at,
            1.1 / (1 + distance) AS score
        ORDER BY score DESC, updated_at DESC
        LIMIT $limit
        """

    def _author_docs_cypher(self) -> str:
        return """
        MATCH (d:Document)-[:AUTHORED_BY]->(a:Author)
        WHERE any(term IN $entities WHERE toLower(a.email) CONTAINS term)
           OR any(term IN $keywords WHERE toLower(a.email) CONTAINS term)
        MATCH (c:Chunk)-[:PART_OF]->(d)
        OPTIONAL MATCH (d)-[:TAGGED_WITH]->(l:Label)
        WITH c, d, a, collect(DISTINCT l.name) AS labels
        RETURN
            c.chunk_id AS chunk_id,
            coalesce(c.source_type, d.source_type, '') AS source_type,
            coalesce(c.source_id, d.doc_id, '') AS source_id,
            coalesce(c.title, d.title, '') AS title,
            coalesce(c.text_preview, '') AS text,
            coalesce(a.email, d.author, '') AS author,
            coalesce(d.url, '') AS url,
            labels AS labels,
            coalesce(d.updated_at, '') AS updated_at,
            0.75 AS score
        ORDER BY updated_at DESC
        LIMIT $limit
        """

    def _keyword_fallback_cypher(self) -> str:
        return """
        MATCH (c:Chunk)
        WHERE any(term IN $keywords WHERE toLower(c.title) CONTAINS term)
           OR any(term IN $keywords WHERE toLower(c.text_preview) CONTAINS term)
        OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
        OPTIONAL MATCH (d)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (d)-[:TAGGED_WITH]->(l:Label)
        WITH c, d, a, collect(DISTINCT l.name) AS labels
        RETURN
            c.chunk_id AS chunk_id,
            coalesce(c.source_type, d.source_type, '') AS source_type,
            coalesce(c.source_id, d.doc_id, '') AS source_id,
            coalesce(c.title, d.title, '') AS title,
            coalesce(c.text_preview, '') AS text,
            coalesce(a.email, d.author, '') AS author,
            coalesce(d.url, '') AS url,
            labels AS labels,
            coalesce(d.updated_at, '') AS updated_at,
            0.5 AS score
        ORDER BY updated_at DESC
        LIMIT $limit
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> list[str]:
        values: list[str] = []
        for _, pattern in _DOMAIN_PATTERNS:
            for match in pattern.finditer(query):
                value = next((g for g in match.groups() if g), match.group(0))
                values.append(value.strip().lower())

        # Quoted phrases often identify exact services, teams, or documents.
        values.extend(m.group(1).strip().lower() for m in re.finditer(r'"([^"]{2,80})"', query))

        return _ordered_unique(v for v in values if v)

    def _keyword_terms(self, query: str) -> list[str]:
        terms = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query)
            if token.lower() not in _STOPWORDS
        ]
        return _ordered_unique(terms)[:12]

    def _normalize_hit(self, row: dict[str, Any], strategy: str, rank: int) -> dict[str, Any]:
        score = row.get("score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        normalized = {
            "chunk_id": row.get("chunk_id", ""),
            "source_type": row.get("source_type", ""),
            "source_id": row.get("source_id", ""),
            "title": row.get("title", ""),
            "text": row.get("text", ""),
            "author": row.get("author", ""),
            "url": row.get("url", ""),
            "labels": row.get("labels") or [],
            "updated_at": row.get("updated_at", ""),
            "score": score,
            "graph_strategy": strategy,
            "graph_rank": rank,
        }
        return normalized

    def _merge_dedup(self, hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for hit in hits:
            chunk_id = hit.get("chunk_id")
            if not chunk_id:
                continue

            existing = merged.get(chunk_id)
            if existing is None:
                hit["graph_strategies"] = [hit.pop("graph_strategy")]
                merged[chunk_id] = hit
                continue

            existing["score"] = max(float(existing.get("score", 0.0)), float(hit.get("score", 0.0)))
            strategy = hit.get("graph_strategy")
            if strategy and strategy not in existing["graph_strategies"]:
                existing["graph_strategies"].append(strategy)

        return sorted(
            merged.values(),
            key=lambda row: (float(row.get("score", 0.0)), str(row.get("updated_at", ""))),
            reverse=True,
        )[:limit]


def _ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
