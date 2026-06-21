"""
neo4j_loader.py
---------------
Writes the knowledge graph to Neo4j Aura.

Node types:
    (:Document)    — one per source document (Confluence page, Jira issue,
                     Slack thread, PDF)
    (:Chunk)       — one per chunk (linked to its parent Document)
    (:Entity)      — de-duplicated named entities (PERSON, ORG, JIRA_KEY …)
    (:Author)      — people who wrote documents / comments
    (:Team)        — teams extracted by entity extractor
    (:Label)       — Confluence labels / Jira labels

Relationship types:
    (Chunk)-[:PART_OF]->(Document)
    (Document)-[:AUTHORED_BY]->(Author)
    (Document)-[:TAGGED_WITH]->(Label)
    (Chunk)-[:MENTIONS]->(Entity)
    (Entity:JIRA_KEY)-[:REFERENCES]->(Document)   cross-source link
    (Document)-[:BLOCKS]->(Document)               from Jira issue links
    (Document)-[:PARENT_OF]->(Document)            Confluence parent / Jira subtask
    (Document)-[:IN_EPIC]->(Document)              Jira epic → issue
    (Entity:PERSON)-[:OWNS]->(Document)            when author = entity person

Graph design follows the "connective tissue" pattern: the same Entity node
(e.g. "alice@company.com") appears in MENTIONS edges across Confluence,
Jira, and Slack nodes — enabling cross-source relational queries.

Env vars:
    NEO4J_URI          bolt://... or neo4j+s://...
    NEO4J_USER         default: neo4j
    NEO4J_PASSWORD     required
    NEO4J_DATABASE     default: neo4j
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# neo4j import
# ---------------------------------------------------------------------------
try:
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, TransientError
    _HAS_NEO4J = True
except ImportError:
    _HAS_NEO4J = False
    logger.warning("neo4j driver not installed — pip install neo4j")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class GraphLoadResult:
    documents_merged: int
    chunks_merged: int
    entities_merged: int
    relationships_created: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "documents_merged": self.documents_merged,
            "chunks_merged": self.chunks_merged,
            "entities_merged": self.entities_merged,
            "relationships_created": self.relationships_created,
            "duration_sec": round(self.duration_sec, 2),
        }


# ---------------------------------------------------------------------------
# Neo4j Loader
# ---------------------------------------------------------------------------

class Neo4jLoader:
    """
    Usage:
        loader = Neo4jLoader()
        loader.ensure_schema()          # run once on new DB
        result = loader.load(chunks, entity_results, raw_docs)
        print(result.to_dict())
        loader.close()
    """

    _MAX_RETRIES = 3
    _BACKOFF = 2.0
    _BATCH_SIZE = 500

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        if not _HAS_NEO4J:
            raise ImportError("neo4j package required: pip install neo4j")

        self._uri      = uri      or os.environ["NEO4J_URI"]
        self._user     = user     or os.environ.get("NEO4J_USER", "neo4j")
        self._password = password or os.environ["NEO4J_PASSWORD"]
        self._database = database or os.environ.get("NEO4J_DATABASE", "neo4j")

        self._driver: Optional[Driver] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """
        Create indexes and constraints. Idempotent — safe to call repeatedly.
        Run once after DB provisioning.
        """
        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.normalized, e.label) IS NODE KEY",
            "CREATE CONSTRAINT author_email IF NOT EXISTS FOR (a:Author) REQUIRE a.email IS UNIQUE",
            "CREATE CONSTRAINT label_name IF NOT EXISTS FOR (l:Label) REQUIRE l.name IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX doc_source_type IF NOT EXISTS FOR (d:Document) ON (d.source_type)",
            "CREATE INDEX doc_updated IF NOT EXISTS FOR (d:Document) ON (d.updated_at)",
            "CREATE INDEX chunk_source_id IF NOT EXISTS FOR (c:Chunk) ON (c.source_id)",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
            "CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)",
        ]
        with self._session() as s:
            for stmt in constraints + indexes:
                try:
                    s.run(stmt)
                except Exception as exc:
                    logger.debug("Schema stmt skipped (%s): %s", exc, stmt[:60])
        logger.info("Neo4jLoader: schema ensured")

    def load(
        self,
        chunks: list,
        entity_results: Optional[list] = None,
        raw_docs: Optional[list] = None,
    ) -> GraphLoadResult:
        """
        Main entry point.

        Args:
            chunks:          list[Chunk]
            entity_results:  list[EntityResult] (optional)
            raw_docs:        original source objects (ConfluencePage, JiraIssue…)
                             used to extract relationships like BLOCKS, IN_EPIC
        """
        start = time.time()

        # 1. Merge Document + Author + Label nodes from raw docs
        docs_merged = 0
        if raw_docs:
            docs_merged = self._merge_documents(raw_docs)

        # 2. Merge Chunk nodes + PART_OF edges
        chunks_merged = self._merge_chunks(chunks)

        # 3. Merge Entity nodes + MENTIONS edges
        entities_merged = 0
        rels_created = 0
        if entity_results:
            entities_merged, rels_created = self._merge_entities(entity_results)

        # 4. Intra-document relationships from raw docs
        if raw_docs:
            extra_rels = self._merge_doc_relationships(raw_docs)
            rels_created += extra_rels

        duration = time.time() - start
        result = GraphLoadResult(
            documents_merged=docs_merged,
            chunks_merged=chunks_merged,
            entities_merged=entities_merged,
            relationships_created=rels_created,
            duration_sec=duration,
        )
        logger.info(
            "Neo4jLoader: docs=%d chunks=%d entities=%d rels=%d (%.1fs)",
            docs_merged, chunks_merged, entities_merged, rels_created, duration,
        )
        return result

    def cypher_query(self, query: str, params: Optional[dict] = None) -> list[dict]:
        """
        Execute a read Cypher query. Used by graph_query.py in retrieval layer.
        Returns list of record dicts.
        """
        with self._session() as s:
            result = s.run(query, params or {})
            return [dict(r) for r in result]

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Private: documents
    # ------------------------------------------------------------------

    def _merge_documents(self, raw_docs: list) -> int:
        rows = []
        for doc in raw_docs:
            d = doc.to_dict() if hasattr(doc, "to_dict") else {}
            source = d.get("source", "unknown")
            doc_id = (
                d.get("page_id") or d.get("thread_ts") or
                d.get("doc_id") or d.get("key") or "unknown"
            )
            rows.append({
                "doc_id":      doc_id,
                "source_type": source,
                "title":       d.get("title", ""),
                "url":         d.get("url", ""),
                "author":      d.get("author_email") or d.get("author", "unknown"),
                "created_at":  d.get("created_at", ""),
                "updated_at":  d.get("updated_at", ""),
                "labels":      d.get("labels", []),
                "doc_type":    d.get("doc_type") or d.get("issue_type", ""),
            })

        if not rows:
            return 0

        cypher = """
        UNWIND $rows AS row
        MERGE (d:Document {doc_id: row.doc_id})
        SET
            d.source_type = row.source_type,
            d.title       = row.title,
            d.url         = row.url,
            d.created_at  = row.created_at,
            d.updated_at  = row.updated_at,
            d.doc_type    = row.doc_type

        MERGE (a:Author {email: row.author})

        MERGE (d)-[:AUTHORED_BY]->(a)

        WITH d, row
        UNWIND row.labels AS lbl
        MERGE (l:Label {name: lbl})
        MERGE (d)-[:TAGGED_WITH]->(l)
        """
        self._run_batched(cypher, rows)
        return len(rows)

    # ------------------------------------------------------------------
    # Private: chunks
    # ------------------------------------------------------------------

    def _merge_chunks(self, chunks: list) -> int:
        rows = [
            {
                "chunk_id":    c.chunk_id,
                "source_id":   c.source_id,
                "source_type": c.source_type,
                "chunk_index": c.chunk_index,
                "title":       c.title,
                "text_preview": c.text[:200],
                "token_estimate": c.token_estimate,
                "section_heading": getattr(c, "section_heading", None),
                "parent_doc_id":   getattr(c, "parent_doc_id", None),
            }
            for c in chunks
        ]

        if not rows:
            return 0

        cypher = """
        UNWIND $rows AS row
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET
            c.source_id       = row.source_id,
            c.source_type     = row.source_type,
            c.chunk_index     = row.chunk_index,
            c.title           = row.title,
            c.text_preview    = row.text_preview,
            c.token_estimate  = row.token_estimate,
            c.section_heading = row.section_heading

        WITH c, row
        MATCH (d:Document {doc_id: row.source_id})
        MERGE (c)-[:PART_OF]->(d)

        WITH c, row
        WHERE row.parent_doc_id IS NOT NULL
        MATCH (parent:Document {doc_id: row.parent_doc_id})
        MERGE (c)-[:PART_OF]->(parent)
        """
        self._run_batched(cypher, rows)
        return len(rows)

    # ------------------------------------------------------------------
    # Private: entities
    # ------------------------------------------------------------------

    def _merge_entities(self, entity_results: list) -> tuple[int, int]:
        rows = []
        for er in entity_results:
            for e in er.entities:
                rows.append({
                    "chunk_id":   er.chunk_id,
                    "text":       e.text,
                    "label":      e.label,
                    "normalized": e.normalized,
                })

        if not rows:
            return 0, 0

        cypher = """
        UNWIND $rows AS row
        MERGE (e:Entity {normalized: row.normalized, label: row.label})
        SET e.text = row.text

        WITH e, row
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MERGE (c)-[:MENTIONS]->(e)

        WITH e, row
        WHERE row.label = 'JIRA_KEY'
        MATCH (d:Document {doc_id: row.normalized})
        MERGE (e)-[:REFERENCES]->(d)
        """
        self._run_batched(cypher, rows)

        unique_entities = len({(r["normalized"], r["label"]) for r in rows})
        return unique_entities, len(rows)

    # ------------------------------------------------------------------
    # Private: document-level relationships
    # ------------------------------------------------------------------

    def _merge_doc_relationships(self, raw_docs: list) -> int:
        rels = 0
        rels += self._merge_jira_relationships(raw_docs)
        rels += self._merge_confluence_parent_relationships(raw_docs)
        return rels

    def _merge_jira_relationships(self, raw_docs: list) -> int:
        """BLOCKS, IN_EPIC from JiraIssue objects."""
        # Detect JiraIssue by duck-typing
        try:
            from ..ingestion.connectors import JiraIssue
        except ImportError:
            return 0

        rows = []
        for doc in raw_docs:
            if not isinstance(doc, JiraIssue):
                continue
            for link in doc.links:
                rows.append({
                    "from_key": doc.key,
                    "to_key":   link.target_key,
                    "rel_type": _sanitize_rel_type(link.link_type),
                })
            if doc.epic_key:
                rows.append({
                    "from_key": doc.epic_key,
                    "to_key":   doc.key,
                    "rel_type": "IN_EPIC",
                })
            if doc.parent_key and doc.parent_key != doc.epic_key:
                rows.append({
                    "from_key": doc.parent_key,
                    "to_key":   doc.key,
                    "rel_type": "PARENT_OF",
                })

        if not rows:
            return 0

        # Group by rel_type and create typed relationships
        for rel_type in {r["rel_type"] for r in rows}:
            batch = [r for r in rows if r["rel_type"] == rel_type]
            cypher = f"""
            UNWIND $rows AS row
            MATCH (a:Document {{doc_id: row.from_key}})
            MATCH (b:Document {{doc_id: row.to_key}})
            MERGE (a)-[:{rel_type}]->(b)
            """
            self._run_batched(cypher, batch)

        return len(rows)

    def _merge_confluence_parent_relationships(self, raw_docs: list) -> int:
        """PARENT_OF from ConfluencePage.parent_id."""
        try:
            from ..ingestion.connectors import ConfluencePage
        except ImportError:
            return 0

        rows = [
            {"parent_id": doc.parent_id, "child_id": doc.page_id}
            for doc in raw_docs
            if isinstance(doc, ConfluencePage) and doc.parent_id
        ]

        if not rows:
            return 0

        cypher = """
        UNWIND $rows AS row
        MATCH (parent:Document {doc_id: row.parent_id})
        MATCH (child:Document  {doc_id: row.child_id})
        MERGE (parent)-[:PARENT_OF]->(child)
        """
        self._run_batched(cypher, rows)
        return len(rows)

    # ------------------------------------------------------------------
    # Private: execution helpers
    # ------------------------------------------------------------------

    def _run_batched(self, cypher: str, rows: list) -> None:
        for batch in _batchify(rows, self._BATCH_SIZE):
            self._run_with_retry(cypher, {"rows": batch})

    def _run_with_retry(self, cypher: str, params: dict) -> None:
        for attempt in range(self._MAX_RETRIES):
            try:
                with self._session() as s:
                    s.run(cypher, params)
                return
            except (ServiceUnavailable, TransientError) as exc:
                wait = self._BACKOFF ** attempt
                logger.warning(
                    "Neo4jLoader: attempt %d failed (%s) — retrying in %.1fs",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
            except Exception as exc:
                logger.error("Neo4jLoader: Cypher error — %s\n%s", exc, cypher[:200])
                raise

    def _session(self) -> Session:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=10,
            )
        return self._driver.session(database=self._database)

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_rel_type(raw: str) -> str:
    """Convert Jira link type string to valid Cypher relationship type."""
    import re
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", raw.upper().strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "RELATES_TO"


def _batchify(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i: i + size]


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    loader = Neo4jLoader()
    loader.ensure_schema()

    # Verify connectivity with a simple query
    result = loader.cypher_query("MATCH (n) RETURN COUNT(n) AS node_count")
    print(f"Total nodes in graph: {result[0]['node_count']}")

    # Example relational query — "who owns the auth service?"
    result = loader.cypher_query("""
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {label: 'SVC_NAME', normalized: 'auth-service'})
              <-[:MENTIONS]-(c2:Chunk)-[:PART_OF]->(d:Document)-[:AUTHORED_BY]->(a:Author)
        RETURN DISTINCT a.email AS owner, d.title AS doc_title
        LIMIT 5
    """)
    print(json.dumps(result, indent=2))

    loader.close()