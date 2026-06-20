"""
alloydb_loader.py
-----------------
Writes Chunk objects + their Vertex AI embeddings to AlloyDB (pgvector).
Also writes EntityResult rows to the entities table.

Two write paths:
    upsert_chunks()   — primary path; calls Vertex AI text-embedding-004,
                        then bulk-upserts into chunks table
    upsert_entities() — companion path; bulk-inserts entity rows

Embedding strategy:
    - Batches chunks into groups of 250 (Vertex AI batch limit)
    - Retries on transient API errors with exponential back-off
    - Skips re-embedding if chunk text_hash already exists in DB
      (idempotent re-ingestion)

Env vars required:
    ALLOYDB_URI           postgresql+asyncpg://user:pass@host:5432/dbname
                          OR standard psycopg2 DSN
    GOOGLE_CLOUD_PROJECT  GCP project for Vertex AI
    VERTEX_LOCATION       e.g. us-central1 (default)

Optional:
    EMBEDDING_MODEL       default: text-embedding-004
    EMBEDDING_BATCH_SIZE  default: 250
    ALLOYDB_POOL_SIZE     default: 10
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Vertex AI import
# ---------------------------------------------------------------------------
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    _HAS_VERTEX = True
except ImportError:
    _HAS_VERTEX = False
    logger.warning(
        "google-cloud-aiplatform not installed — "
        "embeddings will be zeroed. Install: pip install google-cloud-aiplatform"
    )

_EMBEDDING_DIM = 768        # text-embedding-004


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    chunks_upserted: int
    chunks_skipped: int     # already exist with same text_hash
    entities_inserted: int
    embedding_calls: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "chunks_upserted": self.chunks_upserted,
            "chunks_skipped": self.chunks_skipped,
            "entities_inserted": self.entities_inserted,
            "embedding_calls": self.embedding_calls,
            "duration_sec": round(self.duration_sec, 2),
        }


# ---------------------------------------------------------------------------
# AlloyDB Loader
# ---------------------------------------------------------------------------

class AlloyDBLoader:
    """
    Usage:
        loader = AlloyDBLoader()
        result = loader.upsert_chunks(chunks, entity_results)
        print(result.to_dict())
    """

    _EMBED_BATCH = int(os.environ.get("EMBEDDING_BATCH_SIZE", 250))
    _EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-004")
    _POOL_SIZE   = int(os.environ.get("ALLOYDB_POOL_SIZE", 10))
    _MAX_RETRIES = 3
    _BACKOFF_BASE = 2.0

    def __init__(
        self,
        dsn: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self._dsn = dsn or os.environ["ALLOYDB_URI"]
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location or os.environ.get("VERTEX_LOCATION", "us-central1")

        self._conn: Optional[psycopg2.extensions.connection] = None
        self._embed_model: Optional[TextEmbeddingModel] = None

        if _HAS_VERTEX and self._project:
            self._init_vertex()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list,
        entity_results: Optional[list] = None,
        skip_existing: bool = True,
    ) -> LoadResult:
        """
        Main entry point. Embeds chunks and upserts into AlloyDB.

        Args:
            chunks:          list[Chunk] from chunker
            entity_results:  list[EntityResult] from entity_extractor (optional)
            skip_existing:   skip re-embedding chunks with known text_hash
        """
        start = time.time()
        conn = self._get_connection()

        # --- 1. Filter already-ingested chunks (idempotency) ---
        if skip_existing:
            chunks_to_embed, skipped = self._filter_existing(conn, chunks)
        else:
            chunks_to_embed = chunks
            skipped = 0

        # --- 2. Embed in batches ---
        embed_calls = 0
        embeddings: dict[str, list[float]] = {}

        for batch in _batchify(chunks_to_embed, self._EMBED_BATCH):
            batch_embeddings = self._embed_batch([c.text for c in batch])
            embed_calls += 1
            for chunk, emb in zip(batch, batch_embeddings):
                embeddings[chunk.chunk_id] = emb

        # --- 3. Upsert chunks ---
        upserted = self._upsert_chunk_rows(conn, chunks_to_embed, embeddings)

        # --- 4. Upsert entities ---
        entity_count = 0
        if entity_results:
            entity_count = self._insert_entity_rows(conn, entity_results)

        conn.commit()

        duration = time.time() - start
        result = LoadResult(
            chunks_upserted=upserted,
            chunks_skipped=skipped,
            entities_inserted=entity_count,
            embedding_calls=embed_calls,
            duration_sec=duration,
        )

        logger.info(
            "AlloyDBLoader: upserted=%d skipped=%d entities=%d embeds=%d (%.1fs)",
            upserted, skipped, entity_count, embed_calls, duration,
        )
        return result

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 10,
        source_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Pure vector search — used by hybrid_retriever as one signal.
        Returns top-k chunks sorted by cosine similarity.
        """
        conn = self._get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            filter_clause = "AND source_type = %s" if source_type else ""
            params: list = [f"[{','.join(str(x) for x in query_embedding)}]", k]
            if source_type:
                params = [f"[{','.join(str(x) for x in query_embedding)}]",
                          source_type, k]

            sql = f"""
                SELECT
                    chunk_id, source_type, source_id, title, text,
                    author, url, labels, updated_at,
                    1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE embedding IS NOT NULL
                {filter_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            # Adjust params for no-filter case
            if not source_type:
                emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
                cur.execute(sql.replace("AND source_type = %s", ""),
                            [emb_str, emb_str, k])
            else:
                emb_str = f"[{','.join(str(x) for x in query_embedding)}]"
                cur.execute(
                    """
                    SELECT
                        chunk_id, source_type, source_id, title, text,
                        author, url, labels, updated_at,
                        1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE embedding IS NOT NULL AND source_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [emb_str, source_type, emb_str, k],
                )
            return [dict(row) for row in cur.fetchall()]

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 10,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        source_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Calls the hybrid_search() SQL function defined in schema.sql.
        Returns ranked list of chunks with combined_score.
        """
        conn = self._get_connection()
        emb_str = f"[{','.join(str(x) for x in query_embedding)}]"

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM hybrid_search(
                    %s, %s::vector, %s, %s, %s, %s
                )
                """,
                [
                    query_text,
                    emb_str,
                    k,
                    bm25_weight,
                    vector_weight,
                    source_type,
                ],
            )
            return [dict(row) for row in cur.fetchall()]

    def embed_query(self, query_text: str) -> list[float]:
        """Embed a single query string. Used by retrieval agents."""
        results = self._embed_batch([query_text])
        return results[0] if results else [0.0] * _EMBEDDING_DIM

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Private: idempotency filter
    # ------------------------------------------------------------------

    def _filter_existing(
        self, conn, chunks: list
    ) -> tuple[list, int]:
        """Return (chunks_needing_embed, skipped_count)."""
        if not chunks:
            return [], 0

        hashes = {_text_hash(c.text): c for c in chunks}

        with conn.cursor() as cur:
            cur.execute(
                "SELECT text_hash FROM chunks WHERE text_hash = ANY(%s)",
                [list(hashes.keys())],
            )
            existing_hashes = {row[0] for row in cur.fetchall()}

        to_embed = [c for c in chunks if _text_hash(c.text) not in existing_hashes]
        skipped  = len(chunks) - len(to_embed)
        return to_embed, skipped

    # ------------------------------------------------------------------
    # Private: upsert rows
    # ------------------------------------------------------------------

    def _upsert_chunk_rows(
        self,
        conn,
        chunks: list,
        embeddings: dict[str, list[float]],
    ) -> int:
        if not chunks:
            return 0

        rows = []
        for c in chunks:
            emb = embeddings.get(c.chunk_id, [0.0] * _EMBEDDING_DIM)
            emb_str = f"[{','.join(str(x) for x in emb)}]"
            rows.append((
                c.chunk_id,
                c.chunk_index,
                c.source_id,
                c.source_type,
                c.text,
                c.token_estimate,
                c.char_count,
                c.title,
                c.author,
                c.url,
                getattr(c, "section_heading", None),
                getattr(c, "parent_doc_id", None),
                c.labels,
                c.created_at,
                c.updated_at,
                emb_str,
                _text_hash(c.text),
            ))

        sql = """
            INSERT INTO chunks (
                chunk_id, chunk_index, source_id, source_type,
                text, token_estimate, char_count,
                title, author, url, section_heading, parent_doc_id,
                labels, created_at, updated_at, embedding, text_hash
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                text            = EXCLUDED.text,
                token_estimate  = EXCLUDED.token_estimate,
                char_count      = EXCLUDED.char_count,
                title           = EXCLUDED.title,
                author          = EXCLUDED.author,
                url             = EXCLUDED.url,
                section_heading = EXCLUDED.section_heading,
                parent_doc_id   = EXCLUDED.parent_doc_id,
                labels          = EXCLUDED.labels,
                updated_at      = EXCLUDED.updated_at,
                embedding       = EXCLUDED.embedding,
                text_hash       = EXCLUDED.text_hash,
                ingested_at     = NOW()
        """

        with conn.cursor() as cur:
            execute_values(cur, sql, rows, template="""(
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s::vector, %s
            )""")

        return len(rows)

    def _insert_entity_rows(self, conn, entity_results: list) -> int:
        rows = []
        for er in entity_results:
            for e in er.entities:
                rows.append((
                    er.chunk_id,
                    er.source_id,
                    e.text,
                    e.label,
                    e.normalized,
                    getattr(e, "start", None),
                    getattr(e, "end", None),
                ))

        if not rows:
            return 0

        sql = """
            INSERT INTO entities
                (chunk_id, source_id, entity_text, entity_label,
                 normalized, char_start, char_end)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)

        return len(rows)

    # ------------------------------------------------------------------
    # Private: Vertex AI embedding
    # ------------------------------------------------------------------

    def _init_vertex(self) -> None:
        try:
            vertexai.init(project=self._project, location=self._location)
            self._embed_model = TextEmbeddingModel.from_pretrained(self._EMBED_MODEL)
            logger.info("AlloyDBLoader: Vertex AI initialized — model=%s", self._EMBED_MODEL)
        except Exception as exc:
            logger.warning("AlloyDBLoader: Vertex AI init failed — %s", exc)
            self._embed_model = None

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Call Vertex AI text-embedding-004 for a batch of texts.
        Falls back to zero vectors on error.
        """
        if not self._embed_model or not texts:
            return [[0.0] * _EMBEDDING_DIM for _ in texts]

        inputs = [
            TextEmbeddingInput(text=t[:8192], task_type="RETRIEVAL_DOCUMENT")
            for t in texts
        ]

        for attempt in range(self._MAX_RETRIES):
            try:
                response = self._embed_model.get_embeddings(inputs)
                return [r.values for r in response]
            except Exception as exc:
                wait = self._BACKOFF_BASE ** attempt
                logger.warning(
                    "AlloyDBLoader: embedding attempt %d failed (%s) — retrying in %.1fs",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)

        logger.error("AlloyDBLoader: embedding failed after %d retries", self._MAX_RETRIES)
        return [[0.0] * _EMBEDDING_DIM for _ in texts]

    # ------------------------------------------------------------------
    # Private: connection
    # ------------------------------------------------------------------

    def _get_connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                self._dsn,
                connect_timeout=10,
                options="-c statement_timeout=30000",
            )
            self._conn.autocommit = False
            logger.info("AlloyDBLoader: connected to AlloyDB")
        return self._conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    import unicodedata, re
    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def _batchify(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i: i + size]


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    loader = AlloyDBLoader()

    # Test connection
    conn = loader._get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
    print(f"chunks table row count: {count}")

    # Test embed (if Vertex available)
    test_emb = loader.embed_query("What is the auth service architecture?")
    print(f"Embedding dim: {len(test_emb)}, first 5 values: {test_emb[:5]}")

    loader.close()