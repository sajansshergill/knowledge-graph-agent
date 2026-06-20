-- schema.sql
-- ---------------------------------------------------------------------------
-- AlloyDB (PostgreSQL-compatible) schema for EKGA vector + metadata store.
--
-- Run once against your AlloyDB instance:
--   psql $ALLOYDB_URI -f schema.sql
--
-- Extensions required:
--   pgvector  — vector similarity search (HNSW index)
--   uuid-ossp — UUID generation
--   pg_trgm   — trigram index for BM25-style keyword search
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- chunks — primary vector store
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    -- Identity
    chunk_id        TEXT        PRIMARY KEY,
    chunk_index     INTEGER     NOT NULL,
    source_id       TEXT        NOT NULL,
    source_type     TEXT        NOT NULL CHECK (source_type IN ('confluence', 'slack', 'pdf', 'jira')),

    -- Content
    text            TEXT        NOT NULL,
    token_estimate  INTEGER     NOT NULL DEFAULT 0,
    char_count      INTEGER     NOT NULL DEFAULT 0,

    -- Provenance
    title           TEXT        NOT NULL DEFAULT '',
    author          TEXT        NOT NULL DEFAULT 'unknown',
    url             TEXT        NOT NULL DEFAULT '',
    section_heading TEXT,
    parent_doc_id   TEXT,                           -- Jira comment → issue key
    labels          TEXT[]      NOT NULL DEFAULT '{}',

    -- Dates
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Embedding (Vertex AI text-embedding-004 = 768 dims)
    embedding       VECTOR(768),

    -- Dedup fingerprint
    text_hash       TEXT        NOT NULL DEFAULT '',

    -- BM25 search column (populated by trigger)
    ts_content      TSVECTOR
                        GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))
                        ) STORED
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- HNSW vector index (cosine distance) — primary semantic search path
-- m=16, ef_construction=64 is a good starting point for <1M chunks
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- BM25 / full-text search via GIN on tsvector
CREATE INDEX IF NOT EXISTS idx_chunks_ts_content
    ON chunks USING GIN (ts_content);

-- Trigram index on raw text for ILIKE / fuzzy matching
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
    ON chunks USING GIN (text gin_trgm_ops);

-- Filtered queries by source type (common in retrieval agent)
CREATE INDEX IF NOT EXISTS idx_chunks_source_type
    ON chunks (source_type);

-- Recency filter
CREATE INDEX IF NOT EXISTS idx_chunks_updated_at
    ON chunks (updated_at DESC);

-- Dedup lookups
CREATE INDEX IF NOT EXISTS idx_chunks_text_hash
    ON chunks (text_hash)
    WHERE text_hash <> '';

-- Source document lookup (Neo4j cross-reference)
CREATE INDEX IF NOT EXISTS idx_chunks_source_id
    ON chunks (source_id);

-- ---------------------------------------------------------------------------
-- entities — extracted named entities per chunk
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id              BIGSERIAL   PRIMARY KEY,
    chunk_id        TEXT        NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    source_id       TEXT        NOT NULL,
    entity_text     TEXT        NOT NULL,
    entity_label    TEXT        NOT NULL,    -- PERSON | ORG | JIRA_KEY | etc.
    normalized      TEXT        NOT NULL,
    char_start      INTEGER,
    char_end        INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entities_chunk_id
    ON entities (chunk_id);

CREATE INDEX IF NOT EXISTS idx_entities_label
    ON entities (entity_label);

CREATE INDEX IF NOT EXISTS idx_entities_normalized
    ON entities (normalized);

-- Composite: find all chunks mentioning a specific entity
CREATE INDEX IF NOT EXISTS idx_entities_normalized_label
    ON entities (normalized, entity_label);

-- ---------------------------------------------------------------------------
-- ingestion_runs — audit log for each pipeline execution
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    source_type     TEXT,                           -- NULL = all sources
    chunks_ingested INTEGER     NOT NULL DEFAULT 0,
    chunks_skipped  INTEGER     NOT NULL DEFAULT 0,
    chunks_duped    INTEGER     NOT NULL DEFAULT 0,
    entities_found  INTEGER     NOT NULL DEFAULT 0,
    status          TEXT        NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'success', 'failed')),
    error_message   TEXT,
    metadata        JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started
    ON ingestion_runs (started_at DESC);

-- ---------------------------------------------------------------------------
-- eval_traces — per-query LLM-native observability (written by trace_logger)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_traces (
    trace_id        UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text      TEXT        NOT NULL,
    query_type      TEXT,                           -- factual | relational | procedural
    session_id      TEXT,

    -- Agent hop metrics (one row per full query, not per hop)
    total_tokens_in     INTEGER NOT NULL DEFAULT 0,
    total_tokens_out    INTEGER NOT NULL DEFAULT 0,
    total_latency_ms    INTEGER NOT NULL DEFAULT 0,
    total_cost_usd      NUMERIC(10, 6) NOT NULL DEFAULT 0,

    -- Per-hop breakdown stored as JSONB array
    -- [{agent, tokens_in, tokens_out, latency_ms, cost_usd}, ...]
    agent_hops      JSONB       NOT NULL DEFAULT '[]',

    -- Retrieval metadata
    chunks_retrieved    INTEGER NOT NULL DEFAULT 0,
    retrieval_path      TEXT,                       -- bm25 | vector | graph | hybrid

    -- Eval scores (RAGAS)
    faithfulness        NUMERIC(4, 3),
    answer_relevancy    NUMERIC(4, 3),
    context_recall      NUMERIC(4, 3),

    -- Output
    answer_text     TEXT,
    source_chunk_ids TEXT[]  NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_traces_created
    ON eval_traces (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_eval_traces_query_type
    ON eval_traces (query_type);

CREATE INDEX IF NOT EXISTS idx_eval_traces_faithfulness
    ON eval_traces (faithfulness)
    WHERE faithfulness IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Materialized view: cost summary by day (refreshed by observability pipeline)
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_cost_summary AS
SELECT
    DATE_TRUNC('day', created_at)   AS day,
    query_type,
    COUNT(*)                        AS query_count,
    SUM(total_tokens_in)            AS tokens_in,
    SUM(total_tokens_out)           AS tokens_out,
    SUM(total_cost_usd)             AS total_cost_usd,
    AVG(total_cost_usd)             AS avg_cost_per_query,
    PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY total_latency_ms)    AS p50_latency_ms,
    PERCENTILE_CONT(0.95)
        WITHIN GROUP (ORDER BY total_latency_ms)    AS p95_latency_ms,
    AVG(faithfulness)               AS avg_faithfulness,
    AVG(answer_relevancy)           AS avg_relevancy
FROM eval_traces
GROUP BY 1, 2
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_daily_cost_day_type
    ON mv_daily_cost_summary (day, query_type);

-- Refresh command (called by observability pipeline daily):
-- REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_cost_summary;

-- ---------------------------------------------------------------------------
-- Helper function: hybrid_search
-- Combines BM25 rank + cosine similarity for hybrid retrieval
-- Called by hybrid_retriever.py
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text      TEXT,
    query_embedding VECTOR(768),
    match_count     INTEGER DEFAULT 10,
    bm25_weight     FLOAT   DEFAULT 0.3,
    vector_weight   FLOAT   DEFAULT 0.7,
    source_filter   TEXT    DEFAULT NULL   -- optional: 'confluence' | 'jira' etc.
)
RETURNS TABLE (
    chunk_id        TEXT,
    source_type     TEXT,
    source_id       TEXT,
    title           TEXT,
    text            TEXT,
    author          TEXT,
    url             TEXT,
    labels          TEXT[],
    updated_at      TIMESTAMPTZ,
    bm25_score      FLOAT,
    vector_score    FLOAT,
    combined_score  FLOAT
)
LANGUAGE SQL STABLE
AS $$
WITH bm25 AS (
    SELECT
        c.chunk_id,
        ts_rank_cd(c.ts_content, plainto_tsquery('english', query_text)) AS bm25_score
    FROM chunks c
    WHERE
        c.ts_content @@ plainto_tsquery('english', query_text)
        AND (source_filter IS NULL OR c.source_type = source_filter)
    ORDER BY bm25_score DESC
    LIMIT match_count * 3
),
vec AS (
    SELECT
        c.chunk_id,
        1 - (c.embedding <=> query_embedding) AS vector_score
    FROM chunks c
    WHERE
        c.embedding IS NOT NULL
        AND (source_filter IS NULL OR c.source_type = source_filter)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count * 3
),
combined AS (
    SELECT
        COALESCE(b.chunk_id, v.chunk_id)                              AS chunk_id,
        COALESCE(b.bm25_score, 0.0)                                   AS bm25_score,
        COALESCE(v.vector_score, 0.0)                                 AS vector_score,
        (bm25_weight  * COALESCE(b.bm25_score, 0.0))
        + (vector_weight * COALESCE(v.vector_score, 0.0))             AS combined_score
    FROM bm25 b
    FULL OUTER JOIN vec v ON b.chunk_id = v.chunk_id
)
SELECT
    c.chunk_id,
    c.source_type,
    c.source_id,
    c.title,
    c.text,
    c.author,
    c.url,
    c.labels,
    c.updated_at,
    cm.bm25_score::FLOAT,
    cm.vector_score::FLOAT,
    cm.combined_score::FLOAT
FROM combined cm
JOIN chunks c ON cm.chunk_id = c.chunk_id
ORDER BY cm.combined_score DESC
LIMIT match_count;
$$;

-- ---------------------------------------------------------------------------
-- Row-level grants (adjust role names to match your AlloyDB setup)
-- ---------------------------------------------------------------------------
-- GRANT SELECT, INSERT, UPDATE ON chunks          TO ekga_app;
-- GRANT SELECT, INSERT         ON entities        TO ekga_app;
-- GRANT SELECT, INSERT, UPDATE ON ingestion_runs  TO ekga_app;
-- GRANT SELECT, INSERT         ON eval_traces     TO ekga_app;
-- GRANT SELECT                 ON mv_daily_cost_summary TO ekga_app;
-- GRANT EXECUTE ON FUNCTION hybrid_search TO ekga_app;