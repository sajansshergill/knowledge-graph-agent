"""Trace logging for agent runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class TraceRecord:
    trace_id: str
    query_text: str
    query_type: str
    session_id: Optional[str]
    total_latency_ms: int
    retrieval_path: str
    chunks_retrieved: int
    answer_text: str
    source_chunk_ids: list[str] = field(default_factory=list)
    agent_hops: list[dict[str, Any]] = field(default_factory=list)
    eval: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class TraceLogger:
    """Append traces to JSONL locally and optionally mirror to BigQuery."""

    def __init__(self, path: Optional[str] = None, bigquery_table: Optional[str] = None) -> None:
        self.path = Path(path or os.environ.get("TRACE_LOG_PATH", "data/traces/ekga_traces.jsonl"))
        self.bigquery_table = bigquery_table or os.environ.get("BIGQUERY_TRACE_TABLE", "")

    def log_query(self, result: dict[str, Any], session_id: Optional[str] = None) -> TraceRecord:
        citations = result.get("citations", [])
        record = TraceRecord(
            trace_id=result.get("trace_id", ""),
            query_text=result.get("query", ""),
            query_type=result.get("query_type", ""),
            session_id=session_id,
            total_latency_ms=int(result.get("latency_ms", 0)),
            retrieval_path=result.get("retrieval_path", "none"),
            chunks_retrieved=len(citations),
            answer_text=result.get("answer", ""),
            source_chunk_ids=[c.get("chunk_id", "") for c in citations],
            agent_hops=result.get("hops", []),
            eval=result.get("eval", {}),
        )
        self._append_local(record)
        self._insert_bigquery(record)
        return record

    def _append_local(self, record: TraceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def _insert_bigquery(self, record: TraceRecord) -> None:
        if not self.bigquery_table:
            return
        try:
            from google.cloud import bigquery

            client = bigquery.Client()
            client.insert_rows_json(self.bigquery_table, [record.to_dict()])
        except Exception:
            return
