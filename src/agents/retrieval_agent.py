"""Retrieval agent wrapper around HybridRetriever."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetrievalAgentResult:
    query: str
    query_type: str
    chunks: list[Any] = field(default_factory=list)
    retrieval_path: str = "none"
    latency_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "chunks": [_chunk_to_dict(c) for c in self.chunks],
            "retrieval_path": self.retrieval_path,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class RetrievalAgent:
    def __init__(self, retriever: Optional[Any] = None) -> None:
        self._retriever = retriever

    def retrieve(self, query: str, query_type: str, top_k: int = 5) -> RetrievalAgentResult:
        started = time.time()
        try:
            retriever = self._get_retriever()
            result = retriever.retrieve_with_reflection(query, query_type=query_type, top_k=top_k)
            chunks = list(getattr(result, "chunks", []))
            if chunks:
                return RetrievalAgentResult(
                    query=query,
                    query_type=query_type,
                    chunks=chunks,
                    retrieval_path=getattr(result, "retrieval_path", "hybrid"),
                    latency_ms=getattr(result, "latency_ms", int((time.time() - started) * 1000)),
                )
            if _demo_mode_enabled():
                return self._demo_result(query, query_type, top_k, started)
            return RetrievalAgentResult(
                query=query,
                query_type=query_type,
                chunks=chunks,
                retrieval_path=getattr(result, "retrieval_path", "hybrid"),
                latency_ms=getattr(result, "latency_ms", int((time.time() - started) * 1000)),
            )
        except Exception as exc:
            if _demo_mode_enabled():
                return self._demo_result(query, query_type, top_k, started)
            return RetrievalAgentResult(
                query=query,
                query_type=query_type,
                latency_ms=int((time.time() - started) * 1000),
                error=str(exc),
            )

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            from ..retrieval import HybridRetriever

            self._retriever = HybridRetriever()
        return self._retriever

    def _demo_result(self, query: str, query_type: str, top_k: int, started: float) -> RetrievalAgentResult:
        return RetrievalAgentResult(
            query=query,
            query_type=query_type,
            chunks=_rank_demo_chunks(query, query_type, top_k),
            retrieval_path="demo-hybrid-graph",
            latency_ms=int((time.time() - started) * 1000),
        )


def _chunk_to_dict(chunk: Any) -> dict:
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if isinstance(chunk, dict):
        return chunk
    return dict(getattr(chunk, "__dict__", {}))


def _demo_mode_enabled() -> bool:
    return os.environ.get("EKGA_DEMO_MODE", "true").lower() not in {"0", "false", "no"}


def _rank_demo_chunks(query: str, query_type: str, top_k: int) -> list[dict]:
    query_terms = _terms(query)
    ranked = []
    for chunk in _DEMO_CORPUS:
        text_terms = _terms(" ".join(str(v) for v in chunk.values()))
        score = len(query_terms & text_terms) / max(1, len(query_terms))
        if chunk["query_type"] == query_type:
            score += 0.25
        if query_type == "relational" and chunk["source_type"] in {"jira", "confluence"}:
            score += 0.1
        ranked.append(({**chunk, "score": round(score, 3)}, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in ranked[:top_k]]


def _terms(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "what", "why", "how", "was", "were"}
    return {t for t in re.findall(r"[a-z0-9_-]{3,}", text.lower()) if t not in stop}


_DEMO_CORPUS = [
    {
        "chunk_id": "demo-adr-042",
        "source_type": "confluence",
        "source_id": "ADR-042",
        "title": "ADR-042: Auth Service Database Migration",
        "text": (
            "The Auth Platform team approved moving the auth service session and token workload "
            "off PostgreSQL after Q3 load tests showed connection pool exhaustion at peak traffic. "
            "The decision linked ADR-042, Jira PLAT-123, and the migration runbook. AlloyDB/pgvector "
            "remained the knowledge retrieval store while the service path moved to a managed cache-backed design."
        ),
        "author": "alice.chen@example.com",
        "url": "https://example.com/confluence/ADR-042",
        "labels": ["architecture", "auth", "decision"],
        "updated_at": "2026-05-18T14:20:00Z",
        "query_type": "relational",
    },
    {
        "chunk_id": "demo-jira-plat-123",
        "source_type": "jira",
        "source_id": "PLAT-123",
        "title": "PLAT-123: Resolve Auth Connection Pool Bottleneck",
        "text": (
            "PLAT-123 tracks the production risk behind the migration: authentication latency climbed when "
            "PostgreSQL pools saturated during token refresh bursts. The ticket is blocked by load-test signoff "
            "and references ADR-042 plus the rollout checklist owned by Platform Infrastructure."
        ),
        "author": "platform-infra@example.com",
        "url": "https://example.com/jira/PLAT-123",
        "labels": ["platform", "auth", "incident-prevention"],
        "updated_at": "2026-05-21T09:10:00Z",
        "query_type": "relational",
    },
    {
        "chunk_id": "demo-runbook-onboarding",
        "source_type": "confluence",
        "source_id": "RUNBOOK-017",
        "title": "Backend Engineer Onboarding Runbook",
        "text": (
            "Backend onboarding is a five-step path: get IAM access, run the local Docker stack, read ADR-042 "
            "and the service ownership map, shadow one incident review, then ship a low-risk Jira ticket. "
            "The runbook reduces time-to-first-production-change from weeks to days."
        ),
        "author": "devex@example.com",
        "url": "https://example.com/confluence/RUNBOOK-017",
        "labels": ["onboarding", "runbook", "backend"],
        "updated_at": "2026-06-02T16:35:00Z",
        "query_type": "procedural",
    },
    {
        "chunk_id": "demo-slack-rate-limit",
        "source_type": "slack",
        "source_id": "C-PLATFORM-171890",
        "title": "Slack Thread: Rate Limiting Ownership",
        "text": (
            "The rate-limiting module is owned by Platform Infrastructure. The Slack thread links the codeowner, "
            "the Jira epic for adaptive throttling, and the service map entry that routes escalation to the "
            "platform-infra on-call rotation."
        ),
        "author": "morgan.lee@example.com",
        "url": "https://example.com/slack/C-PLATFORM-171890",
        "labels": ["ownership", "rate-limiting", "platform"],
        "updated_at": "2026-05-29T11:00:00Z",
        "query_type": "relational",
    },
    {
        "chunk_id": "demo-pdf-gtm",
        "source_type": "pdf",
        "source_id": "FDE-BUSINESS-CASE",
        "title": "FDE Business Case: Enterprise Knowledge Graph",
        "text": (
            "The value case for EKGA is reducing duplicated engineering research, shortening onboarding, and "
            "preserving architectural memory across Confluence, Jira, Slack, and PDFs. The demo highlights "
            "graph retrieval, citation grounding, and per-agent observability as the differentiators over standard RAG."
        ),
        "author": "field-engineering@example.com",
        "url": "https://example.com/docs/fde-business-case.pdf",
        "labels": ["business-value", "fde", "google-cloud"],
        "updated_at": "2026-06-05T12:00:00Z",
        "query_type": "factual",
    },
]
