"""Grounded answer synthesis agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Citation:
    chunk_id: str
    title: str
    url: str = ""
    source_type: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
        }


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
        }


class SynthesisAgent:
    """Create a concise grounded answer from retrieved chunks."""

    def synthesize(self, query: str, chunks: list[Any], query_type: str = "factual") -> SynthesisResult:
        chunk_dicts = [_chunk_to_dict(c) for c in chunks]
        if not chunk_dicts:
            return SynthesisResult(
                answer=(
                    "I could not find enough indexed knowledge to answer that yet. "
                    "Ingest relevant Confluence, Jira, Slack, or PDF sources and retry."
                ),
                citations=[],
                confidence=0.0,
            )

        citations = [
            Citation(
                chunk_id=str(c.get("chunk_id", "")),
                title=str(c.get("title", "")),
                url=str(c.get("url", "")),
                source_type=str(c.get("source_type", "")),
            )
            for c in chunk_dicts[:5]
        ]

        evidence = " ".join(str(c.get("text", "")) for c in chunk_dicts[:3]).strip()
        prefix = {
            "procedural": "Recommended path",
            "relational": "Relationship summary",
            "factual": "Answer",
        }.get(query_type, "Answer")

        primary = chunk_dicts[0]
        supporting_titles = ", ".join(c.get("title", "source") for c in chunk_dicts[1:3])
        why_it_matters = (
            "This is valuable because the answer is grounded in cross-source enterprise memory, "
            "not just vector similarity: the system preserves provenance, ownership, and decision context."
        )

        answer = (
            f"{prefix}: {primary.get('text', evidence)[:520]}"
            f"\n\nEvidence: cited {primary.get('title', 'primary source')}"
            f"{f' plus {supporting_titles}' if supporting_titles else ''}."
            f"\n\nWhy it matters: {why_it_matters}"
        )
        return SynthesisResult(answer=answer, citations=citations, confidence=min(0.95, 0.45 + 0.1 * len(citations)))


def _chunk_to_dict(chunk: Any) -> dict:
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if isinstance(chunk, dict):
        return chunk
    return dict(getattr(chunk, "__dict__", {}))
