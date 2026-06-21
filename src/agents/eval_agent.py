"""Inline answer evaluation agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    grounded: bool

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "grounded": self.grounded,
        }


class EvalAgent:
    """Cheap lexical evaluation for local CI and inline monitoring."""

    def evaluate(self, query: str, answer: str, chunks: list[Any]) -> EvalResult:
        context = " ".join(str(_chunk_to_dict(c).get("text", "")) for c in chunks)
        faithfulness = _overlap(answer, context)
        relevancy = _overlap(query, answer)
        recall = _overlap(query, context)
        return EvalResult(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_recall=recall,
            grounded=bool(chunks) and faithfulness >= 0.15,
        )


def _overlap(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return round(len(left_terms & right_terms) / len(left_terms), 3)


def _terms(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "what", "why", "how"}
    return {t for t in re.findall(r"[a-z0-9_-]{3,}", text.lower()) if t not in stop}


def _chunk_to_dict(chunk: Any) -> dict:
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if isinstance(chunk, dict):
        return chunk
    return dict(getattr(chunk, "__dict__", {}))
