"""Query classification agent."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RouteDecision:
    query: str
    query_type: str
    confidence: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


class RouterAgent:
    """Classify questions into factual, relational, or procedural retrieval paths."""

    _PROCEDURAL = re.compile(r"\b(how|steps|process|runbook|onboard|checklist|setup|deploy|migrate)\b", re.I)
    _RELATIONAL = re.compile(
        r"\b(why|who owns|owner|depends|dependency|blocked|blocks|related|decision|impact|because|"
        r"which team|references|linked)\b",
        re.I,
    )

    def route(self, query: str) -> RouteDecision:
        clean = query.strip()
        if self._PROCEDURAL.search(clean):
            return RouteDecision(clean, "procedural", 0.82, "procedural language detected")
        if self._RELATIONAL.search(clean):
            return RouteDecision(clean, "relational", 0.78, "relationship or rationale language detected")
        return RouteDecision(clean, "factual", 0.72, "default factual lookup path")

    def __call__(self, query: str) -> RouteDecision:
        return self.route(query)
