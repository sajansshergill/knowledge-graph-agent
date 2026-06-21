"""
query_rewriter.py
-----------------
Lightweight self-reflection query rewriting for retrieval retries.

This intentionally avoids an LLM dependency: the retriever can call it when
initial recall is low and get a deterministic, cheap rewrite.
"""

from __future__ import annotations

import re
from typing import Any


_ACRONYM_EXPANSIONS = {
    "adr": "architecture decision record",
    "api": "application programming interface",
    "ci": "continuous integration",
    "cd": "continuous delivery",
    "db": "database",
    "gke": "google kubernetes engine",
    "iam": "identity access management",
    "jwt": "json web token",
    "k8s": "kubernetes",
    "mcp": "model context protocol",
    "pr": "pull request",
    "rfc": "request for comments",
    "slo": "service level objective",
    "sre": "site reliability engineering",
}

_RELATIONAL_HINTS = {
    "relational": "related dependencies owners decisions blockers references",
    "procedural": "steps process runbook onboarding checklist owner",
    "factual": "definition value setting status summary",
}


class QueryRewriter:
    """Deterministic rewrite helper used by HybridRetriever reflection."""

    def rewrite(self, query: str, query_type: str = "factual", result: Any = None) -> str:
        """
        Expand likely acronyms and add retrieval hints when the first pass is sparse.

        Args:
            query: Original natural language query.
            query_type: factual | relational | procedural.
            result: Optional RetrievalResult. If it contains source titles/entities,
                they are folded in as weak hints.
        """
        rewritten = self._expand_acronyms(query)
        hints = self._result_hints(result)

        query_type_hint = _RELATIONAL_HINTS.get(query_type, "")
        additions = " ".join(part for part in [query_type_hint, hints] if part)

        if additions and additions.lower() not in rewritten.lower():
            rewritten = f"{rewritten} {additions}"

        return _squash_spaces(rewritten)

    def _expand_acronyms(self, query: str) -> str:
        tokens = []
        for token in query.split():
            stripped = re.sub(r"[^A-Za-z0-9]", "", token).lower()
            expansion = _ACRONYM_EXPANSIONS.get(stripped)
            if expansion and expansion not in query.lower():
                tokens.append(f"{token} {expansion}")
            else:
                tokens.append(token)
        return " ".join(tokens)

    def _result_hints(self, result: Any) -> str:
        if result is None:
            return ""

        chunks = getattr(result, "chunks", []) or []
        titles: list[str] = []
        labels: list[str] = []
        for chunk in chunks[:3]:
            title = getattr(chunk, "title", "")
            if title:
                titles.append(title)
            labels.extend(getattr(chunk, "labels", []) or [])

        hints = titles + labels
        deduped = []
        seen = set()
        for hint in hints:
            normalized = str(hint).strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(str(hint).strip())
        return " ".join(deduped[:6])


def _squash_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
