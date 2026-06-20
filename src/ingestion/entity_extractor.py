"""
entity_extractor.py
-------------------
Extracts named entities and domain-specific terms from Chunk text.
Feeds two downstream consumers:

    1. Neo4j graph loader  — entities become nodes; co-occurrence
                             within a chunk creates MENTIONS edges
    2. AlloyDB metadata    — stored alongside the embedding for
                             structured filtering (filter by person,
                             system name, team, etc.)

Entity types extracted:
    Standard NER (spaCy en_core_web_sm):
        PERSON       — people mentioned in text
        ORG          — organizations, teams, companies
        GPE          — locations (for geo-aware queries)
        DATE         — dates and time references
        PRODUCT      — product or service names

    Domain-specific (regex/pattern):
        JIRA_KEY     — e.g. PLAT-123, ENG-42
        GIT_SHA      — 7–40 hex chars preceded by commit/sha
        SVC_NAME     — microservice names (CamelCase or kebab-case
                       prefixed by "service", "svc", "api")
        TEAM         — "<Name> team" or "team <Name>" patterns
        ADR_REF      — "ADR-042", "RFC-7", etc.

Env vars:
    SPACY_MODEL     spaCy model name (default: en_core_web_sm)
                    Use en_core_web_trf for higher accuracy if GPU available.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional spaCy import — graceful degradation to regex-only mode
# ---------------------------------------------------------------------------
try:
    import spacy
    from spacy.language import Language

    _SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
    _NLP: Optional[Language] = None          # lazy-loaded on first call
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False
    logger.warning("spaCy not installed — falling back to regex-only entity extraction")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    text: str               # surface form as it appears in the chunk
    label: str              # entity type (PERSON, JIRA_KEY, etc.)
    start: int              # char offset in chunk text
    end: int                # char offset in chunk text
    normalized: str = ""    # lowercased / canonical form

    def __post_init__(self) -> None:
        if not self.normalized:
            self.normalized = self.text.strip().lower()

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "normalized": self.normalized,
        }


@dataclass
class EntityResult:
    chunk_id: str
    source_id: str
    entities: list[Entity] = field(default_factory=list)

    # Convenience views
    @property
    def by_label(self) -> dict[str, list[Entity]]:
        result: dict[str, list[Entity]] = {}
        for e in self.entities:
            result.setdefault(e.label, []).append(e)
        return result

    @property
    def unique_labels(self) -> set[str]:
        return {e.label for e in self.entities}

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "entities": [e.to_dict() for e in self.entities],
            "entity_count": len(self.entities),
            "label_summary": {
                label: len(ents)
                for label, ents in self.by_label.items()
            },
        }


# ---------------------------------------------------------------------------
# Domain-specific regex patterns
# ---------------------------------------------------------------------------

_DOMAIN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("JIRA_KEY",  re.compile(r"\b([A-Z]{2,8}-\d{1,6})\b")),
    ("ADR_REF",   re.compile(r"\b(ADR|RFC)[-\s]?\d{1,4}\b", re.I)),
    ("GIT_SHA",   re.compile(
        r"\b(?:commit|sha|ref)[:\s]+([0-9a-f]{7,40})\b", re.I
    )),
    ("SVC_NAME",  re.compile(
        r"\b(?:service|svc|api)[:\s]+([a-z][a-z0-9-]{2,30})\b", re.I
    )),
    ("TEAM",      re.compile(
        r"\b(?:([A-Z][A-Za-z\s]{1,20})\s+team|team\s+([A-Z][A-Za-z\s]{1,20}))\b"
    )),
    ("PR_REF",    re.compile(r"\bPR[:\s#]+(\d{1,6})\b", re.I)),
    ("ENV_NAME",  re.compile(
        r"\b(prod(?:uction)?|staging|dev(?:elopment)?|canary|sandbox)\b", re.I
    )),
]

# spaCy labels we care about (filter noise)
_SPACY_LABELS_KEEP = {"PERSON", "ORG", "GPE", "DATE", "PRODUCT", "WORK_OF_ART"}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Usage:
        extractor = EntityExtractor()

        from chunker import Chunk
        result = extractor.extract(chunk)
        print(result.by_label)

        # Batch
        results = extractor.extract_many(chunks)
    """

    def __init__(self, use_spacy: bool = True) -> None:
        self._use_spacy = use_spacy and _HAS_SPACY
        if self._use_spacy:
            self._ensure_nlp_loaded()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, chunk) -> EntityResult:
        """Extract entities from a single Chunk."""
        entities: list[Entity] = []

        # 1. Domain-specific regex (always runs — fast, high precision)
        entities.extend(self._extract_domain(chunk.text))

        # 2. spaCy NER (if available)
        if self._use_spacy:
            entities.extend(self._extract_spacy(chunk.text))

        # Deduplicate by (normalized, label) — keep first occurrence
        seen: set[tuple[str, str]] = set()
        deduped: list[Entity] = []
        for e in entities:
            key = (e.normalized, e.label)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        return EntityResult(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            entities=deduped,
        )

    def extract_many(self, chunks: list) -> list[EntityResult]:
        """Batch extraction. Uses spaCy pipe for efficiency if available."""
        if self._use_spacy and len(chunks) > 10:
            return self._extract_many_spacy_pipe(chunks)

        results = []
        for chunk in chunks:
            try:
                results.append(self.extract(chunk))
            except Exception as exc:
                logger.warning("EntityExtractor: failed on chunk %s — %s", chunk.chunk_id, exc)
        return results

    # ------------------------------------------------------------------
    # Private: domain regex
    # ------------------------------------------------------------------

    def _extract_domain(self, text: str) -> list[Entity]:
        entities: list[Entity] = []

        for label, pattern in _DOMAIN_PATTERNS:
            for match in pattern.finditer(text):
                # Use first non-None group, or full match
                surface = next(
                    (g for g in match.groups() if g is not None),
                    match.group(0)
                ).strip()

                if not surface:
                    continue

                entities.append(Entity(
                    text=surface,
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    normalized=surface.lower(),
                ))

        return entities

    # ------------------------------------------------------------------
    # Private: spaCy NER
    # ------------------------------------------------------------------

    def _extract_spacy(self, text: str) -> list[Entity]:
        global _NLP
        if _NLP is None:
            return []

        entities: list[Entity] = []
        try:
            doc = _NLP(text[:100_000])  # spaCy's default max length guard
            for ent in doc.ents:
                if ent.label_ not in _SPACY_LABELS_KEEP:
                    continue
                if len(ent.text.strip()) < 2:
                    continue
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                ))
        except Exception as exc:
            logger.warning("EntityExtractor: spaCy error — %s", exc)

        return entities

    def _extract_many_spacy_pipe(self, chunks: list) -> list[EntityResult]:
        """Use spaCy nlp.pipe for batched inference (2–4× faster)."""
        global _NLP
        if _NLP is None:
            return [self.extract(c) for c in chunks]

        texts = [c.text[:100_000] for c in chunks]
        results: list[EntityResult] = []

        try:
            for chunk, doc in zip(chunks, _NLP.pipe(texts, batch_size=32)):
                spacy_ents = []
                for ent in doc.ents:
                    if ent.label_ not in _SPACY_LABELS_KEEP:
                        continue
                    if len(ent.text.strip()) < 2:
                        continue
                    spacy_ents.append(Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                    ))

                domain_ents = self._extract_domain(chunk.text)
                all_ents = domain_ents + spacy_ents

                seen: set[tuple[str, str]] = set()
                deduped: list[Entity] = []
                for e in all_ents:
                    key = (e.normalized, e.label)
                    if key not in seen:
                        seen.add(key)
                        deduped.append(e)

                results.append(EntityResult(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    entities=deduped,
                ))
        except Exception as exc:
            logger.error("EntityExtractor: pipe failed — %s", exc)
            return [self.extract(c) for c in chunks]

        return results

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_nlp_loaded() -> None:
        global _NLP
        if _NLP is not None:
            return
        try:
            _NLP = spacy.load(_SPACY_MODEL, disable=["parser", "lemmatizer"])
            logger.info("EntityExtractor: loaded spaCy model '%s'", _SPACY_MODEL)
        except OSError:
            logger.warning(
                "EntityExtractor: model '%s' not found. "
                "Run: python -m spacy download %s",
                _SPACY_MODEL, _SPACY_MODEL,
            )
            _NLP = None


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from dataclasses import dataclass as dc
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO)

    # Minimal Chunk stub for testing without full pipeline
    @dc
    class _FakeChunk:
        chunk_id: str
        source_id: str
        text: str

    sample_text = """
    Alice Chen from the Platform Infrastructure team deployed the auth-service
    to production on 2024-06-15. The fix addressed PLAT-442 and was reviewed
    in PR #2847 (commit sha: a1b2c3d). ADR-012 guided the JWT token strategy.
    The staging environment showed a 40% latency improvement.
    """

    fake_chunk = _FakeChunk(
        chunk_id="test-chunk-001",
        source_id="page-001",
        text=sample_text.strip(),
    )

    extractor = EntityExtractor()
    result = extractor.extract(fake_chunk)

    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nLabel summary: {result.unique_labels}")