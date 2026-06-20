"""
deduplicator.py
---------------
Near-duplicate detection for Chunk objects using MinHash + LSH.

Problem: the same content often appears multiple times across sources:
  - A Confluence page pasted into a Slack thread
  - A Jira description copied from an RFC PDF
  - Confluence pages that share boilerplate headers/footers
  - Re-ingestion of the same document after minor edits

Strategy:
  1. Exact dedup    — SHA-256 of normalized text (catches byte-for-byte copies)
  2. Near-dup LSH   — MinHash Locality Sensitive Hashing (datasketch)
                      Jaccard similarity threshold configurable (default 0.85)
  3. Exact-match on source_id — same source_id + chunk_index always deduped

When a duplicate is detected, the EARLIER chunk (lower created_at) is kept
unless `keep_newer=True` is set.

Env vars:
    DEDUP_THRESHOLD   Jaccard similarity threshold 0.0–1.0 (default 0.85)
    DEDUP_NUM_PERM    MinHash permutations (default 128; higher = more accurate)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional datasketch import
# ---------------------------------------------------------------------------
try:
    from datasketch import MinHash, MinHashLSH
    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False
    logger.warning(
        "datasketch not installed — near-duplicate detection disabled. "
        "Install with: pip install datasketch"
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class DedupResult:
    original_count: int
    unique_count: int
    duplicate_count: int
    exact_duplicates: int
    near_duplicates: int
    duplicate_pairs: list[tuple[str, str]] = field(default_factory=list)
    # (kept_chunk_id, dropped_chunk_id)

    @property
    def reduction_pct(self) -> float:
        if self.original_count == 0:
            return 0.0
        return round((self.duplicate_count / self.original_count) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "original_count": self.original_count,
            "unique_count": self.unique_count,
            "duplicate_count": self.duplicate_count,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "reduction_pct": self.reduction_pct,
            "sample_pairs": self.duplicate_pairs[:10],  # cap for logging
        }


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------

class Deduplicator:
    """
    Usage:
        dedup = Deduplicator(threshold=0.85, num_perm=128)
        unique_chunks, result = dedup.deduplicate(chunks)

        print(result.reduction_pct, "% duplicates removed")

    The returned unique_chunks list is safe to pass directly to the
    AlloyDB and Neo4j loaders.
    """

    _DEFAULT_THRESHOLD = float(os.environ.get("DEDUP_THRESHOLD", 0.85))
    _DEFAULT_NUM_PERM  = int(os.environ.get("DEDUP_NUM_PERM", 128))

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        num_perm: int = _DEFAULT_NUM_PERM,
        keep_newer: bool = False,
        shingle_size: int = 3,
    ) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1]")

        self._threshold = threshold
        self._num_perm = num_perm
        self._keep_newer = keep_newer
        self._shingle_size = shingle_size

        self._use_lsh = _HAS_DATASKETCH

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def deduplicate(self, chunks: list) -> tuple[list, DedupResult]:
        """
        Remove exact and near-duplicate chunks.

        Returns:
            (unique_chunks, DedupResult)
        """
        original_count = len(chunks)
        if not chunks:
            return [], DedupResult(0, 0, 0, 0, 0)

        # Stage 1: exact dedup by normalized text hash
        chunks_after_exact, exact_pairs, exact_count = self._exact_dedup(chunks)

        # Stage 2: near-dup via MinHash LSH
        if self._use_lsh and len(chunks_after_exact) > 1:
            unique_chunks, near_pairs, near_count = self._lsh_dedup(chunks_after_exact)
        else:
            unique_chunks = chunks_after_exact
            near_pairs = []
            near_count = 0

        all_pairs = exact_pairs + near_pairs
        total_dupes = exact_count + near_count

        result = DedupResult(
            original_count=original_count,
            unique_count=len(unique_chunks),
            duplicate_count=total_dupes,
            exact_duplicates=exact_count,
            near_duplicates=near_count,
            duplicate_pairs=all_pairs,
        )

        logger.info(
            "Deduplicator: %d → %d chunks (%.1f%% removed; %d exact, %d near-dup)",
            original_count, len(unique_chunks), result.reduction_pct,
            exact_count, near_count,
        )

        return unique_chunks, result

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute Jaccard similarity between two texts using MinHash.
        Returns float in [0, 1]. Falls back to 0.0 if datasketch unavailable.
        """
        if not self._use_lsh:
            return 0.0
        mh_a = self._build_minhash(_normalize(text_a))
        mh_b = self._build_minhash(_normalize(text_b))
        return mh_a.jaccard(mh_b)

    # ------------------------------------------------------------------
    # Private: exact dedup
    # ------------------------------------------------------------------

    def _exact_dedup(
        self,
        chunks: list,
    ) -> tuple[list, list[tuple[str, str]], int]:
        seen_hashes: dict[str, str] = {}    # hash → chunk_id of first seen
        seen_source_ids: dict[tuple, str] = {}  # (source_id, idx) → chunk_id
        unique: list = []
        pairs: list[tuple[str, str]] = []

        for chunk in chunks:
            # Dedup by (source_id, chunk_index) — same position in same doc
            source_key = (chunk.source_id, chunk.chunk_index)
            if source_key in seen_source_ids:
                pairs.append((seen_source_ids[source_key], chunk.chunk_id))
                continue
            seen_source_ids[source_key] = chunk.chunk_id

            # Dedup by normalized text hash
            text_hash = _text_hash(_normalize(chunk.text))
            if text_hash in seen_hashes:
                kept_id = seen_hashes[text_hash]
                pairs.append((kept_id, chunk.chunk_id))
                continue
            seen_hashes[text_hash] = chunk.chunk_id
            unique.append(chunk)

        return unique, pairs, len(chunks) - len(unique)

    # ------------------------------------------------------------------
    # Private: LSH near-dup
    # ------------------------------------------------------------------

    def _lsh_dedup(
        self,
        chunks: list,
    ) -> tuple[list, list[tuple[str, str]], int]:
        lsh = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
        minhashes: dict[str, MinHash] = {}

        # Build MinHash for each chunk
        for chunk in chunks:
            normalized = _normalize(chunk.text)
            mh = self._build_minhash(normalized)
            minhashes[chunk.chunk_id] = mh

        # Insert and detect duplicates
        dropped: set[str] = set()
        pairs: list[tuple[str, str]] = []

        for chunk in chunks:
            if chunk.chunk_id in dropped:
                continue

            mh = minhashes[chunk.chunk_id]

            try:
                # Query before inserting (avoids self-match)
                candidates = lsh.query(mh)
            except Exception:
                candidates = []

            if candidates:
                # This chunk is a near-duplicate of something already inserted
                kept_id = candidates[0]
                pairs.append((kept_id, chunk.chunk_id))
                dropped.add(chunk.chunk_id)

                # Optionally swap: keep newer
                if self._keep_newer:
                    kept_chunk = next((c for c in chunks if c.chunk_id == kept_id), None)
                    if kept_chunk and hasattr(kept_chunk, "updated_at") and hasattr(chunk, "updated_at"):
                        if chunk.updated_at > kept_chunk.updated_at:
                            # Swap: remove old from LSH, insert new
                            try:
                                lsh.remove(kept_id)
                                lsh.insert(chunk.chunk_id, mh)
                                dropped.discard(chunk.chunk_id)
                                dropped.add(kept_id)
                                pairs[-1] = (chunk.chunk_id, kept_id)
                            except Exception as exc:
                                logger.debug("Deduplicator: swap failed — %s", exc)
            else:
                try:
                    lsh.insert(chunk.chunk_id, mh)
                except Exception as exc:
                    logger.debug("Deduplicator: LSH insert failed for %s — %s",
                                 chunk.chunk_id, exc)

        unique = [c for c in chunks if c.chunk_id not in dropped]
        return unique, pairs, len(dropped)

    # ------------------------------------------------------------------
    # Private: MinHash builder
    # ------------------------------------------------------------------

    def _build_minhash(self, text: str) -> "MinHash":
        mh = MinHash(num_perm=self._num_perm)
        shingles = _shingle(text, self._shingle_size)
        for s in shingles:
            mh.update(s.encode("utf-8"))
        return mh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Normalize text for similarity comparison:
      - Unicode NFC
      - Lowercase
      - Collapse whitespace
      - Strip punctuation runs (keep words only)
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _text_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode()).hexdigest()


def _shingle(text: str, k: int) -> set[str]:
    """Character-level k-shingles on word-token sequence."""
    words = text.split()
    if len(words) < k:
        return {text}
    return {" ".join(words[i: i + k]) for i in range(len(words) - k + 1)}


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from dataclasses import dataclass as dc
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO)

    @dc
    class _FakeChunk:
        chunk_id: str
        source_id: str
        chunk_index: int
        text: str
        updated_at: datetime

    base_text = (
        "The auth service handles authentication and authorization. "
        "It uses JWT tokens with a 15-minute lifetime and rotating refresh tokens. "
        "Deployed on Cloud Run with min-instances=2."
    )
    near_dup_text = (
        "The auth service handles authentication and authorization. "
        "It uses JWT tokens with a 15-minute lifetime and rotating refresh tokens. "
        "Deployed on Cloud Run with min-instances=3."   # only '2' → '3' changed
    )
    unrelated_text = (
        "Kafka is a distributed streaming platform used for real-time data pipelines. "
        "Topics are partitioned for parallel processing across consumer groups."
    )

    now = datetime.now(timezone.utc)
    chunks = [
        _FakeChunk("c1", "page-001", 0, base_text, now),
        _FakeChunk("c2", "page-002", 0, near_dup_text, now),
        _FakeChunk("c3", "page-003", 0, unrelated_text, now),
        _FakeChunk("c4", "page-001", 0, base_text, now),   # exact dup of c1
    ]

    dedup = Deduplicator(threshold=0.85)
    unique, result = dedup.deduplicate(chunks)

    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nKept chunk IDs: {[c.chunk_id for c in unique]}")

    # Pairwise similarity demo
    sim = dedup.similarity(base_text, near_dup_text)
    print(f"\nSimilarity(base, near_dup) = {sim:.3f}")
    sim2 = dedup.similarity(base_text, unrelated_text)
    print(f"Similarity(base, unrelated) = {sim2:.3f}")