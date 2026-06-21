"""
chunker.py
----------
Turns connector documents into storage-ready Chunk objects.

The chunker is intentionally light on dependencies: it uses word-token
estimates so ingestion works before a tokenizer service is configured.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    source_type: str
    chunk_index: int
    title: str
    text: str
    token_estimate: int
    char_count: int
    author: str = "unknown"
    url: str = ""
    section_heading: Optional[str] = None
    parent_doc_id: Optional[str] = None
    labels: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    text_hash: str = ""

    def __post_init__(self) -> None:
        if not self.text_hash:
            self.text_hash = _sha256(_normalize_text(self.text))
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "char_count": self.char_count,
            "author": self.author,
            "url": self.url,
            "section_heading": self.section_heading,
            "parent_doc_id": self.parent_doc_id,
            "labels": self.labels,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "text_hash": self.text_hash,
        }


class Chunker:
    """
    Split connector documents into overlapping chunks.

    Args:
        max_tokens: Approximate tokens per chunk.
        overlap_ratio: Fraction of each chunk repeated in the next chunk.
    """

    def __init__(self, max_tokens: int = 512, overlap_ratio: float = 0.10) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not (0 <= overlap_ratio < 1):
            raise ValueError("overlap_ratio must be in [0, 1)")
        self.max_tokens = max_tokens
        self.overlap_tokens = int(max_tokens * overlap_ratio)

    def chunk_document(self, document: Any) -> list[Chunk]:
        data = _as_dict(document)
        source_type = data.get("source") or data.get("source_type") or "unknown"
        source_id = _source_id(data)
        title = data.get("title") or data.get("summary") or data.get("filename") or source_id
        text = _document_text(data)
        if not text.strip():
            return []

        words = _tokenize(text)
        if not words:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.max_tokens - self.overlap_tokens)
        for index, start in enumerate(range(0, len(words), step)):
            window = words[start: start + self.max_tokens]
            if not window:
                continue

            chunk_text = " ".join(window).strip()
            chunk_id = _chunk_id(source_type, source_id, index, chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    source_type=source_type,
                    chunk_index=index,
                    title=str(title),
                    text=chunk_text,
                    token_estimate=len(window),
                    char_count=len(chunk_text),
                    author=_author(data),
                    url=data.get("url") or data.get("source_path") or "",
                    section_heading=_section_heading(chunk_text),
                    parent_doc_id=data.get("parent_id") or data.get("parent_key"),
                    labels=list(data.get("labels") or data.get("components") or []),
                    created_at=_iso(data.get("created_at")),
                    updated_at=_iso(data.get("updated_at") or data.get("created_at")),
                )
            )

            if start + self.max_tokens >= len(words):
                break
        return chunks

    def chunk_many(self, documents: Iterable[Any]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks


def chunk_document(document: Any, max_tokens: int = 512, overlap_ratio: float = 0.10) -> list[Chunk]:
    return Chunker(max_tokens=max_tokens, overlap_ratio=overlap_ratio).chunk_document(document)


def chunk_many(documents: Iterable[Any], max_tokens: int = 512, overlap_ratio: float = 0.10) -> list[Chunk]:
    return Chunker(max_tokens=max_tokens, overlap_ratio=overlap_ratio).chunk_many(documents)


def _as_dict(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        return dict(document)
    if hasattr(document, "to_dict"):
        return dict(document.to_dict())
    return dict(getattr(document, "__dict__", {}))


def _document_text(data: dict[str, Any]) -> str:
    for key in ("full_text", "body_text", "description", "text", "content"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _source_id(data: dict[str, Any]) -> str:
    for key in ("page_id", "thread_ts", "doc_id", "key", "issue_id", "source_id", "filename"):
        value = data.get(key)
        if value:
            return str(value)
    return _sha256(_document_text(data))[:16]


def _author(data: dict[str, Any]) -> str:
    return str(
        data.get("author_email")
        or data.get("author")
        or data.get("assignee_email")
        or data.get("reporter_email")
        or "unknown"
    )


def _section_heading(text: str) -> Optional[str]:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    return None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", _normalize_text(text))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_id(source_type: str, source_id: str, index: int, text: str) -> str:
    digest = _sha256(f"{source_type}:{source_id}:{index}:{text}")[:16]
    return f"{source_type}:{source_id}:{index}:{digest}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iso(value: Any) -> str:
    if value is None or value == "":
        return _now_iso()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
