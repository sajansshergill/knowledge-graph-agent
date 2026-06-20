"""
pdf_connector.py
----------------
Ingests PDF documents from a local directory or GCS bucket.
Extracts text per page, detects document type (ADR, RFC, runbook,
spec, meeting-notes) from filename + content heuristics, and
returns PDFDocument dataclasses ready for the chunker.

Handles:
  - Text-based PDFs (pdfplumber)
  - Scanned/image PDFs (pytesseract OCR fallback)
  - Password-protected PDFs (skip with warning)
  - GCS bucket source (google-cloud-storage)

Env vars:
    PDF_SOURCE_DIR        local folder path  (used if set)
    GCS_BUCKET_NAME       GCS bucket name    (used if PDF_SOURCE_DIR not set)
    GCS_PDF_PREFIX        optional GCS key prefix, e.g. "docs/pdfs/"
    GOOGLE_CLOUD_PROJECT  required for GCS mode
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False
    logger.warning("pdfplumber not installed — PDF text extraction unavailable")

try:
    from PIL import Image
    import pytesseract
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

try:
    from google.cloud import storage as gcs
    _HAS_GCS = True
except ImportError:
    _HAS_GCS = False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class PDFPage:
    page_number: int        # 1-indexed
    text: str
    char_count: int
    extraction_method: str  # "pdfplumber" | "ocr" | "empty"


@dataclass
class PDFDocument:
    doc_id: str             # sha256 of file content (stable, dedup-friendly)
    filename: str
    source_path: str        # local path or gs:// URI
    doc_type: str           # "adr" | "rfc" | "runbook" | "spec" | "meeting_notes" | "unknown"
    title: str
    pages: list[PDFPage]
    page_count: int
    author: str
    created_at: datetime
    full_text: str          # all pages joined
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": "pdf",
            "doc_id": self.doc_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "doc_type": self.doc_type,
            "title": self.title,
            "page_count": self.page_count,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "full_text": self.full_text,
            "labels": self.labels,
        }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class PDFConnector:
    """
    Usage — local directory:
        connector = PDFConnector(source_dir="/data/docs")
        docs = connector.fetch_all()

    Usage — GCS bucket:
        connector = PDFConnector()  # reads GCS_BUCKET_NAME from env
        docs = connector.fetch_all()

    Usage — single file:
        doc = connector.fetch_file("/path/to/adr-042.pdf")
    """

    _SUPPORTED_EXTS = {".pdf"}
    _MIN_TEXT_CHARS = 50    # below this, try OCR fallback

    # Filename pattern → doc_type
    _TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"adr[-_\s]?\d+", re.I),          "adr"),
        (re.compile(r"rfc[-_\s]?\d+", re.I),           "rfc"),
        (re.compile(r"runbook",        re.I),           "runbook"),
        (re.compile(r"spec|design[-_]doc", re.I),       "spec"),
        (re.compile(r"meeting|notes|minutes", re.I),    "meeting_notes"),
        (re.compile(r"onboarding|getting[-_]started", re.I), "runbook"),
    ]

    def __init__(
        self,
        source_dir: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        gcs_prefix: Optional[str] = None,
    ) -> None:
        self._source_dir = source_dir or os.environ.get("PDF_SOURCE_DIR")
        self._gcs_bucket = gcs_bucket or os.environ.get("GCS_BUCKET_NAME")
        self._gcs_prefix = gcs_prefix or os.environ.get("GCS_PDF_PREFIX", "")

        if not _HAS_PDFPLUMBER:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch_all(self) -> list[PDFDocument]:
        """Fetch all PDFs from configured source (local dir or GCS)."""
        if self._source_dir:
            return self._fetch_from_dir(Path(self._source_dir))
        elif self._gcs_bucket:
            return self._fetch_from_gcs()
        else:
            raise EnvironmentError(
                "Set PDF_SOURCE_DIR or GCS_BUCKET_NAME"
            )

    def fetch_file(self, path: str) -> Optional[PDFDocument]:
        """Extract a single PDF by file path."""
        return self._process_pdf(Path(path), source_path=path)

    # ------------------------------------------------------------------
    # Private: sources
    # ------------------------------------------------------------------

    def _fetch_from_dir(self, directory: Path) -> list[PDFDocument]:
        if not directory.exists():
            raise FileNotFoundError(f"PDF source dir not found: {directory}")

        pdf_files = sorted(directory.rglob("*.pdf"))
        docs: list[PDFDocument] = []

        for pdf_path in pdf_files:
            doc = self._process_pdf(pdf_path, source_path=str(pdf_path))
            if doc:
                docs.append(doc)

        logger.info("PDFConnector: processed %d/%d PDFs from %s",
                    len(docs), len(pdf_files), directory)
        return docs

    def _fetch_from_gcs(self) -> list[PDFDocument]:
        if not _HAS_GCS:
            raise ImportError("Install google-cloud-storage: pip install google-cloud-storage")

        client = gcs.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        bucket = client.bucket(self._gcs_bucket)
        blobs = bucket.list_blobs(prefix=self._gcs_prefix)

        docs: list[PDFDocument] = []

        for blob in blobs:
            if not blob.name.lower().endswith(".pdf"):
                continue

            gcs_uri = f"gs://{self._gcs_bucket}/{blob.name}"
            logger.info("PDFConnector: downloading %s", gcs_uri)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                blob.download_to_filename(tmp.name)
                doc = self._process_pdf(Path(tmp.name), source_path=gcs_uri)
                if doc:
                    # Override filename with GCS blob name
                    doc.filename = Path(blob.name).name
                    docs.append(doc)
            Path(tmp.name).unlink(missing_ok=True)

        logger.info("PDFConnector: processed %d PDFs from gs://%s/%s",
                    len(docs), self._gcs_bucket, self._gcs_prefix)
        return docs

    # ------------------------------------------------------------------
    # Private: extraction
    # ------------------------------------------------------------------

    def _process_pdf(self, path: Path, source_path: str) -> Optional[PDFDocument]:
        try:
            raw_bytes = path.read_bytes()
        except (OSError, PermissionError) as exc:
            logger.warning("PDFConnector: cannot read %s — %s", path, exc)
            return None

        doc_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        try:
            with pdfplumber.open(str(path)) as pdf:
                # Password-protected check
                if pdf.pages is None:
                    logger.warning("PDFConnector: skipping encrypted PDF %s", path.name)
                    return None

                pages: list[PDFPage] = []
                all_text_parts: list[str] = []

                for i, pdf_page in enumerate(pdf.pages, start=1):
                    text, method = self._extract_page_text(pdf_page, path, i)
                    pages.append(PDFPage(
                        page_number=i,
                        text=text,
                        char_count=len(text),
                        extraction_method=method,
                    ))
                    all_text_parts.append(text)

                full_text = "\n\n".join(all_text_parts).strip()

                # Metadata from PDF info dict
                meta = pdf.metadata or {}
                author = meta.get("Author", "unknown") or "unknown"
                raw_created = meta.get("CreationDate", "")
                created_at = _parse_pdf_date(raw_created) or datetime.now(timezone.utc)
                pdf_title = meta.get("Title", "") or path.stem

        except Exception as exc:
            logger.error("PDFConnector: failed to process %s — %s", path.name, exc)
            return None

        doc_type = self._classify_doc(path.name, full_text)
        title = _clean_title(pdf_title) or path.stem

        return PDFDocument(
            doc_id=doc_id,
            filename=path.name,
            source_path=source_path,
            doc_type=doc_type,
            title=title,
            pages=pages,
            page_count=len(pages),
            author=author,
            created_at=created_at,
            full_text=full_text,
            labels=[doc_type] if doc_type != "unknown" else [],
        )

    def _extract_page_text(
        self,
        pdf_page,
        path: Path,
        page_num: int,
    ) -> tuple[str, str]:
        """
        Try pdfplumber first; if text is too short, fall back to OCR.
        Returns (text, method).
        """
        text = pdf_page.extract_text() or ""
        text = text.strip()

        if len(text) >= self._MIN_TEXT_CHARS:
            return text, "pdfplumber"

        # OCR fallback
        if _HAS_OCR:
            try:
                img = pdf_page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    logger.debug("PDFConnector: OCR fallback p%d of %s", page_num, path.name)
                    return ocr_text, "ocr"
            except Exception as exc:
                logger.debug("PDFConnector: OCR failed p%d — %s", page_num, exc)

        return text, "empty" if not text else "pdfplumber"

    # ------------------------------------------------------------------
    # Private: classification
    # ------------------------------------------------------------------

    def _classify_doc(self, filename: str, text: str) -> str:
        """Heuristic: check filename first, then first 500 chars of text."""
        probe = filename + " " + text[:500]
        for pattern, doc_type in self._TYPE_PATTERNS:
            if pattern.search(probe):
                return doc_type
        return "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pdf_date(raw: str) -> Optional[datetime]:
    """Parse PDF CreationDate string like D:20240115120000+00'00'"""
    if not raw:
        return None
    # Strip D: prefix
    raw = raw.lstrip("D:").replace("'", "")
    formats = [
        "%Y%m%d%H%M%S%z",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw[:len(fmt.replace("%z",""))], fmt.replace("%z",""))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _clean_title(title: str) -> str:
    """Remove common PDF title noise."""
    if not title:
        return ""
    title = re.sub(r"\s+", " ", title).strip()
    # Remove file extension if accidentally in title
    title = re.sub(r"\.pdf$", "", title, flags=re.I)
    return title


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        connector = PDFConnector(source_dir=sys.argv[1])
    else:
        # Fallback: look for any PDF in CWD
        connector = PDFConnector(source_dir=".")

    docs = connector.fetch_all()
    print(f"\nProcessed {len(docs)} PDFs\n")

    for doc in docs[:3]:
        d = doc.to_dict()
        d["full_text"] = d["full_text"][:300] + "..."   # truncate for display
        print(json.dumps(d, indent=2, default=str))
        print("---")