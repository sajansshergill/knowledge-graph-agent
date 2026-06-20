"""
confluence_connector.py
-----------------------
Pulls pages from a Confluence Cloud space via REST API v2.
Handles pagination, incremental sync (last_modified cursor), and
rate-limit back-off. Returns a list of ConfluencePage dataclasses
ready for the chunker.

Auth: HTTP Basic (email + API token).
Env vars required:
    CONFLUENCE_BASE_URL   e.g. https://your-org.atlassian.net
    CONFLUENCE_EMAIL      e.g. sajan@company.com
    CONFLUENCE_API_TOKEN  Atlassian API token (not password)
    CONFLUENCE_SPACE_KEY  e.g. ENG
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class ConfluencePage:
    page_id: str
    title: str
    space_key: str
    body_text: str                      # stripped plain text (no HTML)
    author_email: str
    created_at: datetime
    updated_at: datetime
    url: str
    labels: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    # metadata forwarded to the graph loader
    def to_dict(self) -> dict:
        return {
            "source": "confluence",
            "page_id": self.page_id,
            "title": self.title,
            "space_key": self.space_key,
            "body_text": self.body_text,
            "author_email": self.author_email,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "url": self.url,
            "labels": self.labels,
            "parent_id": self.parent_id,
        }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class ConfluenceConnector:
    """
    Fetches pages from one Confluence space.

    Usage:
        connector = ConfluenceConnector()
        pages = connector.fetch_all(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
        for page in pages:
            print(page.title, len(page.body_text))
    """

    _PAGE_LIMIT = 50            # Confluence v2 max per call
    _RETRY_BACKOFF = 2.0        # seconds, doubles on each retry
    _MAX_RETRIES = 4

    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        space_key: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.environ["CONFLUENCE_BASE_URL"]).rstrip("/")
        self.space_key = space_key or os.environ["CONFLUENCE_SPACE_KEY"]
        email = email or os.environ["CONFLUENCE_EMAIL"]
        token = api_token or os.environ["CONFLUENCE_API_TOKEN"]

        self._session = self._build_session(email, token)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        since: Optional[datetime] = None,
    ) -> list[ConfluencePage]:
        """
        Return all pages in the space, optionally filtered to those
        updated after `since` (UTC-aware datetime).
        """
        pages: list[ConfluencePage] = []
        for raw in self._paginate(since):
            page = self._parse_page(raw)
            if page:
                pages.append(page)
        logger.info("ConfluenceConnector: fetched %d pages from space=%s", len(pages), self.space_key)
        return pages

    def fetch_page(self, page_id: str) -> Optional[ConfluencePage]:
        """Fetch a single page by ID."""
        url = f"{self.base_url}/wiki/api/v2/pages/{page_id}"
        params = {"body-format": "atlas_doc_format", "include-labels": "true"}
        resp = self._get(url, params=params)
        if resp is None:
            return None
        return self._parse_page(resp.json())

    # ------------------------------------------------------------------
    # Private: pagination
    # ------------------------------------------------------------------

    def _paginate(self, since: Optional[datetime]) -> Generator[dict, None, None]:
        url = f"{self.base_url}/wiki/api/v2/spaces/{self.space_key}/pages"
        params: dict = {
            "limit": self._PAGE_LIMIT,
            "body-format": "storage",   # XHTML-like; we strip tags below
            "include-labels": "true",
        }

        while True:
            resp = self._get(url, params=params)
            if resp is None:
                break

            data = resp.json()
            results = data.get("results", [])

            for item in results:
                updated_raw = item.get("version", {}).get("createdAt", "")
                if since and updated_raw:
                    updated_dt = _parse_iso(updated_raw)
                    if updated_dt and updated_dt < since:
                        continue        # skip pages older than cursor
                yield item

            # Follow cursor-based pagination
            next_url = data.get("_links", {}).get("next")
            if not next_url:
                break
            # next_url is a relative path
            url = self.base_url + next_url
            params = {}     # already encoded in next_url

    # ------------------------------------------------------------------
    # Private: parsing
    # ------------------------------------------------------------------

    def _parse_page(self, raw: dict) -> Optional[ConfluencePage]:
        try:
            page_id = raw["id"]
            title = raw.get("title", "")
            space_key = raw.get("spaceId", self.space_key)  # v2 returns spaceId

            # Body — storage format is XHTML; strip tags for plain text
            body_storage = (
                raw.get("body", {})
                   .get("storage", {})
                   .get("value", "")
            )
            body_text = _strip_html(body_storage)

            # Author
            author_email = (
                raw.get("version", {})
                   .get("authorId", "unknown")
            )

            # Dates
            created_at = _parse_iso(raw.get("createdAt", "")) or datetime.now(timezone.utc)
            updated_at = _parse_iso(
                raw.get("version", {}).get("createdAt", "")
            ) or created_at

            # URL
            web_ui = raw.get("_links", {}).get("webui", "")
            url = f"{self.base_url}/wiki{web_ui}" if web_ui else ""

            # Labels
            labels = [
                lbl.get("name", "")
                for lbl in raw.get("labels", {}).get("results", [])
            ]

            # Parent
            parent_id = raw.get("parentId")

            return ConfluencePage(
                page_id=page_id,
                title=title,
                space_key=space_key,
                body_text=body_text,
                author_email=author_email,
                created_at=created_at,
                updated_at=updated_at,
                url=url,
                labels=labels,
                parent_id=parent_id,
            )
        except Exception as exc:
            logger.warning("ConfluenceConnector: failed to parse page — %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private: HTTP
    # ------------------------------------------------------------------

    def _build_session(self, email: str, token: str) -> requests.Session:
        session = requests.Session()
        session.auth = (email, token)
        session.headers.update({"Accept": "application/json"})

        retry = Retry(
            total=self._MAX_RETRIES,
            backoff_factor=self._RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                logger.warning("ConfluenceConnector: rate limited, sleeping %ds", retry_after)
                time.sleep(retry_after)
                resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.error("ConfluenceConnector: GET %s failed — %s", url, exc)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Minimal HTML tag stripper — no external deps required."""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse ISO-8601 string to UTC-aware datetime."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    connector = ConfluenceConnector()
    pages = connector.fetch_all()

    print(f"\nFetched {len(pages)} pages\n")
    for p in pages[:3]:
        print(json.dumps(p.to_dict(), indent=2, default=str))