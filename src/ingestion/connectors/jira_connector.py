"""
jira_connector.py
-----------------
Fetches issues from Jira Cloud via REST API v3.
Supports JQL filtering, incremental sync by updatedDate,
and relationship extraction (blocks/is-blocked-by, epic links,
parent/child) for the Neo4j graph loader.

Env vars required:
    JIRA_BASE_URL     e.g. https://your-org.atlassian.net
    JIRA_EMAIL        e.g. sajan@company.com
    JIRA_API_TOKEN    Atlassian API token
    JIRA_PROJECT_KEY  e.g. PLAT  (comma-separated for multiple)

Optional:
    JIRA_JQL_EXTRA    extra JQL appended to default query
                      e.g. 'AND labels = "architecture"'
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class JiraComment:
    comment_id: str
    author_email: str
    body: str
    created_at: datetime
    updated_at: datetime


@dataclass
class JiraLink:
    link_type: str          # "blocks" | "is blocked by" | "relates to" | etc.
    target_key: str         # e.g. PLAT-42


@dataclass
class JiraIssue:
    issue_id: str           # numeric Jira ID
    key: str                # e.g. PLAT-123
    summary: str
    description: str
    issue_type: str         # Bug | Story | Task | Epic | Spike
    status: str             # To Do | In Progress | Done | Blocked
    priority: str           # Highest | High | Medium | Low | Lowest
    assignee_email: str
    reporter_email: str
    project_key: str
    epic_key: Optional[str]
    parent_key: Optional[str]
    labels: list[str]
    components: list[str]
    comments: list[JiraComment]
    links: list[JiraLink]
    created_at: datetime
    updated_at: datetime
    resolution_date: Optional[datetime]
    url: str

    @property
    def full_text(self) -> str:
        """Concatenate all text fields for chunking."""
        parts = [f"[{self.key}] {self.summary}"]
        if self.description:
            parts.append(self.description)
        for c in self.comments:
            parts.append(f"Comment by {c.author_email}:\n{c.body}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "source": "jira",
            "issue_id": self.issue_id,
            "key": self.key,
            "summary": self.summary,
            "full_text": self.full_text,
            "issue_type": self.issue_type,
            "status": self.status,
            "priority": self.priority,
            "assignee_email": self.assignee_email,
            "reporter_email": self.reporter_email,
            "project_key": self.project_key,
            "epic_key": self.epic_key,
            "parent_key": self.parent_key,
            "labels": self.labels,
            "components": self.components,
            "comment_count": len(self.comments),
            "links": [{"type": l.link_type, "target": l.target_key} for l in self.links],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolution_date": self.resolution_date.isoformat() if self.resolution_date else None,
            "url": self.url,
        }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class JiraConnector:
    """
    Usage:
        connector = JiraConnector()

        # All issues in configured project(s)
        issues = connector.fetch_all()

        # Incremental: only issues updated since a date
        from datetime import datetime, timezone
        issues = connector.fetch_all(since=datetime(2024, 6, 1, tzinfo=timezone.utc))

        # Custom JQL
        issues = connector.fetch_by_jql('project = PLAT AND status = Blocked')
    """

    _PAGE_SIZE = 100        # Jira max per page
    _MAX_RETRIES = 4
    _BACKOFF = 2.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        project_key: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.environ["JIRA_BASE_URL"]).rstrip("/")
        self._project_key = project_key or os.environ.get("JIRA_PROJECT_KEY", "")
        email = email or os.environ["JIRA_EMAIL"]
        token = api_token or os.environ["JIRA_API_TOKEN"]

        self._session = self._build_session(email, token)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch_all(self, since: Optional[datetime] = None) -> list[JiraIssue]:
        """
        Fetch all issues in configured project(s), optionally
        filtered to those updated after `since`.
        """
        project_keys = [k.strip() for k in self._project_key.split(",") if k.strip()]
        if not project_keys:
            raise EnvironmentError("JIRA_PROJECT_KEY not set")

        project_clause = " OR ".join(f'project = "{k}"' for k in project_keys)
        jql = f"({project_clause})"

        if since:
            ts = since.strftime("%Y-%m-%d %H:%M")
            jql += f' AND updated >= "{ts}"'

        extra = os.environ.get("JIRA_JQL_EXTRA", "").strip()
        if extra:
            jql += f" {extra}"

        jql += " ORDER BY updated DESC"
        return self.fetch_by_jql(jql)

    def fetch_by_jql(self, jql: str) -> list[JiraIssue]:
        """Fetch issues matching arbitrary JQL."""
        issues: list[JiraIssue] = []
        start_at = 0

        logger.info("JiraConnector: JQL = %s", jql)

        while True:
            batch = self._search(jql, start_at)
            if batch is None:
                break

            raw_issues = batch.get("issues", [])
            for raw in raw_issues:
                issue = self._parse_issue(raw)
                if issue:
                    issues.append(issue)

            total = batch.get("total", 0)
            start_at += len(raw_issues)
            logger.info("JiraConnector: fetched %d / %d", start_at, total)

            if start_at >= total or not raw_issues:
                break

        logger.info("JiraConnector: total issues fetched = %d", len(issues))
        return issues

    def fetch_issue(self, key: str) -> Optional[JiraIssue]:
        """Fetch a single issue by key (e.g. 'PLAT-42')."""
        url = f"{self.base_url}/rest/api/3/issue/{key}"
        params = {
            "expand": "renderedFields,names,changelog",
            "fields": ",".join(self._FIELDS),
        }
        resp = self._get(url, params)
        if resp is None:
            return None
        return self._parse_issue(resp.json())

    # ------------------------------------------------------------------
    # Private: API
    # ------------------------------------------------------------------

    _FIELDS = [
        "summary", "description", "issuetype", "status", "priority",
        "assignee", "reporter", "project", "parent", "labels",
        "components", "comment", "issuelinks", "created", "updated",
        "resolutiondate", "customfield_10014",  # Epic Link (classic projects)
        "customfield_10016",                     # Story Points (informational)
    ]

    def _search(self, jql: str, start_at: int) -> Optional[dict]:
        url = f"{self.base_url}/rest/api/3/search"
        payload = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": self._PAGE_SIZE,
            "fields": self._FIELDS,
        }
        try:
            resp = self._session.post(url, json=payload, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                logger.warning("JiraConnector: rate limited — sleeping %ds", retry_after)
                time.sleep(retry_after)
                resp = self._session.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("JiraConnector: search failed — %s", exc)
            return None

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.error("JiraConnector: GET %s failed — %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Private: parsing
    # ------------------------------------------------------------------

    def _parse_issue(self, raw: dict) -> Optional[JiraIssue]:
        try:
            fields = raw.get("fields", {})
            issue_id = raw["id"]
            key = raw["key"]
            project_key = fields.get("project", {}).get("key", self._project_key)

            # Text
            summary = fields.get("summary", "")
            description = _extract_adf_text(fields.get("description")) or ""

            # Type / status / priority
            issue_type = fields.get("issuetype", {}).get("name", "unknown")
            status = fields.get("status", {}).get("name", "unknown")
            priority = fields.get("priority", {}).get("name", "unknown")

            # People
            assignee_email = _extract_email(fields.get("assignee"))
            reporter_email = _extract_email(fields.get("reporter"))

            # Hierarchy
            epic_key = (
                fields.get("customfield_10014")        # classic epic link
                or _epic_from_parent(fields)            # next-gen parent
            )
            parent_raw = fields.get("parent")
            parent_key = parent_raw.get("key") if parent_raw else None

            # Labels / components
            labels = fields.get("labels", [])
            components = [c.get("name", "") for c in fields.get("components", [])]

            # Comments
            comments = [
                self._parse_comment(c)
                for c in fields.get("comment", {}).get("comments", [])
            ]

            # Issue links (blocks / is blocked by / relates to)
            links: list[JiraLink] = []
            for link in fields.get("issuelinks", []):
                link_type_name = link.get("type", {}).get("name", "relates to")
                if "inwardIssue" in link:
                    links.append(JiraLink(
                        link_type=link.get("type", {}).get("inward", link_type_name),
                        target_key=link["inwardIssue"]["key"],
                    ))
                if "outwardIssue" in link:
                    links.append(JiraLink(
                        link_type=link.get("type", {}).get("outward", link_type_name),
                        target_key=link["outwardIssue"]["key"],
                    ))

            # Dates
            created_at = _parse_iso(fields.get("created", "")) or datetime.now(timezone.utc)
            updated_at = _parse_iso(fields.get("updated", "")) or created_at
            resolution_date = _parse_iso(fields.get("resolutiondate", ""))

            url = f"{self.base_url}/browse/{key}"

            return JiraIssue(
                issue_id=issue_id,
                key=key,
                summary=summary,
                description=description,
                issue_type=issue_type,
                status=status,
                priority=priority,
                assignee_email=assignee_email,
                reporter_email=reporter_email,
                project_key=project_key,
                epic_key=epic_key,
                parent_key=parent_key,
                labels=labels,
                components=components,
                comments=comments,
                links=links,
                created_at=created_at,
                updated_at=updated_at,
                resolution_date=resolution_date,
                url=url,
            )

        except Exception as exc:
            logger.warning("JiraConnector: failed to parse issue %s — %s",
                           raw.get("key", "?"), exc)
            return None

    def _parse_comment(self, raw: dict) -> JiraComment:
        return JiraComment(
            comment_id=raw.get("id", ""),
            author_email=_extract_email(raw.get("author")),
            body=_extract_adf_text(raw.get("body")) or "",
            created_at=_parse_iso(raw.get("created", "")) or datetime.now(timezone.utc),
            updated_at=_parse_iso(raw.get("updated", "")) or datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Private: session
    # ------------------------------------------------------------------

    def _build_session(self, email: str, token: str) -> requests.Session:
        session = requests.Session()
        session.auth = (email, token)
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        retry = Retry(
            total=self._MAX_RETRIES,
            backoff_factor=self._BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_email(person: Optional[dict]) -> str:
    if not person:
        return "unknown"
    return person.get("emailAddress") or person.get("displayName") or "unknown"


def _epic_from_parent(fields: dict) -> Optional[str]:
    """Next-gen projects store epic as a parent with type=Epic."""
    parent = fields.get("parent")
    if not parent:
        return None
    if parent.get("fields", {}).get("issuetype", {}).get("name") == "Epic":
        return parent.get("key")
    return None


def _extract_adf_text(node: Optional[dict], _depth: int = 0) -> str:
    """
    Recursively extract plain text from Atlassian Document Format (ADF) JSON.
    Jira v3 returns description/body as ADF, not plain text.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node

    node_type = node.get("type", "")
    text = node.get("text", "")

    if node_type == "text":
        return text

    parts: list[str] = []
    for child in node.get("content", []):
        parts.append(_extract_adf_text(child, _depth + 1))

    joined = " ".join(p for p in parts if p)

    # Add newlines around block-level nodes
    if node_type in ("paragraph", "heading", "bulletList", "orderedList", "listItem",
                     "blockquote", "codeBlock", "panel"):
        joined = "\n" + joined + "\n"

    return joined


def _parse_iso(ts: str) -> Optional[datetime]:
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
    import sys
    logging.basicConfig(level=logging.INFO)

    connector = JiraConnector()

    if len(sys.argv) > 1:
        # Single issue mode: python jira_connector.py PLAT-42
        issue = connector.fetch_issue(sys.argv[1])
        if issue:
            print(json.dumps(issue.to_dict(), indent=2, default=str))
    else:
        issues = connector.fetch_all()
        print(f"\nFetched {len(issues)} issues\n")
        for issue in issues[:3]:
            d = issue.to_dict()
            d["full_text"] = d["full_text"][:300] + "..."
            print(json.dumps(d, indent=2, default=str))
            print("---")