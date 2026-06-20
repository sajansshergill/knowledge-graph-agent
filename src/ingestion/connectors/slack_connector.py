"""
slack_connector.py
------------------
Reads a Slack export directory (the ZIP you download from Slack admin)
OR queries the Slack Web API for live channel history.

Two modes:
    1. EXPORT MODE  — point at an unzipped Slack export folder
                      (channels/ sub-dirs with JSON message files)
    2. API MODE     — query channels live via Bot Token + Conversations API

Env vars (API mode):
    SLACK_BOT_TOKEN       xoxb-...
    SLACK_CHANNEL_IDS     comma-separated channel IDs, e.g. C123,C456
                          (leave blank to scan all public channels)

Returns a list of SlackThread dataclasses — each thread is a parent
message plus its replies, collapsed into one logical unit for chunking.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class SlackMessage:
    ts: str                 # Slack timestamp — unique message ID
    user: str               # user ID or display name from export
    text: str
    channel_id: str
    channel_name: str
    posted_at: datetime
    thread_ts: Optional[str] = None     # set if this is a reply
    reactions: list[str] = field(default_factory=list)


@dataclass
class SlackThread:
    thread_ts: str
    channel_id: str
    channel_name: str
    parent: SlackMessage
    replies: list[SlackMessage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate parent + replies into one document string."""
        parts = [f"[{self.parent.user}]: {self.parent.text}"]
        for r in self.replies:
            parts.append(f"  ↳ [{r.user}]: {r.text}")
        return "\n".join(parts)

    @property
    def updated_at(self) -> datetime:
        if self.replies:
            return self.replies[-1].posted_at
        return self.parent.posted_at

    def to_dict(self) -> dict:
        return {
            "source": "slack",
            "thread_ts": self.thread_ts,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "full_text": self.full_text,
            "participant_count": len({self.parent.user} | {r.user for r in self.replies}),
            "reply_count": len(self.replies),
            "created_at": self.parent.posted_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class SlackConnector:
    """
    Usage — export mode:
        connector = SlackConnector(export_dir="/path/to/slack-export")
        threads = connector.fetch_from_export()

    Usage — API mode:
        connector = SlackConnector()
        threads = connector.fetch_from_api(channel_ids=["C123", "C456"])
    """

    _API_BASE = "https://slack.com/api"
    _MAX_RETRIES = 3
    _BACKOFF = 1.5
    _PAGE_SIZE = 200        # Slack conversations.history max

    def __init__(
        self,
        bot_token: Optional[str] = None,
        export_dir: Optional[str] = None,
    ) -> None:
        self._export_dir = Path(export_dir) if export_dir else None
        self._token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Mode 1: Export
    # ------------------------------------------------------------------

    def fetch_from_export(
        self,
        min_thread_length: int = 1,
    ) -> list[SlackThread]:
        """
        Parse an unzipped Slack export directory.
        Structure expected:
            export_dir/
                channels.json
                users.json
                {channel-name}/
                    2024-01-01.json
                    2024-01-02.json
                    ...
        """
        if not self._export_dir or not self._export_dir.exists():
            raise ValueError(f"export_dir not found: {self._export_dir}")

        users = self._load_user_map()
        channel_meta = self._load_channel_map()

        threads: list[SlackThread] = []

        for channel_dir in sorted(self._export_dir.iterdir()):
            if not channel_dir.is_dir():
                continue
            channel_name = channel_dir.name
            channel_id = channel_meta.get(channel_name, {}).get("id", channel_name)

            # Collect all messages from daily JSON files
            raw_messages: list[dict] = []
            for json_file in sorted(channel_dir.glob("*.json")):
                try:
                    raw_messages.extend(json.loads(json_file.read_text()))
                except json.JSONDecodeError as exc:
                    logger.warning("SlackConnector: bad JSON in %s — %s", json_file, exc)

            channel_threads = self._group_into_threads(
                raw_messages, channel_id, channel_name, users
            )
            filtered = [t for t in channel_threads if len(t.replies) >= min_thread_length - 1]
            threads.extend(filtered)

        logger.info("SlackConnector (export): %d threads across %d channels", len(threads), len(list(self._export_dir.iterdir())))
        return threads

    # ------------------------------------------------------------------
    # Mode 2: API
    # ------------------------------------------------------------------

    def fetch_from_api(
        self,
        channel_ids: Optional[list[str]] = None,
        oldest: Optional[datetime] = None,
    ) -> list[SlackThread]:
        """
        Pull channel history via Slack Web API.
        Requires SLACK_BOT_TOKEN with channels:history + channels:read scopes.
        """
        if not self._token:
            raise EnvironmentError("SLACK_BOT_TOKEN not set")

        ids = channel_ids or self._channel_ids_from_env()
        if not ids:
            ids = self._list_all_channels()

        oldest_ts = str(oldest.timestamp()) if oldest else "0"
        threads: list[SlackThread] = []

        for cid in ids:
            cname = self._resolve_channel_name(cid)
            messages = self._fetch_channel_history(cid, oldest_ts)
            channel_threads = self._group_into_threads(messages, cid, cname, {})
            threads.extend(channel_threads)
            logger.info("SlackConnector (api): channel=%s threads=%d", cname, len(channel_threads))

        return threads

    # ------------------------------------------------------------------
    # Private: grouping
    # ------------------------------------------------------------------

    def _group_into_threads(
        self,
        raw: list[dict],
        channel_id: str,
        channel_name: str,
        users: dict[str, str],
    ) -> list[SlackThread]:
        """Group flat message list into threads by thread_ts."""
        parents: dict[str, SlackMessage] = {}
        replies: dict[str, list[SlackMessage]] = {}

        for item in raw:
            if item.get("type") != "message":
                continue
            if item.get("subtype") in ("channel_join", "channel_leave", "bot_message"):
                continue

            msg = self._parse_message(item, channel_id, channel_name, users)
            ts = item.get("ts", "")
            thread_ts = item.get("thread_ts", ts)

            if ts == thread_ts:
                parents[ts] = msg
            else:
                replies.setdefault(thread_ts, []).append(msg)

        threads = []
        for ts, parent in parents.items():
            thread = SlackThread(
                thread_ts=ts,
                channel_id=channel_id,
                channel_name=channel_name,
                parent=parent,
                replies=sorted(replies.get(ts, []), key=lambda m: m.ts),
            )
            threads.append(thread)

        return threads

    def _parse_message(
        self,
        item: dict,
        channel_id: str,
        channel_name: str,
        users: dict[str, str],
    ) -> SlackMessage:
        ts = item.get("ts", "0")
        user_id = item.get("user", item.get("username", "unknown"))
        user_display = users.get(user_id, user_id)

        reactions = [
            r.get("name", "") for r in item.get("reactions", [])
        ]

        return SlackMessage(
            ts=ts,
            user=user_display,
            text=item.get("text", ""),
            channel_id=channel_id,
            channel_name=channel_name,
            posted_at=_ts_to_dt(ts),
            thread_ts=item.get("thread_ts"),
            reactions=reactions,
        )

    # ------------------------------------------------------------------
    # Private: export helpers
    # ------------------------------------------------------------------

    def _load_user_map(self) -> dict[str, str]:
        """Returns {user_id: display_name} from users.json."""
        users_file = self._export_dir / "users.json"
        if not users_file.exists():
            return {}
        try:
            raw = json.loads(users_file.read_text())
            return {
                u["id"]: u.get("profile", {}).get("display_name") or u.get("name", u["id"])
                for u in raw
                if "id" in u
            }
        except Exception as exc:
            logger.warning("SlackConnector: could not load users.json — %s", exc)
            return {}

    def _load_channel_map(self) -> dict[str, dict]:
        """Returns {channel_name: {id, ...}} from channels.json."""
        channels_file = self._export_dir / "channels.json"
        if not channels_file.exists():
            return {}
        try:
            raw = json.loads(channels_file.read_text())
            return {c["name"]: c for c in raw if "name" in c}
        except Exception as exc:
            logger.warning("SlackConnector: could not load channels.json — %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Private: API helpers
    # ------------------------------------------------------------------

    def _fetch_channel_history(self, channel_id: str, oldest: str) -> list[dict]:
        messages: list[dict] = []
        cursor: Optional[str] = None

        while True:
            params: dict = {
                "channel": channel_id,
                "limit": self._PAGE_SIZE,
                "oldest": oldest,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._api_get("conversations.history", params)
            if not data.get("ok"):
                logger.error("SlackConnector: conversations.history error — %s", data.get("error"))
                break

            messages.extend(data.get("messages", []))

            # Fetch replies for threaded messages
            for msg in data.get("messages", []):
                if msg.get("reply_count", 0) > 0:
                    replies = self._fetch_thread_replies(channel_id, msg["ts"])
                    messages.extend(replies[1:])    # skip parent (already in messages)

            meta = data.get("response_metadata", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        return messages

    def _fetch_thread_replies(self, channel_id: str, thread_ts: str) -> list[dict]:
        data = self._api_get("conversations.replies", {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": 200,
        })
        if not data.get("ok"):
            return []
        return data.get("messages", [])

    def _list_all_channels(self) -> list[str]:
        data = self._api_get("conversations.list", {"limit": 200, "types": "public_channel"})
        if not data.get("ok"):
            return []
        return [c["id"] for c in data.get("channels", [])]

    def _resolve_channel_name(self, channel_id: str) -> str:
        data = self._api_get("conversations.info", {"channel": channel_id})
        if data.get("ok"):
            return data.get("channel", {}).get("name", channel_id)
        return channel_id

    def _channel_ids_from_env(self) -> list[str]:
        raw = os.environ.get("SLACK_CHANNEL_IDS", "")
        return [c.strip() for c in raw.split(",") if c.strip()]

    def _api_get(self, method: str, params: dict) -> dict:
        url = f"{self._API_BASE}/{method}"
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Slack rate-limit: Tier 3 = 50 req/min
            if not data.get("ok") and data.get("error") == "ratelimited":
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning("SlackConnector: rate limited, sleeping %ds", retry_after)
                time.sleep(retry_after)
                return self._api_get(method, params)
            return data
        except requests.RequestException as exc:
            logger.error("SlackConnector: API call %s failed — %s", method, exc)
            return {"ok": False, "error": str(exc)}

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self._MAX_RETRIES,
            backoff_factor=self._BACKOFF,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_dt(ts: str) -> datetime:
    """Convert Slack float-string timestamp to UTC datetime."""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (ValueError, OSError):
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        # Export mode: python slack_connector.py /path/to/export
        connector = SlackConnector(export_dir=sys.argv[1])
        threads = connector.fetch_from_export()
    else:
        # API mode
        connector = SlackConnector()
        threads = connector.fetch_from_api()

    print(f"\nFetched {len(threads)} threads\n")
    for t in threads[:3]:
        print(json.dumps(t.to_dict(), indent=2, default=str))
        print("---")