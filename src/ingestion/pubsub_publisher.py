"""
pubsub_publisher.py
-------------------
Small helper for publishing ingestion jobs to Pub/Sub or generating synthetic
documents for local demos.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


@dataclass
class IngestionMessage:
    message_id: str
    source: str
    payload: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "source": self.source,
            "payload": self.payload,
            "created_at": self.created_at,
        }


class PubSubPublisher:
    def __init__(
        self,
        project_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.topic_id = topic_id or os.environ.get("PUBSUB_TOPIC", "ekga-ingestion")
        self.dry_run = dry_run
        self._publisher = None
        self._topic_path = ""

        if not dry_run and self.project_id:
            try:
                from google.cloud import pubsub_v1

                self._publisher = pubsub_v1.PublisherClient()
                self._topic_path = self._publisher.topic_path(self.project_id, self.topic_id)
            except ImportError:
                self.dry_run = True

    def publish(self, source: str, payload: dict[str, Any]) -> IngestionMessage:
        message = IngestionMessage(
            message_id=str(uuid.uuid4()),
            source=source,
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        data = json.dumps(message.to_dict(), sort_keys=True).encode("utf-8")

        if self._publisher and not self.dry_run:
            future = self._publisher.publish(self._topic_path, data)
            message.message_id = future.result()
        return message

    def publish_many(self, source: str, payloads: Iterable[dict[str, Any]]) -> list[IngestionMessage]:
        return [self.publish(source, payload) for payload in payloads]


def synthetic_documents(count: int = 10) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    topics = ["auth service", "payments service", "rate limiting", "onboarding", "incident response"]
    for i in range(count):
        topic = topics[i % len(topics)]
        docs.append(
            {
                "source": "confluence",
                "page_id": f"SYN-{i + 1:04d}",
                "title": f"{topic.title()} Knowledge Note",
                "body_text": (
                    f"This synthetic document explains {topic}. "
                    f"ADR-{40 + i} records the decision, PLAT-{100 + i} tracks rollout, "
                    f"and the Platform team owns follow-up work."
                ),
                "author_email": f"user{i % 5}@example.com",
                "labels": ["synthetic", topic.replace(" ", "-")],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "url": f"https://example.com/wiki/SYN-{i + 1:04d}",
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish EKGA ingestion messages")
    parser.add_argument("--source", default="synthetic", help="source type to publish")
    parser.add_argument("--docs", type=int, default=10, help="number of synthetic docs")
    parser.add_argument("--dry-run", action="store_true", help="print messages instead of publishing")
    args = parser.parse_args()

    publisher = PubSubPublisher(dry_run=args.dry_run or args.source == "synthetic")
    payloads = synthetic_documents(args.docs) if args.source == "synthetic" else []
    messages = publisher.publish_many(args.source, payloads)
    print(json.dumps([m.to_dict() for m in messages], indent=2))


if __name__ == "__main__":
    main()
