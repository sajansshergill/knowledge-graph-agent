from src.ingestion.chunker import Chunker
from src.ingestion.pubsub_publisher import PubSubPublisher, synthetic_documents


def test_chunker_creates_storage_ready_chunks():
    doc = {
        "source": "confluence",
        "page_id": "DOC-1",
        "title": "Auth ADR",
        "body_text": " ".join(f"word{i}" for i in range(30)),
        "author_email": "alice@example.com",
        "labels": ["architecture"],
        "url": "https://example.com/doc",
    }

    chunks = Chunker(max_tokens=10, overlap_ratio=0.2).chunk_document(doc)

    assert len(chunks) == 4
    assert chunks[0].source_id == "DOC-1"
    assert chunks[0].source_type == "confluence"
    assert chunks[0].author == "alice@example.com"
    assert chunks[0].labels == ["architecture"]
    assert chunks[0].text_hash


def test_pubsub_publisher_dry_run_returns_messages():
    docs = synthetic_documents(2)
    publisher = PubSubPublisher(dry_run=True)

    messages = publisher.publish_many("synthetic", docs)

    assert len(messages) == 2
    assert messages[0].source == "synthetic"
    assert messages[0].payload["source"] == "confluence"
