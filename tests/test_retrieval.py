from src.retrieval import GraphQuery, QueryRewriter


class FakeNeo4j:
    def __init__(self):
        self.calls = []

    def cypher_query(self, query, params=None):
        params = params or {}
        self.calls.append((query, params))

        if "MATCH (start:Document)" in query:
            return [
                {
                    "chunk_id": "jira-1",
                    "source_type": "jira",
                    "source_id": "PLAT-123",
                    "title": "Auth migration blocker",
                    "text": "PLAT-123 blocks the auth service migration",
                    "author": "alice@example.com",
                    "url": "https://jira.example/PLAT-123",
                    "labels": ["auth"],
                    "updated_at": "2026-06-01T00:00:00Z",
                    "score": 1.1,
                }
            ]

        if "MATCH (e:Entity)" in query:
            return [
                {
                    "chunk_id": "entity-1",
                    "source_type": "confluence",
                    "source_id": "DOC-1",
                    "title": "Auth service ADR",
                    "text": "The auth service moved because of connection pressure.",
                    "author": "bob@example.com",
                    "url": "https://docs.example/auth",
                    "labels": ["architecture"],
                    "updated_at": "2026-06-02T00:00:00Z",
                    "score": 1.0,
                },
                {
                    "chunk_id": "jira-1",
                    "source_type": "jira",
                    "source_id": "PLAT-123",
                    "title": "Auth migration blocker",
                    "text": "Duplicate lower score should be merged.",
                    "author": "alice@example.com",
                    "url": "https://jira.example/PLAT-123",
                    "labels": ["auth"],
                    "updated_at": "2026-06-01T00:00:00Z",
                    "score": 0.7,
                },
            ]

        if "MATCH (c:Chunk)" in query:
            return [
                {
                    "chunk_id": "kw-1",
                    "source_type": "confluence",
                    "source_id": "DOC-2",
                    "title": "Onboarding checklist",
                    "text": "Follow this onboarding checklist.",
                    "author": "carol@example.com",
                    "url": "https://docs.example/onboarding",
                    "labels": ["onboarding"],
                    "updated_at": "2026-06-03T00:00:00Z",
                    "score": 0.5,
                }
            ]

        return []


def test_graph_query_runs_entity_and_relationship_strategies():
    fake = FakeNeo4j()
    graph = GraphQuery(neo4j=fake, limit=10, max_hops=2)

    hits = graph.search("Why did service auth depend on PLAT-123?", query_type="relational")

    assert [hit["chunk_id"] for hit in hits] == ["jira-1", "entity-1"]
    assert hits[0]["score"] == 1.1
    assert "jira_chain" in hits[0]["graph_strategies"]
    assert any("entity_lookup" in hit["graph_strategies"] for hit in hits)
    assert graph.explain()["entities"] == ["plat-123", "auth"]
    assert len(fake.calls) >= 3


def test_graph_query_uses_keyword_fallback_without_entities():
    fake = FakeNeo4j()
    graph = GraphQuery(neo4j=fake)

    hits = graph.search("onboarding checklist", query_type="relational")

    assert hits[0]["chunk_id"] == "kw-1"
    assert hits[0]["graph_strategies"] == ["keyword_fallback"]
    assert graph.explain()["keywords"] == ["onboarding", "checklist"]


def test_query_rewriter_expands_acronyms_and_adds_type_hints():
    rewritten = QueryRewriter().rewrite("What is the JWT lifetime?", "factual")

    assert "json web token" in rewritten.lower()
    assert "definition value setting status summary" in rewritten
