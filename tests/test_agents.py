from src.agents import Orchestrator, RetrievalAgent, RouterAgent


class FakeRetriever:
    def retrieve_with_reflection(self, query, query_type="factual", top_k=5):
        class Result:
            retrieval_path = "hybrid"
            latency_ms = 3
            chunks = [
                {
                    "chunk_id": "c1",
                    "source_type": "confluence",
                    "title": "Auth ADR",
                    "text": "The auth service moved because connection pools were exhausted.",
                    "url": "https://example.com/auth",
                    "labels": ["auth"],
                }
            ]

        return Result()


def test_router_classifies_query_types():
    router = RouterAgent()

    assert router.route("How do I onboard?").query_type == "procedural"
    assert router.route("Why did auth move?").query_type == "relational"
    assert router.route("What is JWT lifetime?").query_type == "factual"


def test_orchestrator_runs_agent_flow_with_fake_retriever():
    retrieval = RetrievalAgent(retriever=FakeRetriever())
    orchestrator = Orchestrator(retrieval=retrieval)

    result = orchestrator.query("Why did auth move?", top_k=1)

    assert result.query_type == "relational"
    assert result.retrieval_path == "hybrid"
    assert result.citations[0]["chunk_id"] == "c1"
    assert len(result.hops) == 4


def test_retrieval_agent_uses_demo_fallback_without_live_stores(monkeypatch):
    monkeypatch.setenv("EKGA_DEMO_MODE", "true")
    retrieval = RetrievalAgent(retriever=None)

    result = retrieval.retrieve("Why was the auth service migrated off PostgreSQL?", "relational", top_k=2)

    assert result.retrieval_path == "demo-hybrid-graph"
    assert result.chunks
    assert result.chunks[0]["chunk_id"] == "demo-adr-042"
