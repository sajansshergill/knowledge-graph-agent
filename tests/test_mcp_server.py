from fastapi.testclient import TestClient

from src.mcp_server.server import app


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint_returns_agent_payload():
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is the auth service?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["query_type"] == "factual"
    assert "answer" in body
    assert len(body["hops"]) == 4
