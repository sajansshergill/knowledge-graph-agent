"""FastAPI server wrapping the EKGA orchestrator."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..agents import Orchestrator
from .schemas import HealthResponse, QueryRequest, QueryResponse


app = FastAPI(
    title="EKGA MCP Server",
    version="0.1.0",
    description="Enterprise Knowledge Graph Agent API",
)

_orchestrator = Orchestrator()


@app.get("/")
def root() -> dict:
    return {
        "service": "ekga-mcp-server",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "query": "/query",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = _orchestrator.query(
            request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        )
        return QueryResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
