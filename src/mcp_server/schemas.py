"""Pydantic schemas for the EKGA API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    session_id: Optional[str] = None


class CitationSchema(BaseModel):
    chunk_id: str
    title: str
    url: str = ""
    source_type: str = ""


class AgentHopSchema(BaseModel):
    agent_name: str
    latency_ms: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    trace_id: str
    query: str
    query_type: str
    answer: str
    citations: list[CitationSchema] = Field(default_factory=list)
    retrieval_path: str = "none"
    eval: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int
    hops: list[AgentHopSchema] = Field(default_factory=list)
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "ekga-mcp-server"
