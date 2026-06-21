"""Multi-agent orchestration entry point."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .eval_agent import EvalAgent
from .retrieval_agent import RetrievalAgent
from .router_agent import RouterAgent
from .synthesis_agent import SynthesisAgent


@dataclass
class AgentHop:
    agent_name: str
    latency_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class OrchestratorResult:
    trace_id: str
    query: str
    query_type: str
    answer: str
    citations: list[dict]
    retrieval_path: str
    eval: dict
    latency_ms: int
    hops: list[AgentHop] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "query_type": self.query_type,
            "answer": self.answer,
            "citations": self.citations,
            "retrieval_path": self.retrieval_path,
            "eval": self.eval,
            "latency_ms": self.latency_ms,
            "hops": [h.to_dict() for h in self.hops],
            "error": self.error,
        }


class Orchestrator:
    def __init__(
        self,
        router: Optional[RouterAgent] = None,
        retrieval: Optional[RetrievalAgent] = None,
        synthesis: Optional[SynthesisAgent] = None,
        evaluator: Optional[EvalAgent] = None,
        trace_logger: Optional[Any] = None,
    ) -> None:
        self.router = router or RouterAgent()
        self.retrieval = retrieval or RetrievalAgent()
        self.synthesis = synthesis or SynthesisAgent()
        self.evaluator = evaluator or EvalAgent()
        self.trace_logger = trace_logger

    def query(self, query: str, top_k: int = 5, session_id: Optional[str] = None) -> OrchestratorResult:
        trace_id = str(uuid.uuid4())
        started = time.time()
        hops: list[AgentHop] = []

        route, latency = _timed(lambda: self.router.route(query))
        hops.append(AgentHop("router", latency, route.to_dict()))

        retrieval_result, latency = _timed(lambda: self.retrieval.retrieve(query, route.query_type, top_k=top_k))
        hops.append(AgentHop("retrieval", latency, retrieval_result.to_dict()))

        synthesis_result, latency = _timed(
            lambda: self.synthesis.synthesize(query, retrieval_result.chunks, route.query_type)
        )
        hops.append(AgentHop("synthesis", latency, synthesis_result.to_dict()))

        eval_result, latency = _timed(
            lambda: self.evaluator.evaluate(query, synthesis_result.answer, retrieval_result.chunks)
        )
        hops.append(AgentHop("eval", latency, eval_result.to_dict()))

        result = OrchestratorResult(
            trace_id=trace_id,
            query=query,
            query_type=route.query_type,
            answer=synthesis_result.answer,
            citations=[c.to_dict() for c in synthesis_result.citations],
            retrieval_path=retrieval_result.retrieval_path,
            eval=eval_result.to_dict(),
            latency_ms=int((time.time() - started) * 1000),
            hops=hops,
            error=retrieval_result.error,
        )
        self._log_trace(result, session_id)
        return result

    def _log_trace(self, result: OrchestratorResult, session_id: Optional[str]) -> None:
        if not self.trace_logger:
            return
        try:
            self.trace_logger.log_query(result.to_dict(), session_id=session_id)
        except Exception:
            return


def _timed(fn):
    started = time.time()
    result = fn()
    return result, int((time.time() - started) * 1000)
