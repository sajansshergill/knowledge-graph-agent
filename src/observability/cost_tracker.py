"""Token and cost estimation utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass


MODEL_PRICING_PER_1K = {
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "text-embedding-004": {"input": 0.000025, "output": 0.0},
}


@dataclass
class CostEstimate:
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
        }


class CostTracker:
    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(re.findall(r"\S+", text)) * 1.3)) if text else 0

    def estimate_cost(self, model: str, input_text: str = "", output_text: str = "") -> CostEstimate:
        tokens_in = self.estimate_tokens(input_text)
        tokens_out = self.estimate_tokens(output_text)
        pricing = MODEL_PRICING_PER_1K.get(model, MODEL_PRICING_PER_1K["gemini-1.5-flash"])
        cost = (tokens_in / 1000 * pricing["input"]) + (tokens_out / 1000 * pricing["output"])
        return CostEstimate(model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=round(cost, 6))
