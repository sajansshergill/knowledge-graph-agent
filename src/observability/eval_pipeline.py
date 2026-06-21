"""Offline evaluation pipeline for EKGA answers."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class EvalCaseResult:
    query: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
        }


@dataclass
class EvalSummary:
    case_count: int
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    passed: bool
    results: list[EvalCaseResult]

    def to_dict(self) -> dict:
        return {
            "case_count": self.case_count,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }


class EvalPipeline:
    def __init__(self, orchestrator: Optional[Any] = None, min_faithfulness: float = 0.80) -> None:
        self.orchestrator = orchestrator
        self.min_faithfulness = min_faithfulness

    def run_dataset(self, cases: Iterable[dict[str, Any]]) -> EvalSummary:
        results: list[EvalCaseResult] = []
        for case in cases:
            query = case.get("query", "")
            expected = case.get("answer") or case.get("expected_answer") or ""
            contexts = case.get("contexts") or case.get("context") or []
            if isinstance(contexts, str):
                contexts = [contexts]

            actual = case.get("actual_answer")
            if actual is None and self.orchestrator:
                actual = self.orchestrator.query(query).answer
            actual = actual or expected

            context_text = " ".join(contexts) or expected
            results.append(
                EvalCaseResult(
                    query=query,
                    faithfulness=_overlap(actual, context_text),
                    answer_relevancy=_overlap(query, actual),
                    context_recall=_overlap(expected, context_text),
                )
            )

        return self._summary(results)

    def run_file(self, dataset_path: str) -> EvalSummary:
        data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        cases = data.get("cases", data) if isinstance(data, dict) else data
        return self.run_dataset(cases)

    def _summary(self, results: list[EvalCaseResult]) -> EvalSummary:
        count = len(results)
        if count == 0:
            return EvalSummary(0, 0.0, 0.0, 0.0, False, [])
        faithfulness = round(sum(r.faithfulness for r in results) / count, 3)
        relevancy = round(sum(r.answer_relevancy for r in results) / count, 3)
        recall = round(sum(r.context_recall for r in results) / count, 3)
        return EvalSummary(count, faithfulness, relevancy, recall, faithfulness >= self.min_faithfulness, results)


def _overlap(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return round(len(left_terms & right_terms) / len(left_terms), 3)


def _terms(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "what", "why", "how"}
    return {t for t in re.findall(r"[a-z0-9_-]{3,}", text.lower()) if t not in stop}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EKGA evaluation")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--min-faithfulness", type=float, default=0.80)
    args = parser.parse_args()
    summary = EvalPipeline(min_faithfulness=args.min_faithfulness).run_file(args.dataset)
    print(json.dumps(summary.to_dict(), indent=2))
    raise SystemExit(0 if summary.passed else 1)


if __name__ == "__main__":
    main()
