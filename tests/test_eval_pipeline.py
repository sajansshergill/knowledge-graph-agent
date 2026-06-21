from src.observability import CostTracker, EvalPipeline


def test_eval_pipeline_scores_dataset():
    summary = EvalPipeline(min_faithfulness=0.1).run_dataset(
        [
            {
                "query": "Why did auth move?",
                "answer": "Auth moved because connection pools were exhausted.",
                "contexts": ["The auth service moved because connection pools were exhausted."],
            }
        ]
    )

    assert summary.case_count == 1
    assert summary.faithfulness > 0
    assert summary.passed


def test_cost_tracker_estimates_cost():
    estimate = CostTracker().estimate_cost("gemini-1.5-flash", "hello world", "short answer")

    assert estimate.tokens_in > 0
    assert estimate.cost_usd >= 0
