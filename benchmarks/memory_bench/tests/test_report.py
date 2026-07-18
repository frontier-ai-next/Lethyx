from memory_bench.core.report import build_comparison_diagnostics, build_report
from memory_bench.core.types import CaseResult, Judgment, Usage


def _case(category: str, score: float, correct: bool = False) -> CaseResult:
    return CaseResult(
        provider="p",
        benchmark="b",
        episode_id=f"ep-{category}-{score}",
        probe_id="q",
        category=category,
        status="ok",
        judgment=Judgment(correct=correct, score=score, metric="f1"),
        retrieval={"recall_at_10": score, "mrr": score, "ndcg_at_10": score},
        usage=Usage(prompt_tokens=1, completion_tokens=1, calls=1),
    )


def test_benchmark_rollup_reports_row_and_category_weighted_means() -> None:
    rows = [
        _case("large", 0.0),
        _case("large", 0.0),
        _case("large", 0.0),
        _case("small", 1.0, correct=True),
    ]

    [rollup] = build_report(rows)["rollups"]

    assert rollup["mean_score"] == 0.25
    assert rollup["mean_score_row_weighted"] == 0.25
    assert rollup["mean_score_category_weighted"] == 0.5
    assert rollup["recall_at_k"] == 0.25
    assert rollup["recall_at_k_category_weighted"] == 0.5


def test_benchmark_rollup_includes_top_level_mrr_and_ndcg() -> None:
    rows = [_case("a", 0.25), _case("b", 0.75)]

    [rollup] = build_report(rows)["rollups"]

    assert rollup["mean_mrr"] == 0.5
    assert rollup["mean_ndcg"] == 0.5


def test_report_includes_lethyx_underperformance_diagnostics() -> None:
    rows = [
        CaseResult(
            provider="lethyx-mempalace",
            benchmark="agent_quality",
            episode_id="ep1",
            probe_id="p1",
            category="cross_session_fact",
            status="ok",
            judgment=Judgment(correct=False, score=0.0, metric="f1"),
            retrieval={"recall_at_10": 0.0},
            diagnostics={
                "failure_mode": "retrieval_miss",
                "missing_evidence": ["e1"],
                "answer_text": "I don't know",
            },
        ),
        CaseResult(
            provider="baseline-bm25",
            benchmark="agent_quality",
            episode_id="ep1",
            probe_id="p1",
            category="cross_session_fact",
            status="ok",
            judgment=Judgment(correct=True, score=1.0, metric="f1"),
            retrieval={"recall_at_10": 1.0},
            diagnostics={"failure_mode": "ok", "answer_text": "Nina"},
        ),
    ]

    diagnostics = build_comparison_diagnostics(rows)
    report = build_report(rows)

    assert len(diagnostics) == 1
    assert diagnostics[0]["provider"] == "lethyx-mempalace"
    assert diagnostics[0]["reason"] == "retrieval_miss"
    assert report["diagnostics"]["summary"]["n_lethyx_underperforming_cases"] == 1
