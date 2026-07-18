"""End-to-end baseline test: NoMemory provider + EchoResponder.

No network, no LLM, no Lethyx dependency.  Tests the full pipeline:
runner -> provider -> responder -> judge -> CaseResult.
"""

from memory_bench.adapters.baselines import NoMemoryProvider, FullContextProvider
from memory_bench.adapters.responder_llm import EchoResponder
from memory_bench.adapters.judges import F1Judge, AbstentionJudge
from memory_bench.benchmarks.recall_driver import run_recall_episode
from memory_bench.core.types import (
    CaseResult,
    DatasetSpec,
    Episode,
    JudgeConfig,
    LimitsSpec,
    MemoryRecord,
    Probe,
    ResponderSpec,
    RunSpec,
    Session,
)


def make_spec(**overrides) -> RunSpec:
    defaults = dict(
        providers=(),
        benchmarks=("locomo",),
        dataset=DatasetSpec(name=""),
        responder=ResponderSpec(),
        judge=JudgeConfig(use_llm_judge=False),
        limits=LimitsSpec(recall_k=3),
        out_dir="runs/test_baseline",
    )
    defaults.update(overrides)
    return RunSpec(**defaults)


def make_episode(records=None) -> Episode:
    recs = records or [
        MemoryRecord(id="r0", content="Alice lives in Paris.", evidence_id="ev0"),
        MemoryRecord(id="r1", content="Bob works at a bakery.", evidence_id="ev1"),
    ]
    sess = Session(id="s0", ts="2024-01-01", records=tuple(recs))
    return Episode(id="ep_test", sessions=(sess,))


def make_probe(question="Where does Alice live?", gold=("Paris",), abstain=False) -> Probe:
    return Probe(
        id="p0",
        question=question,
        gold=gold,
        category="test",
        evidence_ids=("ev0",),
        abstain=abstain,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_context_correct_answer():
    """FullContext provider + EchoResponder configured to answer correctly."""
    provider = FullContextProvider()
    store = provider.namespace("ep_test")
    ep = make_episode()
    store.write_many(ep.all_records)

    responder = EchoResponder({"Where does Alice live?": "Paris"})
    judge = F1Judge()
    spec = make_spec()
    probe = make_probe()

    result = run_recall_episode(
        store=store,
        episode=ep,
        probe=probe,
        responder=responder,
        judge=judge,
        spec=spec,
        provider_id="baseline-full-context",
        benchmark_name="test",
    )

    assert result.status == "ok"
    assert result.judgment is not None
    assert result.judgment.correct
    assert result.judgment.score > 0.5
    store.close()


def test_no_memory_always_wrong():
    """NoMemory provider: retrieved context is empty, so echo returns 'I don't know'."""
    provider = NoMemoryProvider()
    store = provider.namespace("ep_test")
    ep = make_episode()
    store.write_many(ep.all_records)

    # EchoResponder maps the question to nothing -> returns default "I don't know"
    responder = EchoResponder()
    judge = F1Judge()
    spec = make_spec()
    probe = make_probe()

    result = run_recall_episode(
        store=store,
        episode=ep,
        probe=probe,
        responder=responder,
        judge=judge,
        spec=spec,
        provider_id="baseline-no-memory",
        benchmark_name="test",
    )

    assert result.status == "ok"
    assert result.judgment is not None
    # "I don't know" has near-zero F1 against "Paris"
    assert result.judgment.score < 0.2
    store.close()


def test_abstention_probe_correct():
    """Abstention probe: model should say it doesn't know."""
    provider = NoMemoryProvider()
    store = provider.namespace("ep_abs")
    ep = make_episode()
    store.write_many(ep.all_records)

    responder = EchoResponder(default="I don't know")
    judge = AbstentionJudge()
    spec = make_spec()
    probe = make_probe(question="Does Alice have a dog?", gold=(), abstain=True)

    result = run_recall_episode(
        store=store,
        episode=ep,
        probe=probe,
        responder=responder,
        judge=judge,
        spec=spec,
        provider_id="no_memory",
        benchmark_name="test",
    )

    assert result.status == "ok"
    assert result.judgment is not None
    assert result.judgment.correct
    store.close()


def test_case_result_serialisation():
    """CaseResult roundtrips through to_dict/from_dict."""
    from memory_bench.core.types import Judgment, Usage

    cr = CaseResult(
        provider="test",
        benchmark="locomo",
        episode_id="ep1",
        probe_id="p1",
        category="single_hop",
        status="ok",
        judgment=Judgment(correct=True, score=0.85, metric="f1"),
        retrieval={"recall_at_10": 0.5},
        usage=Usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.001, calls=1),
        latency_ms=123.4,
        diagnostics={"failure_mode": "ok", "answer_text": "Alice"},
    )
    d = cr.to_dict()
    cr2 = CaseResult.from_dict(d)
    assert cr2.provider == cr.provider
    assert cr2.judgment is not None
    assert abs(cr2.judgment.score - cr.judgment.score) < 0.001
    assert cr2.usage.prompt_tokens == 100
    assert cr2.diagnostics["answer_text"] == "Alice"


def test_report_from_results():
    """Build a report from a set of fake CaseResults."""
    from memory_bench.core.report import build_report, render_markdown
    from memory_bench.core.types import Judgment

    results = [
        CaseResult(
            provider="p1",
            benchmark="locomo",
            episode_id="ep1",
            probe_id="q1",
            category="single_hop",
            status="ok",
            judgment=Judgment(correct=True, score=0.9, metric="f1"),
        ),
        CaseResult(
            provider="p1",
            benchmark="locomo",
            episode_id="ep1",
            probe_id="q2",
            category="temporal",
            status="ok",
            judgment=Judgment(correct=False, score=0.1, metric="f1"),
        ),
    ]
    report = build_report(results)
    assert report["schema"] == "memory_bench/report/v1"
    assert len(report["rollups"]) > 0
    md = render_markdown(report)
    assert "locomo" in md
    assert "p1" in md


def test_runner_jsonl_resume_handles_unicode_line_separator(tmp_path):
    """JSONL resume should not split valid JSON strings containing U+2028."""
    from memory_bench.core.runner import _append_result, _read_existing
    from memory_bench.core.types import Judgment

    row = CaseResult(
        provider="p1",
        benchmark="locomo",
        episode_id="ep1",
        probe_id="q1",
        category="single_hop",
        status="ok",
        judgment=Judgment(correct=True, score=1.0, metric="f1"),
        diagnostics={"answer_text": "before\u2028after"},
    )
    path = tmp_path / "provider" / "locomo.jsonl"

    _append_result(path, row)

    assert _read_existing(path) == {"ep1::q1"}
    assert path.read_text(encoding="utf-8").count("\n") == 1
