from memory_bench.adapters.judges import F1Judge, LLMJudge
from memory_bench.benchmarks.longmemeval import LongMemEvalBenchmark
from memory_bench.core.runner import _resolve_judge
from memory_bench.core.types import Answer, DatasetSpec, JudgeConfig, Probe, RunSpec

import pytest


def _spec(judge: JudgeConfig) -> RunSpec:
    return RunSpec(providers=(), benchmarks=(), dataset=DatasetSpec(name=""), judge=judge)


def test_resolve_judge_uses_f1_when_llm_judge_disabled():
    spec = _spec(JudgeConfig(use_llm_judge=False))

    judge = _resolve_judge(LongMemEvalBenchmark(), spec)

    assert isinstance(judge, F1Judge)


def test_resolve_judge_passes_configured_llm_judge_model():
    spec = _spec(JudgeConfig(llm_judge_model="judge-model", use_llm_judge=True))

    judge = _resolve_judge(LongMemEvalBenchmark(), spec)

    assert isinstance(judge, LLMJudge)
    assert judge._config.model == "judge-model"


def test_llm_judge_fails_closed_instead_of_falling_back_to_f1(monkeypatch, tmp_path):
    import memory_bench.adapters.judges as judges

    def boom(*args, **kwargs):
        raise RuntimeError("model denied")

    monkeypatch.setattr(judges, "call_llm", boom)
    judge = LLMJudge(model="gemma-4-31B-it", cache_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="LLM judge failed"):
        judge.judge(
            Probe(id="p", question="What is the answer?", gold=("answer",)),
            Answer(text="answer"),
        )
