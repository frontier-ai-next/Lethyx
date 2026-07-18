"""Agentic runner regression tests."""

from memory_bench.benchmarks.agentic_driver import run_agentic_episode
from memory_bench.core.types import (
    DatasetSpec,
    Episode,
    JudgeConfig,
    LimitsSpec,
    Probe,
    ResponderSpec,
    RunSpec,
    Session,
)


class _OneStepEnv:
    def reset(self):
        return {"observation": "start"}

    def step(self, action: str):
        return {"observation": "done"}, 1.0, True, {}

    def score(self) -> float:
        return 1.0


class _Bench:
    name = "agentic_test"


class _Provider:
    name = "bm25"


class _Store:
    def retrieve(self, *args, **kwargs):
        return []

    def write(self, record) -> None:
        self.last_record = record


def test_agentic_episode_uses_provider_entry_id_for_consistent_reports() -> None:
    spec = RunSpec(
        providers=(),
        benchmarks=("memoryarena",),
        dataset=DatasetSpec(name="memoryarena"),
        responder=ResponderSpec(),
        judge=JudgeConfig(use_llm_judge=False),
        limits=LimitsSpec(recall_k=3),
    )
    episode = Episode(id="ep", sessions=(Session(id="s0", ts="", records=()),))
    probe = Probe(
        id="p0",
        question="do it",
        category="test",
        kind="action",
        metadata={"env": _OneStepEnv(), "success_threshold": 0.5},
    )

    result = run_agentic_episode(
        provider=_Provider(),
        provider_id="baseline-bm25",
        bench=_Bench(),
        episode=episode,
        probe=probe,
        store=_Store(),
        spec=spec,
    )

    assert result.status == "ok"
    assert result.provider == "baseline-bm25"
