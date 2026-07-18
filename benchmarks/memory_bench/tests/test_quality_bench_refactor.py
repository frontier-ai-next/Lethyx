"""Simple quality-benchmark launcher and native-provider regressions."""

from memory_bench.core.types import (
    DatasetSpec,
    JudgeConfig,
    LimitsSpec,
    MemoryRecord,
    ResponderSpec,
    RunSpec,
)

import pytest


def _spec(**dataset_overrides) -> RunSpec:
    return RunSpec(
        providers=(),
        benchmarks=("agent_quality",),
        dataset=DatasetSpec(name="agent_quality", **dataset_overrides),
        responder=ResponderSpec(),
        judge=JudgeConfig(use_llm_judge=False),
        limits=LimitsSpec(recall_k=3),
    )


def test_agent_quality_benchmark_is_self_contained_and_limited() -> None:
    from memory_bench.benchmarks.agent_quality import AgentQualityBenchmark

    bench = AgentQualityBenchmark()
    items = list(bench.load(_spec(max_episodes=1, max_probes_per_episode=3)))

    assert len(items) == 1
    episode, probes = items[0]
    assert episode.all_records
    assert len(probes) == 3
    assert {probe.category for probe in probes} <= {
        "cross_session_fact",
        "preference_update",
        "procedural_memory",
        "conflict_resolution",
        "abstention",
    }
    assert all(probe.evidence_ids or probe.abstain for probe in probes)


def test_quality_bench_suite_is_small_but_covers_lethyx_and_clean_backends() -> None:
    from memory_bench.scripts.quality_bench import build_suite_toml

    suite = build_suite_toml(
        max_episodes=1,
        max_probes_per_episode=2,
        include_external=False,
        include_raw_hindsight=False,
        use_llm_judge=False,
    )

    assert '"agent_quality"' in suite
    for provider_id in (
        "lethyx-mempalace",
        "lethyx-hindsight",
        "lethyx-mnemosyne",
        "raw-mempalace",
        "raw-mnemosyne",
        "baseline-no-memory",
        "baseline-full-context",
        "baseline-bm25",
    ):
        assert f'id = "{provider_id}"' in suite
    assert 'id = "raw-hindsight"' not in suite
    assert "max_parallel = 1" in suite
    assert "use_llm_judge = false" in suite


def test_raw_mnemosyne_provider_round_trips_memory(tmp_path) -> None:
    from memory_bench.adapters.lethyx_provider import RawMnemosyneProvider

    provider = RawMnemosyneProvider({"repo_root": str(tmp_path)})
    store = provider.namespace("agent_quality/ep")
    records = [
        MemoryRecord(
            id="r0",
            content="The deployment codename is ORCHID and the rollback owner is Nina.",
            evidence_id="ev-orchid",
            kind="observation",
        )
    ]

    store.write_many(records)
    hits = store.retrieve("Which codename has rollback owner Nina?", k=3)

    assert any("ORCHID" in hit.text for hit in hits)
    assert any(hit.source_id == "ev-orchid" for hit in hits)
    store.close()


def test_bm25_provider_fails_closed_when_rank_bm25_is_missing(monkeypatch) -> None:
    from memory_bench.adapters import baselines

    store = baselines._BM25Store()
    store.write_many(
        [
            MemoryRecord(
                id="r0",
                content="The deployment codename is ORCHID and the rollback owner is Nina.",
                evidence_id="ev-orchid",
            )
        ]
    )
    monkeypatch.setattr(baselines, "_try_import_bm25", lambda: None)

    with pytest.raises(RuntimeError, match="rank-bm25"):
        store.retrieve("rollback owner", k=1)
