import sqlite3

import pytest

from lethyx.memory.local_service import build_local_service_identity
from lethyx.memory.operational_slo import (
    BACKENDS,
    CONCURRENCY_LEVELS,
    CORPUS_TIERS,
    DEFAULT_ONE_HOP_RECALL_BUDGET,
    DEFAULT_RECONSTRUCTIVE_RECALL_BUDGET,
    DEFAULT_SERVICE_SLO_MATRIX,
    QUERY_MODES,
    CostSample,
    OperationSample,
    OperationalSloLedger,
    PrefixCacheSample,
    RecallBudget,
    RecallUsage,
    StorageSample,
    decide_recall_budget,
    service_slo_contract_revision,
)


def _identity(tmp_path):
    return build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="release",
        backend="mempalace",
        caller_role="operator",
    )


def test_numeric_slo_matrix_covers_every_backend_tier_concurrency_and_mode() -> None:
    expected = len(BACKENDS) * len(CORPUS_TIERS) * len(CONCURRENCY_LEVELS) * len(QUERY_MODES)
    keys = {
        (row.backend, row.corpus_tier, row.concurrency, row.query_mode)
        for row in DEFAULT_SERVICE_SLO_MATRIX
    }

    assert len(DEFAULT_SERVICE_SLO_MATRIX) == expected == 432
    assert len(keys) == expected
    assert all(
        0 < row.p50_ms <= row.p95_ms <= row.p99_ms
        and row.min_throughput_per_second > 0
        and row.max_queue_lag_ms > 0
        and row.max_projection_lag_ms > 0
        and row.rto_ms > 0
        and 0 < row.max_memory_turn_fraction <= 0.75
        for row in DEFAULT_SERVICE_SLO_MATRIX
    )
    assert service_slo_contract_revision().startswith("service-slo:sha256:")
    assert DEFAULT_ONE_HOP_RECALL_BUDGET.max_steps == 2
    assert DEFAULT_RECONSTRUCTIVE_RECALL_BUDGET.max_steps == 6


def test_budget_allows_bounded_widening_and_abstains_without_hidden_overrun() -> None:
    budget = RecallBudget(
        max_latency_ms=1_000,
        max_steps=4,
        max_tokens=1_000,
        max_candidates=20,
    )
    usage = RecallUsage(latency_ms=300, steps=1, tokens=300, candidates=4)

    assert decide_recall_budget(budget, usage).action == "continue"
    assert (
        decide_recall_budget(
            budget,
            usage,
            widening_cost=RecallUsage(
                latency_ms=500,
                steps=2,
                tokens=500,
                candidates=10,
            ),
        ).action
        == "widen"
    )
    refused = decide_recall_budget(
        budget,
        usage,
        widening_cost=RecallUsage(
            latency_ms=800,
            steps=4,
            tokens=800,
            candidates=20,
        ),
    )
    assert refused.action == "abstain"
    assert refused.reason == "widening_budget_unavailable"
    exhausted = decide_recall_budget(
        budget,
        RecallUsage(latency_ms=1_001, steps=1, tokens=1, candidates=1),
    )
    assert exhausted.action == "abstain"
    assert exhausted.reason == "hard_budget_exhausted"


def test_latency_report_separates_cold_warm_reconstruction_and_model(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    modes = {
        "cold_start": "cold_exact",
        "warm_recall": "warm_exact",
        "reconstruction": "warm_reconstructive",
        "model": "warm_hybrid",
    }
    for component, mode in modes.items():
        ledger.record_operation(
            OperationSample(
                backend="mempalace",
                corpus_tier="1k",
                concurrency=1,
                query_mode=mode,
                component=component,
                latency_ms=10,
                operation_count=10,
                window_ms=100,
                queue_lag_ms=5,
                projection_lag_ms=6,
                recovery_time_ms=7,
                agent_turn_ms=100,
            )
        )

    report = ledger.latency_report()
    groups = report["groups"]
    assert {group["component"] for group in groups} == set(modes)
    assert all(group["p50_ms"] == group["p95_ms"] == group["p99_ms"] == 10 for group in groups)
    assert all(group["memory_latency_fraction_of_agent_turn"] == 0.1 for group in groups)
    assert report["all_native_qualified"] is True
    assert report["all_observed_slos_passed"] is True


def test_unavailable_or_sidecar_native_sample_can_never_pass_slo(tmp_path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="release",
        backend="hindsight",
        caller_role="operator",
    )
    ledger = OperationalSloLedger(identity)
    ledger.record_operation(
        OperationSample(
            backend="hindsight",
            corpus_tier="1k",
            concurrency=1,
            query_mode="warm_exact",
            component="warm_recall",
            latency_ms=1,
            operation_count=1_000,
            window_ms=1,
            native_ok=False,
            effective_mode="sidecar",
        )
    )

    report = ledger.latency_report()
    assert report["all_native_qualified"] is False
    assert report["all_observed_slos_passed"] is False
    assert report["groups"][0]["sample_count"] == 0
    assert report["groups"][0]["slo_passed"] is False


def test_one_hop_memory_latency_cannot_dominate_a_measured_agent_turn(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    ledger.record_operation(
        OperationSample(
            backend="mempalace",
            corpus_tier="1k",
            concurrency=1,
            query_mode="warm_exact",
            component="warm_recall",
            latency_ms=200,
            operation_count=10,
            window_ms=100,
            agent_turn_ms=300,
        )
    )

    group = ledger.latency_report()["groups"][0]
    assert group["memory_latency_fraction_of_agent_turn"] > 0.5
    assert group["slo_passed"] is False


def test_headline_cost_requires_online_and_every_background_category(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    categories = (
        "capture",
        "embedding",
        "online_recall",
        "model",
        "consolidation",
        "policy_qualification",
        "rebuild",
        "background_maintenance",
    )
    for category in categories:
        ledger.record_cost(
            CostSample(
                category=category,
                latency_ms=1,
                compute_ms=2,
                tokens=3,
                bytes_written=4,
            )
        )

    report = ledger.headline_cost_report()
    assert report["complete"] is True
    assert report["missing_categories"] == []
    assert report["totals"] == {
        "latency_ms": 8.0,
        "compute_ms": 16.0,
        "tokens": 24,
        "bytes_written": 32,
    }
    assert set(report["background"]) == {
        "capture",
        "embedding",
        "consolidation",
        "policy_qualification",
        "rebuild",
        "background_maintenance",
    }


def test_storage_and_prefix_reports_include_full_operating_cost(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    ledger.record_storage(
        StorageSample(
            canonical_bytes=100,
            index_bytes=40,
            artifact_bytes=10,
            total_write_bytes=200,
            catalog_revision="catalog-1",
            guidance_tokens=20,
            compaction_compute_ms=5,
            refinement_compute_ms=7,
        )
    )
    ledger.record_storage(
        StorageSample(
            canonical_bytes=200,
            index_bytes=90,
            artifact_bytes=30,
            total_write_bytes=500,
            catalog_revision="catalog-2",
            guidance_tokens=25,
            compaction_compute_ms=15,
            refinement_compute_ms=27,
        )
    )
    for source in ("static", "working_fold"):
        ledger.record_prefix_cache(
            PrefixCacheSample(
                source=source,
                eligible_tokens=100,
                reused_tokens=80,
                invalidated_tokens=20,
                input_tokens=80,
                baseline_input_tokens=100,
                total_compute_ms=90,
                baseline_total_compute_ms=100,
            )
        )
    ledger.record_prefix_cache(
        PrefixCacheSample(
            source="dynamic_guidance",
            eligible_tokens=100,
            reused_tokens=20,
            invalidated_tokens=80,
            input_tokens=70,
            baseline_input_tokens=100,
            total_compute_ms=120,
            baseline_total_compute_ms=100,
        )
    )

    storage = ledger.storage_report()
    assert storage["storage_growth_bytes"] == {
        "canonical": 100,
        "index": 50,
        "artifact": 20,
    }
    assert storage["write_amplification"] == 3.0
    assert storage["catalog_churn"] == 1
    assert storage["guidance_overhead_tokens"] == 25
    assert storage["compaction_compute_ms"] == 10.0
    assert storage["refinement_compute_ms"] == 20.0

    prefix = ledger.prefix_cache_report()
    assert prefix["complete"] is True
    assert prefix["sources"]["working_fold"]["invalidation_rate"] == 0.2
    assert prefix["sources"]["dynamic_guidance"]["input_tokens_saved"] == 30
    assert prefix["sources"]["dynamic_guidance"]["optimization_accepted"] is False
    assert prefix["all_optimizations_accepted"] is False


def test_operating_ledger_is_content_free(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    secret_revision = "catalog:subject-that-must-not-persist"
    ledger.record_storage(
        StorageSample(
            canonical_bytes=1,
            index_bytes=1,
            artifact_bytes=0,
            total_write_bytes=1,
            catalog_revision=secret_revision,
            guidance_tokens=0,
            compaction_compute_ms=0,
            refinement_compute_ms=0,
        )
    )

    connection = sqlite3.connect(ledger.path)
    try:
        persisted = connection.execute(
            "SELECT catalog_revision_sha256 FROM storage_samples"
        ).fetchone()[0]
    finally:
        connection.close()
    assert secret_revision not in persisted
    assert len(persisted) == 64


def test_partial_provider_cost_and_unpaired_prefix_never_claim_zero_or_savings(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    ledger.record_cost(
        CostSample(
            category="model",
            latency_ms=120,
            compute_ms=None,
            tokens=250,
            bytes_written=None,
            observation_id="phase-1",
        )
    )
    ledger.record_cost(
        CostSample(
            category="model",
            latency_ms=120,
            compute_ms=None,
            tokens=250,
            bytes_written=None,
            observation_id="phase-1",
        )
    )
    cost = ledger.headline_cost_report()
    assert cost["categories"]["model"]["tokens"] == 250
    assert cost["categories"]["model"]["compute_ms"] is None
    assert cost["complete"] is False
    with pytest.raises(ValueError, match="observation"):
        ledger.record_cost(
            CostSample(
                category="model",
                latency_ms=120,
                compute_ms=None,
                tokens=999,
                bytes_written=None,
                observation_id="phase-1",
            )
        )
    ledger.record_prefix_cache(
        PrefixCacheSample(
            source="dynamic_guidance",
            eligible_tokens=None,
            reused_tokens=None,
            invalidated_tokens=None,
            input_tokens=210,
            baseline_input_tokens=None,
            total_compute_ms=None,
            baseline_total_compute_ms=None,
            observation_id="phase-1",
        )
    )
    prefix = ledger.prefix_cache_report()
    assert prefix["complete"] is False
    assert prefix["sources"]["dynamic_guidance"]["input_tokens_saved"] is None
    assert prefix["sources"]["dynamic_guidance"]["reuse_rate"] is None
    assert prefix["sources"]["dynamic_guidance"]["optimization_accepted"] is None
    assert prefix["all_optimizations_accepted"] is False


def test_slo_additive_observations_preserve_legacy_cost_rows(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "INSERT INTO cost_samples(category,latency_ms,compute_ms,tokens,bytes_written,recorded_at) VALUES(?,?,?,?,?,?)",
            ("model", 10, 5, 20, 0, "2026-09-14T00:00:00Z"),
        )
    ledger.record_cost(CostSample("model", None, None, 30, None, "new-phase"))
    report = ledger.headline_cost_report()
    assert report["categories"]["model"]["tokens"] == 50
    assert report["categories"]["model"]["latency_ms"] is None
    sample = PrefixCacheSample("static", None, None, None, 25, None, None, None, "new-phase")
    ledger.record_prefix_cache(sample)
    ledger.record_prefix_cache(sample)
    assert ledger.prefix_cache_report()["sources"]["static"]["sample_count"] == 1


def test_public_runtime_usage_is_partial_idempotent_and_scope_bound(tmp_path) -> None:
    from lethyx import MemoryClient

    client = MemoryClient(
        tmp_path, workspace_id="cost-test", profile_id="umbrella", backend="mempalace"
    )
    observed = client.observe_runtime_cost(
        observation_id="sealed-work:1",
        input_tokens=100,
        total_tokens=120,
    )
    repeated = client.observe_runtime_cost(
        observation_id="sealed-work:1",
        input_tokens=100,
        total_tokens=120,
    )
    assert observed == repeated
    assert observed["cost"]["categories"]["model"]["tokens"] == 120
    assert observed["cost"]["categories"]["model"]["latency_ms"] is None
    assert observed["prefix_cache"]["sources"]["request"]["optimization_accepted"] is None
    other = MemoryClient(tmp_path, workspace_id="other", profile_id="umbrella", backend="mempalace")
    isolated = other.observe_runtime_cost(
        observation_id="sealed-work:1",
        input_tokens=10,
        total_tokens=12,
    )
    assert isolated["cost"]["categories"]["model"]["tokens"] == 12
