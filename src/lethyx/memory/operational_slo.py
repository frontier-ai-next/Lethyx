"""Content-free operating SLOs, budgets, and full-cost telemetry."""

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from lethyx.memory.local_service import LocalServiceIdentity, local_service_runtime_paths


BackendName = Literal["mempalace", "hindsight", "mnemosyne", "zvec"]
CorpusTier = Literal["1k", "10k", "100k"]
QueryMode = Literal[
    "cold_exact",
    "cold_hybrid",
    "cold_temporal",
    "cold_reconstructive",
    "warm_exact",
    "warm_hybrid",
    "warm_temporal",
    "warm_reconstructive",
    "write",
    "correct",
    "erase",
    "rebuild",
]
LatencyComponent = Literal["cold_start", "warm_recall", "reconstruction", "model"]
CostCategory = Literal[
    "capture",
    "embedding",
    "online_recall",
    "model",
    "consolidation",
    "policy_qualification",
    "rebuild",
    "background_maintenance",
]
PrefixSource = Literal["static", "dynamic_guidance", "working_fold", "request"]

BACKENDS: tuple[BackendName, ...] = ("mempalace", "hindsight", "mnemosyne", "zvec")
CORPUS_TIERS: tuple[CorpusTier, ...] = ("1k", "10k", "100k")
CONCURRENCY_LEVELS = (1, 8, 32)
QUERY_MODES: tuple[QueryMode, ...] = (
    "cold_exact",
    "cold_hybrid",
    "cold_temporal",
    "cold_reconstructive",
    "warm_exact",
    "warm_hybrid",
    "warm_temporal",
    "warm_reconstructive",
    "write",
    "correct",
    "erase",
    "rebuild",
)
_COMPONENTS = ("cold_start", "warm_recall", "reconstruction", "model")
_COST_CATEGORIES: tuple[CostCategory, ...] = (
    "capture",
    "embedding",
    "online_recall",
    "model",
    "consolidation",
    "policy_qualification",
    "rebuild",
    "background_maintenance",
)
_PREFIX_SOURCES = ("static", "dynamic_guidance", "working_fold", "request")
_BASE_P95_MS: dict[QueryMode, int] = {
    "cold_exact": 2_000,
    "cold_hybrid": 3_000,
    "cold_temporal": 3_000,
    "cold_reconstructive": 8_000,
    "warm_exact": 250,
    "warm_hybrid": 750,
    "warm_temporal": 750,
    "warm_reconstructive": 3_000,
    "write": 1_000,
    "correct": 1_500,
    "erase": 3_000,
    "rebuild": 30_000,
}
_TIER_SCALE = {"1k": 1.0, "10k": 2.0, "100k": 5.0}
_CONCURRENCY_SCALE = {1: 1.0, 8: 1.75, 32: 3.5}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS operation_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend TEXT NOT NULL,
    corpus_tier TEXT NOT NULL,
    concurrency INTEGER NOT NULL,
    query_mode TEXT NOT NULL,
    component TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    steps INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    candidates INTEGER NOT NULL,
    operation_count INTEGER NOT NULL,
    window_ms REAL NOT NULL,
    native_ok INTEGER NOT NULL,
    effective_mode TEXT NOT NULL,
    queue_lag_ms REAL NOT NULL,
    projection_lag_ms REAL NOT NULL,
    recovery_time_ms REAL NOT NULL,
    agent_turn_ms REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cost_samples (
    cost_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    compute_ms REAL NOT NULL,
    tokens INTEGER NOT NULL,
    bytes_written INTEGER NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS storage_samples (
    storage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_bytes INTEGER NOT NULL,
    index_bytes INTEGER NOT NULL,
    artifact_bytes INTEGER NOT NULL,
    total_write_bytes INTEGER NOT NULL,
    catalog_revision_sha256 TEXT NOT NULL,
    guidance_tokens INTEGER NOT NULL,
    compaction_compute_ms REAL NOT NULL,
    refinement_compute_ms REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prefix_samples (
    prefix_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    eligible_tokens INTEGER NOT NULL,
    reused_tokens INTEGER NOT NULL,
    invalidated_tokens INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    baseline_input_tokens INTEGER NOT NULL,
    total_compute_ms REAL NOT NULL,
    baseline_total_compute_ms REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cost_observations (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT UNIQUE,
    category TEXT NOT NULL,
    latency_ms REAL, compute_ms REAL, tokens INTEGER, bytes_written INTEGER,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prefix_observations (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT UNIQUE,
    source TEXT NOT NULL,
    eligible_tokens INTEGER, reused_tokens INTEGER, invalidated_tokens INTEGER,
    input_tokens INTEGER, baseline_input_tokens INTEGER,
    total_compute_ms REAL, baseline_total_compute_ms REAL,
    recorded_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ServiceSloThreshold:
    backend: BackendName
    corpus_tier: CorpusTier
    concurrency: int
    query_mode: QueryMode
    p50_ms: int
    p95_ms: int
    p99_ms: int
    min_throughput_per_second: float
    max_queue_lag_ms: int
    max_projection_lag_ms: int
    rto_ms: int
    max_memory_turn_fraction: float


def default_service_slo_matrix() -> tuple[ServiceSloThreshold, ...]:
    """Return the complete numeric 3x3x3x12 release threshold matrix."""

    rows: list[ServiceSloThreshold] = []
    for backend in BACKENDS:
        backend_scale = 1.0 if backend == "mempalace" else 1.25
        for tier in CORPUS_TIERS:
            for concurrency in CONCURRENCY_LEVELS:
                scale = backend_scale * _TIER_SCALE[tier] * _CONCURRENCY_SCALE[concurrency]
                for mode in QUERY_MODES:
                    p95 = max(1, round(_BASE_P95_MS[mode] * scale))
                    rows.append(
                        ServiceSloThreshold(
                            backend=backend,
                            corpus_tier=tier,
                            concurrency=concurrency,
                            query_mode=mode,
                            p50_ms=max(1, round(p95 * 0.55)),
                            p95_ms=p95,
                            p99_ms=max(p95, round(p95 * 1.7)),
                            min_throughput_per_second=max(
                                0.01,
                                round(concurrency * 1_000 / p95, 4),
                            ),
                            max_queue_lag_ms=max(1_000, p95 * 2),
                            max_projection_lag_ms=max(2_000, p95 * 3),
                            rto_ms=max(5_000, p95 * (4 if mode == "rebuild" else 2)),
                            max_memory_turn_fraction=(0.75 if "reconstructive" in mode else 0.25),
                        )
                    )
    return tuple(rows)


DEFAULT_SERVICE_SLO_MATRIX = default_service_slo_matrix()


def service_slo_contract_revision() -> str:
    payload = json.dumps(
        [row.__dict__ for row in DEFAULT_SERVICE_SLO_MATRIX],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "service-slo:sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecallBudget:
    max_latency_ms: int
    max_steps: int
    max_tokens: int
    max_candidates: int

    def __post_init__(self) -> None:
        for name in ("max_latency_ms", "max_steps", "max_tokens", "max_candidates"):
            _positive_int(getattr(self, name), field_name=name)


@dataclass(frozen=True)
class RecallUsage:
    latency_ms: float
    steps: int
    tokens: int
    candidates: int

    def __post_init__(self) -> None:
        _non_negative_number(self.latency_ms, field_name="latency_ms")
        for name in ("steps", "tokens", "candidates"):
            _non_negative_int(getattr(self, name), field_name=name)


@dataclass(frozen=True)
class RecallBudgetDecision:
    action: Literal["continue", "widen", "abstain"]
    reason: str
    remaining_latency_ms: float
    remaining_steps: int
    remaining_tokens: int
    remaining_candidates: int


def decide_recall_budget(
    budget: RecallBudget,
    usage: RecallUsage,
    *,
    widening_cost: RecallUsage | None = None,
) -> RecallBudgetDecision:
    """Continue, widen once when it fits, or explicitly abstain at a hard bound."""

    if not isinstance(budget, RecallBudget) or not isinstance(usage, RecallUsage):
        raise TypeError("budget and usage must be typed recall budget values")
    remaining = (
        budget.max_latency_ms - usage.latency_ms,
        budget.max_steps - usage.steps,
        budget.max_tokens - usage.tokens,
        budget.max_candidates - usage.candidates,
    )
    if any(value < 0 for value in remaining):
        return RecallBudgetDecision("abstain", "hard_budget_exhausted", *remaining)
    if widening_cost is None:
        return RecallBudgetDecision("continue", "within_budget", *remaining)
    if not isinstance(widening_cost, RecallUsage):
        raise TypeError("widening_cost must be a RecallUsage")
    required = (
        widening_cost.latency_ms,
        widening_cost.steps,
        widening_cost.tokens,
        widening_cost.candidates,
    )
    if all(available >= needed for available, needed in zip(remaining, required, strict=True)):
        return RecallBudgetDecision("widen", "bounded_widening_available", *remaining)
    return RecallBudgetDecision("abstain", "widening_budget_unavailable", *remaining)


@dataclass(frozen=True)
class OperationSample:
    backend: BackendName
    corpus_tier: CorpusTier
    concurrency: int
    query_mode: QueryMode
    component: LatencyComponent
    latency_ms: float
    steps: int = 0
    tokens: int = 0
    candidates: int = 0
    operation_count: int = 1
    window_ms: float = 1.0
    native_ok: bool = True
    effective_mode: str = "selected_native"
    queue_lag_ms: float = 0.0
    projection_lag_ms: float = 0.0
    recovery_time_ms: float = 0.0
    agent_turn_ms: float = 0.0

    def __post_init__(self) -> None:
        _axis(self.backend, BACKENDS, field_name="backend")
        _axis(self.corpus_tier, CORPUS_TIERS, field_name="corpus_tier")
        _axis(self.concurrency, CONCURRENCY_LEVELS, field_name="concurrency")
        _axis(self.query_mode, QUERY_MODES, field_name="query_mode")
        _axis(self.component, _COMPONENTS, field_name="component")
        _non_negative_number(self.latency_ms, field_name="latency_ms")
        _non_negative_number(self.window_ms, field_name="window_ms")
        for name in (
            "queue_lag_ms",
            "projection_lag_ms",
            "recovery_time_ms",
            "agent_turn_ms",
        ):
            _non_negative_number(getattr(self, name), field_name=name)
        if self.agent_turn_ms and self.latency_ms > self.agent_turn_ms:
            raise ValueError("memory latency cannot exceed the measured agent turn")
        for name in ("steps", "tokens", "candidates"):
            _non_negative_int(getattr(self, name), field_name=name)
        _positive_int(self.operation_count, field_name="operation_count")
        if not isinstance(self.native_ok, bool):
            raise ValueError("native_ok must be a bool")
        if not str(self.effective_mode or "").strip():
            raise ValueError("effective_mode is required")


@dataclass(frozen=True)
class CostSample:
    category: CostCategory
    latency_ms: float | None
    compute_ms: float | None
    tokens: int | None
    bytes_written: int | None
    observation_id: str = ""

    def __post_init__(self) -> None:
        _axis(self.category, _COST_CATEGORIES, field_name="category")
        _observation_id(self.observation_id)
        for name in ("latency_ms", "compute_ms"):
            if getattr(self, name) is not None:
                _non_negative_number(getattr(self, name), field_name=name)
        for name in ("tokens", "bytes_written"):
            if getattr(self, name) is not None:
                _non_negative_int(getattr(self, name), field_name=name)


@dataclass(frozen=True)
class StorageSample:
    canonical_bytes: int
    index_bytes: int
    artifact_bytes: int
    total_write_bytes: int
    catalog_revision: str
    guidance_tokens: int
    compaction_compute_ms: float
    refinement_compute_ms: float

    def __post_init__(self) -> None:
        for name in (
            "canonical_bytes",
            "index_bytes",
            "artifact_bytes",
            "total_write_bytes",
            "guidance_tokens",
        ):
            _non_negative_int(getattr(self, name), field_name=name)
        for name in ("compaction_compute_ms", "refinement_compute_ms"):
            _non_negative_number(getattr(self, name), field_name=name)
        if not str(self.catalog_revision or "").strip():
            raise ValueError("catalog_revision is required")


@dataclass(frozen=True)
class PrefixCacheSample:
    source: PrefixSource
    eligible_tokens: int | None
    reused_tokens: int | None
    invalidated_tokens: int | None
    input_tokens: int | None
    baseline_input_tokens: int | None
    total_compute_ms: float | None
    baseline_total_compute_ms: float | None
    observation_id: str = ""

    def __post_init__(self) -> None:
        _axis(self.source, _PREFIX_SOURCES, field_name="source")
        _observation_id(self.observation_id)
        for name in (
            "eligible_tokens",
            "reused_tokens",
            "invalidated_tokens",
            "input_tokens",
            "baseline_input_tokens",
        ):
            if getattr(self, name) is not None:
                _non_negative_int(getattr(self, name), field_name=name)
        if self.eligible_tokens is not None and (
            (self.reused_tokens or 0) + (self.invalidated_tokens or 0) > self.eligible_tokens
        ):
            raise ValueError("prefix cache accounting exceeds eligible tokens")
        for name in ("total_compute_ms", "baseline_total_compute_ms"):
            if getattr(self, name) is not None:
                _non_negative_number(getattr(self, name), field_name=name)


class OperationalSloLedger:
    """Durable numeric-only telemetry for release SLO and total-cost reports."""

    def __init__(self, identity: LocalServiceIdentity) -> None:
        if not isinstance(identity, LocalServiceIdentity):
            raise TypeError("identity must be a LocalServiceIdentity")
        self.identity = identity
        registry = local_service_runtime_paths(identity).registry
        digest = hashlib.sha256(
            f"{identity.repo_root.resolve()}|{identity.workspace_id}|{identity.backend}".encode()
        ).hexdigest()[:20]
        self.path = registry.with_name(f"{registry.stem}.{digest}.slo.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    def record_operation(self, sample: OperationSample) -> None:
        if not isinstance(sample, OperationSample):
            raise TypeError("sample must be an OperationSample")
        if sample.backend != self.identity.backend:
            raise ValueError("operation sample backend does not match selected service backend")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO operation_samples(backend,corpus_tier,concurrency,query_mode,component,latency_ms,steps,tokens,candidates,operation_count,window_ms,native_ok,effective_mode,queue_lag_ms,projection_lag_ms,recovery_time_ms,agent_turn_ms,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sample.backend,
                    sample.corpus_tier,
                    sample.concurrency,
                    sample.query_mode,
                    sample.component,
                    sample.latency_ms,
                    sample.steps,
                    sample.tokens,
                    sample.candidates,
                    sample.operation_count,
                    sample.window_ms,
                    int(sample.native_ok),
                    sample.effective_mode,
                    sample.queue_lag_ms,
                    sample.projection_lag_ms,
                    sample.recovery_time_ms,
                    sample.agent_turn_ms,
                    _now(),
                ),
            )

    def record_cost(self, sample: CostSample) -> None:
        if not isinstance(sample, CostSample):
            raise TypeError("sample must be a CostSample")
        values = (
            sample.category,
            sample.latency_ms,
            sample.compute_ms,
            sample.tokens,
            sample.bytes_written,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if sample.observation_id:
                existing = connection.execute(
                    "SELECT category,latency_ms,compute_ms,tokens,bytes_written FROM cost_observations WHERE observation_id=?",
                    (sample.observation_id,),
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != values:
                        raise ValueError("cost observation identity conflicts with retained values")
                    return
            connection.execute(
                "INSERT INTO cost_observations(observation_id,category,latency_ms,compute_ms,tokens,bytes_written,recorded_at) VALUES(?,?,?,?,?,?,?)",
                (sample.observation_id or None, *values, _now()),
            )

    def record_storage(self, sample: StorageSample) -> None:
        if not isinstance(sample, StorageSample):
            raise TypeError("sample must be a StorageSample")
        revision_hash = hashlib.sha256(sample.catalog_revision.encode()).hexdigest()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO storage_samples(canonical_bytes,index_bytes,artifact_bytes,total_write_bytes,catalog_revision_sha256,guidance_tokens,compaction_compute_ms,refinement_compute_ms,recorded_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    sample.canonical_bytes,
                    sample.index_bytes,
                    sample.artifact_bytes,
                    sample.total_write_bytes,
                    revision_hash,
                    sample.guidance_tokens,
                    sample.compaction_compute_ms,
                    sample.refinement_compute_ms,
                    _now(),
                ),
            )

    def record_prefix_cache(self, sample: PrefixCacheSample) -> None:
        if not isinstance(sample, PrefixCacheSample):
            raise TypeError("sample must be a PrefixCacheSample")
        values = (
            sample.source,
            sample.eligible_tokens,
            sample.reused_tokens,
            sample.invalidated_tokens,
            sample.input_tokens,
            sample.baseline_input_tokens,
            sample.total_compute_ms,
            sample.baseline_total_compute_ms,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if sample.observation_id:
                existing = connection.execute(
                    "SELECT source,eligible_tokens,reused_tokens,invalidated_tokens,input_tokens,baseline_input_tokens,total_compute_ms,baseline_total_compute_ms FROM prefix_observations WHERE observation_id=?",
                    (sample.observation_id,),
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != values:
                        raise ValueError(
                            "prefix observation identity conflicts with retained values"
                        )
                    return
            connection.execute(
                "INSERT INTO prefix_observations(observation_id,source,eligible_tokens,reused_tokens,invalidated_tokens,input_tokens,baseline_input_tokens,total_compute_ms,baseline_total_compute_ms,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (sample.observation_id or None, *values, _now()),
            )

    def latency_report(self) -> dict[str, object]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT backend,corpus_tier,concurrency,query_mode,component,latency_ms,operation_count,window_ms,native_ok,effective_mode,queue_lag_ms,projection_lag_ms,recovery_time_ms,agent_turn_ms FROM operation_samples ORDER BY sample_id"
            ).fetchall()
        groups: dict[
            tuple[BackendName, CorpusTier, int, QueryMode, LatencyComponent],
            list[tuple[float, int, float, float, float, float, float]],
        ] = {}
        unavailable: set[tuple[BackendName, CorpusTier, int, QueryMode, LatencyComponent]] = set()
        for (
            backend,
            tier,
            concurrency,
            mode,
            component,
            latency,
            count,
            window,
            native,
            effective,
            queue_lag,
            projection_lag,
            recovery,
            agent_turn,
        ) in rows:
            _axis(backend, BACKENDS, field_name="backend")
            _axis(tier, CORPUS_TIERS, field_name="corpus_tier")
            _axis(concurrency, CONCURRENCY_LEVELS, field_name="concurrency")
            _axis(mode, QUERY_MODES, field_name="query_mode")
            _axis(component, _COMPONENTS, field_name="component")
            key: tuple[BackendName, CorpusTier, int, QueryMode, LatencyComponent] = (
                backend,
                tier,
                concurrency,
                mode,
                component,
            )
            if not native or effective != "selected_native":
                unavailable.add(key)
                continue
            groups.setdefault(key, []).append(
                (
                    float(latency),
                    int(count),
                    float(window),
                    float(queue_lag),
                    float(projection_lag),
                    float(recovery),
                    float(agent_turn),
                )
            )
        summaries: list[dict[str, object]] = []
        thresholds = {
            (row.backend, row.corpus_tier, row.concurrency, row.query_mode): row
            for row in DEFAULT_SERVICE_SLO_MATRIX
        }
        for key in sorted(set(groups) | unavailable):
            samples = groups.get(key, [])
            latencies = [item[0] for item in samples]
            threshold = thresholds[key[:4]]
            throughput = (
                sum(item[1] for item in samples) * 1_000 / sum(item[2] for item in samples)
                if samples and sum(item[2] for item in samples) > 0
                else 0.0
            )
            native_qualified = key not in unavailable and bool(samples)
            p50 = _percentile(latencies, 0.50)
            p95 = _percentile(latencies, 0.95)
            p99 = _percentile(latencies, 0.99)
            max_queue_lag = max((item[3] for item in samples), default=0.0)
            max_projection_lag = max((item[4] for item in samples), default=0.0)
            max_recovery = max((item[5] for item in samples), default=0.0)
            turn_time = sum(item[6] for item in samples)
            memory_turn_fraction = (
                sum(item[0] for item in samples) / turn_time if turn_time > 0 else None
            )
            recall_mode = str(key[3]).startswith(("cold_", "warm_"))
            summaries.append(
                {
                    "backend": key[0],
                    "corpus_tier": key[1],
                    "concurrency": key[2],
                    "query_mode": key[3],
                    "component": key[4],
                    "sample_count": len(samples),
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "throughput_per_second": throughput,
                    "max_queue_lag_ms": max_queue_lag,
                    "max_projection_lag_ms": max_projection_lag,
                    "max_recovery_time_ms": max_recovery,
                    "memory_latency_fraction_of_agent_turn": memory_turn_fraction,
                    "native_qualified": native_qualified,
                    "slo_passed": bool(
                        native_qualified
                        and p50 is not None
                        and p50 <= threshold.p50_ms
                        and p95 is not None
                        and p95 <= threshold.p95_ms
                        and p99 is not None
                        and p99 <= threshold.p99_ms
                        and throughput >= threshold.min_throughput_per_second
                        and max_queue_lag <= threshold.max_queue_lag_ms
                        and max_projection_lag <= threshold.max_projection_lag_ms
                        and max_recovery <= threshold.rto_ms
                        and (
                            not recall_mode
                            or (
                                memory_turn_fraction is not None
                                and memory_turn_fraction <= threshold.max_memory_turn_fraction
                            )
                        )
                    ),
                }
            )
        return {
            "slo_contract_revision": service_slo_contract_revision(),
            "groups": summaries,
            "all_native_qualified": bool(summaries)
            and all(item["native_qualified"] for item in summaries),
            "all_observed_slos_passed": bool(summaries)
            and all(item["slo_passed"] for item in summaries),
        }

    def headline_cost_report(self) -> dict[str, object]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT category,latency_ms,compute_ms,tokens,bytes_written FROM cost_samples UNION ALL SELECT category,latency_ms,compute_ms,tokens,bytes_written FROM cost_observations"
            ).fetchall()
        fields = ("latency_ms", "compute_ms", "tokens", "bytes_written")
        categories = {
            category: {
                field: _complete_sum(row[index] for row in rows if row[0] == category)
                for index, field in enumerate(fields, 1)
            }
            for category in dict.fromkeys(row[0] for row in rows)
        }
        missing = sorted(set(_COST_CATEGORIES) - set(categories))
        totals = {
            field: _complete_sum(row[index] for row in rows)
            for index, field in enumerate(fields, 1)
        }
        coverage = {
            field: {"observed": sum(row[index] is not None for row in rows), "total": len(rows)}
            for index, field in enumerate(fields, 1)
        }
        return {
            "complete": not missing and all(value is not None for value in totals.values()),
            "missing_categories": missing,
            "categories": categories,
            "background": {
                category: values
                for category, values in categories.items()
                if category not in {"online_recall", "model"}
            },
            "totals": totals,
            "field_coverage": coverage,
        }

    def storage_report(self) -> dict[str, object]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT canonical_bytes,index_bytes,artifact_bytes,total_write_bytes,catalog_revision_sha256,guidance_tokens,compaction_compute_ms,refinement_compute_ms FROM storage_samples ORDER BY storage_id"
            ).fetchall()
        if len(rows) < 2:
            return {"complete": False, "sample_count": len(rows)}
        first, last = rows[0], rows[-1]
        canonical_growth = int(last[0]) - int(first[0])
        write_growth = int(last[3]) - int(first[3])
        amplification = write_growth / canonical_growth if canonical_growth > 0 else None
        return {
            "complete": True,
            "sample_count": len(rows),
            "storage_growth_bytes": {
                "canonical": canonical_growth,
                "index": int(last[1]) - int(first[1]),
                "artifact": int(last[2]) - int(first[2]),
            },
            "write_amplification": amplification,
            "catalog_churn": sum(
                previous[4] != current[4] for previous, current in zip(rows, rows[1:])
            ),
            "guidance_overhead_tokens": int(last[5]),
            "compaction_compute_ms": float(last[6]) - float(first[6]),
            "refinement_compute_ms": float(last[7]) - float(first[7]),
        }

    def prefix_cache_report(self) -> dict[str, object]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT source,eligible_tokens,reused_tokens,invalidated_tokens,input_tokens,baseline_input_tokens,total_compute_ms,baseline_total_compute_ms FROM prefix_samples UNION ALL SELECT source,eligible_tokens,reused_tokens,invalidated_tokens,input_tokens,baseline_input_tokens,total_compute_ms,baseline_total_compute_ms FROM prefix_observations"
            ).fetchall()
        by_source: dict[str, dict[str, float | int | bool | None]] = {}
        for source in _PREFIX_SOURCES:
            selected = [row for row in rows if row[0] == source]
            if not selected:
                continue
            (
                eligible,
                reused,
                invalidated,
                input_tokens,
                baseline_tokens,
                compute,
                baseline_compute,
            ) = (_complete_sum(row[index] for row in selected) for index in range(1, 8))
            paired = all(
                value is not None
                for value in (input_tokens, baseline_tokens, compute, baseline_compute)
            )
            by_source[source] = {
                "sample_count": len(selected),
                "eligible_tokens": eligible,
                "reuse_rate": reused / eligible if eligible and reused is not None else None,
                "invalidation_rate": invalidated / eligible
                if eligible and invalidated is not None
                else None,
                "input_tokens_saved": (
                    baseline_tokens - input_tokens
                    if baseline_tokens is not None and input_tokens is not None
                    else None
                ),
                "total_compute_ms": compute,
                "baseline_total_compute_ms": baseline_compute,
                "optimization_accepted": (
                    input_tokens <= baseline_tokens and compute <= baseline_compute
                    if input_tokens is not None
                    and baseline_tokens is not None
                    and compute is not None
                    and baseline_compute is not None
                    else None
                ),
                "comparison_complete": paired,
            }
        return {
            "complete": set(_PREFIX_SOURCES[:3]).issubset(by_source)
            and all(
                row["comparison_complete"]
                and row["reuse_rate"] is not None
                and row["invalidation_rate"] is not None
                for row in by_source.values()
            ),
            "sources": by_source,
            "all_optimizations_accepted": bool(by_source)
            and all(row["optimization_accepted"] is True for row in by_source.values()),
        }

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _axis(value: object, allowed: Iterable[object], *, field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} is invalid")


def _positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _non_negative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _non_negative_number(value: object, *, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")


def _now() -> str:
    return datetime.now(UTC).isoformat()


DEFAULT_ONE_HOP_RECALL_BUDGET = RecallBudget(
    max_latency_ms=1_000,
    max_steps=2,
    max_tokens=2_048,
    max_candidates=32,
)
DEFAULT_RECONSTRUCTIVE_RECALL_BUDGET = RecallBudget(
    max_latency_ms=3_000,
    max_steps=6,
    max_tokens=4_096,
    max_candidates=64,
)


def _complete_sum(values: Iterable[int | float | None]) -> int | float | None:
    observed = tuple(values)
    if not observed or any(value is None for value in observed):
        return None
    return sum(value for value in observed if value is not None)


def _observation_id(value: str) -> None:
    if not isinstance(value, str) or len(value) > 256 or (value and not value.strip()):
        raise ValueError("observation_id must be empty or bounded non-blank text")
