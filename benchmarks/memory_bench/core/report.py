"""Report generation for memory_bench runs.

Schema: memory_bench/report/v1

Report structure:
  {
    "schema": "memory_bench/report/v1",
    "totals": {...},
    "rollups": [...],          # per (provider, benchmark, category)
    "failures": [...]
  }

Also provides an optional bridge to the bench_common TaskResult format so
results can be fed into the existing benchkit report pipeline.
"""

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from memory_bench.core.metrics import mean, pct
from memory_bench.core.types import CaseResult


def _recall_value(row: CaseResult) -> float:
    for key, value in row.retrieval.items():
        if str(key).startswith("recall_at_"):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _score(row: CaseResult) -> float:
    if row.status != "ok" or row.judgment is None:
        return -1.0
    return float(row.judgment.score)


def _correct(row: CaseResult) -> bool:
    return bool(row.status == "ok" and row.judgment and row.judgment.correct)


def _case_key(row: CaseResult) -> tuple[str, str, str]:
    return (row.benchmark, row.episode_id, row.probe_id)


def _is_lethyx_provider(provider: str) -> bool:
    return provider.startswith("lethyx")


def _classify_underperformance(row: CaseResult, best: CaseResult) -> str:
    diag = row.diagnostics or {}
    if row.status == "error":
        return "provider_error"
    failure_mode = str(diag.get("failure_mode") or "")
    if failure_mode and failure_mode != "ok":
        return failure_mode
    missing = diag.get("missing_evidence") or []
    if missing:
        return "retrieval_miss"
    if _recall_value(best) > _recall_value(row):
        return "retrieval_quality_gap"
    if (
        row.judgment is not None
        and best.judgment is not None
        and best.judgment.score > row.judgment.score
    ):
        return "answer_or_judge_gap"
    return "metric_gap"


def _provider_error_entry(row: CaseResult) -> dict[str, Any]:
    return {
        "case_id": f"{row.benchmark}/{row.episode_id}/{row.probe_id}",
        "benchmark": row.benchmark,
        "episode_id": row.episode_id,
        "probe_id": row.probe_id,
        "category": row.category,
        "provider": row.provider,
        "best_provider": "",
        "reason": "provider_error",
        "score": 0.0,
        "best_score": 0.0,
        "score_delta": 0.0,
        "correct": False,
        "best_correct": False,
        "status": row.status,
        "best_status": "",
        "recall_at_k": _recall_value(row),
        "best_recall_at_k": 0.0,
        "recall_delta": 0.0,
        "error": row.error,
        "lethyx_diagnostics": row.diagnostics,
        "best_diagnostics": {},
        "all_provider_scores": [
            {
                "provider": row.provider,
                "status": row.status,
                "correct": False,
                "score": 0.0,
                "recall_at_k": _recall_value(row),
                "failure_mode": (row.diagnostics or {}).get("failure_mode", ""),
            }
        ],
    }


def build_comparison_diagnostics(results: Iterable[CaseResult]) -> list[dict[str, Any]]:
    """Return per-case Lethyx underperformance diagnostics.

    A row is emitted when a provider whose id starts with ``lethyx`` scores below
    the best provider on the same (benchmark, episode, probe). The entry carries
    the Lethyx trace captured by the driver plus a compact competitor summary so
    failed cases can be debugged without re-running the benchmark.
    """
    groups: dict[tuple[str, str, str], list[CaseResult]] = defaultdict(list)
    for row in results:
        groups[_case_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for (benchmark, episode_id, probe_id), cases in sorted(groups.items()):
        if len(cases) < 2:
            for row in cases:
                if _is_lethyx_provider(row.provider) and row.status == "error":
                    out.append(_provider_error_entry(row))
            continue
        best = max(
            cases,
            key=lambda row: (
                int(row.status == "ok"),
                int(_correct(row)),
                _score(row),
                _recall_value(row),
                -float(row.latency_ms or 0.0),
            ),
        )
        best_score = _score(best)
        best_correct = _correct(best)
        for row in cases:
            if not _is_lethyx_provider(row.provider):
                continue
            row_score = _score(row)
            underperforming = False
            if row.status != "ok" and best.status == "ok":
                underperforming = True
            elif best_correct and not _correct(row):
                underperforming = True
            elif best_score > row_score + 1e-9:
                underperforming = True
            if not underperforming:
                continue

            all_scores = []
            for candidate in sorted(cases, key=lambda item: (-_score(item), item.provider)):
                all_scores.append(
                    {
                        "provider": candidate.provider,
                        "status": candidate.status,
                        "correct": _correct(candidate),
                        "score": max(_score(candidate), 0.0),
                        "recall_at_k": _recall_value(candidate),
                        "failure_mode": (candidate.diagnostics or {}).get("failure_mode", ""),
                    }
                )

            out.append(
                {
                    "case_id": f"{benchmark}/{episode_id}/{probe_id}",
                    "benchmark": benchmark,
                    "episode_id": episode_id,
                    "probe_id": probe_id,
                    "category": row.category,
                    "provider": row.provider,
                    "best_provider": best.provider,
                    "reason": _classify_underperformance(row, best),
                    "score": max(row_score, 0.0),
                    "best_score": max(best_score, 0.0),
                    "score_delta": round(max(best_score, 0.0) - max(row_score, 0.0), 6),
                    "correct": _correct(row),
                    "best_correct": best_correct,
                    "status": row.status,
                    "best_status": best.status,
                    "recall_at_k": _recall_value(row),
                    "best_recall_at_k": _recall_value(best),
                    "recall_delta": round(_recall_value(best) - _recall_value(row), 6),
                    "lethyx_diagnostics": row.diagnostics,
                    "best_diagnostics": {
                        "provider": best.provider,
                        "failure_mode": (best.diagnostics or {}).get("failure_mode", ""),
                        "answer_text": (best.diagnostics or {}).get("answer_text", ""),
                        "retrieved_ids": (best.diagnostics or {}).get("retrieved_ids", []),
                        "missing_evidence": (best.diagnostics or {}).get("missing_evidence", []),
                        "evidence_ranks": (best.diagnostics or {}).get("evidence_ranks", {}),
                    },
                    "all_provider_scores": all_scores,
                }
            )
    return out


def _diagnostics_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason = Counter(str(entry.get("reason") or "unknown") for entry in entries)
    by_provider = Counter(str(entry.get("provider") or "unknown") for entry in entries)
    return {
        "n_lethyx_underperforming_cases": len(entries),
        "by_reason": dict(sorted(by_reason.items())),
        "by_provider": dict(sorted(by_provider.items())),
    }


# ---------------------------------------------------------------------------
# Rollup dataclass
# ---------------------------------------------------------------------------


@dataclass
class CategoryRollup:
    """Aggregated scores for one (provider, benchmark, category) group."""

    provider: str
    benchmark: str
    category: str
    n: int
    n_correct: int
    accuracy: float  # n_correct / n
    mean_score: float  # mean continuous score (F1, nugget, etc.)
    abstention_acc: float  # only meaningful for abstention category
    recall_at_k: float  # mean retrieval recall@k (0 if not computed)
    mean_mrr: float
    mean_ndcg: float
    mean_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    n_errors: int
    n_skipped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "benchmark": self.benchmark,
            "category": self.category,
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": round(self.accuracy, 4),
            "mean_score": round(self.mean_score, 4),
            "abstention_acc": round(self.abstention_acc, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg": round(self.mean_ndcg, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "n_errors": self.n_errors,
            "n_skipped": self.n_skipped,
        }


@dataclass
class BenchmarkRollup:
    """Aggregated scores for one (provider, benchmark) pair — all categories combined."""

    provider: str
    benchmark: str
    n: int
    n_correct: int
    accuracy: float
    mean_score: float
    recall_at_k: float
    mean_mrr: float
    mean_ndcg: float
    mean_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    n_errors: int
    by_category: list[CategoryRollup]
    mean_score_row_weighted: float = 0.0
    mean_score_category_weighted: float = 0.0
    recall_at_k_row_weighted: float = 0.0
    recall_at_k_category_weighted: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "benchmark": self.benchmark,
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": round(self.accuracy, 4),
            "mean_score": round(self.mean_score, 4),
            "mean_score_row_weighted": round(self.mean_score_row_weighted, 4),
            "mean_score_category_weighted": round(self.mean_score_category_weighted, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "recall_at_k_row_weighted": round(self.recall_at_k_row_weighted, 4),
            "recall_at_k_category_weighted": round(self.recall_at_k_category_weighted, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg": round(self.mean_ndcg, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "n_errors": self.n_errors,
            "by_category": [c.to_dict() for c in self.by_category],
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _recall_key(retrieval: dict[str, Any]) -> tuple[str, float]:
    """Find the recall@k key and value in retrieval dict."""
    for k, v in retrieval.items():
        if k.startswith("recall_at_"):
            return k, float(v)
    return "recall_at_k", 0.0


def aggregate(results: Iterable[CaseResult]) -> list[BenchmarkRollup]:
    rows = list(results)
    # Group by (provider, benchmark, category)
    groups: dict[tuple[str, str, str], list[CaseResult]] = defaultdict(list)
    for row in rows:
        groups[(row.provider, row.benchmark, row.category)].append(row)

    # Build per-benchmark groups
    bench_groups: dict[tuple[str, str], list[CategoryRollup]] = defaultdict(list)
    bench_cases: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)

    for row in rows:
        bench_cases[(row.provider, row.benchmark)].append(row)

    for (provider, benchmark, category), cases in sorted(groups.items()):
        ok_cases = [c for c in cases if c.status == "ok"]
        err_cases = [c for c in cases if c.status == "error"]
        skip_cases = [c for c in cases if c.status == "skipped"]

        scores = [c.judgment.score for c in ok_cases if c.judgment is not None]
        correct = [c.judgment.correct for c in ok_cases if c.judgment is not None]

        # Abstention accuracy
        abs_cases = [c for c in ok_cases if c.judgment and c.judgment.metric == "abstention"]
        abs_acc = (
            mean([1.0 if c.judgment.correct else 0.0 for c in abs_cases]) if abs_cases else 0.0
        )

        # Retrieval metrics
        recalls = []
        mrrs = []
        ndcgs = []
        for c in ok_cases:
            _, rv = _recall_key(c.retrieval)
            recalls.append(rv)
            mrrs.append(float(c.retrieval.get("mrr", 0.0)))
            for k2, v2 in c.retrieval.items():
                if k2.startswith("ndcg_at_"):
                    ndcgs.append(float(v2))
                    break

        latencies = [c.latency_ms for c in ok_cases]
        total_cost = sum(c.usage.cost_usd for c in cases)
        total_tokens = sum(c.usage.total_tokens for c in cases)

        cr = CategoryRollup(
            provider=provider,
            benchmark=benchmark,
            category=category,
            n=len(cases),
            n_correct=sum(1 for b in correct if b),
            accuracy=pct(sum(1 for b in correct if b), len(correct)) / 100.0 if correct else 0.0,
            mean_score=mean(scores),
            abstention_acc=abs_acc,
            recall_at_k=mean(recalls),
            mean_mrr=mean(mrrs),
            mean_ndcg=mean(ndcgs),
            mean_latency_ms=mean(latencies),
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            n_errors=len(err_cases),
            n_skipped=len(skip_cases),
        )
        bench_groups[(provider, benchmark)].append(cr)

    rollups: list[BenchmarkRollup] = []
    for (provider, benchmark), cat_rollups in sorted(bench_groups.items()):
        cases = bench_cases[(provider, benchmark)]
        ok_cases = [c for c in cases if c.status == "ok"]
        n = sum(c.n for c in cat_rollups)
        n_correct = sum(c.n_correct for c in cat_rollups)
        score_values = [c.judgment.score for c in ok_cases if c.judgment is not None]
        recall_values: list[float] = []
        mrr_values: list[float] = []
        ndcg_values: list[float] = []
        for c in ok_cases:
            _, rv = _recall_key(c.retrieval)
            recall_values.append(rv)
            mrr_values.append(float(c.retrieval.get("mrr", 0.0)))
            for k2, v2 in c.retrieval.items():
                if k2.startswith("ndcg_at_"):
                    ndcg_values.append(float(v2))
                    break
        mean_score_row_weighted = mean(score_values)
        mean_score_category_weighted = mean([c.mean_score for c in cat_rollups if c.n > 0])
        recall_row_weighted = mean(recall_values)
        recall_category_weighted = mean([c.recall_at_k for c in cat_rollups if c.n > 0])
        lat = mean([c.latency_ms for c in ok_cases])
        cost = sum(c.total_cost_usd for c in cat_rollups)
        tokens = sum(c.total_tokens for c in cat_rollups)
        errors = sum(c.n_errors for c in cat_rollups)
        rollups.append(
            BenchmarkRollup(
                provider=provider,
                benchmark=benchmark,
                n=n,
                n_correct=n_correct,
                accuracy=(n_correct / n) if n else 0.0,
                mean_score=mean_score_row_weighted,
                recall_at_k=recall_row_weighted,
                mean_mrr=mean(mrr_values),
                mean_ndcg=mean(ndcg_values),
                mean_latency_ms=lat,
                total_cost_usd=cost,
                total_tokens=tokens,
                n_errors=errors,
                by_category=cat_rollups,
                mean_score_row_weighted=mean_score_row_weighted,
                mean_score_category_weighted=mean_score_category_weighted,
                recall_at_k_row_weighted=recall_row_weighted,
                recall_at_k_category_weighted=recall_category_weighted,
            )
        )
    return rollups


def build_report(results: Iterable[CaseResult]) -> dict[str, Any]:
    rows = list(results)
    rollups = aggregate(rows)
    providers = sorted({r.provider for r in rows})
    benchmarks = sorted({r.benchmark for r in rows})
    comparison_diagnostics = build_comparison_diagnostics(rows)
    return {
        "schema": "memory_bench/report/v1",
        "totals": {
            "n_rows": len(rows),
            "n_providers": len(providers),
            "n_benchmarks": len(benchmarks),
            "providers": providers,
            "benchmarks": benchmarks,
        },
        "rollups": [r.to_dict() for r in rollups],
        "diagnostics": {
            "summary": _diagnostics_summary(comparison_diagnostics),
            "underperforming_cases": comparison_diagnostics,
        },
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# memory_bench report",
        "",
        f"Schema: `{report.get('schema', '')}`",
        "",
    ]
    totals = report.get("totals", {})
    lines.append(
        f"Providers: {', '.join(totals.get('providers', []))}  |  "
        f"Benchmarks: {', '.join(totals.get('benchmarks', []))}  |  "
        f"Rows: {totals.get('n_rows', 0)}"
    )
    lines.append("")

    diagnostics = report.get("diagnostics", {}) or {}
    diag_summary = diagnostics.get("summary", {}) or {}
    underperforming = diagnostics.get("underperforming_cases", []) or []
    lines.append("## Diagnostics")
    lines.append("")
    lines.append(
        "Lethyx underperforming cases: "
        f"{diag_summary.get('n_lethyx_underperforming_cases', len(underperforming))}"
    )
    by_reason = diag_summary.get("by_reason", {}) or {}
    if by_reason:
        lines.append(
            "Reasons: "
            + ", ".join(f"{reason}={count}" for reason, count in sorted(by_reason.items()))
        )
    if underperforming:
        lines.append("")
        lines.append("| provider | case | reason | best_provider | score_delta | recall_delta |")
        lines.append("|---|---|---|---|---:|---:|")
        for entry in underperforming[:20]:
            lines.append(
                f"| {entry.get('provider', '')} "
                f"| {entry.get('case_id', '')} "
                f"| {entry.get('reason', '')} "
                f"| {entry.get('best_provider', '')} "
                f"| {float(entry.get('score_delta', 0.0)):.3f} "
                f"| {float(entry.get('recall_delta', 0.0)):.3f} |"
            )
        if len(underperforming) > 20:
            lines.append(f"\nShowing first 20 of {len(underperforming)}; see `diagnostics.jsonl`.")
    lines.append("")

    # Per-benchmark summary tables
    rollups: list[dict[str, Any]] = report.get("rollups", [])
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rollups:
        by_bench[r["benchmark"]].append(r)

    for bench in sorted(by_bench):
        lines.append(f"## {bench}")
        lines.append("")
        lines.append(
            "| provider | n | accuracy | mean_score | recall@k | mrr | cost_usd | errors |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(by_bench[bench], key=lambda x: -x["accuracy"]):
            lines.append(
                f"| {r['provider']} | {r['n']} "
                f"| {r['accuracy']:.3f} "
                f"| {r['mean_score']:.3f} "
                f"| {r['recall_at_k']:.3f} "
                f"| {r.get('mean_mrr', 0.0):.3f} "
                f"| {r['total_cost_usd']:.4f} "
                f"| {r['n_errors']} |"
            )
        lines.append("")

        # Category breakdown
        for r in sorted(by_bench[bench], key=lambda x: x["provider"]):
            cats = r.get("by_category", [])
            if len(cats) <= 1:
                continue
            lines.append(f"### {r['provider']} — by category")
            lines.append("")
            lines.append("| category | n | accuracy | mean_score | recall@k |")
            lines.append("|---|---:|---:|---:|---:|")
            for c in sorted(cats, key=lambda x: x["category"]):
                lines.append(
                    f"| {c['category'] or '(all)'} | {c['n']} "
                    f"| {c['accuracy']:.3f} "
                    f"| {c['mean_score']:.3f} "
                    f"| {c['recall_at_k']:.3f} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def render_csv(report: dict[str, Any]) -> str:
    """Flat CSV of all category-level rollup rows."""
    rollups: list[dict[str, Any]] = report.get("rollups", [])
    all_cats: list[dict[str, Any]] = []
    for r in rollups:
        for c in r.get("by_category", [r]):
            row = dict(r)
            row.update(c)
            all_cats.append(row)

    if not all_cats:
        return ""

    buf = io.StringIO()
    fields = [
        "provider",
        "benchmark",
        "category",
        "n",
        "n_correct",
        "accuracy",
        "mean_score",
        "mean_score_row_weighted",
        "mean_score_category_weighted",
        "recall_at_k",
        "recall_at_k_row_weighted",
        "recall_at_k_category_weighted",
        "mean_mrr",
        "mean_ndcg",
        "mean_latency_ms",
        "total_cost_usd",
        "total_tokens",
        "n_errors",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_cats)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_report(report: dict[str, Any], *, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    csv_path = out_dir / "report.csv"
    diagnostics_path = out_dir / "diagnostics.jsonl"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    csv_path.write_text(render_csv(report), encoding="utf-8")
    underperforming = (report.get("diagnostics", {}) or {}).get("underperforming_cases", []) or []
    with diagnostics_path.open("w", encoding="utf-8") as fh:
        for entry in underperforming:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return json_path, md_path, csv_path


# ---------------------------------------------------------------------------
# Optional bridge to bench_common.results.TaskResult
# ---------------------------------------------------------------------------


def to_taskresult(result: CaseResult) -> Any:
    """Convert a CaseResult to a bench_common TaskResult for benchkit integration.

    Returns None if bench_common is not available.
    """
    try:
        from bench_common.results import TaskResult
    except ImportError:
        return None

    status = "passed" if (result.judgment and result.judgment.correct) else "failed"
    if result.status == "error":
        status = "error"
    elif result.status == "skipped":
        status = "skipped"

    return TaskResult(
        bench=f"memory-bench/{result.benchmark}",
        mode=result.provider,
        task_id=f"{result.episode_id}::{result.probe_id}",
        status=status,
        metrics={
            "category": result.category,
            "score": result.judgment.score if result.judgment else 0.0,
            "metric": result.judgment.metric if result.judgment else "",
            **result.retrieval,
        },
        wall_s=result.latency_ms / 1000.0 if result.latency_ms else None,
        error=result.error or None,
    )
