"""Matrix runner: provider × benchmark × episode.

Responsibilities:
- Resolve providers/benchmarks/judges from the registry.
- For each (provider, benchmark, episode) create an isolated MemoryStore
  namespace, ingest the episode, then evaluate each probe.
- Collect CaseResult rows and write them to JSONL incrementally.
- Support resume (skip existing per-(provider, benchmark) JSONL files).
- Deterministic seeding and optional concurrency with per-provider locking.
"""

import concurrent.futures
import importlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_bench.core.types import CaseResult, RunSpec

log = logging.getLogger(__name__)

_PROVIDER_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_MUTEX = threading.Lock()


def _provider_lock(provider_id: str) -> threading.Lock:
    with _LOCKS_MUTEX:
        if provider_id not in _PROVIDER_LOCKS:
            _PROVIDER_LOCKS[provider_id] = threading.Lock()
        return _PROVIDER_LOCKS[provider_id]


def _jsonl_path(out_dir: Path, provider_id: str, benchmark_name: str) -> Path:
    return out_dir / provider_id / f"{benchmark_name}.jsonl"


def _iter_jsonl_physical_lines(path: Path):
    """Yield physical JSONL records without treating Unicode separators as new rows."""
    for raw_line in path.read_bytes().split(b"\n"):
        line = raw_line.strip()
        if line:
            yield line.decode("utf-8")


def _read_existing(path: Path) -> set[str]:
    """Return set of probe_ids already in a JSONL result file."""
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in _iter_jsonl_physical_lines(path):
        try:
            row = json.loads(line)
            ep = str(row.get("episode_id", ""))
            pr = str(row.get("probe_id", ""))
            done.add(f"{ep}::{pr}")
        except Exception:
            pass
    return done


def _append_result(path: Path, result: CaseResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_dict(), ensure_ascii=True) + "\n")


def _resolve_provider(spec_entry: Any) -> Any:
    from memory_bench.core.registry import get_provider

    factory = get_provider(spec_entry.adapter)
    if callable(factory):
        try:
            return factory(spec_entry.params)
        except TypeError:
            return factory()
    return factory


def _resolve_benchmark(name: str) -> Any:
    from memory_bench.core.registry import get_benchmark

    factory = get_benchmark(name)
    if callable(factory):
        try:
            return factory()
        except TypeError:
            return factory
    return factory


def _resolve_responder(spec: RunSpec) -> Any:
    from memory_bench.adapters.responder_llm import LLMResponder
    from memory_bench.core.llm import LLMConfig

    cfg = LLMConfig(
        model=spec.responder.model,
        temperature=spec.responder.temperature,
        max_tokens=spec.responder.max_tokens,
        cache_dir=str(Path(spec.cache_dir) / "llm"),
    )
    return LLMResponder(cfg)


def _resolve_judge(bench: Any, spec: RunSpec) -> Any:
    from memory_bench.adapters.judges import F1Judge, LLMJudge
    from memory_bench.core.registry import get_judge

    judge_name = getattr(bench, "default_judge", "f1")
    if judge_name == "llm_judge":
        if not spec.judge.use_llm_judge:
            return F1Judge()
        return LLMJudge(
            model=spec.judge.llm_judge_model,
            cache_dir=str(Path(spec.cache_dir)),
        )

    factory = get_judge(judge_name)
    if callable(factory):
        try:
            return factory()
        except TypeError:
            return factory
    return factory


def _prewarm_worker_imports(spec: RunSpec) -> None:
    """Import optional heavy memory modules before worker threads start.

    Native memory providers lazily import Lethyx/Hindsight/MemPalace modules.
    With many worker threads those imports can contend on Python module locks and
    look like a benchmark stall.  Importing them once on the main thread keeps
    worker startup deterministic while preserving optional-backend fail-soft
    behavior.
    """
    wants_mempalace = False
    wants_hindsight = False
    wants_mnemosyne = False
    for provider in spec.providers:
        adapter = str(provider.adapter)
        params = dict(provider.params or {})
        backend = str(params.get("backend") or "").strip().lower()
        raw_backends = str(params.get("backends") or "").strip().lower()
        backends = {part.strip() for part in raw_backends.split(",") if part.strip()}
        if adapter == "lethyx-composite" and not backend and not backends:
            backends = {"mempalace", "hindsight", "mnemosyne"}

        wants_mempalace = wants_mempalace or adapter == "raw-mempalace"
        wants_hindsight = wants_hindsight or adapter == "raw-hindsight"
        wants_mnemosyne = wants_mnemosyne or adapter == "raw-mnemosyne"
        if adapter in {"lethyx", "lethyx-composite"}:
            wants_mempalace = wants_mempalace or backend == "mempalace" or "mempalace" in backends
            wants_hindsight = wants_hindsight or backend == "hindsight" or "hindsight" in backends
            wants_mnemosyne = wants_mnemosyne or backend == "mnemosyne" or "mnemosyne" in backends

    modules: list[str] = []
    if wants_mempalace or wants_hindsight or wants_mnemosyne:
        modules.extend(
            [
                "lethyx.memory.backends",
                "lethyx.memory.backends.contracts",
            ]
        )
    if wants_mempalace:
        modules.extend(
            [
                "lethyx.memory.palace.facade",
                "lethyx.memory.palace.tiers",
            ]
        )
    if wants_hindsight:
        modules.extend(
            [
                "lethyx.memory.backends.hindsight",
                "lethyx.memory.hindsight.config",
                "hindsight",
                "hindsight_client",
            ]
        )
    if wants_mnemosyne:
        modules.extend(
            [
                "lethyx.memory.backends.mnemosyne",
                "mnemosyne",
            ]
        )

    for module in dict.fromkeys(modules):
        try:
            importlib.import_module(module)
        except Exception as exc:
            log.debug("prewarm import skipped for %s: %s", module, exc)


def run_pair(
    provider_entry: Any,
    benchmark_name: str,
    spec: RunSpec,
    out_dir: Path,
) -> list[CaseResult]:
    """Run all episodes of one benchmark against one provider.

    Returns the list of CaseResult rows produced.  Results are also written
    to JSONL incrementally so progress survives interruption.
    """
    from memory_bench.benchmarks.recall_driver import run_recall_episode
    from memory_bench.benchmarks.agentic_driver import run_agentic_episode

    provider = _resolve_provider(provider_entry)
    bench = _resolve_benchmark(benchmark_name)
    responder = _resolve_responder(spec)
    judge = _resolve_judge(bench, spec)

    jsonl = _jsonl_path(out_dir, provider_entry.id, benchmark_name)
    done_keys = set() if spec.limits.redo_existing else _read_existing(jsonl)
    results: list[CaseResult] = []
    lock = _provider_lock(provider_entry.id)

    kind = getattr(bench, "kind", "recall")

    for episode, probes in bench.load(spec):
        # Filter already-done probes (resume)
        pending_probes = [p for p in probes if f"{episode.id}::{p.id}" not in done_keys]
        if not pending_probes:
            log.debug(
                "[%s/%s] episode %s already done, skipping",
                provider_entry.id,
                benchmark_name,
                episode.id,
            )
            continue

        ns = f"{benchmark_name}/{episode.id}"

        with lock:
            store = provider.namespace(ns)

        try:
            # Ingest episode records
            with lock:
                store.write_many(episode.all_records)

            for probe in pending_probes:
                key = f"{episode.id}::{probe.id}"
                t0 = time.monotonic()
                try:
                    use_agentic_driver = kind == "agentic" or (
                        kind == "mixed" and probe.kind == "action"
                    )
                    if use_agentic_driver:
                        row = run_agentic_episode(
                            provider=provider,
                            provider_id=provider_entry.id,
                            bench=bench,
                            episode=episode,
                            probe=probe,
                            store=store,
                            spec=spec,
                        )
                    else:
                        row = run_recall_episode(
                            store=store,
                            episode=episode,
                            probe=probe,
                            responder=responder,
                            judge=judge,
                            spec=spec,
                            provider_id=provider_entry.id,
                            benchmark_name=benchmark_name,
                        )
                except Exception as exc:
                    log.error(
                        "[%s/%s] probe %s error: %s",
                        provider_entry.id,
                        benchmark_name,
                        probe.id,
                        exc,
                        exc_info=True,
                    )
                    row = CaseResult(
                        provider=provider_entry.id,
                        benchmark=benchmark_name,
                        episode_id=episode.id,
                        probe_id=probe.id,
                        category=probe.category,
                        status="error",
                        error=str(exc),
                        latency_ms=(time.monotonic() - t0) * 1000,
                        diagnostics={
                            "question": getattr(probe, "question", ""),
                            "failure_mode": "probe_exception",
                            "exception_type": type(exc).__name__,
                        },
                    )

                _append_result(jsonl, row)
                results.append(row)
                done_keys.add(key)

        finally:
            with lock:
                try:
                    store.close()
                except Exception:
                    pass

    return results


def _safe_run_pair(
    provider_entry: Any,
    benchmark_name: str,
    spec: RunSpec,
    out_dir: Path,
) -> list[CaseResult]:
    try:
        return run_pair(provider_entry, benchmark_name, spec, out_dir)
    except Exception as exc:
        log.error("job (%s × %s) failed: %s", provider_entry.id, benchmark_name, exc, exc_info=True)
        return [
            CaseResult(
                provider=provider_entry.id,
                benchmark=benchmark_name,
                episode_id="__job__",
                probe_id="__job__",
                category="",
                status="error",
                error=str(exc),
                diagnostics={
                    "failure_mode": "job_exception",
                    "exception_type": type(exc).__name__,
                },
            )
        ]


def run_suite(spec: RunSpec, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Run the full provider × benchmark matrix for one RunSpec.

    Returns a report dict (also written to out_dir/report.json).
    """
    from memory_bench.core import registry
    from memory_bench.core.report import build_report, write_report
    from memory_bench.core.manifest import build_manifest, write_manifest

    registry.bootstrap()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(spec.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting suite: %d providers × %d benchmarks",
        len(spec.providers),
        len(spec.benchmarks),
    )
    _prewarm_worker_imports(spec)

    jobs = [(prov, bench) for prov in spec.providers for bench in spec.benchmarks]

    all_results: list[CaseResult] = []

    if spec.limits.max_parallel <= 1:
        for prov, bench in jobs:
            all_results.extend(_safe_run_pair(prov, bench, spec, out_dir))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=spec.limits.max_parallel) as pool:
            futures = {
                pool.submit(_safe_run_pair, prov, bench, spec, out_dir): (prov, bench)
                for prov, bench in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

    # Write combined results JSONL
    combined = out_dir / "results.jsonl"
    combined.parent.mkdir(parents=True, exist_ok=True)
    with combined.open("w", encoding="utf-8") as fh:
        for row in all_results:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    report = build_report(all_results)
    write_report(report, out_dir=out_dir)

    # Build manifest
    providers_resolved = []
    dataset_paths: dict[str, Path] = {}
    try:
        from memory_bench.core.datasets import resolve_path

        for bench_name in spec.benchmarks:
            p = resolve_path(bench_name, strict=False)
            if p:
                dataset_paths[bench_name] = p
        for prov_spec in spec.providers:
            try:
                providers_resolved.append(_resolve_provider(prov_spec))
            except Exception:
                pass
    except Exception as exc:
        log.debug("manifest data collection partial: %s", exc)

    manifest = build_manifest(
        spec=spec,
        providers=providers_resolved,
        dataset_paths=dataset_paths,
        repo_root=repo_root or Path.cwd(),
    )
    write_manifest(manifest, out_dir)

    log.info("Suite complete. Output: %s", out_dir)
    return {"out_dir": str(out_dir), "report": report, "manifest": manifest}


def read_results_from_dir(out_dir: Path) -> list[CaseResult]:
    """Read all CaseResult rows from per-(provider, benchmark) JSONL files."""
    results: list[CaseResult] = []
    combined = out_dir / "results.jsonl"
    if combined.is_file():
        for line in _iter_jsonl_physical_lines(combined):
            try:
                results.append(CaseResult.from_dict(json.loads(line)))
            except Exception as exc:
                log.debug("skip malformed result line in %s: %s", combined, exc)
        return results

    for jsonl in sorted(out_dir.rglob("*.jsonl")):
        if jsonl.name == "results.jsonl":
            continue
        for line in _iter_jsonl_physical_lines(jsonl):
            try:
                results.append(CaseResult.from_dict(json.loads(line)))
            except Exception as exc:
                log.debug("skip malformed result line in %s: %s", jsonl, exc)
    return results
