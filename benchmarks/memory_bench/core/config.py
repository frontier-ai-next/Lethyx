"""Parse suite.toml into RunSpec.

suite.toml is the provider-agnostic configuration file, replacing the
lethyx-specific modes.toml for memory benchmark purposes.
"""

import tomllib
from pathlib import Path
from typing import Any

from memory_bench.core.types import (
    DatasetSpec,
    JudgeConfig,
    LimitsSpec,
    ProviderSpec,
    ResponderSpec,
    RunSpec,
)


def _parse_provider(entry: dict[str, Any]) -> ProviderSpec:
    pid = str(entry.get("id") or "").strip()
    adapter = str(entry.get("adapter") or "").strip()
    if not pid or not adapter:
        raise ValueError(f"[[provider]] entry must have 'id' and 'adapter': {entry}")
    params = {k: v for k, v in entry.items() if k not in ("id", "adapter")}
    return ProviderSpec(id=pid, adapter=adapter, params=params)


def parse_suite(data: dict[str, Any], *, base_dir: Path | None = None) -> list[RunSpec]:
    """Parse a decoded suite.toml into a list of RunSpec objects.

    Each entry in ``[[run]]`` (or the implicit single run from top-level sections)
    produces one RunSpec.  If no ``[[run]]`` table is present the entire file is
    treated as one run.
    """
    runs_data: list[dict[str, Any]] = data.get("run") or [data]
    results: list[RunSpec] = []

    # Global defaults
    g_providers: list[dict[str, Any]] = list(data.get("provider") or [])
    g_responder = data.get("responder") or {}
    g_judge = data.get("judge") or {}
    g_limits = data.get("limits") or {}
    g_suite = data.get("suite") or {}
    g_dataset = data.get("dataset") or {}

    for run_data in runs_data:
        providers_raw = list(run_data.get("provider") or g_providers)
        providers = tuple(_parse_provider(p) for p in providers_raw)

        resp_raw = {**g_responder, **(run_data.get("responder") or {})}
        responder = ResponderSpec(
            model=str(resp_raw.get("model") or ""),
            temperature=float(resp_raw.get("temperature") or 0.0),
            max_tokens=int(resp_raw.get("max_tokens") or 512),
        )

        judge_raw = {**g_judge, **(run_data.get("judge") or {})}
        judge = JudgeConfig(
            llm_judge_model=str(judge_raw.get("llm_judge_model") or ""),
            use_llm_judge=bool(judge_raw.get("use_llm_judge", True)),
        )

        lim_raw = {**g_limits, **(run_data.get("limits") or {})}
        limits = LimitsSpec(
            recall_k=int(lim_raw.get("recall_k") or 10),
            max_context_tokens=int(lim_raw.get("max_context_tokens") or 4000),
            seed=int(lim_raw.get("seed") or 0),
            max_parallel=int(lim_raw.get("max_parallel") or 1),
            redo_existing=bool(lim_raw.get("redo_existing", False)),
        )

        suite_raw = {**g_suite, **(run_data.get("suite") or {})}
        benchmarks = tuple(str(b) for b in (suite_raw.get("benchmarks") or []))

        ds_raw = {**g_dataset, **(run_data.get("dataset") or {})}
        tiers = tuple(str(t) for t in (ds_raw.get("tiers") or []))
        dataset = DatasetSpec(
            name=str(ds_raw.get("name") or ""),
            path=str(ds_raw.get("path") or ""),
            tiers=tiers,
            max_episodes=int(ds_raw.get("max_episodes") or 0),
            max_probes_per_episode=int(ds_raw.get("max_probes_per_episode") or 0),
        )

        out_dir = str(run_data.get("out_dir") or data.get("out_dir") or "runs/memory_bench")
        cache_dir = str(run_data.get("cache_dir") or data.get("cache_dir") or ".memory_bench_cache")

        results.append(
            RunSpec(
                providers=providers,
                benchmarks=benchmarks,
                dataset=dataset,
                responder=responder,
                judge=judge,
                limits=limits,
                out_dir=out_dir,
                cache_dir=cache_dir,
            )
        )

    return results


def load_suite(path: str | Path) -> list[RunSpec]:
    """Load and parse a suite.toml file from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"suite.toml not found: {p}")
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    return parse_suite(data, base_dir=p.parent)
