"""Load memory scenarios from YAML files."""

from pathlib import Path
from typing import Any

import yaml

from lethyx.evals.memory_scenarios.fixtures import SCENARIOS_DIR
from lethyx.evals.memory_scenarios.models import MemoryScenario, ScenarioSeed, ScenarioStep

LLM_SCENARIOS_DIR = SCENARIOS_DIR / "llm"


def _parse_seed(raw: dict[str, Any] | None) -> ScenarioSeed:
    raw = raw or {}
    return ScenarioSeed(
        manager_core={k: str(v) for k, v in (raw.get("manager_core") or {}).items()},
        workspace_core={k: str(v) for k, v in (raw.get("workspace_core") or {}).items()},
        palace=dict(raw.get("palace") or {}),
        workspace_fixture=str(raw.get("workspace_fixture") or "default"),
        manager_fixture=str(raw.get("manager_fixture") or "manager_default"),
        extra_workspaces={str(k): str(v) for k, v in (raw.get("extra_workspaces") or {}).items()},
    )


def _parse_steps(raw: list[dict[str, Any]] | None) -> list[ScenarioStep]:
    steps: list[ScenarioStep] = []
    for row in raw or []:
        steps.append(
            ScenarioStep(
                id=str(row.get("id") or row.get("action") or "step"),
                action=str(row.get("action") or ""),
                phase=row.get("phase"),
                run_id=str(row.get("run_id") or "run-scenario"),
                workspace_id=row.get("workspace_id"),
                args=dict(row.get("args") or {}),
            )
        )
    return steps


def load_scenario(path: Path) -> MemoryScenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenario_id = str(data.get("id") or path.stem)
    return MemoryScenario(
        id=scenario_id,
        description=str(data.get("description") or ""),
        workspace=str(data.get("workspace") or "test"),
        mode=str(data.get("mode") or "phase_prompt"),
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        seed=_parse_seed(data.get("seed")),
        steps=_parse_steps(data.get("steps")),
        assertions=dict(data.get("assert") or {}),
        source_path=path,
        raw_seed=dict(data.get("seed") or {}),
        llm=dict(data.get("llm") or {}),
        requires_no_volatile_stub=bool(data.get("requires_no_volatile_stub")),
    )


def list_scenario_paths(
    scenarios_dir: Path | None = None,
    *,
    include_llm: bool = False,
) -> list[Path]:
    root = scenarios_dir or SCENARIOS_DIR
    if not root.is_dir():
        return []
    paths = sorted(root.glob("*.yaml"))
    if include_llm and LLM_SCENARIOS_DIR.is_dir():
        paths.extend(sorted(LLM_SCENARIOS_DIR.glob("*.yaml")))
    return paths


def load_all_scenarios(
    scenarios_dir: Path | None = None,
    *,
    include_llm: bool = False,
) -> list[MemoryScenario]:
    return [load_scenario(p) for p in list_scenario_paths(scenarios_dir, include_llm=include_llm)]


def load_all_llm_scenarios() -> list[MemoryScenario]:
    if not LLM_SCENARIOS_DIR.is_dir():
        return []
    return [load_scenario(p) for p in sorted(LLM_SCENARIOS_DIR.glob("*.yaml"))]


def load_scenario_by_id(
    scenario_id: str,
    scenarios_dir: Path | None = None,
) -> MemoryScenario:
    for path in list_scenario_paths(scenarios_dir):
        if path.stem == scenario_id or path.name == scenario_id:
            return load_scenario(path)
        loaded = load_scenario(path)
        if loaded.id == scenario_id:
            return loaded
    raise FileNotFoundError(f"memory scenario not found: {scenario_id}")
