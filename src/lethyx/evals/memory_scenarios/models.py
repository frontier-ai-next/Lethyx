"""Data models for memory scenario harness."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScenarioSeed:
    manager_core: dict[str, str] = field(default_factory=dict)
    workspace_core: dict[str, str] = field(default_factory=dict)
    palace: dict[str, Any] = field(default_factory=dict)
    workspace_fixture: str = "default"
    manager_fixture: str = "manager_default"
    extra_workspaces: dict[str, str] = field(default_factory=dict)


@dataclass
class ScenarioStep:
    id: str
    action: str
    phase: str | None = None
    run_id: str = "run-scenario"
    workspace_id: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryScenario:
    id: str
    description: str
    workspace: str
    mode: str
    env: dict[str, str]
    seed: ScenarioSeed
    steps: list[ScenarioStep]
    assertions: dict[str, Any]
    source_path: Path | None = None
    raw_seed: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    requires_no_volatile_stub: bool = False


@dataclass
class ScenarioStepResult:
    step_id: str
    action: str
    ok: bool
    prompt: str = ""
    task: dict[str, Any] = field(default_factory=dict)
    overlays: dict[str, Any] = field(default_factory=dict)
    bundle: dict[str, Any] = field(default_factory=dict)
    injection_report: dict[str, Any] = field(default_factory=dict)
    palace_before: dict[str, Any] = field(default_factory=dict)
    palace_after: dict[str, Any] = field(default_factory=dict)
    snapshot_before: dict[str, Any] = field(default_factory=dict)
    snapshot_after: dict[str, Any] = field(default_factory=dict)
    files_changed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class MemoryScenarioResult:
    scenario_id: str
    ok: bool
    repo_root: Path
    workspace_id: str
    step_results: list[ScenarioStepResult]
    invariant_failures: list[str]
    report_dir: Path
    dashboard: dict[str, Any] = field(default_factory=dict)

    @property
    def summary_text(self) -> str:
        if self.scenario_id == "__aggregate__":
            if self.ok:
                return "All memory scenarios passed"
            return "\n".join(self.invariant_failures[:30]) or "Memory scenarios failed"
        if self.ok:
            return f"SCENARIO {self.scenario_id} PASS"
        lines = [f"SCENARIO {self.scenario_id} FAIL"]
        for msg in self.invariant_failures[:20]:
            lines.append(f"  ✗ {msg}")
        lines.append(f"  artifacts: {self.report_dir / 'report.md'}")
        return "\n".join(lines)
