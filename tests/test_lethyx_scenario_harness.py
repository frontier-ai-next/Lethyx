import json
import tempfile
from pathlib import Path

from lethyx.evals.memory_scenarios.phase_tasks import (
    PhaseNode,
    build_phase_task,
    load_manifest,
)
from lethyx.evals.memory_scenarios.runner import MemoryScenarioRunner


def test_scenario_runner_imports_without_host_orchestrator_package() -> None:
    assert MemoryScenarioRunner.__name__ == "MemoryScenarioRunner"


def test_phase_task_builder_writes_framework_neutral_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        drive = repo / "workspaces" / "demo" / ".memory" / "drive"
        manifest = load_manifest(repo / "missing-manifest.yaml")
        task = build_phase_task(
            phase_node=PhaseNode(id="research", manifest_id="research"),
            manifest=manifest,
            workspace_id="demo",
            run_id="run-test",
            repo_root=repo,
            drive_root=drive,
        )

        report_path = drive / "state" / "memory_injection_report_latest.json"
        bundle_path = drive / "state" / "llm_input_bundle_latest.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

        assert task["phase_id"] == "research"
        assert "## [ALWAYS-LOADED MEMORY]" in task["input"]
        assert "## Phase instructions" in task["input"]
        assert task["context_overlays"]["memory_injection_contract"]["mode"] == "LETHYX_owned"
        assert report["phase_id"] == "research"
        assert bundle["task"]["workspace_id"] == "demo"
