"""Human-readable and JSON reports for memory scenarios."""

import json
from pathlib import Path
from typing import Any

from lethyx.evals.memory_scenarios.models import MemoryScenario, MemoryScenarioResult


def write_scenario_artifacts(
    report_dir: Path,
    scenario: MemoryScenario,
    result: MemoryScenarioResult,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    if scenario.source_path and scenario.source_path.is_file():
        (report_dir / "scenario.yaml").write_text(
            scenario.source_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    payload = {
        "scenario_id": result.scenario_id,
        "ok": result.ok,
        "workspace_id": result.workspace_id,
        "invariant_failures": result.invariant_failures,
        "steps": [
            {
                "step_id": s.step_id,
                "action": s.action,
                "ok": s.ok,
                "errors": s.errors,
            }
            for s in result.step_results
        ],
        "dashboard": result.dashboard,
    }
    (report_dir / "result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(render_report_md(scenario, result), encoding="utf-8")


def render_report_md(scenario: MemoryScenario, result: MemoryScenarioResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    lines = [
        f"# Memory Scenario: {result.scenario_id}",
        "",
        f"Status: {status}",
        "",
        scenario.description.strip(),
        "",
    ]
    if result.invariant_failures:
        lines.append("## Failed invariants")
        for msg in result.invariant_failures:
            lines.append(f"- {msg}")
        lines.append("")

    for step in result.step_results:
        lines.append(f"## Step: {step.step_id} ({step.action})")
        lines.append(f"- ok: {step.ok}")
        if step.errors:
            for err in step.errors:
                lines.append(f"- error: {err}")
        report = step.injection_report
        if report:
            lines.append("### BKB / memory included")
            lines.append("| id | reason | directive |")
            lines.append("|---|---|---|")
            for row in report.get("included") or []:
                if isinstance(row, dict):
                    lines.append(
                        f"| {row.get('id')} | {row.get('reason')} | {row.get('directive')} |"
                    )
            lines.append("### BKB / memory skipped")
            lines.append("| id | reason |")
            lines.append("|---|---|")
            for row in report.get("skipped") or []:
                if isinstance(row, dict):
                    lines.append(f"| {row.get('id')} | {row.get('reason')} |")
        prompt = step.prompt or str((step.task or {}).get("input") or "")
        if prompt:
            markers = [
                ("ALWAYS-LOADED", "[ALWAYS-LOADED MEMORY]"),
                ("Phase instructions", "Phase instructions"),
                ("Supplemental", "Supplemental"),
            ]
            lines.append("### Prompt order")
            lines.append("| marker | position |")
            lines.append("|---|---:|")
            for label, needle in markers:
                pos = prompt.find(needle)
                lines.append(f"| {label} | {pos if pos >= 0 else 'missing'} |")
        before_p = (step.snapshot_before.get("palace") or {}).get("count", 0)
        after_p = (step.snapshot_after.get("palace") or {}).get("count", 0)
        if before_p or after_p:
            lines.append("### Palace diff")
            lines.append("| before | after | delta |")
            lines.append(f"| {before_p} | {after_p} | {after_p - before_p} |")
        lines.append("")

    lines.append("## Artifacts")
    for step in result.step_results:
        lines.append(f"- prompt_{step.step_id}.txt")
        lines.append(f"- memory_injection_report_{step.step_id}.json")
    return "\n".join(lines)


def print_terminal_summary(results: list[MemoryScenarioResult]) -> str:
    lines = ["Memory Scenario Harness", ""]
    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    for r in results:
        mark = "OK" if r.ok else "FAIL"
        lines.append(f"{mark} {r.scenario_id}")
        if not r.ok:
            for msg in r.invariant_failures[:5]:
                lines.append(f"  - {msg}")
            lines.append(f"  report: {r.report_dir / 'report.md'}")
    lines.append("")
    lines.append(f"Summary: {passed} passed, {failed} failed")
    text = "\n".join(lines)
    print(text)
    return text


def build_dashboard(results: list[MemoryScenarioResult], reports_dir: Path) -> dict[str, Any]:
    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    top_failures = []
    for r in results:
        if not r.ok:
            for msg in r.invariant_failures[:3]:
                top_failures.append({"scenario": r.scenario_id, "step": "", "invariant": msg})
    return {
        "ok": failed == 0,
        "scenarios": len(results),
        "passed": passed,
        "failed": failed,
        "top_failures": top_failures[:10],
        "reports_dir": str(reports_dir),
    }
