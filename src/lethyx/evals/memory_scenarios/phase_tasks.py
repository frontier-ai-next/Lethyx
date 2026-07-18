"""Framework-neutral phase task helpers for the memory scenario harness."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lethyx.memory.palace.recall import RecallBundle
from lethyx.memory.proactive.compiler import ProactiveMemoryCompiler
from lethyx.memory.proactive.models import ProactiveMemoryOverlay


@dataclass(frozen=True)
class PhaseBudget:
    max_tokens: int = 4000


@dataclass(frozen=True)
class PhaseTools:
    allowed: tuple[str, ...] = ()
    available: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseManifest:
    id: str
    description: str = ""
    instructions: str = ""
    budgets: PhaseBudget = field(default_factory=PhaseBudget)
    tools: PhaseTools = field(default_factory=PhaseTools)


@dataclass(frozen=True)
class PhaseNode:
    id: str
    manifest_id: str = ""
    status: str = "running"

    def to_payload(self) -> dict[str, str]:
        return {
            "id": self.id,
            "manifest_id": self.manifest_id or self.id,
            "status": self.status,
        }


def load_manifest(path: Path) -> PhaseManifest:
    """Load a phase manifest, falling back to a minimal local manifest."""

    data: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded

    phase_id = str(data.get("id") or path.stem or "task").strip() or "task"
    budgets = data.get("budgets") if isinstance(data.get("budgets"), dict) else {}
    tools = data.get("tools") if isinstance(data.get("tools"), dict) else {}
    allowed = tools.get("allowed") or tools.get("available") or data.get("allowed_tools") or ()
    available = tools.get("available") or allowed

    return PhaseManifest(
        id=phase_id,
        description=str(
            data.get("description") or f"Run the {phase_id} phase with Lethyx advisory memory."
        ),
        instructions=str(data.get("instructions") or ""),
        budgets=PhaseBudget(max_tokens=int(budgets.get("max_tokens") or 4000)),
        tools=PhaseTools(
            allowed=tuple(str(item) for item in allowed),
            available=tuple(str(item) for item in available),
        ),
    )


def render_phase_user_prompt(
    manifest: PhaseManifest,
    recall_bundle: RecallBundle | None = None,
    *,
    proactive_overlay: ProactiveMemoryOverlay | None = None,
) -> str:
    """Render the prompt shape expected by the scenario assertions."""

    lines: list[str] = []
    if proactive_overlay is not None:
        lines.append(proactive_overlay.render_markdown())
        lines.append("")

    lines.append("## Phase instructions")
    lines.append(manifest.description.strip() or f"Phase: {manifest.id}")
    if manifest.instructions.strip():
        lines.append("")
        lines.append(manifest.instructions.strip())

    bundle = recall_bundle or RecallBundle()
    nodes = bundle.all_nodes()
    if nodes:
        lines.append("")
        lines.append("## Supplemental recall (NON-DIRECTIVE)")
        for node in nodes[:8]:
            text = str(node.get("text") or node.get("content") or "").strip()
            source = str(node.get("source_id") or node.get("id") or "").strip()
            if text:
                prefix = f"- source_id={source}: " if source else "- "
                lines.append(prefix + text[:800])

    return "\n".join(lines).strip()


def build_phase_task(
    *,
    phase_node: PhaseNode,
    manifest: PhaseManifest,
    workspace_id: str,
    run_id: str,
    palace: Any = None,
    repo_root: Path,
    drive_root: Path,
) -> dict[str, Any]:
    """Build a scenario task without depending on a host orchestration runtime."""

    del palace
    phase_id = phase_node.id or manifest.id
    drive_root.mkdir(parents=True, exist_ok=True)
    state_dir = drive_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    overlay = ProactiveMemoryCompiler().build_overlay(
        repo_root=repo_root,
        workspace_id=workspace_id,
        run_id=run_id,
        phase_id=phase_id,
        subtask_id=None,
        task_brief="scenario",
        manifest=manifest,
        drive_root=drive_root,
    )
    prompt = render_phase_user_prompt(
        manifest,
        RecallBundle(),
        proactive_overlay=overlay,
    )
    overlay_payload = overlay.to_payload()
    audit = _injection_audit(overlay_payload)
    overlay_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    contract = {
        "mode": "LETHYX_owned",
        "proactive_overlay_injected": True,
        "proactive_overlay_hash": overlay_hash,
        "retrieval_is_supplemental_only": True,
    }
    injection_report = {
        "workspace_id": workspace_id,
        "run_id": run_id,
        "phase_id": phase_id,
        "overlay_hash": overlay_hash,
        "included": [
            {
                **row,
                "directive": row.get("directive") or row.get("id") or "",
            }
            for row in audit.get("included_bkb", [])
            if isinstance(row, dict)
        ],
        "skipped": [row for row in audit.get("skipped_bkb", []) if isinstance(row, dict)],
    }
    overlays = {
        "phase_node": phase_node.to_payload(),
        "proactive_memory": overlay_payload,
        "memory_injection_contract": contract,
        "prevent_host_auto_core_overlay": True,
        "proactive_memory_rendered_in_task_input": True,
    }
    task = {
        "id": f"{run_id}:{phase_id}",
        "task_id": f"{run_id}:{phase_id}",
        "workspace_id": workspace_id,
        "phase_id": phase_id,
        "input": prompt,
        "context_overlays": overlays,
    }
    bundle_path = state_dir / "llm_input_bundle_latest.json"
    bundle = {
        "task": {
            "id": task["id"],
            "workspace_id": workspace_id,
            "phase_id": phase_id,
            "input": prompt,
        },
        "context_overlays": overlays,
        "memory_injection_report": injection_report,
    }
    overlays["llm_input_bundle"] = {
        "available": True,
        "path": str(bundle_path),
    }
    bundle["context_overlays"] = overlays

    (state_dir / "memory_injection_report_latest.json").write_text(
        json.dumps(injection_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return task


def _injection_audit(payload: dict[str, Any]) -> dict[str, Any]:
    audit = payload.get("injection_audit")
    if isinstance(audit, dict):
        return audit
    telemetry = payload.get("telemetry")
    if isinstance(telemetry, dict) and isinstance(telemetry.get("injection_audit"), dict):
        return telemetry["injection_audit"]
    return {}
