"""Build unified phase platform context from capability_declaration."""

from pathlib import Path
from typing import Any

from lethyx.contracts.capability_declaration import (
    load_capability_declaration,
    proof_required_capabilities,
)
from lethyx.contracts.models import PlanIR
from lethyx.contracts.runtime_probes import effective_runtime_capabilities
from lethyx.workspace_registry.charter import load_workspace_charter


def overlay_hints_from_declaration(
    drive_root: Path | None,
    workspace_root: Path,
) -> dict[str, Any]:
    declaration = load_capability_declaration(drive_root)
    charter = load_workspace_charter(workspace_root)
    recommended_skills: list[str] = []
    detected_domains: list[str] = []
    if declaration is not None:
        recommended_skills = list(declaration.recommended_skills)
        llm_entry = declaration.capabilities.get("llm_runtime") or declaration.capabilities.get(
            "llm_api"
        )
        if llm_entry is not None and llm_entry.available:
            detected_domains.append("multi_agent_gmas")
    policies = charter.get("policies") if isinstance(charter.get("policies"), dict) else {}
    if policies.get("multi_agent_gmas") is True:
        detected_domains.append("multi_agent_gmas")
    return {
        "recommended_skills": recommended_skills,
        "detected_domains": sorted(set(detected_domains)),
        "charter_policies": policies,
    }


def build_platform_context_envelope(
    *,
    drive_root: Path | None,
    workspace_root: Path,
    plan: PlanIR | None = None,
) -> dict[str, Any]:
    declaration = load_capability_declaration(drive_root)
    effective = effective_runtime_capabilities(drive_root)
    matrix: dict[str, list[str]] = {}
    if plan is not None:
        for subtask in plan.subtasks:
            if subtask.proof is not None:
                matrix[subtask.id] = sorted(proof_required_capabilities(subtask.proof))
    payload: dict[str, Any] = {
        "effective_capabilities": effective,
        "proof_capability_matrix": matrix,
    }
    if declaration is not None:
        payload["capability_declaration"] = declaration.to_dict()
    payload.update(overlay_hints_from_declaration(drive_root, workspace_root))
    return payload


def capability_gate_recovery_hint(issues: list[Any]) -> str | None:
    codes = {str(getattr(item, "code", "") or "") for item in issues}
    if "missing_capability_declaration" in codes:
        return "run preflight capability probes and persist capability_declaration before submitting the plan."
    if "capability_probe_failed" in codes:
        return "route to execute env_capability_repair or apply_plan_revision_patch for proof.required_capabilities; do not loop back to research for a local runtime/proof repair."
    return None
