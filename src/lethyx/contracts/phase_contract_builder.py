"""Merge system, workspace, capability, and phase rules before agent spawn."""

from dataclasses import dataclass, field
from typing import Any

from lethyx.contracts.runtime_probes import proof_requires_capability


@dataclass(frozen=True)
class PhaseContractResult:
    capability_envelope: dict[str, Any]
    conflicts: list[dict[str, str]] = field(default_factory=list)
    diagnostic: str = ""

    @property
    def ok(self) -> bool:
        return not self.conflicts


def _workspace_write_allowed(manifest: Any) -> bool:
    allowed = set(getattr(manifest, "allowed_tools", ()) or ())
    return bool(
        allowed
        & {
            "apply_workspace_patch",
            "delete_workspace_file",
            "replace_workspace_file",
            "repo_write_commit",
            "update_workspace_seed",
            "update_workspace_from_instance",
            "commit_workspace_changes",
        }
    )


def build_phase_contract(
    *,
    manifest: Any,
    phase_id: str,
    workspace_policy: dict[str, Any] | None = None,
    runtime_capabilities: dict[str, bool] | None = None,
    active_subtask: dict[str, Any] | None = None,
    plan_proofs: list[dict[str, Any]] | None = None,
) -> PhaseContractResult:
    ws_policy = workspace_policy or {}
    caps = runtime_capabilities or {}
    write_allowed = _workspace_write_allowed(manifest)
    forbidden_paths = [".git/", ".memory/"]
    workspace_toml_rule: dict[str, str] = {"write": "forbidden"}
    forbidden_paths.append("workspace.toml")

    shell_allowed = bool(
        set(getattr(manifest, "allowed_tools", ()) or ())
        & {"shell", "terminal_session", "run_shell", "run_workspace_command"}
    )

    envelope: dict[str, Any] = {
        "phase": phase_id,
        "workspace_write": {
            "allowed": write_allowed,
            "allowed_paths": "declared_subtask_scope",
            "forbidden_paths": forbidden_paths,
            "workspace_toml": workspace_toml_rule,
        },
        "shell": {"allowed": shell_allowed},
        "memory_write": {
            "allowed_kinds": ["observation", "completion_memory"],
            "durable_requires_verified_evidence": True,
        },
        "verification": {
            "candidate_workspace_writable": write_allowed,
            "evaluator_writable": False,
        },
        "runtime_capabilities": caps,
    }
    if active_subtask:
        declared: list[str] = []
        for key in ("files_to_create", "files_to_change", "files_affected"):
            raw = active_subtask.get(key)
            if isinstance(raw, str) and raw.strip():
                declared.append(raw.strip())
            elif isinstance(raw, (list, tuple)):
                declared.extend(str(item).strip() for item in raw if str(item).strip())
        proof = active_subtask.get("proof")
        proof_payload = proof if isinstance(proof, dict) else {}
        oracle = proof_payload.get("oracle") if isinstance(proof_payload, dict) else {}
        anti_gaming = proof_payload.get("anti_gaming") if isinstance(proof_payload, dict) else {}
        harness_options = (
            proof_payload.get("harness_options") if isinstance(proof_payload, dict) else {}
        )
        envelope["active_subtask"] = {
            "id": str(active_subtask.get("id") or ""),
            "allowed_files": sorted(set(declared)),
            "proof_contract": proof_payload,
            "oracle_contract": oracle if isinstance(oracle, dict) else {},
            "oracle_freeze_policy": {
                "no_test_tampering": (
                    "no_test_tampering"
                    in {
                        str(item)
                        for item in (
                            oracle.get("required_properties") if isinstance(oracle, dict) else []
                        )
                    }
                ),
                "allows_test_only_change": bool(
                    isinstance(anti_gaming, dict) and anti_gaming.get("allows_test_only_change")
                ),
                "proof_revision_requires": [
                    "request_watcher_review verdict=bad_test_contract",
                    "apply_plan_revision_patch typed proof patch with required_deltas",
                    "fresh run_subtask_proof after proof contract revision",
                ],
            },
            "runtime_contract": {
                "harness_profile": str(proof_payload.get("harness_profile") or "")
                if isinstance(proof_payload, dict)
                else "",
                "required_capabilities": list(proof_payload.get("required_capabilities") or [])
                if isinstance(proof_payload, dict)
                else [],
                "harness_options": harness_options if isinstance(harness_options, dict) else {},
            },
            "memory_scope": active_subtask.get("memory_scope")
            if isinstance(active_subtask.get("memory_scope"), dict)
            else {},
            "allowed_tools": list(active_subtask.get("allowed_tools") or []),
            "allowed_skills": list(active_subtask.get("allowed_skills") or []),
            "codeptr_refs": list(active_subtask.get("codeptr_refs") or []),
            "mcp_refs": list(active_subtask.get("mcp_refs") or []),
        }

    conflicts: list[dict[str, str]] = []
    charter_forbidden = ws_policy.get("forbidden_paths")
    if isinstance(charter_forbidden, list):
        for path in charter_forbidden:
            conflicts.append(
                {
                    "code": "policy_conflict",
                    "message": f"Charter forbids `{path}` but phase `{phase_id}` allows conflicting writes.",
                }
            )

    for proof in plan_proofs or []:
        capability_issue = proof_requires_capability(proof, caps)
        if capability_issue and caps:
            conflicts.append(
                {
                    "code": "capability_unavailable",
                    "message": capability_issue,
                }
            )

    diagnostic = ""
    if conflicts:
        diagnostic = "; ".join(item["message"] for item in conflicts)
    return PhaseContractResult(
        capability_envelope=envelope,
        conflicts=conflicts,
        diagnostic=diagnostic,
    )
