"""Supervisor post-phase hooks for verify durable memory and reflexion BKB proposals."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from lethyx.contracts import VerificationReportRef
from lethyx.contracts.validators import validate_verification_report_ref
from lethyx.memory.backends.base import ReflectionQuery
from lethyx.memory.backends.factory import create_durable_backend
from lethyx.memory.hindsight.candidates import (
    build_reflection_question,
    proposal_queue_dir,
    write_hindsight_candidates_as_pending_proposals,
)
from lethyx.memory.proactive.promotion import (
    ProposedBkbPatch,
    accept_bkb_patch,
    reject_bkb_patch,
)

log = logging.getLogger(__name__)


def _read_control_records(
    drive_root: Path,
    *,
    task_id: str,
    phase_started_at: float | None,
) -> list[dict[str, Any]]:
    state_dir = drive_root / "state"
    records: list[dict[str, Any]] = []
    ledger = state_dir / "phase_control_signals.jsonl"
    if ledger.is_file():
        try:
            for line in ledger.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    records.append(row)
        except OSError:
            log.debug("Failed to read phase control ledger", exc_info=True)
    single = state_dir / "phase_control_signal.json"
    if single.is_file():
        try:
            row = json.loads(single.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                records.append(row)
        except (OSError, json.JSONDecodeError):
            pass

    filtered: list[dict[str, Any]] = []
    for row in records:
        row_task = str(row.get("task_id") or "")
        if task_id and row_task and row_task != task_id:
            continue
        ts = row.get("timestamp") or row.get("ts")
        if phase_started_at is not None and ts is not None:
            try:
                if float(ts) < float(phase_started_at):
                    continue
            except (TypeError, ValueError):
                pass
        filtered.append(row)
    return filtered


def promote_durable_row_is_valid(payload: dict[str, Any]) -> bool:
    if payload.get("saved") is not True:
        return False
    store = str(payload.get("durable_store") or payload.get("store") or "")
    node_id = str(payload.get("durable_node_id") or payload.get("canonical_id") or "").strip()
    return store == "palace.durable" and bool(node_id)


def has_valid_promote_durable(
    tools_log_path: Path,
    *,
    task_id: str,
) -> bool:
    if not task_id or not tools_log_path.is_file():
        return False
    try:
        for line in tools_log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("tool") or "") != "promote_to_durable":
                continue
            if str(row.get("task_id") or "") != task_id:
                continue
            payload = row.get("result")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    continue
            if isinstance(payload, dict) and promote_durable_row_is_valid(payload):
                return True
    except OSError:
        log.debug("Failed reading tools log for promote_to_durable", exc_info=True)
    return False


def mirror_verify_durable_if_needed(
    *,
    repo_root: Path,
    drive_root: Path,
    workspace_id: str,
    task_id: str,
    phase_started_at: float | None,
    tools_log_path: Path,
) -> str | None:
    """Write palace.durable verification_report when agent did not promote."""
    if has_valid_promote_durable(tools_log_path, task_id=task_id):
        return None

    report_ref: VerificationReportRef | None = None
    details = ""
    for row in reversed(
        _read_control_records(drive_root, task_id=task_id, phase_started_at=phase_started_at)
    ):
        if str(row.get("kind") or "") != "submit_verification":
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if str(payload.get("status") or "") != "pass":
            continue
        raw_ref = payload.get("verification_report_ref")
        if isinstance(raw_ref, dict):
            report_ref = VerificationReportRef.from_mapping(raw_ref)
            details = str(payload.get("details") or "")
            break
    if report_ref is None:
        return None

    from lethyx.contracts import build_workspace_context
    from lethyx.memory.paths import workspace_root as ws_root_for

    ws_root = ws_root_for(repo_root, workspace_id) if workspace_id else repo_root
    context = build_workspace_context(
        repo_root=str(repo_root.resolve()),
        workspace_root=str(ws_root.resolve()),
        workspace_id=workspace_id,
    )
    issues = validate_verification_report_ref(
        report_ref,
        context=context,
        phase="verify",
    )
    if issues:
        log.warning("Skipping auto durable mirror: %s", issues[0].message)
        return None

    evidence = report_ref.evidence_ref(phase="verify")
    body = details.strip() or f"Verification report {report_ref.report_id}"
    run_id = task_id.split(":", 1)[0] if ":" in task_id else task_id
    try:
        from lethyx.memory.kernel.models import memory_event_from_tool_write
        from lethyx.memory.kernel.writer import write_memory_event

        event = memory_event_from_tool_write(
            content=body,
            title="Verification report",
            memory_kind="verification_report",
            workspace_id=workspace_id,
            tags=["durable", "verification_report"],
            scope="cross_run_durable",
            tier="warm",
            trust_level="public_verified",
            evidence_refs=[evidence],
            lifecycle="active",
            surface="supplemental_evidence",
            source_backend="lethyx.auto_mirror",
            verified=True,
            phase_id="verify",
            run_id=run_id,
            palace_store="palace.durable",
        )
        result = write_memory_event(repo_root, event, workspace_id=workspace_id)
        if not result.saved:
            detail = (
                list(result.policy_issues)
                if result.policy_issues
                else [result.error or "write rejected"]
            )
            log.warning("Auto durable mirror blocked: %s", detail)
            return None
        node_id = result.canonical_id
    except Exception as exc:
        log.warning("Auto durable mirror failed: %s", exc)
        return None
    return node_id


def process_reflexion_bkb_patch(
    *,
    repo_root: Path,
    drive_root: Path,
    workspace_id: str,
    run_id: str = "",
) -> dict[str, Any] | None:
    """Supervisor accept/reject of proposed_bkb_patch.json after reflexion."""
    patch_path = drive_root / "state" / "proposed_bkb_patch.json"
    result: dict[str, Any] | None = None
    try:
        if patch_path.is_file():
            try:
                doc = json.loads(patch_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Invalid proposed_bkb_patch.json: %s", exc)
                doc = None
            if isinstance(doc, dict) and str(doc.get("status") or "") == "candidate":
                ws = str(doc.get("workspace_id") or workspace_id or "")
                patch = ProposedBkbPatch(
                    patch_id=str(doc.get("patch_id") or ""),
                    rules=list(doc.get("rules") or []),
                    source_evidence=list(doc.get("source_evidence") or []),
                    actor="supervisor",
                    run_id=str(doc.get("run_id") or run_id or ""),
                    phase_id=str(doc.get("phase_id") or "reflexion"),
                    workspace_id=ws,
                )
                target = "workspace" if ws else "manager"
                try:
                    result = accept_bkb_patch(
                        repo_root,
                        patch,
                        target=target,
                        drive_root=drive_root,
                    )
                    doc["status"] = "accepted"
                except ValueError as exc:
                    reject_bkb_patch(
                        repo_root,
                        patch,
                        reason=str(exc),
                        target=target,
                    )
                    doc["status"] = "rejected"
                    doc["reject_reason"] = str(exc)
                    result = {"accepted": False, "reason": str(exc)}
                try:
                    patch_path.write_text(
                        json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
    finally:
        hindsight_result = _queue_hindsight_reflection_candidates(
            repo_root=repo_root,
            drive_root=drive_root,
            workspace_id=workspace_id,
            run_id=run_id,
        )
        if hindsight_result and result is not None:
            result["hindsight_candidates"] = hindsight_result
        elif hindsight_result:
            result = {"accepted": None, "hindsight_candidates": hindsight_result}
    return result


def _queue_hindsight_reflection_candidates(
    *,
    repo_root: Path,
    drive_root: Path,
    workspace_id: str,
    run_id: str,
) -> dict[str, Any] | None:
    if os.getenv("LETHYX_HINDSIGHT_REFLECT_ENABLED", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    try:
        max_candidates = int(os.getenv("LETHYX_HINDSIGHT_MAX_CANDIDATES", "3"))
    except ValueError:
        max_candidates = 3
    max_candidates = max(1, min(20, max_candidates))
    try:
        backend = create_durable_backend(
            repo_root=repo_root,
            workspace_id=workspace_id,
        )
        candidates = backend.reflect_candidates(
            ReflectionQuery(
                question=build_reflection_question(max_candidates=max_candidates),
                workspace_id=workspace_id,
                run_id=run_id,
                phase_id="reflexion",
                tags=[
                    f"workspace:{workspace_id}",
                    "source:Lethyx",
                    "trust:supervisor_verified",
                ]
                if workspace_id
                else ["source:Lethyx", "trust:supervisor_verified"],
                max_candidates=max_candidates,
                budget="mid",
            )
        )
    except Exception as exc:
        log.warning("Hindsight reflection candidate generation failed: %s", exc)
        return {"generated": 0, "queued": 0, "error": str(exc)}
    if not candidates:
        return {"generated": 0, "queued": 0}
    result = write_hindsight_candidates_as_pending_proposals(
        drive_root=drive_root,
        repo_root=repo_root,
        workspace_id=workspace_id,
        run_id=run_id,
        phase_id="reflexion",
        candidates=list(candidates),
    )
    if os.getenv("LETHYX_HINDSIGHT_AUTO_ACCEPT_CANDIDATES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        result["auto_accept"] = _auto_accept_hindsight_candidates(
            repo_root=repo_root,
            drive_root=drive_root,
            workspace_id=workspace_id,
            run_id=run_id,
        )
    return result


def _auto_accept_hindsight_candidates(
    *,
    repo_root: Path,
    drive_root: Path,
    workspace_id: str,
    run_id: str,
) -> dict[str, Any]:
    accepted = 0
    rejected = 0
    for path in proposal_queue_dir(drive_root).glob("*.candidate.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if str(doc.get("source") or "") != "hindsight":
            continue
        if run_id and str(doc.get("run_id") or "") != run_id:
            continue
        ws = str(doc.get("workspace_id") or workspace_id or "")
        patch = ProposedBkbPatch(
            patch_id=str(doc.get("patch_id") or path.stem),
            rules=list(doc.get("rules") or []),
            source_evidence=list(doc.get("source_evidence") or []),
            actor="supervisor",
            run_id=str(doc.get("run_id") or run_id or ""),
            phase_id=str(doc.get("phase_id") or "reflexion"),
            workspace_id=ws,
        )
        target = "workspace" if ws else "manager"
        try:
            accept_bkb_patch(
                repo_root,
                patch,
                target=target,
                drive_root=drive_root,
            )
            doc["status"] = "accepted"
            accepted += 1
            new_path = path.with_name(path.name.replace(".candidate.json", ".accepted.json"))
        except ValueError as exc:
            reject_bkb_patch(repo_root, patch, reason=str(exc), target=target)
            doc["status"] = "rejected"
            doc["reject_reason"] = str(exc)
            rejected += 1
            new_path = path.with_name(path.name.replace(".candidate.json", ".rejected.json"))
        try:
            new_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return {"accepted": accepted, "rejected": rejected}
