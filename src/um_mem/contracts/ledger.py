"""Append-only supervisor evidence ledger used by memory authority checks."""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from um_mem.contracts.hashing import diff_hash, hash_value, workspace_hash
from um_mem.contracts.models import (
    CandidateVerificationSubject,
    EvidenceRef,
    MemorySemanticReview,
    VerificationReportRef,
)


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    event_hash: str
    path: Path


def append_supervisor_ledger_event(
    *,
    repo_root: Path,
    workspace_id: str,
    actor: str,
    stage_id: str,
    tool: str,
    result: dict[str, Any],
    result_hash: str = "",
) -> LedgerEvent:
    """Append a typed verification event and return its evidence identity."""

    ledger_path = _ledger_path(repo_root, workspace_id)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    event_id = f"ledger-{uuid.uuid4().hex[:12]}"
    payload = {
        "event_id": event_id,
        "ts": time.time(),
        "workspace_id": workspace_id,
        "actor": actor,
        "stage_id": stage_id,
        "tool": tool,
        "result": result,
    }
    if result_hash:
        payload["result_hash"] = str(result_hash)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["event_hash"] = event_hash
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return LedgerEvent(event_id=event_id, event_hash=event_hash, path=ledger_path)


def append_verification_report_ledger_event(
    *,
    repo_root: Path,
    workspace_id: str,
    verifier_id: str,
    report_hash: str,
    passed: bool,
    stage_id: str = "memory_promotion",
    workspace_root: Path | None = None,
    changed_files: tuple[str, ...] = (),
    produced_after_event_id: str = "",
    memory_review: MemorySemanticReview | None = None,
) -> VerificationReportRef:
    """Append a verifier-owned report whose typed ref can pass EvidenceResolver."""

    if memory_review is not None and (
        not isinstance(memory_review, MemorySemanticReview)
        or verifier_id != memory_review.review_authority_id
        or report_hash != hash_value(asdict(memory_review))
        or not passed
    ):
        raise ValueError("semantic report must bind its exact independent reviewed subject")
    root = Path(workspace_root or repo_root).resolve()
    current_workspace_hash = workspace_hash(root)
    current_diff_hash = diff_hash(root, changed_files)
    result_hash = hash_value(
        {
            "report_hash": report_hash,
            "passed": passed,
            "workspace_hash": current_workspace_hash,
            "diff_hash": current_diff_hash,
        }
    )
    event = append_supervisor_ledger_event(
        repo_root=Path(repo_root),
        workspace_id=workspace_id,
        actor="verifier",
        stage_id=stage_id,
        tool="verification_report",
        result={
            "verifier_id": verifier_id,
            "report_hash": report_hash,
            "passed": passed,
            "workspace_hash": current_workspace_hash,
            "diff_hash": current_diff_hash,
            "produced_after_event_id": produced_after_event_id,
            **({"memory_review": asdict(memory_review)} if memory_review is not None else {}),
        },
        result_hash=result_hash,
    )
    return VerificationReportRef(
        report_id=event.event_id,
        report_hash=report_hash,
        workspace_hash=current_workspace_hash,
        diff_hash=current_diff_hash,
        produced_after_event_id=produced_after_event_id,
        verifier_id=verifier_id,
        passed=passed,
        ledger_hash=event.event_hash,
    )


def append_memory_application_verification_ledger_event(
    *,
    repo_root: Path,
    workspace_id: str,
    verifier_id: str,
    decision_id: str,
    action_trace_id: str,
    memory_id: str,
    usage_event_id: str,
    task_signature: str,
    outcome: str,
    passed: bool,
) -> EvidenceRef:
    """Bind a verifier-owned result to one observed application."""

    identities = (
        workspace_id,
        verifier_id,
        decision_id,
        action_trace_id,
        memory_id,
        usage_event_id,
        task_signature,
    )
    if any(
        not isinstance(value, str) or not value or value != value.strip() for value in identities
    ):
        raise ValueError("application verification requires exact identities")
    if type(passed) is not bool:
        raise TypeError("application verification passed must be bool")
    if outcome not in {"helped", "hurt"}:
        raise ValueError("application verification outcome must be helped or hurt")
    result = {
        "verifier_id": verifier_id,
        "decision_id": decision_id,
        "action_trace_id": action_trace_id,
        "memory_id": memory_id,
        "usage_event_id": usage_event_id,
        "task_signature": task_signature,
        "outcome": outcome,
        "passed": passed,
    }
    event = append_supervisor_ledger_event(
        repo_root=Path(repo_root),
        workspace_id=workspace_id,
        actor="verifier",
        stage_id="memory_application",
        tool="memory_application_verification",
        result=result,
        result_hash=hash_value(result),
    )
    return EvidenceRef(
        ref_type="verification_report",
        ref_id=event.event_id,
        hash=event.event_hash,
        produced_by="verifier",
        producer_id=verifier_id,
    )


def append_candidate_semantic_verification_ledger_event(
    *,
    repo_root: Path,
    subject: CandidateVerificationSubject,
    verifier_id: str,
    verifier_model_version: str,
    proposer_model_version: str,
    coverage: bool,
    preservation: bool,
    faithfulness: bool,
    instruction_safety: bool,
    reasons: tuple[str, ...],
    workspace_root: Path | None = None,
    changed_files: tuple[str, ...] = (),
    produced_after_event_id: str = "",
    verifier_prompt_version: str = "candidate-review/1",
) -> VerificationReportRef:
    if not isinstance(subject, CandidateVerificationSubject):
        raise TypeError("candidate semantic verification requires a typed subject")
    identities = (
        verifier_id,
        verifier_model_version,
        proposer_model_version,
        verifier_prompt_version,
    )
    if any(
        not isinstance(value, str) or (not value or value != value.strip()) for value in identities
    ):
        raise ValueError("candidate semantic verification requires verifier and model identities")
    outcomes = (coverage, preservation, faithfulness, instruction_safety)
    if any(type(value) is not bool for value in outcomes):
        raise ValueError("candidate semantic outcomes must be bool values")
    if not reasons or any(not isinstance(value, str) or not value.strip() for value in reasons):
        raise ValueError("candidate semantic verification requires reasons")
    semantic_result = {
        "subject": asdict(subject),
        "semantic_subject_hash": hash_value(asdict(subject)),
        "verifier_id": verifier_id,
        "verifier_model_version": verifier_model_version,
        "proposer_model_version": proposer_model_version,
        "verifier_prompt_version": verifier_prompt_version,
        "coverage": coverage,
        "preservation": preservation,
        "faithfulness": faithfulness,
        "instruction_safety": instruction_safety,
        "reasons": reasons,
    }
    report_hash = hash_value(semantic_result)
    passed = all(outcomes) and verifier_model_version != proposer_model_version
    root = Path(workspace_root or repo_root).resolve()
    current_workspace_hash = workspace_hash(root)
    current_diff_hash = diff_hash(root, changed_files)
    event = append_supervisor_ledger_event(
        repo_root=repo_root,
        workspace_id=subject.workspace_id,
        actor="verifier",
        stage_id="candidate_semantic_verification",
        tool="candidate_semantic_verification",
        result={
            **semantic_result,
            "passed": passed,
            "workspace_hash": current_workspace_hash,
            "diff_hash": current_diff_hash,
            "produced_after_event_id": produced_after_event_id,
        },
        result_hash=hash_value(
            {
                "report_hash": report_hash,
                "passed": passed,
                "workspace_hash": current_workspace_hash,
                "diff_hash": current_diff_hash,
            }
        ),
    )
    return VerificationReportRef(
        event.event_id,
        report_hash,
        current_workspace_hash,
        current_diff_hash,
        produced_after_event_id,
        verifier_id,
        passed,
        event.event_hash,
    )


def append_consolidation_semantic_verification_ledger_event(
    *,
    repo_root: Path,
    workspace_id: str,
    verifier_id: str,
    verifier_model_version: str,
    proposer_model_version: str,
    source_ids: tuple[str, ...],
    content: str,
    title: str,
    requested_block_id: str,
    coverage: bool,
    preservation: bool,
    faithfulness: bool,
    reasons: tuple[str, ...],
    workspace_root: Path | None = None,
    changed_files: tuple[str, ...] = (),
    produced_after_event_id: str = "",
) -> VerificationReportRef:
    """Persist verifier-owned semantics bound to one exact consolidation proposal."""

    verifier = str(verifier_id or "").strip()
    verifier_model = str(verifier_model_version or "").strip()
    proposer_model = str(proposer_model_version or "").strip()
    sources = tuple(sorted(dict.fromkeys(str(value or "").strip() for value in source_ids)))
    compact_content = str(content or "").strip()
    compact_title = str(title or "").strip()
    target_id = str(requested_block_id or "").strip()
    compact_reasons = tuple(str(value or "").strip() for value in reasons)
    if not verifier or not verifier_model or not proposer_model:
        raise ValueError("semantic verification requires verifier and model identities")
    if len(sources) < 2 or any(not value for value in sources):
        raise ValueError("semantic verification requires at least two source IDs")
    if not compact_content or not compact_title or not target_id:
        raise ValueError("semantic verification requires a complete proposal subject")
    if any(type(value) is not bool for value in (coverage, preservation, faithfulness)):
        raise ValueError("semantic verification outcomes must be bool values")
    if not compact_reasons or any(not value for value in compact_reasons):
        raise ValueError("semantic verification requires non-empty reasons")

    subject_hash = hash_value(
        {
            "source_ids": sources,
            "content": compact_content,
            "title": compact_title,
            "requested_block_id": target_id,
        }
    )
    semantic_result = {
        "semantic_subject_hash": subject_hash,
        "verifier_id": verifier,
        "verifier_model_version": verifier_model,
        "proposer_model_version": proposer_model,
        "coverage": coverage,
        "preservation": preservation,
        "faithfulness": faithfulness,
        "reasons": compact_reasons,
    }
    report_hash = hash_value(semantic_result)
    passed = coverage and preservation and faithfulness and verifier_model != proposer_model
    root = Path(workspace_root or repo_root).resolve()
    current_workspace_hash = workspace_hash(root)
    current_diff_hash = diff_hash(root, changed_files)
    result_hash = hash_value(
        {
            "report_hash": report_hash,
            "passed": passed,
            "workspace_hash": current_workspace_hash,
            "diff_hash": current_diff_hash,
        }
    )
    event = append_supervisor_ledger_event(
        repo_root=Path(repo_root),
        workspace_id=workspace_id,
        actor="verifier",
        stage_id="consolidation_semantic_verification",
        tool="consolidation_semantic_verification",
        result={
            **semantic_result,
            "passed": passed,
            "workspace_hash": current_workspace_hash,
            "diff_hash": current_diff_hash,
            "produced_after_event_id": produced_after_event_id,
        },
        result_hash=result_hash,
    )
    return VerificationReportRef(
        report_id=event.event_id,
        report_hash=report_hash,
        workspace_hash=current_workspace_hash,
        diff_hash=current_diff_hash,
        produced_after_event_id=produced_after_event_id,
        verifier_id=verifier,
        passed=passed,
        ledger_hash=event.event_hash,
    )


def read_supervisor_ledger_events(*, repo_root: Path, workspace_id: str) -> list[dict[str, Any]]:
    """Read valid object rows from one workspace's supervisor ledger."""

    path = _ledger_path(repo_root, workspace_id)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def find_supervisor_ledger_event(
    *,
    repo_root: Path,
    workspace_id: str,
    event_id: str,
) -> dict[str, Any] | None:
    """Find a ledger event by event id or content hash."""

    needle = str(event_id or "")
    for row in read_supervisor_ledger_events(repo_root=repo_root, workspace_id=workspace_id):
        if needle in {str(row.get("event_id") or ""), str(row.get("event_hash") or "")}:
            return row
    return None


def _ledger_path(repo_root: Path, workspace_id: str) -> Path:
    return (
        Path(repo_root)
        / "workspaces"
        / str(workspace_id or "default")
        / ".memory"
        / "drive"
        / "logs"
        / "supervisor_ledger.jsonl"
    )
