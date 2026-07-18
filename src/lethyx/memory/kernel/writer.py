"""Single write path for backend-neutral memory events."""

from pathlib import Path

from lethyx.contracts.models import json_ready
from lethyx.memory.backends.contracts import memory_event_to_block_record
from lethyx.memory.backends.factory import create_durable_backend
from lethyx.memory.graph.builder import add_memory_event
from lethyx.memory.kernel.models import (
    MemoryEvent,
    MemoryWriteResult,
    validate_memory_event_for_write,
)
from lethyx.memory.kernel.policy import memory_write_policy_issues
from lethyx.memory.kernel.telemetry import record_memory_event
from lethyx.memory.paths import normalize_workspace_id


def _preset_block_id(event: MemoryEvent) -> str:
    return str(event.metadata.get("block_id") or event.metadata.get("canonical_id") or "").strip()


def write_memory_event(
    repo_root: str | Path,
    event: MemoryEvent,
    *,
    workspace_id: str | None = None,
) -> MemoryWriteResult:
    """Validate policy and write through the selected backend only."""
    root = Path(repo_root)
    ws = normalize_workspace_id(workspace_id or event.workspace_id)
    issues = tuple(
        validate_memory_event_for_write(event)
        + memory_write_policy_issues(
            kind=event.memory_kind,
            tags=list(event.tags),
            metadata={
                "scope": event.scope,
                "tier": event.tier,
                "trust_level": event.trust_level,
                "verified": event.verified,
                "evidence_refs": [json_ready(ref) for ref in event.evidence_refs],
                "verify_run_id": event.metadata.get("verify_run_id"),
            },
        )
    )
    preset_id = _preset_block_id(event)
    if issues:
        record_memory_event(
            root,
            event_type="memory_write_rejected",
            workspace_id=ws,
            run_id=event.run_id,
            phase_id=event.phase_id,
            status="blocked",
            data={"block_id": preset_id, "memory_kind": event.memory_kind, "issues": list(issues)},
        )
        return MemoryWriteResult(
            saved=False, canonical_id=preset_id, store="", backend="", policy_issues=issues
        )

    backend = create_durable_backend(root, workspace_id=ws)
    try:
        block = memory_event_to_block_record(event)
        if preset_id:
            from dataclasses import replace

            block = replace(block, block_id=preset_id)
        result = backend.retain_block(block)
    except Exception as exc:
        record_memory_event(
            root,
            event_type="memory_write_failed",
            workspace_id=ws,
            run_id=event.run_id,
            phase_id=event.phase_id,
            status="failed",
            error=str(exc),
            data={
                "block_id": preset_id,
                "memory_kind": event.memory_kind,
                "surface": event.surface,
            },
        )
        return MemoryWriteResult(
            saved=False, canonical_id=preset_id, store="", backend="", error=str(exc)
        )

    block_id = str(result.get("block_id") or preset_id or "")
    backend_id = str(result.get("backend") or backend.health().get("backend") or "")
    saved = bool(result.get("ok") or result.get("saved"))
    record_memory_event(
        root,
        event_type="memory_write_committed" if saved else "memory_write_failed",
        workspace_id=ws,
        run_id=event.run_id,
        phase_id=event.phase_id,
        status="ok" if saved else "failed",
        error=str(result.get("error") or ""),
        data={
            "block_id": block_id,
            "backend": backend_id,
            "memory_kind": event.memory_kind,
            "surface": event.surface,
        },
    )
    drive_root = event.metadata.get("drive_root") if isinstance(event.metadata, dict) else None
    if saved and drive_root:
        add_memory_event(event, block_id, Path(str(drive_root)))
    return MemoryWriteResult(
        saved=saved,
        canonical_id=block_id,
        store=str(result.get("store") or ""),
        backend=backend_id,
        error=str(result.get("error") or ""),
    )
