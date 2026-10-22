"""Single write path for backend-neutral memory events."""

import hashlib
import sqlite3
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from um_mem.contracts import (
    EvidenceRef,
    MemoryBlockRecord,
    MemoryEvent,
    RecordBlockMembership,
    is_protected_memory,
)
from um_mem.contracts.evidence import EvidenceResolver
from um_mem.contracts.ledger import find_supervisor_ledger_event
from um_mem.contracts.models import WorkspaceContext
from um_mem.memory.backends.contracts import (
    ALLOWED_EDGE_TYPES,
    CanonicalHeadConflict,
    CanonicalHeadReservation,
    EdgeType,
    MemoryEdgeRecord,
    memory_event_to_block_record,
)
from um_mem.memory.backends.factory import create_durable_backend
from um_mem.memory.blocks.catalog import compile_semantic_catalog, sync_block_descriptor_index
from um_mem.memory.blocks.registry import load_memory_blocks, load_workspace_topology
from um_mem.memory.erase_journal import (
    EraseRequestConflict,
    EraseRequestState,
    open_erase_request,
    record_erase_request_outcome,
)
from um_mem.memory.kernel.models import MemoryWriteResult, block_record_to_event
from um_mem.memory.kernel.telemetry import record_memory_event
from um_mem.memory.mutation import (
    CanonicalMutationIdentity,
    MemoryEraseCertificate,
    MemoryEraseRequest,
    MutationDecision,
    plan_lifecycle_mutation,
    plan_memory_mutation,
    resolve_canonical_revision_head,
)
from um_mem.memory.paths import normalize_workspace_id, workspace_memory_root

_ERASE_AUTHORIZING_ACTORS = frozenset({"supervisor", "verifier", "watcher", "harness"})
_ERASE_RESUMABLE_STATUSES = frozenset({"requested", "purging"})
_ERASE_NATIVE_FAILURE_CODES = frozenset(
    {
        "native_exact_id_not_found",
        "native_exact_id_not_deleted",
        "native_purge_unavailable",
        "native_purge_unverified",
        "native_record_workspace_mismatch",
        "native_edges_remaining",
        "raw_evidence_not_retained",
        "retraction_failed",
    }
)


def _preset_block_id(event: MemoryEvent) -> str:
    return str(event.block_id or "").strip()


def _block_from_event(event: MemoryEvent, *, preset_id: str = "") -> MemoryBlockRecord:
    block = memory_event_to_block_record(event)
    return replace(
        block,
        block_id=preset_id or block.block_id,
        sensitivity=event.sensitivity,
        content_mode=event.content_mode,
        operational_risk=event.operational_risk,
        revision_of=event.revision_of,
        supersedes_ids=event.supersedes_ids,
        expires_at=event.expires_at,
    )


def transition_memory_status(
    repo_root: str | Path,
    memory_id: str,
    status: str,
    *,
    workspace_id: str,
    reason: str = "",
    evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...] = (),
    authority: str | None = None,
    retention: str | None = None,
    delivery_policy: str | None = None,
    backend_instance: Any | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    if expected_revision is not None and (
        not isinstance(expected_revision, str) or not expected_revision.strip()
    ):
        raise ValueError("expected_revision must be a nonempty revision identity")
    root = Path(repo_root)
    ws = normalize_workspace_id(workspace_id)
    backend = (
        backend_instance
        if backend_instance is not None
        else create_durable_backend(root, workspace_id=ws)
    )
    getter = getattr(backend, "get_block", None)
    if not callable(getter):
        raise RuntimeError("selected memory backend does not support status lookup")
    root_record = getter(memory_id)
    if not isinstance(root_record, MemoryBlockRecord):
        record_memory_event(
            root,
            event_type="memory_status_transition",
            workspace_id=ws,
            status="blocked",
            operation="status_transition",
            decision="canonical_mutation",
            outcome="not_found",
            selected_ids=[memory_id],
        )
        return {"ok": False, "error": "not_found", "block_id": memory_id}
    if root_record.workspace_id != ws:
        raise ValueError("memory block workspace does not match the selected writer workspace")
    raw_edges_for = getattr(backend, "edges_for", None)
    current = resolve_canonical_revision_head(
        root_record,
        canonical_key=root_record.canonical_key,
        workspace_id=ws,
        get_block=getter,
        edges_for=raw_edges_for if callable(raw_edges_for) else None,
    )
    if status == "retracted":
        if retention not in (None, "archive"):
            raise ValueError("retracted memory requires archive retention")
        if delivery_policy not in (None, "hidden"):
            raise ValueError("retracted memory requires hidden delivery_policy")
        retention = "archive"
        delivery_policy = "hidden"
    if (
        status == "retracted"
        and current.status == "retracted"
        and expected_revision is not None
        and root_record.revision_id == expected_revision
        and current.revision_of == memory_id
        and authority in (None, current.authority)
    ):
        if _normal_write_is_pending(backend, current.block_id):
            recovered = _write_memory_event(
                root,
                block_record_to_event(current),
                workspace_id=ws,
                backend_instance=backend,
                mutation=MutationDecision(record=current, action="retry"),
            )
            return {**recovered.raw, "saved": recovered.saved, "already_retracted": True}
        return {
            "ok": True,
            "saved": True,
            "already_retracted": True,
            "block_id": current.block_id,
            "backend": str(getattr(backend, "backend_id", "")),
        }
    if expected_revision is not None and (
        current.block_id != memory_id or current.revision_id != expected_revision
    ):
        return {
            "ok": False,
            "saved": False,
            "error": "expected_revision_conflict",
            "block_id": memory_id,
        }
    membership = current.block_membership
    if membership is not None:
        catalog = compile_semantic_catalog(
            load_memory_blocks(
                root,
                workspace_id=ws,
                profile_id=current.profile_id,
                strict=True,
            )
        )
        current = replace(
            current,
            block_membership=RecordBlockMembership(
                primary_block_id=membership.primary_block_id,
                secondary_block_ids=membership.secondary_block_ids,
                catalog_revision_id=catalog.revision_id,
            ),
        )
    current_identity: CanonicalMutationIdentity | None = None
    lookup_identity = getattr(backend, "canonical_mutation_identity", None)
    if callable(lookup_identity):
        current_identity = lookup_identity(successor_id=current.block_id, workspace_id=ws)
        if current_identity is not None and not isinstance(
            current_identity, CanonicalMutationIdentity
        ):
            raise RuntimeError(
                "selected memory backend returned an invalid canonical mutation identity"
            )
    try:
        action = {
            "deprecated": "deprecate",
            "retracted": "retract",
            "accepted": "promote",
            "quarantined": "quarantine",
        }[status]
    except KeyError as exc:
        raise ValueError(f"unsupported Order15 lifecycle status: {status}") from exc
    decision = plan_lifecycle_mutation(
        current,
        action=action,
        status=status,
        reason=reason,
        evidence_refs=tuple(evidence_refs),
        authority=authority,
        retention=retention,
        delivery_policy=delivery_policy,
        current_identity=current_identity,
    )
    write = _write_memory_event(
        root,
        block_record_to_event(decision.record),
        workspace_id=ws,
        backend_instance=backend,
        mutation=decision,
    )
    result = dict(write.raw)
    result.setdefault("ok", write.saved)
    result.setdefault("saved", write.saved)
    result.setdefault("block_id", write.canonical_id)
    result.setdefault("backend", write.backend)
    saved = write.saved
    record_memory_event(
        root,
        event_type="memory_status_transition",
        workspace_id=ws,
        run_id=current.run_id,
        stage_id=current.stage_id,
        backend=str(result.get("backend") or ""),
        status="ok" if saved else "failed",
        error=str(result.get("error") or ""),
        operation="status_transition",
        decision="canonical_mutation",
        outcome=str(status),
        data={
            "from_status": current.status,
            "to_status": status,
            "reason_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest()[:16]
            if reason
            else "",
        },
        selected_ids=[memory_id],
    )
    return result


def _erase_rejected(
    root: Path,
    request: MemoryEraseRequest,
    *,
    workspace_id: str,
    error: str,
    status: str = "blocked",
    erase_request: EraseRequestState | None = None,
) -> dict[str, Any]:
    record_memory_event(
        root,
        event_type="memory_erase_rejected",
        workspace_id=workspace_id,
        status=status,
        error=error,
        operation="governed_erase",
        decision="authorization_checked",
        outcome=error,
        data={
            "request_id_sha256": hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()[:16]
            if request.request_id
            else "",
            "reason_sha256": hashlib.sha256(request.reason.encode("utf-8")).hexdigest()[:16]
            if request.reason
            else "",
        },
        selected_ids=[request.memory_id],
    )
    result: dict[str, Any] = {"ok": False, "error": error}
    if erase_request is not None:
        result["erase_request"] = erase_request.as_dict()
    return result


def _erase_authorization_is_valid(
    root: Path,
    request: MemoryEraseRequest,
    *,
    workspace_id: str,
) -> bool:
    authorization = request.authorization
    if (
        not isinstance(authorization, EvidenceRef)
        or authorization.ref_type != "ledger_event"
        or authorization.produced_by not in _ERASE_AUTHORIZING_ACTORS
    ):
        return False
    refs = (authorization, *request.evidence_refs)
    if not all(isinstance(ref, EvidenceRef) for ref in refs):
        return False
    context = WorkspaceContext(
        repo_root=str(root.resolve()),
        workspace_root=str(root.resolve()),
        workspace_id=workspace_id,
    )
    resolver = EvidenceResolver(context)
    if any(resolver.validate_ref(ref) for ref in refs):
        return False
    row = find_supervisor_ledger_event(
        repo_root=root,
        workspace_id=workspace_id,
        event_id=authorization.ref_id,
    )
    if row is None or str(row.get("actor") or "") != authorization.produced_by:
        return False
    approval = row.get("result")
    if not isinstance(approval, dict):
        return False
    target_hash = hashlib.sha256(request.memory_id.encode("utf-8")).hexdigest()
    reason_hash = hashlib.sha256(request.reason.encode("utf-8")).hexdigest()
    return (
        str(row.get("tool") or "") == "authorize_memory_erase"
        and approval.get("action") == "erase_memory"
        and approval.get("approved") is True
        and str(approval.get("request_id") or "") == request.request_id
        and str(approval.get("memory_id_sha256") or "") == target_hash
        and str(approval.get("reason_sha256") or "") == reason_hash
    )


def _erase_request_fingerprint(
    request: MemoryEraseRequest,
    *,
    workspace_id: str,
    backend: str,
) -> str:
    refs = (request.authorization, *request.evidence_refs)
    reference_fingerprints = sorted(
        "|".join(
            (
                str(ref.ref_type),
                hashlib.sha256(ref.ref_id.encode("utf-8")).hexdigest(),
                str(ref.hash or ""),
                str(ref.produced_by),
            )
        )
        for ref in refs
    )
    payload = "\n".join(
        (
            hashlib.sha256(workspace_id.encode("utf-8")).hexdigest(),
            hashlib.sha256(request.memory_id.encode("utf-8")).hexdigest(),
            hashlib.sha256(request.reason.encode("utf-8")).hexdigest(),
            backend,
            *reference_fingerprints,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_erase_backend(backend: Any, requested_backend: str) -> str:
    return str(requested_backend or getattr(backend, "backend_id", "") or "selected")


def _record_erase_outcome(
    root: Path,
    erase_request: EraseRequestState,
    *,
    status: str,
    error: str,
) -> EraseRequestState | None:
    try:
        return record_erase_request_outcome(
            root,
            erase_request,
            status=status,
            error=error,
        )
    except (LookupError, OSError, sqlite3.Error):
        return None


def _completed_erase_certificate(erase_request: EraseRequestState) -> dict[str, Any]:
    certificate_payload = f"{erase_request.fingerprint_sha256}\n{erase_request.request_id_sha256}\n{erase_request.target_id_sha256}\n{erase_request.backend}\n{erase_request.updated_at}\nnative_exact_id_absent\nnative_edges_absent\nraw_evidence_retained\nretracted"
    certificate = MemoryEraseCertificate(
        certificate_sha256=hashlib.sha256(certificate_payload.encode("utf-8")).hexdigest(),
        request_id_sha256=erase_request.request_id_sha256,
        target_id_sha256=erase_request.target_id_sha256,
        backend=erase_request.backend,
        issued_at=erase_request.updated_at,
        native_exact_id_absent=True,
        native_edges_absent=True,
        raw_evidence_retained=True,
        sidecar_status="retracted",
    )
    return certificate.as_dict()


def _erase_purge_failure(
    root: Path,
    request: MemoryEraseRequest,
    *,
    workspace_id: str,
    error: str,
    erase_request: EraseRequestState,
) -> dict[str, Any]:
    persisted = _record_erase_outcome(
        root,
        erase_request,
        status="purging",
        error=error,
    )
    if persisted is None:
        return _erase_rejected(
            root,
            request,
            workspace_id=workspace_id,
            error="erase_journal_unavailable",
            status="unavailable",
        )
    return _erase_rejected(
        root,
        request,
        workspace_id=workspace_id,
        error=error,
        status="unavailable",
        erase_request=persisted,
    )


def erase_memory(
    repo_root: str | Path,
    request: MemoryEraseRequest,
    *,
    workspace_id: str,
    backend_instance: Any | None = None,
    backend_id: str = "",
) -> dict[str, Any]:
    root = Path(repo_root)
    ws = normalize_workspace_id(workspace_id)
    if (
        not request.request_id.strip()
        or not request.memory_id.strip()
        or not request.reason.strip()
        or normalize_workspace_id(request.workspace_id) != ws
    ):
        return _erase_rejected(root, request, workspace_id=ws, error="invalid_erase_request")
    if not _erase_authorization_is_valid(root, request, workspace_id=ws):
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="invalid_erase_authorization",
        )
    backend = (
        backend_instance
        if backend_instance is not None
        else create_durable_backend(root, workspace_id=ws)
    )
    selected_backend = _selected_erase_backend(backend, backend_id)
    workspace_id_sha256 = hashlib.sha256(ws.encode("utf-8")).hexdigest()
    request_id_sha256 = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()
    target_id_sha256 = hashlib.sha256(request.memory_id.encode("utf-8")).hexdigest()
    fingerprint_sha256 = _erase_request_fingerprint(
        request,
        workspace_id=ws,
        backend=selected_backend,
    )
    erase_request: EraseRequestState | None
    try:
        erase_request, created = open_erase_request(
            root,
            workspace_id_sha256=workspace_id_sha256,
            request_id_sha256=request_id_sha256,
            fingerprint_sha256=fingerprint_sha256,
            target_id_sha256=target_id_sha256,
            backend=selected_backend,
        )
    except EraseRequestConflict:
        return _erase_rejected(root, request, workspace_id=ws, error="erase_request_conflict")
    except (OSError, sqlite3.Error):
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="erase_journal_unavailable",
            status="unavailable",
        )
    completed = not created and erase_request.status == "completed"
    if not created and not completed and erase_request.status not in _ERASE_RESUMABLE_STATUSES:
        return {
            "ok": False,
            "error": erase_request.error,
            "erase_request": erase_request.as_dict(),
        }
    resuming_native_purge = erase_request.status == "purging"
    getter = getattr(backend, "get_block", None)
    if not callable(getter):
        erase_request = _record_erase_outcome(
            root,
            erase_request,
            status="unavailable",
            error="native_purge_unsupported",
        )
        if erase_request is None:
            return _erase_rejected(
                root,
                request,
                workspace_id=ws,
                error="erase_journal_unavailable",
                status="unavailable",
            )
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="native_purge_unsupported",
            status="unavailable",
            erase_request=erase_request,
        )
    requested = getter(request.memory_id)
    if not isinstance(requested, MemoryBlockRecord):
        erase_request = _record_erase_outcome(
            root,
            erase_request,
            status="blocked",
            error="not_found",
        )
        if erase_request is None:
            return _erase_rejected(
                root,
                request,
                workspace_id=ws,
                error="erase_journal_unavailable",
                status="unavailable",
            )
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="not_found",
            erase_request=erase_request,
        )
    if requested.workspace_id != ws:
        erase_request = _record_erase_outcome(
            root,
            erase_request,
            status="blocked",
            error="workspace_mismatch",
        )
        if erase_request is None:
            return _erase_rejected(
                root,
                request,
                workspace_id=ws,
                error="erase_journal_unavailable",
                status="unavailable",
            )
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="workspace_mismatch",
            erase_request=erase_request,
        )
    edges_for = getattr(backend, "edges_for", None)
    try:
        current = resolve_canonical_revision_head(
            requested,
            canonical_key=requested.canonical_key,
            workspace_id=ws,
            get_block=getter,
            edges_for=edges_for if callable(edges_for) else None,
        )
    except ValueError:
        return {
            "ok": False,
            "error": "canonical_head_unavailable",
            "erase_request": erase_request.as_dict(),
        }
    if completed:
        if (
            current.status == "retracted"
            and current.retention == "archive"
            and current.delivery_policy == "hidden"
        ):
            return {
                "ok": True,
                "block_id": current.block_id,
                "erase_request": erase_request.as_dict(),
                "certificate": _completed_erase_certificate(erase_request),
            }
        return {
            "ok": False,
            "error": "completed_erase_head_invalid",
            "erase_request": erase_request.as_dict(),
        }
    purger = getattr(backend, "purge_exact_native_artifacts", None)
    if selected_backend == "mempalace":
        from um_mem.memory.backends.mempalace import MemPalaceBackend

        is_mempalace_backend = isinstance(backend, MemPalaceBackend)
    else:
        is_mempalace_backend = False
    if selected_backend != "mempalace" or not is_mempalace_backend or not callable(purger):
        erase_request = _record_erase_outcome(
            root,
            erase_request,
            status="unavailable",
            error="native_purge_unsupported",
        )
        if erase_request is None:
            return _erase_rejected(
                root,
                request,
                workspace_id=ws,
                error="erase_journal_unavailable",
                status="unavailable",
            )
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="native_purge_unsupported",
            status="unavailable",
            erase_request=erase_request,
        )
    erase_request = _record_erase_outcome(
        root,
        erase_request,
        status="purging",
        error="",
    )
    if erase_request is None:
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="erase_journal_unavailable",
            status="unavailable",
        )
    original_content = current.content
    retained = current
    if (
        current.status != "retracted"
        or current.retention != "archive"
        or current.delivery_policy != "hidden"
    ):
        try:
            retraction = transition_memory_status(
                root,
                current.block_id,
                "retracted",
                workspace_id=ws,
                reason="governed erase authorized",
                evidence_refs=(request.authorization, *request.evidence_refs),
                retention="archive",
                delivery_policy="hidden",
                backend_instance=backend,
            )
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            return _erase_purge_failure(
                root,
                request,
                workspace_id=ws,
                error="retraction_failed",
                erase_request=erase_request,
            )
        if not bool(retraction.get("ok")):
            return _erase_purge_failure(
                root,
                request,
                workspace_id=ws,
                error="retraction_failed",
                erase_request=erase_request,
            )
        retained = getter(str(retraction.get("block_id") or ""))
    if (
        not isinstance(retained, MemoryBlockRecord)
        or retained.workspace_id != ws
        or retained.status != "retracted"
        or retained.retention != "archive"
        or retained.content != original_content
    ):
        return _erase_purge_failure(
            root,
            request,
            workspace_id=ws,
            error="raw_evidence_not_retained",
            erase_request=erase_request,
        )
    purged_ids: set[str] = set()
    for purge_target in (requested, retained):
        if purge_target.block_id in purged_ids:
            continue
        purged_ids.add(purge_target.block_id)
        try:
            # Governed erasure must close queued projection before certifying absence.
            # Ordinary writes acknowledge durability early; this lifecycle boundary
            # deliberately waits for the exact native receipt before deleting it.
            status_reader = getattr(backend, "write_outbox_status", None)
            if callable(status_reader) and status_reader(purge_target.block_id) == "pending":
                backend.retain_block(purge_target)
                if status_reader(purge_target.block_id) != "committed":
                    raise RuntimeError("erase projection is still pending")
            purge_result = dict(
                purger(purge_target, allow_native_absent=resuming_native_purge) or {}
            )
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            return _erase_purge_failure(
                root,
                request,
                workspace_id=ws,
                error="native_purge_unavailable",
                erase_request=erase_request,
            )
        native_error = str(purge_result.get("error") or "")
        if (
            not bool(purge_result.get("ok"))
            or purge_result.get("native_exact_id_absent") is not True
            or purge_result.get("native_edges_absent") is not True
        ):
            return _erase_purge_failure(
                root,
                request,
                workspace_id=ws,
                error=(
                    native_error
                    if native_error in _ERASE_NATIVE_FAILURE_CODES
                    else "native_purge_unverified"
                ),
                erase_request=erase_request,
            )
    erase_request = _record_erase_outcome(
        root,
        erase_request,
        status="completed",
        error="",
    )
    if erase_request is None:
        return _erase_rejected(
            root,
            request,
            workspace_id=ws,
            error="erase_journal_unavailable",
            status="unavailable",
        )
    certificate = _completed_erase_certificate(erase_request)
    record_memory_event(
        root,
        event_type="memory_erase_completed",
        workspace_id=ws,
        run_id=retained.run_id,
        stage_id=retained.stage_id,
        backend=selected_backend,
        status="ok",
        operation="governed_erase",
        decision="exact_native_closure",
        outcome="completed",
        data={
            "certificate_sha256": certificate["certificate_sha256"],
            "native_exact_id_absent": True,
            "native_edges_absent": True,
            "raw_evidence_retained": True,
        },
        selected_ids=[request.memory_id],
    )
    return {
        "ok": True,
        "block_id": retained.block_id,
        "erase_request": erase_request.as_dict(),
        "certificate": certificate,
    }


def write_memory_edge(
    repo_root: str | Path,
    edge: MemoryEdgeRecord,
    *,
    workspace_id: str,
    backend_instance: Any | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    ws = normalize_workspace_id(workspace_id)
    edge_type = str(edge.edge_type or "").strip()
    if edge_type not in ALLOWED_EDGE_TYPES:
        raise ValueError(f"unsupported memory edge type: {edge_type!r}")
    if edge_type == EdgeType.SUPERSEDES:
        raise ValueError("supersession edges are owned by the canonical memory writer")
    if not edge.evidence_refs or not all(
        isinstance(evidence_ref, EvidenceRef) for evidence_ref in edge.evidence_refs
    ):
        raise ValueError("public memory edges require typed evidence refs")
    selected_edge = replace(edge, workspace_id=ws)
    backend = (
        backend_instance
        if backend_instance is not None
        else create_durable_backend(root, workspace_id=ws)
    )
    getter = getattr(backend, "get_block", None)
    if not callable(getter):
        raise RuntimeError("selected memory backend does not support edge endpoint lookup")
    endpoints = [getter(selected_edge.src_id), getter(selected_edge.dst_id)]
    if not all(isinstance(endpoint, MemoryBlockRecord) for endpoint in endpoints):
        raise ValueError("memory edge endpoints must exist in the selected workspace")
    if any(endpoint.workspace_id != ws for endpoint in endpoints):
        raise ValueError("memory edge endpoints must match the selected writer workspace")
    result = dict(backend.add_edge(selected_edge) or {})
    saved = bool(result.get("ok"))
    record_memory_event(
        root,
        event_type="memory_edge_transition",
        workspace_id=ws,
        run_id=selected_edge.run_id,
        backend=str(result.get("backend") or ""),
        status="ok" if saved else "failed",
        error=str(result.get("error") or ""),
        operation="edge_write",
        decision="canonical_mutation",
        outcome=edge_type,
        data={
            "edge_type": edge_type,
            "reason_sha256": hashlib.sha256(selected_edge.reason.encode("utf-8")).hexdigest()[:16]
            if selected_edge.reason
            else "",
        },
        selected_ids=[selected_edge.src_id, selected_edge.dst_id],
    )
    return result


def _fail_canonical_head(
    backend: Any,
    reservation: CanonicalHeadReservation | None,
    *,
    error: str,
) -> None:
    if reservation is None:
        return
    fail_canonical_head = getattr(backend, "fail_canonical_head", None)
    if not callable(fail_canonical_head):
        raise RuntimeError("selected backend cannot fail a canonical head reservation")
    fail_canonical_head(reservation, error=error or "canonical_mutation_rejected")


def _normal_write_is_pending(backend: Any, block_id: str) -> bool:
    write_outbox_status = getattr(backend, "write_outbox_status", None)
    return callable(write_outbox_status) and write_outbox_status(block_id) == "pending"


def _record_block_membership_issues(
    repo_root: Path,
    record: MemoryBlockRecord,
    *,
    workspace_id: str,
    catalog_blocks: list[Any] | None = None,
    catalog_revision_id: str = "",
    require_current_revision: bool = True,
    retained_record: MemoryBlockRecord | None = None,
) -> tuple[str, ...]:
    membership = record.block_membership
    if membership is None:
        return ("canonical record requires typed block_membership",)
    if (
        membership.record_id != record.block_id
        or membership.record_revision_id != record.revision_id
        or membership.workspace_id != workspace_id
    ):
        return ("block_membership identity does not match canonical record",)
    blocks = catalog_blocks
    current_revision = catalog_revision_id
    if blocks is None or not current_revision:
        try:
            blocks = load_memory_blocks(
                repo_root,
                workspace_id=workspace_id,
                profile_id=record.profile_id,
                strict=True,
            )
            current_revision = compile_semantic_catalog(blocks).revision_id
        except (OSError, TypeError, ValueError) as exc:
            return (f"memory block catalog is invalid: {exc}",)
    if require_current_revision and membership.catalog_revision_id != current_revision:
        return ("block_membership catalog revision is stale",)
    assert blocks is not None
    by_id = {block.block_id: block for block in blocks}
    retained_membership = retained_record.block_membership if retained_record is not None else None
    if (
        record.status == "retracted"
        and retained_record is not None
        and retained_record.workspace_id == record.workspace_id
        and retained_record.profile_id == record.profile_id
        and retained_record.block_id == record.revision_of
        and retained_membership is not None
        and (membership.primary_block_id, membership.secondary_block_ids)
        == (retained_membership.primary_block_id, retained_membership.secondary_block_ids)
        and any(
            identity not in by_id
            for identity in (membership.primary_block_id, *membership.secondary_block_ids)
        )
    ):
        # Only the exact predecessor's retained leaves can accept a hidden tombstone.
        # The active catalog and ordinary knowledge-write eligibility are unchanged.
        by_id.update(
            {
                state.block_id: state.block
                for state in load_workspace_topology(
                    repo_root, workspace_id, profile_id=record.profile_id, include_archived=True
                )
                if state.status == "archived"
            }
        )
    issues: list[str] = []
    for block_id in (membership.primary_block_id, *membership.secondary_block_ids):
        block = by_id.get(block_id)
        if block is None:
            issues.append(f"block_membership references unknown block: {block_id}")
        elif block.index_only:
            issues.append(f"block_membership references index-only block: {block_id}")
    return tuple(issues)


def _rebase_stale_membership_mutation(
    mutation: MutationDecision,
    *,
    catalog_revision_id: str,
    backend: Any,
) -> MutationDecision:
    """Replan an unwritten mutation against a newer, compatible topology snapshot."""

    membership = mutation.record.block_membership
    if (
        membership is None
        or membership.catalog_revision_id == catalog_revision_id
        or mutation.action == "retry"
        or mutation.consolidation_receipt is not None
    ):
        return mutation
    existing: MemoryBlockRecord | None = None
    first_correction = (
        mutation.action == "correction"
        and not mutation.superseded_block_id
        and not mutation.record.revision_of
    )
    if mutation.action != "insert" and not first_correction:
        predecessor_id = mutation.superseded_block_id or mutation.record.revision_of
        get_block = getattr(backend, "get_block", None)
        if not predecessor_id or not callable(get_block):
            raise RuntimeError("selected backend cannot rebase a stale canonical mutation")
        candidate = get_block(predecessor_id)
        if not isinstance(candidate, MemoryBlockRecord):
            raise ValueError("stale canonical mutation predecessor is unavailable")
        existing = candidate
    rebased = replace(
        mutation.record,
        block_membership=RecordBlockMembership(
            primary_block_id=membership.primary_block_id,
            secondary_block_ids=membership.secondary_block_ids,
            catalog_revision_id=catalog_revision_id,
        ),
    )
    refreshed = plan_memory_mutation(
        rebased,
        existing=existing,
        action=mutation.action,
        rollback_target_id=mutation.rollback_target_id,
        affected_view_ids=mutation.affected_view_ids,
        consolidation_receipt=mutation.consolidation_receipt,
        publication_receipt=mutation.publication_receipt,
    )
    if mutation.amendment_receipt is not None:
        receipt = replace(
            mutation.amendment_receipt,
            record_id=refreshed.record.block_id,
            revision_id=refreshed.record.revision_id,
        )
        refreshed = replace(refreshed, amendment_receipt=receipt)
    return refreshed


def _write_memory_event(
    repo_root: str | Path,
    event: MemoryEvent,
    *,
    workspace_id: str | None = None,
    backend_instance: Any | None = None,
    mutation: MutationDecision | None = None,
) -> MemoryWriteResult:
    """Validate policy and write through the selected backend only."""
    root = Path(repo_root)
    ws = normalize_workspace_id(workspace_id or event.workspace_id)
    protected = is_protected_memory(
        event.sensitivity,
        event.content_mode,
        event.operational_risk,
    )
    issues_list = (
        ["protected or non-injectable memory cannot enter selected backend storage"]
        if protected
        else []
    )
    event_ws = normalize_workspace_id(event.workspace_id)
    if event_ws and event_ws != ws:
        issues_list.append("memory event workspace_id does not match selected writer workspace")
    preset_id = _preset_block_id(event)
    block: MemoryBlockRecord | None = None
    if not issues_list:
        try:
            block = _block_from_event(event, preset_id=preset_id)
        except ValueError as exc:
            issues_list.append(str(exc))
    issues = tuple(dict.fromkeys(issue for issue in issues_list if issue))
    if issues:
        record_memory_event(
            root,
            event_type="memory_write_rejected",
            workspace_id=ws,
            run_id=event.run_id,
            stage_id=event.stage_id,
            status="blocked",
            data={
                "block_id": "" if protected else preset_id,
                "memory_kind": event.memory_kind,
                "issues": list(issues),
            },
        )
        return MemoryWriteResult(
            saved=False,
            canonical_id=preset_id,
            backend="",
            policy_issues=issues,
            raw={"ok": False, "block_id": preset_id, "policy_issues": list(issues)},
        )

    backend = (
        backend_instance
        if backend_instance is not None
        else create_durable_backend(root, workspace_id=ws)
    )
    assert block is not None
    if mutation is None:
        predecessor_ids = tuple(
            str(predecessor_id).strip()
            for predecessor_id in event.supersedes_ids
            if str(predecessor_id).strip() and str(predecessor_id).strip() != block.block_id
        )
        if len(predecessor_ids) > 1:
            raise ValueError("canonical mutation accepts exactly one predecessor")
        existing: MemoryBlockRecord | None = None
        if predecessor_ids:
            get_block = getattr(backend, "get_block", None)
            if not callable(get_block):
                raise RuntimeError("selected backend cannot resolve a canonical predecessor")
            candidate = get_block(predecessor_ids[0])
            if not isinstance(candidate, MemoryBlockRecord):
                raise ValueError("canonical mutation predecessor is unavailable")
            existing = candidate
        mutation = plan_memory_mutation(
            block,
            existing=existing,
            action="supersede" if existing is not None else "",
        )
        if existing is None and preset_id:
            mutation = replace(
                mutation,
                record=replace(mutation.record, block_id=preset_id),
            )
        block = mutation.record
    elif (
        mutation.record.block_id != block.block_id
        or mutation.record.canonical_key != block.canonical_key
    ):
        raise ValueError("canonical mutation record does not match memory event")
    block = mutation.record
    try:
        catalog_blocks = load_memory_blocks(
            root,
            workspace_id=ws,
            profile_id=block.profile_id,
            strict=True,
        )
        catalog_revision_id = compile_semantic_catalog(catalog_blocks).revision_id
    except (OSError, TypeError, ValueError) as exc:
        membership_issues: tuple[str, ...] = (f"memory block catalog is invalid: {exc}",)
    else:
        mutation = _rebase_stale_membership_mutation(
            mutation,
            catalog_revision_id=catalog_revision_id,
            backend=backend,
        )
        block = mutation.record
        retained_record = None
        if mutation.action in {"retract", "retry"} and block.status == "retracted":
            predecessor = backend.get_block(block.revision_of)
            if isinstance(predecessor, MemoryBlockRecord):
                retained_record = predecessor
        membership_issues = _record_block_membership_issues(
            root,
            block,
            workspace_id=ws,
            catalog_blocks=catalog_blocks,
            catalog_revision_id=catalog_revision_id,
            require_current_revision=mutation.action != "retry",
            retained_record=retained_record,
        )
    if membership_issues:
        record_memory_event(
            root,
            event_type="memory_write_rejected",
            workspace_id=ws,
            run_id=event.run_id,
            stage_id=event.stage_id,
            status="blocked",
            data={
                "block_id": block.block_id,
                "memory_kind": event.memory_kind,
                "issues": list(membership_issues),
            },
        )
        return MemoryWriteResult(
            saved=False,
            canonical_id=block.block_id,
            backend="",
            policy_issues=membership_issues,
            raw={
                "ok": False,
                "block_id": block.block_id,
                "policy_issues": list(membership_issues),
            },
        )
    if mutation.action == "retry":
        write_outbox_status = getattr(backend, "write_outbox_status", None)
        status = write_outbox_status(block.block_id) if callable(write_outbox_status) else None
        if status == "pending":
            try:
                enqueue_projection = getattr(
                    backend, "enqueue_native_projection", backend.retain_block
                )
                retry_result = dict(enqueue_projection(block) or {})
            except Exception as exc:
                if (
                    callable(write_outbox_status)
                    and write_outbox_status(block.block_id) == "pending"
                ):
                    return MemoryWriteResult(
                        saved=True,
                        canonical_id=block.block_id,
                        backend=str(getattr(backend, "backend_id", "")),
                        raw={
                            "ok": True,
                            "saved": True,
                            "projection_pending": True,
                            "native_written": False,
                            "projection_error": str(exc),
                            "block_id": block.block_id,
                            "backend": str(getattr(backend, "backend_id", "")),
                        },
                    )
                raise
            retry_result.update(
                {
                    "ok": True,
                    "saved": True,
                    "replayed": True,
                    "block_id": block.block_id,
                }
            )
            post_status = (
                write_outbox_status(block.block_id) if callable(write_outbox_status) else None
            )
            if post_status == "committed":
                retry_result["native_written"] = True
                retry_result["projection_pending"] = False
            elif post_status == "pending":
                retry_result["native_written"] = False
                retry_result["projection_pending"] = True
            else:
                return MemoryWriteResult(
                    saved=False,
                    canonical_id=block.block_id,
                    backend=str(getattr(backend, "backend_id", "")),
                    error="retry_write_journal_not_committed",
                    raw={
                        "ok": False,
                        "block_id": block.block_id,
                        "error": "retry_write_journal_not_committed",
                    },
                )
            return MemoryWriteResult(
                saved=True,
                canonical_id=block.block_id,
                backend=str(getattr(backend, "backend_id", "")),
                raw=retry_result,
            )
        if status == "committed":
            return MemoryWriteResult(
                saved=True,
                canonical_id=block.block_id,
                backend=str(getattr(backend, "backend_id", "")),
                raw={
                    "ok": True,
                    "saved": True,
                    "replayed": True,
                    "native_written": True,
                    "block_id": block.block_id,
                },
            )
        return MemoryWriteResult(
            saved=False,
            canonical_id=block.block_id,
            backend=str(getattr(backend, "backend_id", "")),
            error="retry_write_journal_missing",
            raw={
                "ok": False,
                "block_id": block.block_id,
                "error": "retry_write_journal_missing",
            },
        )
    reservation: CanonicalHeadReservation | None = None
    canonical_committed = False
    atomic_commit = False
    canonical_result: dict[str, Any] = {}
    try:
        reserve_canonical_head = getattr(backend, "reserve_canonical_head", None)
        if not callable(reserve_canonical_head):
            raise CanonicalHeadConflict("selected_backend_has_no_canonical_head_cas")
        atomic_gate = getattr(backend, "atomic_canonical_write_gate", None)
        atomic_commit = callable(atomic_gate)
        if callable(atomic_gate):
            canonical_gate = atomic_gate
        else:
            canonical_gate = getattr(backend, "canonical_write_gate", nullcontext)
        with canonical_gate():
            try:
                membership = block.block_membership
                if membership is not None and mutation.action in {
                    "insert",
                    "supersede",
                    "correction",
                    "promote",
                }:
                    profile_key = hashlib.sha256(block.profile_id.encode("utf-8")).hexdigest()
                    catalog_path = (
                        workspace_memory_root(root, ws) / "catalogs" / f"{profile_key}.sqlite"
                    )
                    with sqlite3.connect(
                        f"{catalog_path.resolve().as_uri()}?mode=ro", uri=True
                    ) as catalog:
                        selected = catalog.execute(
                            "SELECT status,backend_partition_id,"
                            "json_extract(payload_json, '$.index_only') "
                            "FROM topology_blocks WHERE block_id=? AND workspace_id=? AND profile_id=?",
                            (membership.primary_block_id, ws, block.profile_id),
                        ).fetchone()
                    if selected is not None and (
                        selected[0] != "active"
                        or bool(selected[2])
                        or selected[1] != membership.backend_partition_id
                    ):
                        raise CanonicalHeadConflict("selected_topology_leaf_changed_before_commit")
                reservation = reserve_canonical_head(
                    workspace_id=ws,
                    canonical_key=block.canonical_key,
                    predecessor_id=mutation.superseded_block_id,
                    successor_id=block.block_id,
                    successor_status=block.status,
                    successor_authority=block.authority,
                    mutation_action=mutation.action,
                    rollback_target_id=mutation.rollback_target_id,
                    mutation=mutation,
                )
                if not isinstance(reservation, CanonicalHeadReservation):
                    raise CanonicalHeadConflict(
                        "selected_backend_invalid_canonical_head_reservation"
                    )
                prepare_canonical_write = getattr(backend, "prepare_canonical_write", None)
                prepared: dict[str, Any] = {}
                if callable(prepare_canonical_write):
                    prepared = dict(prepare_canonical_write(block) or {})
                    if not prepared.get("ok"):
                        raise CanonicalHeadConflict(
                            str(prepared.get("error") or "canonical_write_prepare_rejected")
                        )
                elif str(getattr(backend, "backend_id", "")) in {
                    "mempalace",
                    "hindsight",
                    "mnemosyne",
                }:
                    raise CanonicalHeadConflict("selected_backend_has_no_canonical_write_prepare")
                complete_canonical_head = getattr(backend, "complete_canonical_head", None)
                if not callable(complete_canonical_head):
                    raise CanonicalHeadConflict("selected_backend_has_no_canonical_head_commit")
                canonical_result = dict(complete_canonical_head(reservation, mutation) or {})
                if not canonical_result.get("ok"):
                    raise CanonicalHeadConflict(
                        str(canonical_result.get("error") or "canonical_mutation_rejected")
                    )
            except CanonicalHeadConflict as exc:
                if not atomic_commit:
                    _fail_canonical_head(backend, reservation, error=str(exc))
                reservation = None
                raise
        canonical_committed = True
        enqueue_projection = getattr(backend, "enqueue_native_projection", backend.retain_block)
        result = enqueue_projection(block)
        prepared_revision = int(prepared.get("commit_revision") or 0)
        complete_write_outbox = getattr(backend, "complete_write_outbox", None)
        write_outbox_status = getattr(backend, "write_outbox_status", None)
        if (
            prepared_revision
            and not result.get("projection_pending")
            and callable(complete_write_outbox)
            and callable(write_outbox_status)
            and write_outbox_status(block.block_id) == "pending"
        ):
            complete_write_outbox(prepared_revision)
    except CanonicalHeadConflict as exc:
        error = str(exc)
        if not canonical_committed and not atomic_commit:
            _fail_canonical_head(backend, reservation, error=error)
        record_memory_event(
            root,
            event_type="memory_write_failed",
            workspace_id=ws,
            run_id=event.run_id,
            stage_id=event.stage_id,
            status="failed",
            error=error,
            data={
                "block_id": preset_id,
                "memory_kind": event.memory_kind,
                "authority": event.authority,
            },
        )
        return MemoryWriteResult(
            saved=False,
            canonical_id=preset_id,
            backend="",
            error=error,
            raw={"ok": False, "block_id": preset_id, "error": error},
        )
    except Exception as exc:
        pending_write = _normal_write_is_pending(backend, block.block_id)
        if canonical_committed and pending_write:
            result = {
                "ok": True,
                "saved": True,
                "projection_pending": True,
                "native_written": False,
                "projection_error": str(exc),
                "block_id": block.block_id,
                "backend": str(getattr(backend, "backend_id", "")),
            }
        elif pending_write:
            record_memory_event(
                root,
                event_type="memory_write_failed",
                workspace_id=ws,
                run_id=event.run_id,
                stage_id=event.stage_id,
                status="failed",
                error=str(exc),
                data={
                    "block_id": preset_id,
                    "memory_kind": event.memory_kind,
                    "authority": event.authority,
                },
            )
            return MemoryWriteResult(
                saved=False,
                canonical_id=preset_id,
                backend="",
                error=str(exc),
                raw={"ok": False, "block_id": preset_id, "error": str(exc)},
            )
        else:
            if not canonical_committed and not atomic_commit:
                _fail_canonical_head(backend, reservation, error=str(exc))
            record_memory_event(
                root,
                event_type="memory_write_failed",
                workspace_id=ws,
                run_id=event.run_id,
                stage_id=event.stage_id,
                status="failed",
                error=str(exc),
                data={
                    "block_id": preset_id,
                    "memory_kind": event.memory_kind,
                    "authority": event.authority,
                },
            )
            return MemoryWriteResult(
                saved=False,
                canonical_id=preset_id,
                backend="",
                error=str(exc),
                raw={"ok": False, "block_id": preset_id, "error": str(exc)},
            )

    result = dict(result or {})
    result["canonical_mutation"] = canonical_result
    block_id = str(result.get("block_id") or preset_id or "")
    health: dict[str, Any] = {}
    if not result.get("backend") and hasattr(backend, "health"):
        try:
            health = dict(backend.health() or {})
        except Exception:
            health = {}
    backend_id = str(result.get("backend") or health.get("backend") or "")
    saved = bool(
        not result.get("error")
        and (result.get("ok") or result.get("saved") or result.get("block_id"))
    )
    if saved:
        if reservation is None:
            raise RuntimeError("saved canonical write has no head reservation")
        if block_id != reservation.successor_id:
            result.update(
                {
                    "ok": False,
                    "saved": False,
                    "error": "selected_backend_canonical_head_block_id_mismatch",
                }
            )
            saved = False
    if not saved and not canonical_committed:
        _fail_canonical_head(
            backend, reservation, error=str(result.get("error") or "write_rejected")
        )
    event_data = {
        "block_id": block_id,
        "backend": backend_id,
        "memory_kind": event.memory_kind,
        "authority": event.authority,
        "operation_id": str(block.metadata.get("operation_id") or ""),
        "commit_revision": int(
            result.get("commit_revision")
            or canonical_result.get("commit_revision")
            or prepared.get("commit_revision")
            or 0
        ),
        "durable_status": "committed" if saved else "failed",
        "projection_status": "pending" if result.get("projection_pending") else "committed",
        "projection_revision": int(
            result.get("projection_revision")
            or canonical_result.get("commit_revision")
            or prepared.get("commit_revision")
            or 0
        )
        if not result.get("projection_pending")
        else 0,
        "retryable": bool(result.get("projection_pending") or result.get("retryable")),
        "source_bytes": int(
            block.metadata.get("source_bytes") or len(block.content.encode("utf-8"))
        ),
        "stored_bytes": int(
            block.metadata.get("stored_bytes") or len(block.content.encode("utf-8"))
        ),
        "exact_source_coverage": float(block.metadata.get("exact_source_coverage") or 0.0),
        "records_written": int(result.get("records_written") or 0),
        "blocks_touched": int(result.get("blocks_touched") or 0),
    }
    record_memory_event(
        root,
        event_type="memory_write_committed" if saved else "memory_write_failed",
        workspace_id=ws,
        run_id=event.run_id,
        stage_id=event.stage_id,
        status="ok" if saved else "failed",
        error=str(result.get("error") or ""),
        data=event_data,
    )
    if saved:
        revision_data: dict[str, Any] = {
            "predecessor_count": int(bool(mutation.superseded_block_id))
        }
        canonical_key = block.canonical_key
        if canonical_key:
            revision_data["canonical_key_sha256"] = hashlib.sha256(
                canonical_key.encode("utf-8")
            ).hexdigest()[:16]
        record_memory_event(
            root,
            event_type="memory_revision_committed",
            workspace_id=ws,
            run_id=event.run_id,
            stage_id=event.stage_id,
            backend=backend_id,
            status="ok",
            operation="revision_commit",
            decision="canonical_mutation",
            outcome=mutation.action,
            data=revision_data,
            selected_ids=[
                block_id,
                *([mutation.superseded_block_id] if mutation.superseded_block_id else []),
            ],
        )
        try:
            descriptor_gate = getattr(backend, "descriptor_sync_gate", None)
            gate = (
                descriptor_gate(block.block_membership.primary_block_id)
                if callable(descriptor_gate) and block.block_membership is not None
                else nullcontext()
            )
            with gate:
                index_sync = sync_block_descriptor_index(
                    backend,
                    catalog=compile_semantic_catalog(
                        load_memory_blocks(
                            root,
                            workspace_id=ws,
                            profile_id=block.profile_id,
                        )
                    ),
                    record=block,
                    workspace_id=ws,
                    update_reason=(
                        "consolidation"
                        if mutation.consolidation_receipt is not None
                        else mutation.action
                    ),
                )
            result["logical_index_sync"] = index_sync
            if index_sync.get("status") not in {
                "created",
                "updated",
                "not_applicable",
                "unsupported",
            }:
                raise RuntimeError("logical block descriptor index update failed")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
            result["logical_index_sync"] = {
                "ok": False,
                "status": "degraded",
                "error": str(exc),
            }
            record_memory_event(
                root,
                event_type="memory_index_sync_failed",
                workspace_id=ws,
                run_id=event.run_id,
                stage_id=event.stage_id,
                backend=backend_id,
                status="degraded",
                error=str(exc),
                data={"block_id": block_id},
            )
    if saved:
        result["edge_projection"] = {
            "ok": True,
            "status": "typed_evidence_retained_on_record",
            "projected_edges": 0,
            "non_graph_evidence_refs": len(event.evidence_refs),
        }
    result.pop("store", None)
    operation_id = str(block.metadata.get("operation_id") or "")
    commit_revision = int(
        result.get("commit_revision")
        or canonical_result.get("commit_revision")
        or prepared.get("commit_revision")
        or 0
    )
    projection_pending = bool(result.get("projection_pending"))
    projection_status = "pending" if projection_pending else "committed"
    projection_revision = int(
        result.get("projection_revision") or (commit_revision if not projection_pending else 0)
    )
    source_coverage = float(block.metadata.get("exact_source_coverage") or 0.0)
    result.update(
        {
            "operation_id": operation_id,
            "commit_revision": commit_revision,
            "durable_status": "committed" if saved else "failed",
            "projection_status": projection_status,
            "projection_revision": projection_revision,
            "retryable": bool(projection_pending or result.get("retryable")),
            "source_coverage": source_coverage,
            "write_mode": str(block.metadata.get("write_mode") or "standard"),
            "summary_count": int(block.metadata.get("summary_count") or 0),
            "exact_source_coverage": source_coverage,
            "source_coverage_status": str(block.metadata.get("source_coverage_status") or ""),
            "exact_segment_fidelity": float(block.metadata.get("exact_segment_fidelity") or 0.0),
            "source_segments": list(block.metadata.get("source_segments") or ()),
            "records_written": 1 if saved else 0,
            "blocks_touched": 1 if saved else 0,
        }
    )
    return MemoryWriteResult(
        saved=saved,
        canonical_id=block_id,
        backend=backend_id,
        operation_id=operation_id,
        commit_revision=commit_revision,
        durable_status="committed" if saved else "failed",
        projection_status=projection_status,
        projection_revision=projection_revision,
        retryable=bool(projection_pending or result.get("retryable")),
        source_coverage=source_coverage,
        error=str(result.get("error") or ""),
        raw=result,
    )
