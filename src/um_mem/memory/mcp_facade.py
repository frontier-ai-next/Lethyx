"""Project-scoped UM-MEM facade used by MCP tools and Hermes integrations.

The functions in this module are plain JSON-friendly callables.  The MCP server
wraps them, and tests exercise them directly so the memory authority/isolation
rules stay independent of the transport.
"""

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal, cast

from um_mem import MemoryClient
from um_mem.contracts import (
    AmbiguousEvidenceRefError,
    EvidenceRef,
    HostObservation,
    HostObservationKind,
    MemoryBlockRecord,
    is_protected_memory,
    normalize_turn_child_claims,
)
from um_mem.contracts.memory import (
    CURRENT_SCHEMA_VERSION,
    ContentMode,
    MemoryScope,
    OperationalRisk,
    Sensitivity,
    memory_scope_key,
    validate_memory_safety_axes,
)
from um_mem.integrations import (
    HostCapabilityManifest,
    MemoryServiceCapabilities,
    negotiate_runtime,
)
from um_mem.memory import local_tier
from um_mem.memory.active_mask import (
    ActiveMemoryBlockBudget,
    ActiveMemoryPriority,
    ActiveMemorySelector,
)
from um_mem.memory.backends.base import MemoryBusyError, MemoryHit, NativeProjectionPending
from um_mem.memory.backends.contracts import (
    CanonicalHeadConflict,
    MemoryEdgeRecord,
    block_record_to_memory_hit,
)
from um_mem.memory.backends.query_policy import record_allowed_for_query
from um_mem.memory.catalog_store import (
    BlockEdgeType,
    BlockEvolutionTarget,
    BlockMergeSource,
    MaintenanceObservationType,
)
from um_mem.memory.context_payload import (
    PackedEvidencePayload,
    pack_evidence_payload,
)
from um_mem.memory.kernel.telemetry import record_memory_event
from um_mem.memory.obsidian_export import export_backend_to_obsidian
from um_mem.memory.paths import MANAGER_WORKSPACE_ID, normalize_workspace_id
from um_mem.memory.query_analysis import EvidenceRequirement
from um_mem.memory.recall_plan import RecallPlan, RetrievalMode, SourceHorizon
from um_mem.memory.recall_strategy import (
    allocate_recall_budget,
    provenance_reference_record,
)
from um_mem.memory.scopes import ScopeLineage, bind_scope_chain
from um_mem.memory.selection import MemorySelectionPolicy, SelectionContext

_SUPPRESSED_STATUSES = {
    "quarantined",
    "superseded",
    "deprecated",
    "retracted",
}
_BOUND_LOCAL_SERVICE_CLIENT: ContextVar[MemoryClient | None] = ContextVar(
    "um_mem_bound_local_service_client",
    default=None,
)


def _default_repo_root(repo_root: str | Path | None = None) -> Path:
    raw = str(repo_root or os.getenv("UM_MEM_REPO_ROOT") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path.cwd().resolve()


def _default_workspace(workspace_id: str | None = None) -> str:
    return str(workspace_id or os.getenv("UM_MEM_WORKSPACE") or "default").strip() or "default"


def _default_backend(backend: str | None = None) -> str:
    return str(backend or os.getenv("UM_MEM_BACKEND") or "mempalace").strip() or "mempalace"


def _default_user_id(user_id: str | None = None) -> str:
    requested = str(user_id or "").strip()
    configured = str(os.getenv("UM_MEM_USER_ID") or "").strip()
    if requested and configured and requested != configured:
        raise ValueError("user_id does not match the configured MCP principal")
    return configured or requested


def _default_profile(profile_id: str | None = None) -> str:
    return str(profile_id or os.getenv("UM_MEM_PROFILE") or "default").strip() or "default"


def _client(
    *,
    repo_root: str | Path | None,
    workspace_id: str | None,
    profile_id: str | None = None,
    backend: str | None,
    user_id: str = "",
) -> MemoryClient:
    selected_root = _default_repo_root(repo_root)
    selected_workspace = normalize_workspace_id(_default_workspace(workspace_id))
    selected_profile = _default_profile(profile_id)
    selected_backend = _default_backend(backend)
    selected_user_id = _default_user_id(user_id)
    bound_client = _BOUND_LOCAL_SERVICE_CLIENT.get()
    if (
        bound_client is not None
        and bound_client.repo_root.resolve() == selected_root
        and bound_client.workspace_id == selected_workspace
        and bound_client.profile_id == selected_profile
        and (not selected_user_id or bound_client.user_id == selected_user_id)
        and bound_client.backend_mode == selected_backend
    ):
        return bound_client
    return MemoryClient(
        repo_root=selected_root,
        workspace_id=selected_workspace,
        profile_id=selected_profile,
        user_id=selected_user_id,
        backend=selected_backend,
    )


def _mcp_runtime(*, backend: str, healthy: bool) -> dict[str, Any]:
    """Report an explicit tools-only transport, never a host execution claim."""

    return negotiate_runtime(
        HostCapabilityManifest(
            host_id="mcp",
            adapter_id="um_mem.mcp_facade",
            transport="mcp",
            conformance_passed=True,
        ),
        MemoryServiceCapabilities(
            conformance_passed=healthy,
            healthy=healthy,
            selected_backend=backend,
        ),
        requested_profile="mcp_tools_only",
    ).to_dict()


@contextmanager
def persistent_mcp_client(
    *,
    repo_root: str | Path,
    workspace_id: str,
    profile_id: str = "default",
    backend: str,
    user_id: str = "",
) -> Iterator[MemoryClient]:
    client = _client(
        repo_root=repo_root,
        workspace_id=workspace_id,
        profile_id=profile_id,
        backend=backend,
        user_id=user_id,
    )
    token = _BOUND_LOCAL_SERVICE_CLIENT.set(client)
    try:
        yield client
    finally:
        _BOUND_LOCAL_SERVICE_CLIENT.reset(token)
        _close_client(client)


@contextmanager
def bound_mcp_client(client: MemoryClient) -> Iterator[MemoryClient]:
    """Bind an already authenticated service client for one dispatch."""

    if not isinstance(client, MemoryClient):
        raise TypeError("client must be MemoryClient")
    token = _BOUND_LOCAL_SERVICE_CLIENT.set(client)
    try:
        yield client
    finally:
        _BOUND_LOCAL_SERVICE_CLIENT.reset(token)


def _close_client(client: MemoryClient) -> None:
    manager = client._manager_client_cache
    if manager is not None:
        manager_backend = manager._backend
        close_manager = getattr(manager_backend, "close", None)
        if callable(close_manager):
            close_manager()
        client._manager_client_cache = None
    if client is _BOUND_LOCAL_SERVICE_CLIENT.get():
        return
    backend = getattr(client, "_backend", None)
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _effective_scope_chain(
    client: MemoryClient,
    requested: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    lineage = getattr(client, "scope_lineage", None)
    if isinstance(lineage, ScopeLineage):
        return bind_scope_chain(lineage, tuple(requested or ()))
    return tuple(requested or (f"workspace:{client.workspace_id}",))


def _as_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _as_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _json_object(value: object) -> dict[str, Any]:
    payload = _as_jsonable(value)
    if not isinstance(payload, dict):
        raise TypeError("Memory API response must serialize to a JSON object")
    return payload


def _evidence_ref(value: Any) -> EvidenceRef:
    if isinstance(value, EvidenceRef):
        return value
    if isinstance(value, dict):
        return EvidenceRef.from_mapping(value)
    return EvidenceRef(
        ref_type="session_event",
        ref_id=str(value),
        produced_by="harness",
        producer_id="mcp:um_mem_facade",
    )


def _evidence_refs(values: Iterable[Any]) -> list[EvidenceRef]:
    return [_evidence_ref(value) for value in values]


def _safe_error(exc: BaseException) -> dict[str, Any]:
    message = str(exc)
    if isinstance(exc, NativeProjectionPending):
        reason_code = "native_projection_pending"
    elif "native_projection_rebuild_required" in message:
        reason_code = "native_projection_rebuild_required"
    elif "native_block_scoping_unsupported" in message:
        reason_code = "native_block_scoping_unsupported"
    elif isinstance(exc, PermissionError):
        reason_code = "access_denied"
    elif isinstance(exc, (BlockingIOError, MemoryBusyError)):
        reason_code = "store_lock_unavailable"
    elif isinstance(exc, CanonicalHeadConflict):
        reason_code = "canonical_revision_conflict"
    elif isinstance(exc, AmbiguousEvidenceRefError):
        reason_code = "ambiguous_evidence_ref"
    elif isinstance(exc, TimeoutError):
        reason_code = "transport_timeout"
    elif isinstance(exc, ConnectionError):
        reason_code = "native_transport_failure"
    else:
        reason_code = "memory_operation_failed"
    return {
        "ok": False,
        "error": f"{type(exc).__name__}: {message}",
        "error_type": type(exc).__name__,
        "reason_code": reason_code,
        "retryable": reason_code
        in {
            "native_transport_failure",
            "transport_timeout",
            "store_lock_unavailable",
            "native_projection_pending",
        },
    }


def _record_authority(record: MemoryBlockRecord) -> str:
    return record.authority


def _record_is_passive(record: MemoryBlockRecord) -> bool:
    return (
        record.retention == "archive"
        or record.delivery_policy in {"reflection_only", "hidden"}
        or record.temperature == "cold"
    )


def _record_is_candidate(record: MemoryBlockRecord) -> bool:
    return record.status == "candidate"


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(item) for item in value]
    else:
        parts = [str(value)]
    return {part.strip() for part in parts if part.strip()}


def _record_allowed_for_context(
    record: MemoryBlockRecord,
    *,
    scope_chain: tuple[str, ...],
    run_id: str = "",
    profile_id: str = "default",
    source_horizon: SourceHorizon,
    include_candidates: bool,
    logical_block_ids: list[str] | tuple[str, ...] | None = None,
    record_ids: list[str] | tuple[str, ...] | None = None,
    temporal_mode: str = "current",
    as_of: str = "",
    known_as_of: str = "",
) -> bool:
    return record_allowed_for_query(
        record,
        {
            "query": "",
            "scope_chain": list(scope_chain),
            "run_id": run_id,
            "profile_id": profile_id,
            "source_horizon": source_horizon,
            "include_candidates": include_candidates,
            "temporal_mode": temporal_mode,
            "as_of": as_of,
            "known_as_of": known_as_of,
            "logical_block_ids": list(logical_block_ids or ()),
            "record_ids": list(record_ids or ()),
        },
    )


def _protected_metadata(metadata: dict[str, Any]) -> bool:
    return is_protected_memory(
        metadata.get("sensitivity"),
        metadata.get("content_mode"),
        metadata.get("operational_risk"),
    )


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        sensitivity, content_mode, operational_risk = validate_memory_safety_axes(
            metadata.get("sensitivity"),
            metadata.get("content_mode"),
            metadata.get("operational_risk"),
        )
    except ValueError:
        return {"redacted": True}
    if not is_protected_memory(sensitivity, content_mode, operational_risk):
        return metadata
    return {
        "sensitivity": sensitivity,
        "content_mode": content_mode,
        "operational_risk": operational_risk,
        "redacted": True,
    }


def _response_items_with_retrieval_scores(
    items: list[dict[str, Any]],
    records: Iterable[MemoryBlockRecord],
) -> list[dict[str, Any]]:
    scores: dict[str, Any] = {}
    score_modes: dict[str, str] = {}
    for record in records:
        record_metadata = record.metadata or {}
        scores[record.block_id] = record_metadata.get(
            "retrieval_score", record_metadata.get("native_score")
        )
        score_modes[record.block_id] = str(record_metadata.get("native_score_mode") or "")
    for item in items:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            item["metadata"] = metadata
        block_id = str(metadata.get("block_id") or item.get("source") or "")
        score = scores.get(block_id, item.get("score"))
        if score is not None:
            metadata["retrieval_score"] = score
        if not item.get("score_mode") and score is not None:
            item["score_mode"] = score_modes.get(block_id) or "native_undeclared"
    return items


def _telemetry_native_query_trace(value: Any) -> list[dict[str, Any]]:
    def identifier_hash(raw: object) -> str:
        return "sha256:" + hashlib.sha256(str(raw).encode("utf-8")).hexdigest()

    compact: list[dict[str, Any]] = []
    for raw_item in value or ():
        if not isinstance(raw_item, dict):
            continue
        record_id = str(raw_item.get("record_id") or "")
        candidate_ids = [
            str(item) for item in raw_item.get("candidate_pool_ids") or () if str(item)
        ]
        selected_ids = [
            str(item) for item in raw_item.get("selected_record_ids") or () if str(item)
        ]
        compact.append(
            {
                "native_logical_block_id": str(raw_item.get("native_logical_block_id") or ""),
                "native_rank": raw_item.get("native_rank"),
                "native_score": raw_item.get("native_score"),
                "record_id_hash": identifier_hash(record_id) if record_id else "",
                "candidate_id_hashes": [identifier_hash(item) for item in candidate_ids],
                "selected_id_hashes": [identifier_hash(item) for item in selected_ids],
                "selection_mode": str(raw_item.get("selection_mode") or ""),
            }
        )
    return compact


def _agent_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated internal candidate IDs from the harness-facing payload."""

    diagnostics = dict(value)
    raw_trace = diagnostics.get("native_query_trace")
    if not isinstance(raw_trace, list):
        return _json_object(diagnostics)
    candidate_ids: set[str] = set()
    selected_ids: set[str] = set()
    entry_count = 0
    for item in raw_trace:
        if not isinstance(item, dict):
            continue
        entry_count += 1
        candidate_ids.update(str(raw) for raw in item.get("candidate_pool_ids") or () if str(raw))
        selected_ids.update(str(raw) for raw in item.get("selected_record_ids") or () if str(raw))
    diagnostics["native_query_trace"] = {
        "entry_count": entry_count,
        "candidate_record_count": len(candidate_ids),
        "selected_record_count": len(selected_ids),
    }
    return _json_object(diagnostics)


def _record_explicit_paths(record: MemoryBlockRecord) -> list[str]:
    metadata = dict(record.metadata or {})
    paths: list[str] = []
    for key in ("source_path", "path", "file_path", "source"):
        value = str(metadata.get(key) or "").strip()
        if value and value not in paths:
            paths.append(value.replace("\\", "/"))
    return paths


def _metadata_text_tuple(metadata: dict[str, Any], field_name: str) -> tuple[str, ...]:
    value = metadata.get(field_name) or ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _has_source_speaker(record: MemoryBlockRecord, selected: set[str]) -> bool:
    segments = record.metadata.get("source_segments")
    if not isinstance(segments, (list, tuple)):
        return False
    return any(
        isinstance(segment, dict)
        and str(segment.get("speaker") or "").strip().casefold() in selected
        for segment in segments
    )


def _compact_source_card_text(record: MemoryBlockRecord, text: str) -> str:
    path = next(iter(_record_explicit_paths(record)), "")
    if not path:
        return text
    metadata = dict(record.metadata or {})
    classes = _metadata_text_tuple(metadata, "classes")
    functions = _metadata_text_tuple(metadata, "functions")
    imports = _metadata_text_tuple(metadata, "imports")
    if not any((classes, functions, imports)):
        return text
    header = [f"Path: {path}"]
    if classes:
        header.append("Classes: " + ", ".join(classes))
    if functions:
        header.append("Functions: " + ", ".join(functions))
    if imports:
        header.append("Imports: " + ", ".join(imports))
    header.append("Excerpt:")
    return "\n".join([*header, str(text or "")])


def _hit_from_record(
    record: MemoryBlockRecord, *, force_non_directive: bool = False
) -> dict[str, Any] | None:
    hit = block_record_to_memory_hit(record)
    return _hit_to_payload(hit, record=record, force_non_directive=force_non_directive)


def _hit_to_payload(
    hit: MemoryHit, *, record: MemoryBlockRecord | None = None, force_non_directive: bool = False
) -> dict[str, Any] | None:
    metadata = dict(hit.metadata or {})
    if record is not None:
        metadata["block_id"] = record.block_id
        metadata["workspace_id"] = record.workspace_id
        metadata["status"] = record.status
        metadata["retention"] = record.retention
        metadata["delivery_policy"] = record.delivery_policy
        metadata["temperature"] = record.temperature
        metadata["authority"] = record.authority
        metadata["trust_level"] = record.trust_level
        metadata["scope"] = record.scope
        metadata["run_id"] = record.run_id
        metadata["subtask_id"] = record.subtask_id
        metadata["source_backend"] = record.source_backend
        metadata["sensitivity"] = record.sensitivity
        metadata["content_mode"] = record.content_mode
        metadata["operational_risk"] = record.operational_risk
        metadata["verification_status"] = record.verification_status
        metadata["verification_methods"] = record.verification_methods
        metadata["promotion_decision"] = record.promotion_decision
        metadata["evidence_ref_count"] = len(record.evidence_refs)
    if force_non_directive:
        metadata["authority"] = "advisory"
    if _protected_metadata(metadata):
        return None
    metadata = _safe_metadata(metadata)
    authority = (
        "advisory"
        if force_non_directive
        else (_record_authority(record) if record is not None else "advisory")
    )
    text = str(hit.text or "")
    source_path = ""
    if record is not None:
        text = _compact_source_card_text(record, text)
        source_path = next(iter(_record_explicit_paths(record)), "")
    backend = metadata.get("source_backend") or (
        record.source_backend if record is not None else ""
    )
    return {
        "text": text,
        "title": record.title if record is not None else "",
        "source": hit.source,
        "source_path": source_path,
        "backend": backend,
        "score": hit.score,
        "confidence": hit.score,
        "score_mode": str(
            metadata.get("native_score_mode")
            or ("native_undeclared" if hit.score is not None else "")
        ),
        "rank": metadata.get("rank") or "",
        "why_matched": metadata.get("why_matched", ""),
        "concise_text": text[:500],
        "kind": hit.kind,
        "tags": list(hit.tags),
        "metadata": _as_jsonable(metadata),
        "status": metadata.get("status", ""),
        "delivery_policy": metadata.get("delivery_policy", ""),
        "authority": authority,
    }


def _raw_payloads_from_records(
    records: Iterable[MemoryBlockRecord],
    *,
    query: str = "",
    force_non_directive: bool = False,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in records:
        if is_protected_memory(record.sensitivity, record.content_mode, record.operational_risk):
            continue
        payload = _hit_from_record(record, force_non_directive=force_non_directive)
        if payload is None:
            continue
        payload["status"] = "supplemental non-directive"
        if not payload.get("why_matched"):
            payload["why_matched"] = str(
                (record.metadata or {}).get("recall_plan_reason")
                or (record.metadata or {}).get("retrieval_backend")
                or "memory_recall"
            )
        payloads.append(payload)
    return payloads


def _packed_payload_from_records(
    records: Iterable[MemoryBlockRecord],
    *,
    max_tokens: int,
    query: str = "",
    force_non_directive: bool = False,
    requirements: tuple[EvidenceRequirement, ...] = (),
) -> PackedEvidencePayload:
    return pack_evidence_payload(
        _raw_payloads_from_records(
            records,
            query=query,
            force_non_directive=force_non_directive,
        ),
        max_tokens=max_tokens,
        query=query,
        requirements=requirements,
    )


def _requirements_from_diagnostics(
    diagnostics: dict[str, Any],
) -> tuple[EvidenceRequirement, ...]:
    raw = diagnostics.get("evidence_requirements") or ()
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(EvidenceRequirement.from_mapping(value) for value in raw)


def _record_packed_decision(
    client: MemoryClient,
    records: list[MemoryBlockRecord],
    payload: PackedEvidencePayload,
    *,
    query: str,
    run_id: str,
    scope_chain: tuple[str, ...],
    diagnostics: dict[str, Any],
    started_ns: int,
    catalog_revision: str,
) -> None:
    if not records or not isinstance(client, MemoryClient):
        return
    plan = RecallPlan.compile(
        query,
        query_type=diagnostics.get("query_type"),
        workspace_id=client.workspace_id,
        run_id=run_id,
        scope_chain=scope_chain,
        requirements=_requirements_from_diagnostics(diagnostics),
        max_tokens=payload.token_budget,
        limit=min(8, len(records)),
        reason=str(diagnostics.get("recall_plan_reason") or "mcp_context_recall"),
    )
    decision_id = client._record_context_decision(
        plan=plan,
        hits=tuple(block_record_to_memory_hit(record) for record in records),
        payload=payload,
        latency_ms=(perf_counter_ns() - started_ns) // 1_000_000,
        catalog_revision=catalog_revision,
    )
    if decision_id:
        diagnostics["decision_id"] = decision_id


def _payload_diagnostics(payload: PackedEvidencePayload) -> dict[str, Any]:
    return {
        "truncated_item_count": payload.truncated_item_count,
        "context_char_budget": payload.token_budget * 4,
        "context_token_budget": payload.token_budget,
        "context_payload_chars": len(payload.rendered_context),
        "context_payload_estimated_tokens": payload.estimated_tokens,
        "metadata_overhead_tokens": payload.metadata_tokens,
        "exact_literal_capsule": list(payload.exact_literal_capsule),
        "exact_literal_losses": [asdict(loss) for loss in payload.exact_literal_losses],
        "visibility": [asdict(row) for row in payload.visibility],
    }


def _record_activity_counts(records: Iterable[MemoryBlockRecord]) -> dict[str, int]:
    counts = {
        "active_record_count": 0,
        "passive_record_count": 0,
        "candidate_record_count": 0,
    }
    for record in records:
        if _record_is_candidate(record):
            counts["candidate_record_count"] += 1
        elif _record_is_passive(record):
            counts["passive_record_count"] += 1
        else:
            counts["active_record_count"] += 1
    return counts


def _record_retrieval_provenance(record: MemoryBlockRecord) -> str:
    """Return the retrieval path that produced a recalled record.

    Native Hindsight and Mnemosyne adapters merge their own recall with the
    selected backend's SQLite sidecar.  The sidecar query stamps
    ``retrieval_backend`` at query time, whereas ``native_backend`` can be
    persisted on either kind of record.  Prefer the query-time marker so the
    diagnostics describe the source of the delivered hit rather than where the
    record was originally written.
    """

    metadata = dict(record.metadata or {})
    if str(metadata.get("retrieval_backend") or "").strip():
        return "sqlite_sidecar"
    sidecar_only = metadata.get("sidecar_only")
    if sidecar_only is True or str(sidecar_only or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return "sqlite_sidecar"
    if str(metadata.get("retrieval_provenance") or "").strip() == "selected_lookup":
        return "sqlite_sidecar"
    return "backend_native"


def _recall_provenance_counts(records: Iterable[MemoryBlockRecord]) -> dict[str, int]:
    native_count = 0
    sidecar_count = 0
    for record in records:
        if _record_retrieval_provenance(record) == "sqlite_sidecar":
            sidecar_count += 1
        else:
            native_count += 1
    return {
        "backend_recall_count": native_count + sidecar_count,
        "native_count": native_count,
        "sidecar_count": sidecar_count,
    }


def _set_recall_provenance_diagnostics(
    diagnostics: dict[str, Any],
    records: Iterable[MemoryBlockRecord],
    *,
    stage: str = "",
) -> None:
    suffix = f"_{stage}" if stage else ""
    for key, value in _recall_provenance_counts(records).items():
        diagnostics[f"{key}{suffix}"] = value


def _dedup_records(records: Iterable[MemoryBlockRecord]) -> list[MemoryBlockRecord]:
    by_key: dict[tuple[str, str, str, str], MemoryBlockRecord] = {}
    order: list[tuple[str, str, str, str]] = []
    for record in records:
        key = (record.workspace_id, record.profile_id, record.block_id, record.revision_id)
        if key in by_key:
            continue
        by_key[key] = record
        order.append(key)
    return [by_key[key] for key in order]


def _safe_records_from_native_hits(
    client: MemoryClient,
    *,
    recall_plan: RecallPlan,
    query: str,
    source_horizon: SourceHorizon,
    include_candidates: bool,
    tags: list[str] | tuple[str, ...] | None,
    max_tokens: int,
    limit: int,
    logical_block_ids: list[str] | tuple[str, ...] | None = None,
    record_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[MemoryBlockRecord], dict[str, Any]]:
    backend = client.backend
    requested_record_ids = _string_set(record_ids)
    get_block = getattr(backend, "get_block", None)
    diagnostics: dict[str, Any] = {
        "backend_recall_count": 0,
        "backend_recall_count_before_filter": 0,
        "backend_recall_count_after_safety": 0,
        "native_count": 0,
        "native_count_before_filter": 0,
        "native_count_after_safety": 0,
        "sidecar_count": 0,
        "sidecar_count_before_filter": 0,
        "sidecar_count_after_safety": 0,
        "native_error": "",
        "native_path": "",
    }
    client_recall = getattr(client, "recall", None)
    backend_recall = getattr(backend, "recall", None)

    def selected_lookup(missing_ids: set[str]) -> list[MemoryBlockRecord]:
        if not missing_ids or not callable(backend_recall):
            return []
        # Exact selection can relax query wording, never canonical eligibility.
        # get_block is an audit read and would resurrect rejected dependencies
        # or superseded immutable revisions here.
        payload = replace(
            recall_plan, query="", normalized_query="", record_ids=tuple(sorted(missing_ids))
        ).to_backend_payload()
        return [
            replace(
                record,
                metadata={**dict(record.metadata or {}), "retrieval_provenance": "selected_lookup"},
            )
            for record in backend_recall(payload)
            if isinstance(record, MemoryBlockRecord) and record.block_id in missing_ids
        ]

    temporal_read = bool(
        recall_plan.temporal_mode == "historical" or recall_plan.as_of or recall_plan.known_as_of
    )
    if (not callable(client_recall) or temporal_read) and callable(backend_recall):
        native_records = [
            record
            for record in backend_recall(recall_plan.to_backend_payload())
            if isinstance(record, MemoryBlockRecord)
        ]
        if temporal_read:
            native_records = [
                replace(
                    record,
                    authority="advisory",
                    metadata={
                        **dict(record.metadata or {}),
                        "historical": True,
                        "as_of": recall_plan.as_of,
                        "known_as_of": recall_plan.known_as_of,
                    },
                )
                for record in native_records
            ]
        native_ids = {record.block_id for record in native_records}
        if not temporal_read:
            native_records.extend(selected_lookup(requested_record_ids - native_ids))
        safety_records = [
            record
            for record in native_records
            if _record_allowed_for_context(
                record,
                scope_chain=recall_plan.scope_chain,
                run_id=recall_plan.run_id,
                profile_id=recall_plan.profile_id,
                source_horizon=source_horizon,
                include_candidates=include_candidates,
                logical_block_ids=logical_block_ids,
                record_ids=record_ids,
                temporal_mode=recall_plan.temporal_mode,
                as_of=recall_plan.as_of,
                known_as_of=recall_plan.known_as_of,
            )
        ]
        _set_recall_provenance_diagnostics(diagnostics, native_records, stage="before_filter")
        _set_recall_provenance_diagnostics(diagnostics, safety_records, stage="after_safety")
        _set_recall_provenance_diagnostics(diagnostics, safety_records)
        diagnostics["native_path"] = "backend.recall"
        return safety_records[:limit], diagnostics
    if not callable(get_block):
        return [], diagnostics
    records: list[MemoryBlockRecord] = []
    hits = client.recall(
        recall_plan.query,
        recall_plan=recall_plan,
    )
    for hit in hits:
        hit_metadata = dict(hit.metadata or {})
        block_id = str(hit_metadata.get("block_id") or "")
        if not block_id:
            continue
        record = get_block(block_id)
        if not isinstance(record, MemoryBlockRecord):
            record = provenance_reference_record(hit)
        if not isinstance(record, MemoryBlockRecord):
            continue
        metadata = dict(record.metadata or {})
        if hit.score is not None:
            metadata.setdefault("retrieval_score", hit.score)
            metadata.setdefault("native_score", hit.score)
        for key in (
            "native_score",
            "native_score_mode",
            "retrieval_mode",
            "channel_ranks",
            "rrf_constant",
            "retrieval_score",
            "native_backend",
            "retrieval_backend",
            "sidecar_only",
            "canonical_source_ref",
            "corrective_recall",
            "coverage_complete",
            "coverage_matches",
            "coverage_missing_requirement_ids",
            "coverage_missing_roles",
            "coverage_new_requirement_ids",
            "coverage_new_roles",
            "coverage_original_rank",
            "candidate_pool_ids",
            "native_logical_block_id",
            "native_rank",
            "decision_id",
            "evidence_roles",
            "effective_evidence_requirements",
            "recall_budget",
            "reconstruction_error",
            "reconstruction_performed",
            "reconstruction_status",
            "reverse_provenance_expansion_count",
            "reverse_provenance_depth",
            "reverse_provenance_derived_id",
            "reverse_provenance_reference_only",
            "reverse_provenance_root_id",
            "selected_record_ids",
            "selection_mode",
        ):
            if key in hit_metadata:
                metadata[key] = hit_metadata[key]
        records.append(replace(record, metadata=metadata))
    recalled_ids = {record.block_id for record in records}
    records.extend(selected_lookup(requested_record_ids - recalled_ids))
    safety_records = [
        record
        for record in records
        if _record_allowed_for_context(
            record,
            scope_chain=recall_plan.scope_chain,
            run_id=recall_plan.run_id,
            profile_id=recall_plan.profile_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            logical_block_ids=logical_block_ids,
            record_ids=record_ids,
        )
    ]
    out = safety_records
    _set_recall_provenance_diagnostics(diagnostics, records, stage="before_filter")
    _set_recall_provenance_diagnostics(diagnostics, safety_records, stage="after_safety")
    _set_recall_provenance_diagnostics(diagnostics, out)
    diagnostics["native_path"] = diagnostics.get("native_path") or "client.recall"
    return out, diagnostics


def _recall_records(
    client: MemoryClient,
    *,
    query: str,
    limit: int,
    tags: list[str] | tuple[str, ...] | None,
    source_horizon: SourceHorizon,
    scope_chain: tuple[str, ...],
    include_candidates: bool = False,
    max_tokens: int = 2048,
    run_id: str = "",
    logical_block_ids: list[str] | tuple[str, ...] | None = None,
    record_ids: list[str] | tuple[str, ...] | None = None,
    recall_plan: RecallPlan | None = None,
) -> tuple[list[MemoryBlockRecord], dict[str, Any]]:
    if recall_plan is not None:
        plan = client.compile_recall_plan(
            query,
            recall_plan=recall_plan,
        )
        if plan is None:
            raise RuntimeError("MCP recall plan could not be bound to the client lineage")
    else:
        plan = RecallPlan.compile(
            query,
            workspace_id=client.workspace_id,
            run_id=run_id,
            logical_block_ids=tuple(logical_block_ids or ()),
            record_ids=tuple(record_ids or ()),
            tags=tuple(tags or ()),
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            max_tokens=max_tokens,
            limit=limit,
            reason="mcp_context_recall",
        )
    query = plan.query
    limit = plan.limit
    tags = plan.tags
    source_horizon = plan.source_horizon
    scope_chain = plan.scope_chain
    include_candidates = plan.include_candidates
    max_tokens = plan.max_tokens
    run_id = plan.run_id
    logical_block_ids = plan.logical_block_ids
    record_ids = plan.record_ids
    retrieval_started = perf_counter_ns()
    native, diagnostics = _safe_records_from_native_hits(
        client,
        recall_plan=plan,
        query=query,
        source_horizon=source_horizon,
        include_candidates=include_candidates,
        tags=tags,
        max_tokens=max_tokens,
        limit=limit,
        logical_block_ids=logical_block_ids,
        record_ids=record_ids,
    )
    diagnostics["retrieval_ms"] = (perf_counter_ns() - retrieval_started) / 1_000_000
    ranked = _dedup_records(native)
    diagnostics["corrective_recall_attempted"] = False
    diagnostics["corrective_recall_attempts"] = 0
    diagnostics["corrective_recall_maximum_attempts"] = 0
    diagnostics["reverse_provenance_expansion_count"] = sum(
        1 for record in ranked if record.metadata.get("reverse_provenance_depth") == 1
    )
    diagnostics["reverse_provenance_maximum_depth"] = 1
    trace_metadata = dict(ranked[0].metadata or {}) if ranked else {}
    diagnostics["native_query_trace"] = [
        {
            "record_id": record.block_id,
            "native_logical_block_id": str(record.metadata.get("native_logical_block_id") or ""),
            "native_rank": record.metadata.get("native_rank"),
            "native_score": record.metadata.get("native_score"),
            "native_score_mode": record.metadata.get("native_score_mode"),
            "channel_ranks": record.metadata.get("channel_ranks", {}),
            "candidate_pool_ids": list(record.metadata.get("candidate_pool_ids") or ()),
            "selected_record_ids": list(record.metadata.get("selected_record_ids") or ()),
            "selection_mode": str(record.metadata.get("selection_mode") or ""),
        }
        for record in ranked
    ]
    effective_requirements = trace_metadata.get("effective_evidence_requirements") or [
        requirement.to_mapping() for requirement in plan.requirements
    ]
    diagnostics["evidence_roles"] = list(
        dict.fromkeys(
            str(requirement.get("role") or "")
            for requirement in effective_requirements
            if isinstance(requirement, dict) and str(requirement.get("role") or "")
        )
    )
    diagnostics["evidence_requirements"] = list(effective_requirements)
    diagnostics["evidence_coverage_complete"] = bool(
        ranked and ranked[-1].metadata.get("coverage_complete")
    )
    diagnostics["evidence_coverage_missing_roles"] = (
        list(ranked[-1].metadata.get("coverage_missing_roles") or ())
        if ranked
        else list(plan.evidence_roles)
    )
    diagnostics["evidence_coverage_missing_requirement_ids"] = (
        list(ranked[-1].metadata.get("coverage_missing_requirement_ids") or ())
        if ranked
        else [
            str(requirement.get("requirement_id") or "")
            for requirement in effective_requirements
            if isinstance(requirement, dict) and str(requirement.get("requirement_id") or "")
        ]
    )
    diagnostics["evidence_coverage_matches"] = [
        match
        for record in ranked
        for match in record.metadata.get("coverage_matches") or ()
        if isinstance(match, dict)
    ]
    ranked_ids = [record.block_id for record in ranked]
    diagnostics["candidate_record_ids"] = list(
        trace_metadata.get("candidate_pool_ids") or ranked_ids
    )
    diagnostics["selected_record_ids"] = list(
        trace_metadata.get("selected_record_ids") or ranked_ids[:limit]
    )
    diagnostics["decision_id"] = str(trace_metadata.get("decision_id") or "")
    diagnostics["reconstruction_performed"] = bool(trace_metadata.get("reconstruction_performed"))
    diagnostics["reconstruction_status"] = str(
        trace_metadata.get("reconstruction_status")
        or ("performed" if diagnostics["reconstruction_performed"] else "not_needed")
    )
    diagnostics["reconstruction_error"] = str(trace_metadata.get("reconstruction_error") or "")
    return ranked[:limit], diagnostics


def _compile_context_recall_plan(
    client: MemoryClient,
    *,
    query: str,
    limit: int,
    tags: list[str] | tuple[str, ...] | None,
    source_horizon: SourceHorizon,
    scope_chain: tuple[str, ...],
    include_candidates: bool,
    max_tokens: int,
    run_id: str,
    logical_block_ids: list[str] | tuple[str, ...] | None,
    search_all: bool,
    retrieval_mode: RetrievalMode = "semantic",
    record_ids: list[str] | tuple[str, ...] | None,
    temporal_mode: str = "current",
    as_of: str = "",
    known_as_of: str = "",
    revision_id: str = "",
) -> RecallPlan:
    if not logical_block_ids and not search_all:
        raise ValueError("recall requires logical_block_ids or search_all=True")
    compiler = getattr(client, "compile_recall_plan", None)
    if callable(compiler):
        plan = compiler(
            query,
            limit=limit,
            tags=tags,
            max_tokens=max_tokens,
            run_id=run_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            scope_chain=scope_chain,
            logical_block_ids=logical_block_ids,
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids,
            temporal_mode=temporal_mode,
            as_of=as_of,
            known_as_of=known_as_of,
            revision_id=revision_id,
        )
        if not isinstance(plan, RecallPlan):
            raise TypeError("Memory API compiler must return a RecallPlan")
        return plan
    return RecallPlan.compile(
        query=query,
        workspace_id=getattr(client, "workspace_id", ""),
        run_id=run_id,
        logical_block_ids=tuple(logical_block_ids or ()),
        search_all=search_all,
        retrieval_mode=retrieval_mode,
        record_ids=tuple(record_ids or ()),
        temporal_mode=temporal_mode,
        as_of=as_of,
        known_as_of=known_as_of,
        revision_id=revision_id,
        tags=tuple(tags or ()),
        source_horizon=source_horizon,
        include_candidates=include_candidates,
        scope_chain=scope_chain,
        max_tokens=max_tokens,
        limit=limit,
        reason="explicit_search_all" if search_all else "explicit_block_first_recall",
    )


def _plan_diagnostics(
    plan: RecallPlan | None,
    *,
    source_horizon: SourceHorizon,
    include_candidates: bool,
    max_tokens: int,
    run_id: str = "",
) -> dict[str, Any]:
    budget = allocate_recall_budget(plan) if plan is not None else None
    allocated_tokens = budget.allocated_tokens if budget is not None else max(0, int(max_tokens))
    return {
        "recall_plan_reason": plan.reason if plan is not None and plan.reason else "query_only",
        "normalized_query": plan.normalized_query if plan is not None else "",
        "query_type": plan.query_type if plan is not None else "broad_recall",
        "retrieval_mode": plan.retrieval_mode if plan is not None else "semantic",
        "logical_block_ids": list(plan.logical_block_ids) if plan is not None else [],
        "record_ids": list(plan.record_ids) if plan is not None else [],
        "paths": list(plan.paths) if plan is not None else [],
        "symbols": list(plan.symbols) if plan is not None else [],
        "entities": list(plan.entities) if plan is not None else [],
        "evidence_roles": list(plan.evidence_roles) if plan is not None else [],
        "evidence_requirements": (
            [requirement.to_mapping() for requirement in plan.requirements]
            if plan is not None
            else []
        ),
        "semantic_catalog_revision_id": (
            str(plan.metadata.get("semantic_catalog_revision_id") or "") if plan is not None else ""
        ),
        "source_horizon": plan.source_horizon if plan is not None else source_horizon,
        "scope_chain": list(plan.scope_chain) if plan is not None else [],
        "include_candidates": bool(
            plan.include_candidates if plan is not None else include_candidates
        ),
        "run_id": str(plan.run_id if plan is not None else run_id),
        "as_of": plan.as_of if plan is not None else "",
        "known_as_of": plan.known_as_of if plan is not None else "",
        "requested_context_token_budget": max(0, int(max_tokens or 0)),
        "allocated_context_token_budget": allocated_tokens,
        "recall_budget": budget.to_mapping() if budget is not None else {},
        "context_char_budget": allocated_tokens * 4,
    }


def _apply_plan_to_records(
    records: list[MemoryBlockRecord], plan: RecallPlan | None
) -> list[MemoryBlockRecord]:
    if plan is None or not plan.reason:
        return records
    return [
        replace(
            record,
            metadata={**dict(record.metadata or {}), "recall_plan_reason": plan.reason},
        )
        for record in records
    ]


def _recall_records_scoped(
    client: MemoryClient,
    *,
    query: str,
    limit: int,
    tags: list[str] | tuple[str, ...] | None,
    source_horizon: SourceHorizon,
    scope_chain: tuple[str, ...],
    include_candidates: bool = False,
    max_tokens: int = 2048,
    run_id: str = "",
    logical_block_ids: list[str] | tuple[str, ...] | None = None,
    search_all: bool = False,
    retrieval_mode: RetrievalMode = "semantic",
    record_ids: list[str] | tuple[str, ...] | None = None,
    temporal_mode: str = "current",
    as_of: str = "",
    known_as_of: str = "",
    revision_id: str = "",
    include_manager: bool = False,
    manager_scope_allowed: bool = False,
) -> tuple[list[MemoryBlockRecord], dict[str, Any]]:
    explicit_tags = list(tags or [])
    plan = _compile_context_recall_plan(
        client,
        query=query,
        limit=limit,
        tags=tags,
        source_horizon=source_horizon,
        scope_chain=scope_chain,
        include_candidates=include_candidates,
        max_tokens=max_tokens,
        run_id=run_id,
        logical_block_ids=logical_block_ids,
        search_all=search_all,
        retrieval_mode=retrieval_mode,
        record_ids=record_ids,
        temporal_mode=temporal_mode,
        as_of=as_of,
        known_as_of=known_as_of,
        revision_id=revision_id,
    )
    if plan is not None:
        query = plan.query
        limit = plan.limit
        tags = explicit_tags
        source_horizon = plan.source_horizon
        include_candidates = plan.include_candidates
        scope_chain = plan.scope_chain
        max_tokens = plan.max_tokens
        run_id = plan.run_id
        logical_block_ids = list(plan.logical_block_ids)
        record_ids = list(plan.record_ids)

    records, diagnostics = _recall_records(
        client,
        query=query,
        limit=limit,
        tags=tags,
        source_horizon=source_horizon,
        scope_chain=scope_chain,
        include_candidates=include_candidates,
        max_tokens=max_tokens,
        run_id=run_id,
        logical_block_ids=logical_block_ids,
        record_ids=record_ids,
        recall_plan=plan,
    )
    diagnostics.update(
        _plan_diagnostics(
            plan,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            max_tokens=max_tokens,
            run_id=run_id,
        )
    )
    records = _apply_plan_to_records(records[:limit], plan)
    if include_manager and manager_scope_allowed and client.user_id and len(records) < limit:
        manager_backend = None
        for hit in client._manager_hits(plan):
            block_id = str(hit.metadata.get("block_id") or "")
            if not block_id:
                continue
            if manager_backend is None:
                manager_backend = client._manager_client().backend
            record = manager_backend.get_block(block_id, workspace_id=MANAGER_WORKSPACE_ID)
            if (
                not isinstance(record, MemoryBlockRecord)
                or record.workspace_id != MANAGER_WORKSPACE_ID
                or record.user_id != client.user_id
                or not _record_allowed_for_context(
                    record,
                    scope_chain=(f"user_global:{client.user_id}",),
                    run_id=plan.run_id,
                    profile_id=client.profile_id,
                    source_horizon=plan.source_horizon,
                    include_candidates=plan.include_candidates,
                    record_ids=plan.record_ids,
                    temporal_mode=plan.temporal_mode,
                    as_of=plan.as_of,
                    known_as_of=plan.known_as_of,
                )
            ):
                continue
            records.append(
                replace(
                    record,
                    authority="advisory",
                    metadata={
                        **dict(record.metadata or {}),
                        "memory_level": "manager",
                        **({"retrieval_score": hit.score} if hit.score is not None else {}),
                    },
                )
            )
            if len(records) >= limit:
                break
    diagnostics.update(_record_activity_counts(records))
    return records, diagnostics


def um_mem_health(
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        payload = dict(client.backend.health() or {})
        payload.setdefault("ok", False)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["repo_root"] = str(client.repo_root)
        runtime = _mcp_runtime(
            backend=payload["backend"],
            healthy=payload["ok"] is True,
        )
        payload["integration_mode"] = runtime["integration_mode"]
        payload["deep_agent_support"] = runtime["deep_agent_support"]
        payload["degraded_reasons"] = ["mcp_transport_has_no_host_lifecycle_capture"]
        payload["runtime"] = runtime
        return _json_object(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["repo_root"] = str(client.repo_root)
        runtime = _mcp_runtime(backend=payload["backend"], healthy=False)
        payload["integration_mode"] = runtime["integration_mode"]
        payload["deep_agent_support"] = runtime["deep_agent_support"]
        payload["degraded_reasons"] = ["mcp_transport_has_no_host_lifecycle_capture"]
        payload["runtime"] = runtime
        return payload
    finally:
        _close_client(client)


def um_mem_block_map(
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    depth: int = 1,
    include_meta: bool = False,
    child_limit: int = 50,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        roots = client.block_map(depth=depth, include_meta=include_meta, child_limit=child_limit)
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            "roots": _as_jsonable(roots),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["roots"] = []
        return payload
    finally:
        _close_client(client)


def um_mem_expand_block(
    block_id: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 50,
    cursor: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        expanded = _as_jsonable(client.expand_block(block_id, limit=limit, cursor=cursor))
        if isinstance(expanded, dict):
            for value in (expanded.get("root"), *(expanded.get("children") or ())):
                if not isinstance(value, dict):
                    continue
                # Expansion is a topology operation. Backend partition and metric
                # diagnostics can repeat large native payloads for every child and
                # exceed the authenticated local-service envelope before the scoped
                # harness adapter can compact it. Health remains the explicit surface
                # for those diagnostics; keep navigation content-complete here.
                value.pop("native_partitions", None)
                value.pop("block_metrics", None)
                value.pop("metrics", None)
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            **expanded,
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["children"] = []
        return payload
    finally:
        _close_client(client)


def um_mem_navigation_envelope(
    block_id: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 50,
    cursor: str = "",
    related_cursor: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            "ok": True,
            **_as_jsonable(
                client.navigation_envelope(
                    block_id, limit=limit, cursor=cursor, related_cursor=related_cursor
                )
            ),
        }
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_observe_maintenance(
    block_id: str,
    observation_type: MaintenanceObservationType,
    values: dict[str, int | float],
    threshold_revision: str,
    evidence_ids: list[str],
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            "ok": True,
            "observation": _as_jsonable(
                client.observe_maintenance(
                    block_id,
                    observation_type=observation_type,
                    values=values,
                    threshold_revision=threshold_revision,
                    evidence_ids=tuple(evidence_ids),
                )
            ),
            "mutation_performed": False,
        }
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_create_block(
    name: str,
    synopsis: str,
    parent_id: str,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    scope: str = "workspace",
    index_only: bool = False,
    token_budget: int = 1000,
    expires_at: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                (client.create_index_block if index_only else client.create_leaf_block)(
                    name=name,
                    synopsis=synopsis,
                    parent_id=parent_id,
                    reason=reason,
                    tags=tags or [],
                    metadata=metadata or {},
                    scope=scope,
                    token_budget=token_budget,
                    lifecycle_expires_at=expires_at,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_amend_block(
    block_id: str,
    expected_revision: int,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    name: str | None = None,
    synopsis: str | None = None,
    tags: list[str] | None = None,
    metadata_patch: dict[str, Any] | None = None,
    token_budget: int | None = None,
    expires_at: str | None = None,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                client.amend_block(
                    block_id,
                    expected_revision=expected_revision,
                    name=name,
                    synopsis=synopsis,
                    tags=tags,
                    metadata_patch=metadata_patch,
                    token_budget=token_budget,
                    lifecycle_expires_at=expires_at,
                    reason=reason,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_archive_block(
    block_id: str,
    expected_revision: int,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                client.archive_block(
                    block_id,
                    expected_revision=expected_revision,
                    reason=reason,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_move_block(
    block_id: str,
    parent_id: str,
    expected_revision: int,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                client.move_block(
                    block_id,
                    parent_id=parent_id,
                    expected_revision=expected_revision,
                    reason=reason,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_revive_block(
    block_id: str,
    expected_revision: int,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                client.revive_block(
                    block_id,
                    expected_revision=expected_revision,
                    reason=reason,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_convert_block_to_index(
    block_id: str,
    expected_revision: int,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    """Convert an empty leaf into an index-only parent so it can hold children."""

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                client.convert_block_to_index(
                    block_id,
                    expected_revision=expected_revision,
                    reason=reason,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_delete_block(
    block_id: str,
    expected_revision: int,
    reason: str,
    cascade: bool = False,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                client.delete_block(
                    block_id,
                    expected_revision=expected_revision,
                    reason=reason,
                    cascade=cascade,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_split_block(
    block_id: str,
    expected_revision: int,
    targets: list[dict[str, Any]],
    reason: str,
    operation_id: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        typed_targets = tuple(
            BlockEvolutionTarget(
                name=str(target.get("name") or ""),
                synopsis=str(target.get("synopsis") or ""),
                record_ids=tuple(target.get("record_ids") or ()),
                tags=tuple(target.get("tags") or ()),
                metadata=dict(target.get("metadata") or {}),
                token_budget=int(target.get("token_budget", 1000)),
            )
            for target in targets
        )
        return _json_object(
            client.split_block(
                block_id,
                expected_revision=expected_revision,
                targets=typed_targets,
                reason=reason,
                operation_id=operation_id,
            )
        )
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_merge_blocks(
    sources: list[dict[str, Any]],
    target: dict[str, Any],
    parent_id: str,
    reason: str,
    operation_id: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        typed_sources = tuple(
            BlockMergeSource(
                block_id=str(source.get("block_id") or ""),
                expected_revision=int(source.get("expected_revision", 0)),
            )
            for source in sources
        )
        typed_target = BlockEvolutionTarget(
            name=str(target.get("name") or ""),
            synopsis=str(target.get("synopsis") or ""),
            record_ids=tuple(target.get("record_ids") or ()),
            tags=tuple(target.get("tags") or ()),
            metadata=dict(target.get("metadata") or {}),
            token_budget=int(target.get("token_budget", 1000)),
        )
        return _json_object(
            client.merge_blocks(
                typed_sources,
                target=typed_target,
                parent_id=parent_id,
                reason=reason,
                operation_id=operation_id,
            )
        )
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_pending_topology_evolutions(
    workspace_id: str = "", backend: str = "", repo_root: str = ""
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "pending": _as_jsonable(client.pending_topology_evolutions()),
        }
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_resume_topology_evolution(
    operation_id: str, workspace_id: str = "", backend: str = "", repo_root: str = ""
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return _json_object(client.resume_topology_evolution(operation_id))
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_abort_topology_evolution(
    operation_id: str,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return _json_object(client.abort_topology_evolution(operation_id, reason=reason))
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_add_block_edge(
    src_id: str,
    dst_id: str,
    edge_type: BlockEdgeType,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            "ok": True,
            **_as_jsonable(
                client.add_block_edge(
                    src_id,
                    dst_id,
                    edge_type=edge_type,
                    reason=reason,
                )
            ),
        }
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_remove_block_edge(
    edge_id: str,
    expected_revision: int,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return _json_object(
            client.remove_block_edge(
                edge_id,
                expected_revision=expected_revision,
                reason=reason,
            )
        )
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_get_active_mask(
    mask_id: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        mask = client.get_active_mask(mask_id)
        return {"ok": True, "mask": _as_jsonable(mask) if mask is not None else None}
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def _active_selectors(values: list[dict[str, Any]] | None) -> tuple[ActiveMemorySelector, ...]:
    return tuple(ActiveMemorySelector.from_mapping(value) for value in values or ())


def _active_priorities(values: list[dict[str, Any]] | None) -> tuple[ActiveMemoryPriority, ...]:
    return tuple(ActiveMemoryPriority.from_mapping(value) for value in values or ())


def _active_budgets(values: list[dict[str, Any]] | None) -> tuple[ActiveMemoryBlockBudget, ...]:
    return tuple(ActiveMemoryBlockBudget.from_mapping(value) for value in values or ())


def um_mem_set_active_mask(
    selected_block_ids: list[str],
    token_budget: int,
    activation_reason: str,
    context_refs: list[str] | None = None,
    pinned_record_ids: list[str] | None = None,
    selectors: list[dict[str, Any]] | None = None,
    priorities: list[dict[str, Any]] | None = None,
    block_token_budgets: list[dict[str, Any]] | None = None,
    expires_at: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        mask = client.set_active_mask(
            selected_block_ids=tuple(selected_block_ids),
            token_budget=token_budget,
            activation_reason=activation_reason,
            context_refs=tuple(context_refs or ()),
            pinned_record_ids=tuple(pinned_record_ids or ()),
            selectors=_active_selectors(selectors),
            priorities=_active_priorities(priorities),
            block_token_budgets=_active_budgets(block_token_budgets),
            expires_at=expires_at,
        )
        return {"ok": True, "mask": _as_jsonable(mask)}
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_amend_active_mask(
    mask_id: str,
    expected_revision: int,
    activation_reason: str,
    selected_block_ids: list[str] | None = None,
    context_refs: list[str] | None = None,
    pinned_record_ids: list[str] | None = None,
    selectors: list[dict[str, Any]] | None = None,
    priorities: list[dict[str, Any]] | None = None,
    token_budget: int | None = None,
    block_token_budgets: list[dict[str, Any]] | None = None,
    expires_at: str | None = None,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        mask = client.amend_active_mask(
            mask_id,
            expected_revision=expected_revision,
            activation_reason=activation_reason,
            selected_block_ids=(None if selected_block_ids is None else tuple(selected_block_ids)),
            context_refs=None if context_refs is None else tuple(context_refs),
            pinned_record_ids=(None if pinned_record_ids is None else tuple(pinned_record_ids)),
            selectors=None if selectors is None else _active_selectors(selectors),
            priorities=None if priorities is None else _active_priorities(priorities),
            token_budget=token_budget,
            block_token_budgets=(
                None if block_token_budgets is None else _active_budgets(block_token_budgets)
            ),
            expires_at=expires_at,
        )
        return {"ok": True, "mask": _as_jsonable(mask)}
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_transition_active_mask(
    mask_id: str,
    expected_revision: int,
    transition: str,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        operation = {
            "suspend": client.suspend_active_mask,
            "resume": client.resume_active_mask,
            "deactivate": client.deactivate_active_mask,
        }.get(transition)
        if operation is None:
            raise ValueError("active mask transition is unsupported")
        mask = operation(mask_id, expected_revision=expected_revision, reason=reason)
        return {"ok": True, "mask": _as_jsonable(mask)}
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_resolve_active_mask(
    mask_id: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    expected_revision: int | None = None,
    max_tokens: int | None = None,
    scope_chain: list[str] | None = None,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        lease = client.resolve_active_mask(
            mask_id=mask_id,
            expected_revision=expected_revision,
            max_tokens=max_tokens,
            scope_chain=tuple(scope_chain) if scope_chain is not None else None,
        )
        return {"ok": True, "lease": _as_jsonable(lease)}
    except Exception as exc:
        return {**_safe_error(exc), "workspace_id": client.workspace_id}
    finally:
        _close_client(client)


def um_mem_amend(
    memory_id: str,
    expected_revision: str,
    reason: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    content: str | None = None,
    title: str | None = None,
    tags: list[str] | None = None,
    metadata_patch: dict[str, Any] | None = None,
    retention: str | None = None,
    expires_at: str | None = None,
    evidence_refs: list[Any] | None = None,
    requested_block_id: str = "",
    operation_id: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_as_jsonable(
                client.amend(
                    memory_id,
                    expected_revision=expected_revision,
                    content=content,
                    title=title,
                    tags=tags,
                    metadata_patch=metadata_patch,
                    retention=retention,
                    expires_at=expires_at,
                    reason=reason,
                    evidence_refs=_evidence_refs(evidence_refs or []),
                    requested_block_id=requested_block_id,
                    operation_id=operation_id,
                )
            ),
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_remember(
    content: str,
    operation: str = "add",
    memory_id: str = "",
    repair_of: str = "",
    reason: str = "correction",
    title: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    kind: str = "observation",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    write_mode: str = "standard",
    source_segments: list[dict[str, Any]] | None = None,
    evidence_kind: str = "agent_tool_claim",
    evidence_refs: list[Any] | None = None,
    requested_block_id: str = "",
    source_id: str = "",
    operation_id: str = "",
    sensitivity: Sensitivity = "internal",
    content_mode: ContentMode = "full",
    operational_risk: OperationalRisk = "low",
    child_claims: list[dict[str, object]] | None = None,
    run_id: str = "",
    stage_id: str = "",
    subtask_id: str = "",
    session_id: str = "",
    task_id: str = "",
    parent_session_id: str = "",
    artifact_id: str = "",
    artifact_revision: str = "",
    valid_from: str = "",
    valid_to: str = "",
    project_id: str = "",
    repository_id: str = "",
    checkout: str = "",
    worktree: str = "",
    branch: str = "",
    revision: str = "",
    dirty_state: str = "",
    schema_version: int = CURRENT_SCHEMA_VERSION,
    hard_policy_version: int = 1,
    semantic_catalog_version: str = "",
    _recorded_at: str = "",
    expected_revision: str = "",
) -> dict[str, Any]:
    """Add, repair, or correct agent-authored evidence in one selected logical block."""

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        selected_block_id = str(requested_block_id or "").strip()
        if not selected_block_id:
            raise ValueError("remember requires requested_block_id")
        selected_operation = str(operation or "add").strip().lower()
        if selected_operation not in {"add", "correct", "repair"}:
            raise ValueError("remember operation must be add, correct, or repair")
        if selected_operation == "correct":
            target_id = str(memory_id or "").strip()
            if not target_id:
                raise ValueError("correct operation requires memory_id")
            correction = client.correction(
                target_id,
                content,
                title=title,
                reason=reason,
                evidence_refs=_evidence_refs(evidence_refs or []),
                requested_block_id=selected_block_id,
                metadata=metadata,
                **({"expected_revision": expected_revision} if expected_revision else {}),
            )
            saved = bool(correction.get("saved") or correction.get("ok"))
            correction_id = str(correction.get("block_id") or "")
            result_backend = str(correction.get("backend") or _default_backend(backend))
            canonical_identity: dict[str, Any] = {}
            get_block = getattr(getattr(client, "backend", None), "get_block", None)
            if saved and correction_id and callable(get_block):
                canonical_record = get_block(correction_id)
                if isinstance(canonical_record, MemoryBlockRecord):
                    canonical_identity = {
                        "canonical_key": canonical_record.canonical_key,
                        "revision_id": canonical_record.revision_id,
                        "recorded_at": canonical_record.recorded_at,
                    }
            if saved:
                record_memory_event(
                    client.repo_root,
                    event_type="memory_correction",
                    workspace_id=client.workspace_id,
                    run_id=run_id,
                    backend=result_backend,
                    operation="correction",
                    decision="proposal" if correction.get("pending_host_approval") else "write",
                    outcome="committed",
                    reason="mcp_remember_correct",
                    selected_ids=[correction_id] if correction_id else [],
                )
            return {
                **_as_jsonable(correction),
                **canonical_identity,
                "ok": saved,
                "saved": saved,
                "memory_id": correction_id,
                "backend": result_backend,
                "workspace_id": client.workspace_id,
                "operation": "correct",
                "requested_block_id": selected_block_id,
                "route_rejected": False,
            }
        if selected_operation == "repair":
            repair_reference = str(repair_of or "").strip()
            if not repair_reference:
                raise ValueError("repair operation requires repair_of")
            typed_refs = _evidence_refs(evidence_refs or [])
            if not any(ref.ref_id == repair_reference for ref in typed_refs):
                raise ValueError("repair_of must match an evidence ref id")
            if memory_id:
                raise ValueError("repair operation forbids memory_id")
        else:
            typed_refs = _evidence_refs(evidence_refs or [])
        remember_metadata = dict(metadata or {})
        if source_segments is not None:
            if "source_segments" in remember_metadata:
                raise ValueError("source_segments supplied twice")
            remember_metadata["source_segments"] = source_segments
        if selected_operation == "repair":
            remember_metadata["repair_of"] = repair_reference
        if str(kind or "") == "procedure":
            # A procedure is stamped with the environment it was written
            # against, so a later change can be reported as the reason it went
            # stale instead of a bare number of days.
            from um_mem.memory import procedures as procedure_rules
            from um_mem.memory import semantic as semantic_tier

            remember_metadata = procedure_rules.stamp_metadata(
                remember_metadata,
                procedure_rules.current_descriptor(
                    repo_root=client.repo_root,
                    model=semantic_tier.default_model(),
                    workspace_id=client.workspace_id,
                ),
            )
        if write_mode not in {"standard", "lossless_atomic"}:
            raise ValueError("write_mode must be standard or lossless_atomic")
        result = client.remember(
            content,
            title=title,
            kind=kind,
            tags=tags or [],
            metadata=remember_metadata,
            write_mode=cast(Literal["standard", "lossless_atomic"], write_mode),
            evidence_refs=typed_refs,
            evidence_kind=evidence_kind,
            requested_block_id=selected_block_id,
            source_id=source_id,
            operation_id=operation_id,
            sensitivity=sensitivity,
            content_mode=content_mode,
            operational_risk=operational_risk,
            child_claims=normalize_turn_child_claims(child_claims),
            run_id=run_id,
            stage_id=stage_id,
            subtask_id=subtask_id,
            session_id=session_id,
            task_id=task_id,
            parent_session_id=parent_session_id,
            artifact_id=artifact_id,
            artifact_revision=artifact_revision,
            valid_from=valid_from,
            valid_to=valid_to,
            project_id=project_id,
            repository_id=repository_id,
            checkout=checkout,
            worktree=worktree,
            branch=branch,
            revision=revision,
            dirty_state=dirty_state,
            schema_version=schema_version,
            hard_policy_version=hard_policy_version,
            semantic_catalog_version=semantic_catalog_version,
            _recorded_at=_recorded_at,
        )
        raw = dict(result.raw)
        route_decision = raw.get("route_decision")
        if not isinstance(route_decision, dict):
            route_decision = {}
        route_rejected = bool(
            not result.saved
            and (route_decision.get("accepted") is False or bool(raw.get("policy_issues")))
        )
        result_backend = result.backend or _default_backend(backend)
        if result.saved:
            record_memory_event(
                client.repo_root,
                event_type="memory_capture",
                workspace_id=client.workspace_id,
                run_id=run_id,
                backend=result_backend,
                operation="capture",
                decision="write",
                outcome="committed",
                reason="mcp_remember",
                selected_ids=[str(result.memory_id)] if result.memory_id else [],
                data={
                    "operation": selected_operation,
                    **({"repair_of": repair_reference} if selected_operation == "repair" else {}),
                },
            )
        payload: dict[str, Any] = {
            "ok": result.saved,
            "saved": result.saved,
            "memory_id": result.memory_id,
            "backend": result_backend,
            "workspace_id": client.workspace_id,
            "operation": selected_operation,
            "requested_block_id": selected_block_id,
            "route_decision": _as_jsonable(route_decision),
            "route_rejected": route_rejected,
            "status": str(raw.get("status") or "candidate"),
            "verification_status": str(raw.get("verification_status") or "unverified"),
        }
        # Public callers must distinguish the durable ack from native readiness
        # without interpreting the backend's diagnostic raw payload.
        for field_name in (
            "operation_id",
            "commit_revision",
            "durable_status",
            "projection_status",
            "projection_revision",
            "retryable",
            "source_coverage",
            "source_coverage_status",
            "exact_segment_fidelity",
            "summary_count",
            "records_written",
            "blocks_touched",
        ):
            if field_name in raw:
                payload[field_name] = _as_jsonable(raw[field_name])
        if result.saved:
            payload["actual_block_id"] = str(route_decision.get("block_id") or "")
            try:
                _schedule_semantic_index(
                    client.repo_root,
                    client.workspace_id,
                    str(getattr(client, "profile_id", "default") or "default"),
                    client.read_exact_record(record_id=str(result.memory_id)),
                )
            except Exception:  # noqa: S110 - advisory cache, never blocks the write
                pass
        for field_name in ("scope", "scope_id", "project_id", "user_id"):
            persisted_value = raw.get(field_name)
            if isinstance(persisted_value, str):
                payload[field_name] = persisted_value
        if route_rejected:
            payload["reason"] = str(raw.get("reason") or route_decision.get("reason") or "")
            suggested_route = raw.get("suggested_route")
            if isinstance(suggested_route, dict):
                payload["suggested_route"] = _as_jsonable(suggested_route)
        elif raw:
            payload["raw"] = _as_jsonable(raw)
        return payload
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def _procedure_entry(
    record: Mapping[str, Any], descriptor: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return one stored procedure as a checklist with its freshness."""

    from um_mem.memory import procedures as procedure_rules

    return procedure_rules.procedure_entry(record, descriptor)


def um_mem_procedure(
    query: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 3,
) -> dict[str, Any]:
    """Return stored procedures as checklists with their freshness.

    A procedure is a record of kind ``procedure``. It carries the steps, how
    the result is verified, and when it was last confirmed. A procedure that
    outlived its environment is reported as stale, because a stale procedure
    is worse than no procedure at all.
    """

    from um_mem.memory import procedures as procedure_rules
    from um_mem.memory import semantic as semantic_tier
    from um_mem.memory import text as universal_text

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        safe_limit = max(1, int(limit or 3))
        candidates: list[Mapping[str, Any]] = []
        embedder = semantic_tier.get_embedder(
            cache_dir=semantic_tier.default_cache_dir(client.repo_root)
        )
        if embedder is not None:
            index = semantic_tier.SemanticIndex(
                semantic_tier.semantic_index_path(client.repo_root, client.workspace_id)
            )
            cross_script = semantic_tier.query_uses_non_ascii_script(str(query or ""))
            try:
                ranked = semantic_tier.rank_global(
                    str(query or ""),
                    embedder=embedder,
                    index=index,
                    profile=str(getattr(client, "profile_id", "default") or "default"),
                    floor=semantic_tier.effective_floor(
                        embedder.model,
                        cross_script=cross_script,
                        calibrated=semantic_tier.calibrated_floor(
                            client.repo_root, client.workspace_id, embedder.model
                        ),
                    ),
                    limit=safe_limit * 3,
                    loader=_record_loader(client),
                    kinds=["procedure"],
                )
            except Exception:  # pragma: no cover - advisory tier
                ranked = []
            candidates = [record for _identifier, _score, record in ranked]
        if not candidates:
            window = [
                entry
                for entry in _local_window(client)
                if str(entry.get("kind") or "") == "procedure"
            ]
            if window:
                titles = [str(entry.get("title") or "") for entry in window]
                documents = [
                    f"{entry.get('title') or ''}\n{entry.get('content') or ''}" for entry in window
                ]
                ranked_ids = universal_text.rank(
                    str(query or ""), documents, titles=titles, limit=safe_limit * 3
                )
                candidates = [window[index] for index, _score, _terms in ranked_ids]
        descriptor = procedure_rules.current_descriptor(
            repo_root=client.repo_root,
            model=semantic_tier.default_model(),
            workspace_id=client.workspace_id,
        )
        procedures = [_procedure_entry(record, descriptor) for record in candidates][:safe_limit]
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "query": str(query or ""),
            "count": len(procedures),
            "procedures": procedures,
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["procedures"] = []
        return payload
    finally:
        _close_client(client)


def um_mem_memory_map(
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    profile_id: str = "default",
    output_path: str = "",
) -> dict[str, Any]:
    """Render the memory graph and write it to a file.

    The graph is a snapshot, so it is written on demand: this call refreshes
    the file and returns where it is.
    """

    from um_mem.memory.memory_map import (
        collect_memory_map,
        memory_map_summary,
        write_memory_map,
    )

    try:
        target = str(output_path or "").strip()
        if not target:
            target = str(Path(repo_root or ".").resolve() / "memory-map.html")
        written = write_memory_map(
            repo_root or ".",
            target,
            workspace_id=workspace_id,
            profile_id=profile_id,
            backend=backend,
        )
        data = collect_memory_map(
            repo_root or ".",
            workspace_id=workspace_id,
            profile_id=profile_id,
            backend=backend,
        )
        return {
            "ok": True,
            "workspace_id": workspace_id,
            "path": str(written),
            "summary": memory_map_summary(data),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = workspace_id
        return payload


def um_mem_observe(
    event_id: str,
    event_kind: str,
    raw_evidence: str,
    observed_at: str,
    producer_id: str,
    atomic_claim: str = "",
    source_path: str = "",
    source_uri: str = "",
    artifact_hash: str = "",
    tool_name: str = "",
    workspace_root: str = "",
    changed_files: list[str] | None = None,
    tags: list[str] | None = None,
    speaker: str = "",
    source_occurred_at: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    """Retain one host-only typed observation through the canonical writer."""

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        receipt = client.observe(
            HostObservation(
                event_id=event_id,
                event_kind=cast(HostObservationKind, event_kind),
                raw_evidence=raw_evidence,
                observed_at=observed_at,
                producer_id=producer_id,
                atomic_claim=atomic_claim,
                source_path=source_path,
                source_uri=source_uri,
                artifact_hash=artifact_hash,
                tool_name=tool_name,
                workspace_root=workspace_root,
                changed_files=tuple(changed_files or ()),
                tags=tuple(tags or ()),
                speaker=speaker,
                source_occurred_at=source_occurred_at,
            )
        )
        payload = {
            **_as_jsonable(receipt),
            "saved": bool(receipt.get("witness_id")),
            "backend": _default_backend(backend),
            "workspace_id": client.workspace_id,
        }
        if event_kind == "conversation_turn" and payload.get("ok") is True:
            source_bytes = raw_evidence.encode("utf-8")
            payload.update(
                source_id=event_id,
                source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                source_bytes=len(source_bytes),
            )
        if payload.get("ok") is True:
            selected_ids = [
                str(value)
                for value in (receipt.get("witness_id"), receipt.get("memory_id"))
                if str(value or "")
            ]
            record_memory_event(
                client.repo_root,
                event_type="host_observation_capture",
                workspace_id=client.workspace_id,
                run_id=str(getattr(client.scope_lineage, "session_id", "") or ""),
                backend=_default_backend(backend),
                operation="capture",
                decision=str(receipt.get("route_outcome") or ""),
                outcome="committed",
                reason=str(receipt.get("reason") or ""),
                selected_ids=selected_ids,
                data={
                    "requested_block_id": str(receipt.get("requested_block_id") or ""),
                    "actual_block_id": str(receipt.get("actual_block_id") or ""),
                    "witness_id": str(receipt.get("witness_id") or ""),
                    "route_outcome": str(receipt.get("route_outcome") or ""),
                },
            )
        return payload
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_consolidate_observations(
    source_ids: list[str],
    content: str,
    title: str,
    reason: str,
    verification_reports: list[Any],
    semantic_verification_report: Any,
    requested_block_id: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    """Run one evidence-gated consolidation through the canonical supervisor path."""

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        receipt = client.consolidate_observations(
            source_ids,
            content=content,
            title=title,
            reason=reason,
            verification_reports=verification_reports,
            semantic_verification_report=semantic_verification_report,
            requested_block_id=requested_block_id,
        )
        return {
            **_as_jsonable(receipt),
            "backend": _default_backend(backend),
            "workspace_id": client.workspace_id,
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_read_exact_record(
    record_id: str = "",
    evidence_ref: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        record = client.read_exact_record(record_id=record_id, evidence_ref=evidence_ref)
        return {
            "ok": True,
            "record": _as_jsonable(record),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {
            **_safe_error(exc),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    finally:
        _close_client(client)


def um_mem_read_exact_records(
    record_ids: list[str],
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        records = client.read_exact_records(record_ids)
        return {
            "ok": True,
            "records": [_as_jsonable(record) for record in records],
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {
            **_safe_error(exc),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    finally:
        _close_client(client)


def um_mem_list_records(
    scope: MemoryScope,
    scope_id: str,
    logical_block_id: str = "",
    limit: int = 32,
    cursor: str = "",
    include_retracted: bool = False,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        arguments: dict[str, Any] = {
            "scope": scope,
            "scope_id": scope_id,
            "logical_block_id": logical_block_id,
            "limit": limit,
            "cursor": cursor,
        }
        if include_retracted:
            arguments["include_retracted"] = True
        return {
            "ok": True,
            **_json_object(client.list_records(**arguments)),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {
            **_safe_error(exc),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    finally:
        _close_client(client)


def um_mem_recent(
    limit: int = 8,
    since: str = "",
    scope: MemoryScope = "workspace",
    scope_id: str = "",
    logical_block_id: str = "",
    include_candidates: bool = True,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    """Return the newest canonical records, newest first, for continuity."""

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        records = client.recent_records(
            limit=limit,
            since=since,
            scope=scope,
            scope_id=scope_id,
            logical_block_id=logical_block_id,
            include_candidates=include_candidates,
        )
        return {
            "ok": True,
            "records": [_as_jsonable(record) for record in records],
            "count": len(records),
            "scope": scope,
            "scope_id": scope_id or client.workspace_id,
            "logical_block_id": logical_block_id,
            "since": since,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["records"] = []
        return payload
    finally:
        _close_client(client)


def um_mem_operation_status(
    operation_id: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_json_object(client.operation_status(operation_id)),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {
            **_safe_error(exc),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    finally:
        _close_client(client)


def um_mem_projection_barrier(
    record_id: str,
    commit_revision: int = 0,
    timeout_seconds: float = 30.0,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        return {
            **_json_object(
                client.projection_barrier(
                    record_id, commit_revision=commit_revision, timeout_seconds=timeout_seconds
                )
            ),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    except Exception as exc:
        return {
            **_safe_error(exc),
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
        }
    finally:
        _close_client(client)


def um_mem_read_working_set(
    owner_id: str,
    generation_id: str,
    record_ids: list[str],
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    max_tokens: int = 2048,
    max_bytes: int = 32768,
    scope_chain: list[str] | None = None,
) -> dict[str, Any]:
    """Read exact unpublished records without adding them to accepted recall."""

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        result = client.read_working_set(
            owner_id=owner_id,
            generation_id=generation_id,
            record_ids=record_ids,
            max_tokens=max_tokens,
            max_bytes=max_bytes,
            scope_chain=tuple(scope_chain or ()),
        )
        return {"ok": True, **asdict(result)}
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return _safe_error(exc)
    finally:
        _close_client(client)


def _window_entry(record: Any) -> dict[str, Any]:
    """Return one canonical record in the shape the local tier expects."""

    membership = getattr(record, "block_membership", None)
    primary = str(getattr(membership, "primary_block_id", "") or "")
    metadata = getattr(record, "metadata", None)
    return {
        "record_id": str(getattr(record, "block_id", "") or ""),
        "title": str(getattr(record, "title", "") or ""),
        "content": str(getattr(record, "content", "") or ""),
        "kind": str(getattr(record, "kind", "") or ""),
        "tags": list(getattr(record, "tags", ()) or ()),
        "status": str(getattr(record, "status", "") or ""),
        "scope_key": memory_scope_key(record) if isinstance(record, MemoryBlockRecord) else "",
        "delivery_policy": str(getattr(record, "delivery_policy", "") or ""),
        "revision_id": str(getattr(record, "revision_id", "") or ""),
        "recorded_at": str(getattr(record, "recorded_at", "") or ""),
        "authority": str(getattr(record, "authority", "") or ""),
        "sensitivity": str(getattr(record, "sensitivity", "") or ""),
        "content_mode": str(getattr(record, "content_mode", "") or ""),
        "operational_risk": str(getattr(record, "operational_risk", "") or ""),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
        "block_membership": {"primary_block_id": primary},
    }


def _local_record_allowed(
    client: MemoryClient, record: MemoryBlockRecord, arguments: dict[str, Any] | None
) -> bool:
    policy = {
        "include_candidates": True,
        "scope_chain": [f"workspace:{client.workspace_id}"],
        **(arguments or {}),
        "profile_id": client.profile_id,
    }
    return (
        record_allowed_for_query(record, policy)
        and client.validate_dependencies(record.block_id).status == "valid"
    )


def _local_window(client: Any, arguments: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the newest records as plain dictionaries for the local tier.

    The local tier scores a bounded window of the newest records when the
    native backend cannot answer. It runs in this process, so the embedding
    model stays warm for the next call.
    """

    records = client.recent_records(
        limit=local_tier.FALLBACK_SCAN_LIMIT,
        scope="workspace",
        scope_id=client.workspace_id,
        include_candidates=True,
    )
    return [
        _window_entry(record)
        for record in records
        if _local_record_allowed(client, record, arguments)
    ]


def _scoped_window(
    client: Any, block_ids: Sequence[str], arguments: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return the newest records of the selected leaves for the local tier.

    A block-scoped question must stay inside the caller's leaves, so the tier
    reads those leaves instead of the workspace window and never loads the
    workspace-wide semantic index.
    """

    records: list[Any] = []
    for block_id in list(block_ids)[:8]:
        try:
            records.extend(
                client.recent_records(
                    limit=local_tier.FALLBACK_SCAN_LIMIT,
                    scope="workspace",
                    scope_id=client.workspace_id,
                    logical_block_id=block_id,
                    include_candidates=True,
                )
            )
        except Exception:  # noqa: S112 - one unreadable block must not hide the rest
            continue
    return [
        _window_entry(record)
        for record in records
        if _local_record_allowed(client, record, arguments)
    ]


def _record_loader(client: Any, arguments: dict[str, Any] | None = None) -> Any:
    """Return a loader that reads canonical records by identity.

    The semantic index covers the whole workspace, so a cross-language
    question can reach a record that is not in the newest window. The text is
    always read back from the canonical store, never from the index.
    """

    def load(record_ids: Any) -> list[dict[str, Any]]:
        identifiers = list(
            dict.fromkeys(str(value).strip() for value in record_ids if str(value).strip())
        )
        if not identifiers:
            return []
        try:
            records = client.read_exact_records(identifiers[:128])
        except ValueError:
            records = []
            # An index may still contain a deleted identity. Do not let that
            # one stale entry hide other valid canonical matches in the batch.
            for identifier in identifiers[:128]:
                try:
                    records.append(client.read_exact_record(record_id=identifier))
                except ValueError:
                    continue
        current_ids = client._current_record_id_set(records)
        if current_ids is None:
            return []  # Indexed identities need proof of current membership.
        loaded: list[dict[str, Any]] = []
        for record in records:
            if (
                record.block_id not in current_ids
                or record.scope != "workspace"
                or record.delivery_policy == "hidden"
            ):
                continue
            if not _local_record_allowed(client, record, arguments):
                continue
            status = str(getattr(record, "status", "") or "").casefold()
            if status in {"retracted", "superseded", "deprecated", "expired", "quarantined"}:
                continue
            loaded.append(_window_entry(record))
        return loaded

    return load


def _schedule_semantic_index(
    repo_root: Any, workspace_id: str, profile_id: str, record: Any
) -> None:
    """Add a freshly written record to the semantic index in the background.

    The index is what makes a cross-language question reach old records, so
    every write joins it. The work happens in the background and never delays
    the write itself.
    """

    from um_mem.memory import semantic as semantic_tier

    identifier = str(getattr(record, "block_id", "") or "")
    title = str(getattr(record, "title", "") or "")
    content = str(getattr(record, "content", "") or "")
    if not identifier or not content:
        return

    def run() -> None:
        try:
            embedder = semantic_tier.get_embedder(
                cache_dir=semantic_tier.default_cache_dir(repo_root)
            )
            if embedder is None:
                return
            index = semantic_tier.SemanticIndex(
                semantic_tier.semantic_index_path(repo_root, workspace_id)
            )
            semantic_tier.embed_record(
                index,
                embedder,
                profile=profile_id,
                record_id=identifier,
                title=title,
                content=content,
                kind=str(getattr(record, "kind", "") or ""),
            )
        except Exception:  # pragma: no cover - advisory cache
            return

    threading.Thread(
        target=run,
        name="um-mem-semantic-index",
        daemon=True,
    ).start()


def _apply_local_tier(
    result: dict[str, Any],
    *,
    name: str,
    query: str,
    limit: int,
    arguments: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    """Score the local window when the native backend could not answer."""

    block_ids = [
        str(block_id).strip()
        for block_id in (arguments.get("logical_block_ids") or ())
        if str(block_id).strip()
    ]
    scoped = bool(block_ids)
    return local_tier.apply_local_tier(
        result,
        name=name,
        arguments=arguments,
        records_provider=(
            (lambda: _scoped_window(client, block_ids, arguments))
            if scoped
            else (lambda: _local_window(client, arguments))
        ),
        record_loader=None if scoped else _record_loader(client, arguments),
        repo_root=client.repo_root,
        workspace_id=client.workspace_id,
        profile=str(getattr(client, "profile_id", "default") or "default"),
    )


def um_mem_recall(
    query: str,
    workspace_id: str = "",
    user_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 10,
    tags: list[str] | None = None,
    source_horizon: SourceHorizon = "durable",
    scope_chain: list[str] | None = None,
    include_candidates: bool = False,
    max_tokens: int = 2048,
    logical_block_ids: list[str] | None = None,
    search_all: bool = False,
    retrieval_mode: RetrievalMode = "semantic",
    record_ids: list[str] | None = None,
    source_speakers: list[str] | None = None,
    run_id: str = "",
    temporal_mode: str = "current",
    as_of: str = "",
    known_as_of: str = "",
    revision_id: str = "",
    include_manager: bool = False,
) -> dict[str, Any]:
    started_ns = perf_counter_ns()
    client = _client(
        repo_root=repo_root,
        workspace_id=workspace_id,
        backend=backend,
        user_id=user_id,
    )
    try:
        if source_speakers is not None and (
            not isinstance(source_speakers, list)
            or not 1 <= len(source_speakers) <= 8
            or any(not isinstance(value, str) or not value.strip() for value in source_speakers)
        ):
            raise ValueError("source_speakers must be 1-8 nonempty names")
        catalog_revision = client._catalog_revision_id() if isinstance(client, MemoryClient) else ""
        safe_limit = max(0, int(limit or 0))
        native_limit = max(safe_limit, 64) if source_speakers else safe_limit
        effective_scope_chain = _effective_scope_chain(client, scope_chain)
        records, diagnostics = _recall_records_scoped(
            client,
            query=query,
            limit=native_limit,
            tags=tags or [],
            source_horizon=source_horizon,
            scope_chain=effective_scope_chain,
            include_candidates=include_candidates,
            max_tokens=max_tokens,
            run_id=run_id,
            logical_block_ids=logical_block_ids or [],
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids or [],
            temporal_mode=temporal_mode,
            as_of=as_of,
            known_as_of=known_as_of,
            revision_id=revision_id,
            include_manager=include_manager,
            manager_scope_allowed=scope_chain is None
            or f"user_global:{client.user_id}" in effective_scope_chain,
        )
        if source_speakers:
            selected_speakers = {speaker.strip().casefold() for speaker in source_speakers}
            records = [
                record for record in records if _has_source_speaker(record, selected_speakers)
            ][:safe_limit]
            diagnostics["source_speakers"] = list(source_speakers)
            diagnostics["native_candidate_limit"] = native_limit
            diagnostics["source_speaker_filter_complete"] = False
            diagnostics["source_speaker_filter_stage"] = "after_native_candidate_limit"
        context_budget = int(diagnostics.get("allocated_context_token_budget", max_tokens))
        packing_started = perf_counter_ns()
        packed = _packed_payload_from_records(
            records,
            max_tokens=context_budget,
            query=query,
            requirements=_requirements_from_diagnostics(diagnostics),
        )
        hits = _response_items_with_retrieval_scores(
            [dict(item) for item in packed.items],
            records,
        )
        _record_packed_decision(
            client,
            records,
            packed,
            query=query,
            run_id=run_id,
            scope_chain=effective_scope_chain,
            diagnostics=diagnostics,
            started_ns=started_ns,
            catalog_revision=catalog_revision,
        )
        diagnostics["packing_ms"] = (perf_counter_ns() - packing_started) / 1_000_000
        diagnostics.update(_payload_diagnostics(packed))
        selected_ids: list[str] = []
        for hit in hits:
            metadata = hit.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            block_id = str(metadata.get("block_id") or hit.get("source") or "").strip()
            if block_id and block_id not in selected_ids:
                selected_ids.append(block_id)
        record_memory_event(
            client.repo_root,
            event_type="memory_recall",
            workspace_id=client.workspace_id,
            run_id=run_id,
            backend=_default_backend(backend),
            operation="recall",
            decision="call",
            outcome="returned" if hits else "empty",
            reason="mcp_recall",
            candidate_count=len(records),
            selected_ids=selected_ids,
        )
        result = {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            "source_horizon": source_horizon,
            "scope_chain": list(effective_scope_chain),
            "include_candidates": include_candidates,
            "include_manager": include_manager,
            "run_id": run_id,
            "diagnostics": _agent_diagnostics(diagnostics),
            "hits": hits,
        }
        return _apply_local_tier(
            result,
            name="recall",
            query=query,
            limit=safe_limit,
            arguments={
                "query": query,
                "limit": safe_limit,
                "scope_chain": list(effective_scope_chain),
                "run_id": run_id,
                "tags": list(tags or ()),
                "include_candidates": include_candidates,
                "source_horizon": source_horizon,
                "source_speakers": list(source_speakers or ()),
                "logical_block_ids": list(logical_block_ids or ()),
                "record_ids": list(record_ids or ()),
                "temporal_mode": temporal_mode,
                "as_of": as_of,
                "known_as_of": known_as_of,
                "revision_id": revision_id,
                "max_tokens": max_tokens,
            },
            client=client,
        )
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["hits"] = []
        return payload
    finally:
        _close_client(client)


def um_mem_build_context(
    query: str,
    workspace_id: str = "",
    user_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 8,
    tags: list[str] | None = None,
    source_horizon: SourceHorizon = "durable",
    scope_chain: list[str] | None = None,
    include_candidates: bool = False,
    max_tokens: int = 2048,
    logical_block_ids: list[str] | None = None,
    search_all: bool = False,
    retrieval_mode: RetrievalMode = "semantic",
    record_ids: list[str] | None = None,
    run_id: str = "",
    render_context: bool = False,
    source_speakers: list[str] | None = None,
    temporal_mode: str = "current",
    as_of: str = "",
    known_as_of: str = "",
    revision_id: str = "",
    include_manager: bool = False,
) -> dict[str, Any]:
    started_ns = perf_counter_ns()
    client = _client(
        repo_root=repo_root,
        workspace_id=workspace_id,
        backend=backend,
        user_id=user_id,
    )
    try:
        if source_speakers is not None and (
            not isinstance(source_speakers, list)
            or not 1 <= len(source_speakers) <= 8
            or any(
                not isinstance(speaker, str) or not speaker.strip() for speaker in source_speakers
            )
        ):
            raise ValueError("source_speakers must be 1-8 nonempty names")
        requested_limit = max(0, int(limit or 0))
        native_limit = max(requested_limit, 64) if source_speakers else requested_limit
        catalog_revision = client._catalog_revision_id() if isinstance(client, MemoryClient) else ""
        effective_scope_chain = _effective_scope_chain(client, scope_chain)
        records, diagnostics = _recall_records_scoped(
            client,
            query=query,
            limit=native_limit,
            tags=tags or [],
            source_horizon=source_horizon,
            scope_chain=effective_scope_chain,
            include_candidates=include_candidates,
            max_tokens=max_tokens,
            run_id=run_id,
            logical_block_ids=logical_block_ids or [],
            search_all=search_all,
            retrieval_mode=retrieval_mode,
            record_ids=record_ids or [],
            temporal_mode=temporal_mode,
            as_of=as_of,
            known_as_of=known_as_of,
            revision_id=revision_id,
            include_manager=include_manager,
            manager_scope_allowed=scope_chain is None
            or f"user_global:{client.user_id}" in effective_scope_chain,
        )
        context_budget = int(diagnostics.get("allocated_context_token_budget", max_tokens))
        selection_context = SelectionContext(
            query=query,
            scope_chain=effective_scope_chain,
            run_id=run_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
            historical=temporal_mode == "historical" or bool(known_as_of),
            as_of="" if known_as_of else as_of,
        )
        manager_selection = (
            replace(selection_context, scope_chain=(f"user_global:{client.user_id}",))
            if include_manager and client.user_id
            else None
        )
        renderable_records = [
            record
            for record in records
            if MemorySelectionPolicy().eligible_for_overlay(
                record,
                manager_selection
                if manager_selection is not None
                and record.workspace_id == MANAGER_WORKSPACE_ID
                and record.metadata.get("memory_level") == "manager"
                else selection_context,
            )
        ]
        if source_speakers:
            selected_speakers = {speaker.strip().casefold() for speaker in source_speakers}
            renderable_records = [
                record
                for record in renderable_records
                if _has_source_speaker(record, selected_speakers)
            ][:requested_limit]
            diagnostics["source_speakers"] = list(source_speakers)
            diagnostics["native_candidate_limit"] = native_limit
            diagnostics["source_speaker_filter_complete"] = False
            diagnostics["source_speaker_filter_stage"] = "after_native_candidate_limit"
        packing_started = perf_counter_ns()
        packed = _packed_payload_from_records(
            renderable_records,
            max_tokens=context_budget,
            query=query,
            force_non_directive=True,
            requirements=_requirements_from_diagnostics(diagnostics),
        )
        items = _response_items_with_retrieval_scores(
            [dict(item) for item in packed.items],
            renderable_records,
        )
        _record_packed_decision(
            client,
            renderable_records,
            packed,
            query=query,
            run_id=run_id,
            scope_chain=effective_scope_chain,
            diagnostics=diagnostics,
            started_ns=started_ns,
            catalog_revision=catalog_revision,
        )
        diagnostics["packing_ms"] = (perf_counter_ns() - packing_started) / 1_000_000
        diagnostics.update(_payload_diagnostics(packed))
        rendered_block_ids: list[str] = []
        rendered_path_ids: list[str] = []
        for item in items:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            block_id = str(metadata.get("block_id") or item.get("source") or "").strip()
            path_id = str(
                item.get("source_path") or metadata.get("source_path") or metadata.get("path") or ""
            ).strip()
            if block_id and block_id not in rendered_block_ids:
                rendered_block_ids.append(block_id)
            if path_id and path_id not in rendered_path_ids:
                rendered_path_ids.append(path_id)
        diagnostics["rendered_item_count"] = len(items)
        diagnostics["rendered_block_ids"] = rendered_block_ids
        diagnostics["rendered_path_ids"] = rendered_path_ids
        diagnostics["final_context_chars"] = len(packed.rendered_context)
        payload = {
            "ok": True,
            "workspace_id": client.workspace_id,
            "run_id": run_id,
            "backend": _default_backend(backend),
            "include_manager": include_manager,
            "delivery_policy": "on_demand",
            "authority": "advisory",
            "diagnostics": _agent_diagnostics(diagnostics),
            "items": items,
        }
        if render_context:
            payload["rendered_context"] = packed.rendered_context
            record_memory_event(
                client.repo_root,
                event_type="context_delivery",
                workspace_id=client.workspace_id,
                run_id=run_id,
                backend=_default_backend(backend),
                operation="context_delivery",
                decision="delivery",
                outcome="rendered" if packed.rendered_context else "empty",
                reason="final_packer",
                candidate_count=len(records),
                selected_ids=rendered_block_ids,
                data={
                    "logical_block_ids": list(logical_block_ids or ()),
                    "search_all": bool(search_all),
                    "decision_id": str(diagnostics.get("decision_id") or ""),
                    "native_query_trace": _telemetry_native_query_trace(
                        diagnostics.get("native_query_trace")
                    ),
                },
            )
        return _apply_local_tier(
            payload,
            name="build_context",
            query=query,
            limit=max(1, int(limit or 4)),
            arguments={
                "query": query,
                "limit": max(1, int(limit or 4)),
                "scope_chain": list(effective_scope_chain),
                "tags": list(tags or ()),
                "include_candidates": include_candidates,
                "source_horizon": source_horizon,
                "source_speakers": list(source_speakers or ()),
                "logical_block_ids": list(logical_block_ids or ()),
                "record_ids": list(record_ids or ()),
                "temporal_mode": temporal_mode,
                "as_of": as_of,
                "known_as_of": known_as_of,
                "revision_id": revision_id,
                "max_tokens": max_tokens,
                "render_context": render_context,
            },
            client=client,
        )
    except Exception as exc:
        payload = _safe_error(exc)
        payload.update(
            {
                "workspace_id": client.workspace_id,
                "backend": _default_backend(backend),
                "delivery_policy": "on_demand",
                "authority": "advisory",
                "items": [],
            }
        )
        if render_context:
            payload["rendered_context"] = ""
        return payload
    finally:
        _close_client(client)


def um_mem_build_granted_context(
    workspace_id: str = "",
    user_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 8,
    source_horizon: SourceHorizon = "durable",
    scope_chain: list[str] | None = None,
    include_candidates: bool = False,
    max_tokens: int = 2048,
    logical_block_ids: list[str] | None = None,
    record_ids: list[str] | None = None,
    run_id: str = "",
    render_context: bool = False,
    active_mask_id: str = "",
    active_mask_revision: int | None = None,
    selected_records_only: bool = False,
) -> dict[str, Any]:
    """Pack exact Start-granted records without semantic backend recall.

    This internal local-service operation is deliberately narrower than
    ``build_context``. It accepts only records currently assigned to the exact
    granted logical blocks, applies the normal scope and overlay policy, and
    never searches for granted records. An explicit active mask may resolve its
    own selected backend queries, but cannot broaden the granted record scope.
    """

    started_ns = perf_counter_ns()
    client = _client(
        repo_root=repo_root,
        workspace_id=workspace_id,
        backend=backend,
        user_id=user_id,
    )
    try:
        catalog_revision = client._catalog_revision_id() if isinstance(client, MemoryClient) else ""
        if not isinstance(selected_records_only, bool):
            raise ValueError("selected_records_only must be a bool")
        effective_scope_chain = _effective_scope_chain(client, scope_chain)
        granted_block_ids = tuple(
            dict.fromkeys(
                str(block_id).strip()
                for block_id in (logical_block_ids or ())
                if str(block_id).strip()
            )
        )
        if not granted_block_ids:
            raise ValueError("granted context requires exact logical_block_ids")

        granted_record_ids = tuple(
            dict.fromkeys(
                str(record_id).strip() for record_id in (record_ids or ()) if str(record_id).strip()
            )
        )
        backend_instance = client.backend
        records_for_blocks = getattr(backend_instance, "record_ids_for_logical_blocks", None)
        get_block = getattr(backend_instance, "get_block", None)
        if not callable(records_for_blocks) or not callable(get_block):
            raise RuntimeError("selected backend cannot resolve exact granted context")

        assigned_record_ids = tuple(
            str(record_id).strip()
            for record_id in records_for_blocks(
                granted_block_ids,
                workspace_id=client.workspace_id,
                profile_id=client.profile_id,
                historical=False,
            )
            if str(record_id).strip()
        )
        assigned_set = set(assigned_record_ids)
        selected_record_ids = (
            tuple(record_id for record_id in granted_record_ids if record_id in assigned_set)
            if granted_record_ids or selected_records_only
            else assigned_record_ids
        )
        active_status = "none"
        active_record_ids: tuple[str, ...] = ()
        semantic_recall_performed = False
        if active_mask_id:
            if active_mask_revision is None:
                raise ValueError("granted active context requires an exact mask revision")
            active_lease = client.resolve_active_mask(
                mask_id=active_mask_id,
                expected_revision=active_mask_revision,
                max_tokens=max_tokens,
            )
            active_status = active_lease.trace.resolution_status
            semantic_recall_performed = active_lease.trace.semantic_recall_performed
            if active_status == "resolved":
                active_record_ids = tuple(item.block_id for item in active_lease.items)
            # A stale active selection has no authority to change an independent
            # exact block/record grant. Resolve that grant without this selection.
            selected_set = assigned_set if selected_records_only else set(selected_record_ids)
            selected_record_ids = tuple(
                dict.fromkeys(
                    (
                        *(
                            record_id
                            for record_id in active_record_ids
                            if record_id in selected_set
                        ),
                        *selected_record_ids,
                    )
                )
            )
        records: list[MemoryBlockRecord] = []
        record_limit = max(0, int(limit or 0))
        for record_id in selected_record_ids:
            if len(records) >= record_limit:
                break
            record = get_block(record_id)
            if isinstance(record, MemoryBlockRecord) and _record_allowed_for_context(
                record,
                scope_chain=effective_scope_chain,
                run_id=run_id,
                profile_id=client.profile_id,
                source_horizon=source_horizon,
                include_candidates=include_candidates,
                logical_block_ids=granted_block_ids,
                record_ids=selected_record_ids if selected_records_only else granted_record_ids,
            ):
                records.append(record)
        selection_context = SelectionContext(
            query="",
            scope_chain=effective_scope_chain,
            run_id=run_id,
            source_horizon=source_horizon,
            include_candidates=include_candidates,
        )
        renderable_records = [
            record
            for record in records
            if MemorySelectionPolicy().eligible_for_overlay(record, selection_context)
        ]
        packed = _packed_payload_from_records(
            renderable_records,
            max_tokens=max_tokens,
            force_non_directive=True,
        )
        items = _response_items_with_retrieval_scores(
            [dict(item) for item in packed.items],
            renderable_records,
        )
        rendered_record_ids = [
            str((item.get("metadata") or {}).get("block_id") or item.get("source") or "")
            for item in items
            if isinstance(item, dict)
        ]
        diagnostics = {
            "native_path": "exact_grant_index",
            "selection_mode": "selected_only" if selected_records_only else "granted_blocks",
            "semantic_recall_performed": semantic_recall_performed,
            "active_mask_status": active_status,
            "active_mask_id": active_mask_id,
            "active_mask_revision": active_mask_revision,
            "requested_logical_block_ids": list(granted_block_ids),
            "requested_record_ids": list(granted_record_ids),
            "assigned_record_count": len(assigned_record_ids),
            "selected_record_count": len(records),
            "rendered_record_ids": [record_id for record_id in rendered_record_ids if record_id],
            **_payload_diagnostics(packed),
        }
        _record_packed_decision(
            client,
            renderable_records,
            packed,
            query="",
            run_id=run_id,
            scope_chain=effective_scope_chain,
            diagnostics=diagnostics,
            started_ns=started_ns,
            catalog_revision=catalog_revision,
        )
        payload = {
            "ok": True,
            "workspace_id": client.workspace_id,
            "run_id": run_id,
            "backend": _default_backend(backend),
            "delivery_policy": "start_grant",
            "authority": "advisory",
            "diagnostics": _as_jsonable(diagnostics),
            "items": items,
        }
        if render_context:
            payload["rendered_context"] = packed.rendered_context
            record_memory_event(
                client.repo_root,
                event_type="context_delivery",
                workspace_id=client.workspace_id,
                run_id=run_id,
                backend=_default_backend(backend),
                operation="granted_context_delivery",
                decision="delivery",
                outcome="rendered" if packed.rendered_context else "empty",
                reason="exact_start_grant",
                candidate_count=len(records),
                selected_ids=[record_id for record_id in rendered_record_ids if record_id],
                data={
                    "logical_block_ids": list(granted_block_ids),
                    "semantic_recall_performed": semantic_recall_performed,
                },
            )
        return payload
    except Exception as exc:
        payload = _safe_error(exc)
        payload.update(
            {
                "workspace_id": client.workspace_id,
                "backend": _default_backend(backend),
                "delivery_policy": "start_grant",
                "authority": "advisory",
                "items": [],
            }
        )
        if render_context:
            payload["rendered_context"] = ""
        return payload
    finally:
        _close_client(client)


def um_mem_commit_working_checkpoint(
    checkpoint: dict[str, object],
    verification_report: dict[str, object],
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    from um_mem.contracts import VerificationReportRef, WorkingStateCheckpoint
    from um_mem.memory.working_continuity import commit_working_state_checkpoint

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        state = WorkingStateCheckpoint.from_mapping(checkpoint)
        lineage = client.scope_lineage
        if (state.workspace_id, state.profile_id, state.session_id, state.task_id) != (
            client.workspace_id,
            client.profile_id,
            lineage.session_id,
            lineage.task_id,
        ):
            raise ValueError("working checkpoint is outside the host-bound scope")
        result = commit_working_state_checkpoint(
            client.repo_root,
            state,
            backend=client.backend,
            verification_report=VerificationReportRef.from_mapping(verification_report),
        )
        return {
            "ok": result.saved,
            "saved": result.saved,
            "memory_id": result.canonical_id,
            "backend": result.backend,
            "error": result.error,
        }
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        return _safe_error(exc)
    finally:
        _close_client(client)


def um_mem_verification_subject(
    memory_id: str,
    expected_source_revision_id: str,
    target_block_id: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    from um_mem.contracts import EvidenceResolver, build_workspace_context
    from um_mem.memory.local_service import LOCAL_SERVICE_PROTOCOL_VERSION
    from um_mem.memory.service_receipts import (
        host_tool_mutation_correlation_id,
        read_committed_mutation_receipt,
    )

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        subject = client.verification_subject(
            memory_id,
            expected_source_revision_id=expected_source_revision_id,
            target_block_id=target_block_id,
        )
        candidate = client._require_profile_record(memory_id)
        resolver = EvidenceResolver(
            build_workspace_context(
                repo_root=client.repo_root,
                workspace_root=client.repo_root,
                workspace_id=client.workspace_id,
            )
        )
        sources: list[dict[str, Any] | None] = []
        proposer_model = ""
        for ref in candidate.evidence_refs:
            source = resolver.ledger_row(ref.ref_id)
            if source is None and ref.ref_type == "artifact":
                record = client.backend.get_block(ref.ref_id)
                if (
                    record is not None
                    and record.workspace_id == client.workspace_id
                    and record.profile_id == client.profile_id
                ):
                    source = asdict(record)
                    if (
                        ref.ref_id == candidate.metadata.get("raw_witness_id")
                        and record.producer == "host_observer"
                        and record.metadata.get("record_kind") == "raw_host_witness"
                        and ref.hash
                        == "sha256:" + hashlib.sha256(record.content.encode("utf-8")).hexdigest()
                    ):
                        try:
                            witness = json.loads(record.content)
                        except (ValueError, TypeError):
                            witness = None
                        if isinstance(witness, dict) and witness.get("tool") in {
                            "um_mem_provider_remember",
                            "um_mem_provider_amend",
                        }:
                            author = witness.get("author_model_version")
                            invocation = witness.get("native_tool_call_id")
                            native_session = witness.get("session_id")
                            if (
                                not isinstance(invocation, str)
                                or not invocation
                                or not isinstance(native_session, str)
                                or not native_session
                            ):
                                sources.append(source)
                                continue
                            receipt = read_committed_mutation_receipt(
                                client.repo_root,
                                workspace_id=client.workspace_id,
                                profile_id=client.profile_id,
                                backend=client._backend_id(),
                                caller_role="parent",
                                protocol_version=LOCAL_SERVICE_PROTOCOL_VERSION,
                                mutation_correlation_id=host_tool_mutation_correlation_id(
                                    "hermes:" + native_session, invocation
                                ),
                            )
                            if (
                                receipt is not None
                                and receipt.get("ok") is True
                                and receipt.get("saved") is True
                                and receipt.get("canonical_reference") == candidate.block_id
                                and (receipt.get("revision_id") or receipt.get("revision"))
                                == candidate.revision_id
                                and isinstance(author, str)
                                and author
                                and author == author.strip()
                            ):
                                proposer_model = author
            sources.append(source)
        return {
            "ok": True,
            "subject": asdict(subject),
            "candidate": asdict(candidate),
            "sources": sources,
            "proposer_model_version": proposer_model,
        }
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        return _safe_error(exc)
    finally:
        _close_client(client)


def um_mem_promote(
    memory_id: str,
    reason: str,
    evidence_refs: list[Any],
    authority: str = "advisory",
    target_block_id: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    workspace_root: str = "",
    expected_source_revision_id: str = "",
    operation_id: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        payload = dict(
            client.promote(
                memory_id,
                reason=reason,
                evidence_refs=evidence_refs,
                authority=authority,
                target_block_id=target_block_id,
                workspace_root=workspace_root or None,
                expected_source_revision_id=expected_source_revision_id,
                operation_id=operation_id,
            )
            or {}
        )
        payload.setdefault("ok", bool(payload.get("block_id")))
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = payload.get("backend") or _default_backend(backend)
        return _json_object(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_forget(
    memory_id: str,
    reason: str = "user_requested",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        evidence = [
            EvidenceRef(
                ref_type="session_event",
                ref_id=f"um-mem-forget:{reason or 'user_requested'}",
                produced_by="harness",
                producer_id="mcp:um_mem_forget",
            )
        ]
        payload = dict(
            client.forget(
                memory_id,
                reason=reason or "user_requested",
                evidence_refs=evidence,
            )
            or {}
        )
        payload.setdefault("ok", bool(payload.get("block_id")))
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = payload.get("backend") or _default_backend(backend)
        return _json_object(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_add_edge(
    src_id: str,
    dst_id: str,
    edge_type: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    reason: str = "",
    weight: float = 1.0,
    evidence_refs: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        edge = MemoryEdgeRecord(
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            workspace_id=client.workspace_id,
            reason=reason,
            weight=weight,
            evidence_refs=tuple(_evidence_refs(evidence_refs or [])),
            metadata=metadata or {},
        )
        payload = dict(client.add_edge(edge) or {})
        payload.setdefault("ok", bool(payload.get("src_id") and payload.get("dst_id")))
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = payload.get("backend") or _default_backend(backend)
        return _json_object(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_export_graph(
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    """Export the scoped backend graph through the selected memory client."""

    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        graph = client.export_graph()
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            "blocks": _as_jsonable(graph.get("blocks") or []),
            "edges": _as_jsonable(graph.get("edges") or []),
            "index_node_ids": _as_jsonable(graph.get("index_node_ids") or []),
            "provenance_routes": _as_jsonable(graph.get("provenance_routes") or {}),
            "audit": _as_jsonable(graph.get("audit") or {}),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def um_mem_export_obsidian(
    workspace_id: str = "",
    user_id: str = "",
    backend: str = "",
    repo_root: str = "",
    vault_dir: str = "",
    project_title: str = "",
    scope_chain: list[str] | None = None,
) -> dict[str, Any]:
    client = _client(
        repo_root=repo_root,
        workspace_id=workspace_id,
        backend=backend,
        user_id=user_id,
    )
    try:
        target_vault = str(vault_dir or os.getenv("UM_MEM_OBSIDIAN_VAULT") or "").strip()
        if not target_vault:
            target_vault = str(client.repo_root / "exports" / "obsidian")
        effective_scope_chain = _effective_scope_chain(client, scope_chain)
        result = export_backend_to_obsidian(
            backend=client.backend,
            vault_dir=target_vault,
            workspace_id=client.workspace_id,
            project_title=project_title,
            scope_chain=effective_scope_chain,
        )
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            "vault_dir": str(result.vault_dir),
            "index_path": str(result.index_path),
            "memory_overview_path": str(result.memory_overview_path),
            "blocks_exported": len(result.block_paths),
            "files_written": [str(path) for path in result.written_files],
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)
