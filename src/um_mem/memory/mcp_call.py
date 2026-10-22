"""Canonical dispatcher and long-lived loopback listener for UM-MEM tools.

The listener owns one selected-backend client for a workspace/profile, so MCP, Hermes,
and short-lived host hooks do not create a per-call subprocess or writer path.
"""

import hashlib
import hmac
import json
import logging
import os
import queue
import secrets
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from um_mem.api import AmbiguousEvidenceRefError, MemoryClient
from um_mem.contracts import AuthenticatedHostEvent, EvidenceRef
from um_mem.memory.backends.base import MemoryBusyError
from um_mem.memory.backends.contracts import CanonicalHeadConflict
from um_mem.memory.blocks.registry import load_memory_blocks, use_catalog_snapshot
from um_mem.memory.kernel.telemetry import profile_memory_operation
from um_mem.memory.local_service import (
    _LOCAL_SERVICE_MUTATION_TOOLS,
    LOCAL_SERVICE_API_VERSION,
    LOCAL_SERVICE_LEASE_TTL_SECONDS,
    LOCAL_SERVICE_LISTENER_HOST,
    LOCAL_SERVICE_LISTENER_QUEUE_SIZE,
    LOCAL_SERVICE_LISTENER_TRANSPORT,
    LOCAL_SERVICE_MAX_REQUEST_BYTES,
    LOCAL_SERVICE_MAX_RESPONSE_BYTES,
    LOCAL_SERVICE_SCHEMA_VERSION,
    LocalServiceIdentity,
    _local_service_idle_timeout_seconds,
    _process_is_alive,
    build_local_service_identity,
    clear_local_service_registry,
    local_service_bootstrap_secret,
    local_service_identity_is_authorized,
    local_service_role_allows,
    local_service_role_capabilities,
    write_local_service_registry,
)
from um_mem.memory.mcp_facade import (
    bound_mcp_client,
    persistent_mcp_client,
    um_mem_abort_topology_evolution,
    um_mem_add_block_edge,
    um_mem_add_edge,
    um_mem_amend,
    um_mem_amend_active_mask,
    um_mem_amend_block,
    um_mem_archive_block,
    um_mem_block_map,
    um_mem_build_context,
    um_mem_build_granted_context,
    um_mem_commit_working_checkpoint,
    um_mem_consolidate_observations,
    um_mem_convert_block_to_index,
    um_mem_create_block,
    um_mem_delete_block,
    um_mem_expand_block,
    um_mem_export_graph,
    um_mem_export_obsidian,
    um_mem_forget,
    um_mem_get_active_mask,
    um_mem_health,
    um_mem_list_records,
    um_mem_memory_map,
    um_mem_merge_blocks,
    um_mem_move_block,
    um_mem_navigation_envelope,
    um_mem_observe,
    um_mem_observe_maintenance,
    um_mem_operation_status,
    um_mem_pending_topology_evolutions,
    um_mem_procedure,
    um_mem_projection_barrier,
    um_mem_promote,
    um_mem_read_exact_record,
    um_mem_read_exact_records,
    um_mem_read_working_set,
    um_mem_recall,
    um_mem_recent,
    um_mem_remember,
    um_mem_remove_block_edge,
    um_mem_resolve_active_mask,
    um_mem_resume_topology_evolution,
    um_mem_revive_block,
    um_mem_set_active_mask,
    um_mem_split_block,
    um_mem_transition_active_mask,
    um_mem_verification_subject,
)
from um_mem.memory.operational_slo import service_slo_contract_revision
from um_mem.memory.prospective import ProspectiveTriggerContext
from um_mem.memory.scopes import ScopeLineage, bind_scope_chain
from um_mem.memory.service_policy import (
    CatalogSnapshot,
    ServicePolicyBundleStore,
    baseline_memory_policy_bundle,
    block_revision_id_for_snapshot,
    catalog_snapshot_from_blocks,
    projection_revision_id_for_snapshot,
)
from um_mem.memory.service_receipts import (
    claim_remember_receipt,
    complete_remember_receipt,
)
from um_mem.memory.service_scheduler import (
    AdvisoryDecisionPlan,
    HostStateEvent,
    ServiceScheduler,
)
from um_mem.memory.service_work_queue import (
    DurableServiceWorkQueue,
    ServiceQueueWorker,
    ServiceWorkItem,
    WorkLane,
)

_LOG = logging.getLogger(__name__)


def _local_service_rebuild_indexes(
    *,
    repo_root: str,
    workspace_id: str,
    backend: str,
    **_arguments: Any,
) -> dict[str, Any]:
    """Run the existing deterministic rebuild inside the single writer."""

    with persistent_mcp_client(
        repo_root=repo_root,
        workspace_id=workspace_id,
        backend=backend,
    ) as client:
        return client.rebuild_indexes()


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "health": um_mem_health,
    "block_map": um_mem_block_map,
    "expand_block": um_mem_expand_block,
    "navigation_envelope": um_mem_navigation_envelope,
    "create_block": um_mem_create_block,
    "convert_block_to_index": um_mem_convert_block_to_index,
    "amend_block": um_mem_amend_block,
    "move_block": um_mem_move_block,
    "archive_block": um_mem_archive_block,
    "revive_block": um_mem_revive_block,
    "delete_block": um_mem_delete_block,
    "split_block": um_mem_split_block,
    "merge_blocks": um_mem_merge_blocks,
    "pending_topology_evolutions": um_mem_pending_topology_evolutions,
    "resume_topology_evolution": um_mem_resume_topology_evolution,
    "abort_topology_evolution": um_mem_abort_topology_evolution,
    "add_block_edge": um_mem_add_block_edge,
    "remove_block_edge": um_mem_remove_block_edge,
    "get_active_mask": um_mem_get_active_mask,
    "set_active_mask": um_mem_set_active_mask,
    "amend_active_mask": um_mem_amend_active_mask,
    "transition_active_mask": um_mem_transition_active_mask,
    "resolve_active_mask": um_mem_resolve_active_mask,
    "observe_maintenance": um_mem_observe_maintenance,
    "observe": um_mem_observe,
    "remember": um_mem_remember,
    "amend": um_mem_amend,
    "recall": um_mem_recall,
    "procedure": um_mem_procedure,
    "memory_map": um_mem_memory_map,
    "recent": um_mem_recent,
    "read_working_set": um_mem_read_working_set,
    "read_exact_record": um_mem_read_exact_record,
    "read_exact_records": um_mem_read_exact_records,
    "list_records": um_mem_list_records,
    "operation_status": um_mem_operation_status,
    "projection_barrier": um_mem_projection_barrier,
    "build_context": um_mem_build_context,
    "build_granted_context": um_mem_build_granted_context,
    "promote": um_mem_promote,
    "commit_working_checkpoint": um_mem_commit_working_checkpoint,
    "verification_subject": um_mem_verification_subject,
    "consolidate_observations": um_mem_consolidate_observations,
    "forget": um_mem_forget,
    "add_edge": um_mem_add_edge,
    "export_graph": um_mem_export_graph,
    "export_obsidian": um_mem_export_obsidian,
    "rebuild_indexes": _local_service_rebuild_indexes,
}
_LOCAL_SERVICE_HOST_PATH_FIELDS = frozenset(
    {
        "repo_root",
        "vault_dir",
        "index_path",
        "memory_overview_path",
        "files_written",
    }
)
_HOST_STATE_FIELDS = frozenset(
    {
        "event_id",
        "event_type",
        "run_id",
        "stage_id",
        "occurred_at",
        "signal_codes",
        "capability_codes",
        "retry_count",
        "agent_kind",
        "changed_paths",
        "artifacts",
        "previous_stage_id",
        "failure_signatures",
        "env_flags",
    }
)
_HOST_STATE_REQUIRED_FIELDS = frozenset({"event_id", "event_type", "run_id", "stage_id"})


def _host_state_event_from_request(
    identity: LocalServiceIdentity,
    *,
    policy_bundle_revision: str,
    event: object,
) -> HostStateEvent:
    if not isinstance(event, dict) or not _HOST_STATE_REQUIRED_FIELDS.issubset(event):
        raise ValueError("local service host state is invalid")
    if set(event).difference(_HOST_STATE_FIELDS):
        raise ValueError("local service host state has unsupported fields")
    occurred_at = event.get("occurred_at")
    if occurred_at is None:
        timestamp = datetime.now(UTC)
    elif isinstance(occurred_at, str):
        timestamp = datetime.fromisoformat(occurred_at)
    else:
        raise ValueError("local service host state occurred_at is invalid")
    if timestamp.tzinfo is None:
        raise ValueError("local service host state occurred_at is invalid")
    return HostStateEvent(
        event_id=event["event_id"],
        event_type=event["event_type"],
        workspace_id=identity.workspace_id,
        run_id=event["run_id"],
        stage_id=event["stage_id"],
        state_revision=policy_bundle_revision,
        occurred_at=timestamp,
        signal_codes=event.get("signal_codes", ()),
        capability_codes=event.get("capability_codes", ()),
        retry_count=event.get("retry_count", 0),
        agent_kind=event.get("agent_kind", "generic_agent"),
        changed_paths=event.get("changed_paths", ()),
        artifacts=event.get("artifacts", ()),
        previous_stage_id=event.get("previous_stage_id", ""),
        failure_signatures=event.get("failure_signatures", ()),
        env_flags=event.get("env_flags", ()),
    )


def _host_state_scheduler_plan(
    identity: LocalServiceIdentity,
    event: HostStateEvent,
    *,
    client: MemoryClient | None = None,
) -> AdvisoryDecisionPlan:
    """Capture a verified prospective decision once without executing a host action."""

    authentication = EvidenceRef(
        ref_type="session_event",
        ref_id=f"local-service:{event.event_id}",
        hash=event.fingerprint_sha256,
        produced_by="harness",
        producer_id=f"local-service:{identity.workspace_id}",
    )
    host_event = AuthenticatedHostEvent(
        event_id=event.event_id,
        event_type=event.event_type,
        principal_id=identity.caller_role,
        channel_id="local-service",
        authentication_evidence_ref=authentication,
        payload={
            "stage_id": event.stage_id,
            "run_id": event.run_id,
            "agent_kind": event.agent_kind,
            "changed_paths": event.changed_paths,
            "artifacts": event.artifacts,
            "run_state_flags": event.signal_codes,
            "previous_stage_id": event.previous_stage_id,
            "failure_signatures": event.failure_signatures,
            "tools_available": event.capability_codes,
            "env_flags": event.env_flags,
            "retry_count": event.retry_count,
        },
    )
    trigger_context = ProspectiveTriggerContext(
        host_event=host_event,
        tool_state={"available": event.capability_codes},
        test_state={"failure_signatures": event.failure_signatures},
        file_state={"changed_paths": event.changed_paths, "artifacts": event.artifacts},
        registered_condition={"signals": event.signal_codes, "env_flags": event.env_flags},
        scope="run",
        scope_id=event.run_id,
        authority_ceiling="advisory",
        observed_at=event.occurred_at.isoformat(),
    )
    candidate_lookup = getattr(client, "prospective_trigger_candidates", None)
    decisions = candidate_lookup(trigger_context) if callable(candidate_lookup) else []
    matched_decisions = [decision for decision in decisions if decision.matched]
    matched_rule_sha256s = {
        _evidence_sha256("prospective-obligation", decision.obligation_id)
        for decision in matched_decisions
    }
    matched_action_sha256s = {
        _evidence_sha256("prospective-cue", cue)
        for decision in matched_decisions
        for cue in decision.matched_cues
    }
    matched_rule_evidence = tuple(sorted(matched_rule_sha256s))
    matched_action_evidence = tuple(sorted(matched_action_sha256s))
    deferred_owner_evidence: tuple[str, ...] = ()
    plan_digest = hashlib.sha256(
        json.dumps(
            {
                "policy_bundle_revision": event.state_revision,
                "matched_rule_sha256s": matched_rule_evidence,
                "matched_action_sha256s": matched_action_evidence,
                "deferred_trigger_owner_refs": deferred_owner_evidence,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    matched = bool(matched_rule_evidence)
    return AdvisoryDecisionPlan(
        decision_revision=f"{event.state_revision}:host-state:{plan_digest}",
        activation_due_at=event.occurred_at,
        activation_reason_code="prospective_match" if matched else "prospective_no_match",
        consolidation_due_at=event.occurred_at,
        consolidation_reason_code="consolidation_assessment",
        matched_rule_sha256s=matched_rule_evidence,
        matched_action_sha256s=matched_action_evidence,
        deferred_trigger_owner_refs=deferred_owner_evidence,
    )


def _evidence_sha256(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}:{value}".encode()).hexdigest()


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        return {"ok": False, "error": f"unknown UM-MEM MCP tool: {name}"}
    try:
        return tool(**arguments)
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def _dispatch_local_service_request(
    identity: LocalServiceIdentity,
    raw_request: str,
    *,
    client: MemoryClient,
    expected_lease: str,
    trusted_scope_lineage: ScopeLineage | None = None,
    before_durable_commit: Callable[[], bool] | None = None,
) -> dict[str, object]:
    request_id = ""
    try:
        request = json.loads(raw_request)
    except json.JSONDecodeError:
        return _dispatch_envelope(
            identity, request_id, {"ok": False, "error": "invalid request JSON"}
        )
    if not isinstance(request, dict):
        return _dispatch_envelope(
            identity, request_id, {"ok": False, "error": "request must be an object"}
        )
    request_id_value = request.get("request_id")
    if isinstance(request_id_value, str):
        request_id = request_id_value
    if request.get("protocol_version") != identity.protocol_version:
        return _dispatch_envelope(
            identity,
            request_id,
            {"ok": False, "error": "local service protocol version mismatch"},
        )
    if not request_id:
        return _dispatch_envelope(
            identity, request_id, {"ok": False, "error": "request_id is required"}
        )
    if request.get("lease") != expected_lease:
        return _dispatch_envelope(
            identity,
            request_id,
            {"ok": False, "error": "local service lease rejected"},
        )
    if request.get("caller_role") != identity.caller_role:
        return _dispatch_envelope(
            identity,
            request_id,
            {"ok": False, "error": "local service caller role mismatch"},
        )
    tool = request.get("tool")
    if not isinstance(tool, str) or tool not in TOOLS:
        return _dispatch_envelope(
            identity, request_id, {"ok": False, "error": "unknown UM-MEM MCP tool"}
        )
    if not local_service_role_allows(identity.caller_role, tool):
        return _dispatch_envelope(
            identity,
            request_id,
            {"ok": False, "error": "local service caller role is not allowed for tool"},
        )
    arguments = request.get("arguments")
    if not isinstance(arguments, dict):
        return _dispatch_envelope(
            identity, request_id, {"ok": False, "error": "arguments must be an object"}
        )
    try:
        requested_identity = build_local_service_identity(
            repo_root=(
                identity.repo_root
                if arguments.get("repo_root") in (None, "")
                else arguments.get("repo_root")
            ),
            workspace_id=(
                identity.workspace_id
                if arguments.get("workspace_id") in (None, "")
                else arguments.get("workspace_id")
            ),
            profile_id=identity.profile_id,
            backend=(
                identity.backend
                if arguments.get("backend") in (None, "")
                else arguments.get("backend")
            ),
            caller_role=identity.caller_role,
        )
    except ValueError as exc:
        return _dispatch_envelope(
            identity,
            request_id,
            {"ok": False, "error": f"local service identity rejected: {exc}"},
        )
    if requested_identity != identity:
        return _dispatch_envelope(
            identity,
            request_id,
            {"ok": False, "error": "request identity does not match local service listener"},
        )
    effective_arguments = identity.bound_arguments(arguments)
    if trusted_scope_lineage is not None and tool in {
        "remember",
        "recall",
        "build_context",
        "build_granted_context",
    }:
        effective_arguments.setdefault("run_id", trusted_scope_lineage.session_id)
    receipt_arguments = effective_arguments
    receipt_fingerprint_arguments = (
        {key: value for key, value in effective_arguments.items() if key != "observed_at"}
        if tool == "observe"
        else effective_arguments
    )
    recovered_pending_receipt = False
    if tool in _LOCAL_SERVICE_MUTATION_TOOLS:
        try:
            mutation_correlation_id = _remember_mutation_correlation_id(
                request,
                request_id,
            )
        except ValueError:
            return _dispatch_envelope(
                identity,
                request_id,
                {"ok": False, "error": "local service mutation correlation rejected"},
            )
        if before_durable_commit is not None and not before_durable_commit():
            return _dispatch_envelope(
                identity,
                request_id,
                {
                    "ok": False,
                    "error": "local service request cancelled before durable claim",
                    "reason_code": "dispatch_cancelled",
                    "request_id": request_id,
                    "operation_id": mutation_correlation_id,
                    "durable_status": "not_started",
                    "retryable": True,
                },
            )
        try:
            receipt_status, remembered, receipt_recorded_at = claim_remember_receipt(
                identity.repo_root,
                workspace_id=identity.workspace_id,
                profile_id=identity.profile_id,
                backend=identity.backend,
                caller_role=identity.caller_role,
                protocol_version=identity.protocol_version,
                mutation_correlation_id=mutation_correlation_id,
                arguments=receipt_fingerprint_arguments,
            )
            receipt_arguments = (
                _receipt_remember_arguments(
                    effective_arguments,
                    recorded_at=receipt_recorded_at,
                    operation_id=mutation_correlation_id,
                )
                if tool == "remember"
                else (
                    {**effective_arguments, "operation_id": mutation_correlation_id}
                    if tool in {"promote", "amend"}
                    else effective_arguments
                )
            )
        except ValueError:
            return _dispatch_envelope(
                identity,
                request_id,
                {
                    "ok": False,
                    "error": "local service remember receipt conflict",
                    "error_type": "ValueError",
                    "reason_code": "mutation_receipt_conflict",
                    "retryable": False,
                },
            )
        except (OSError, sqlite3.Error, RuntimeError, TypeError):
            return _dispatch_envelope(
                identity,
                request_id,
                {"ok": False, "error": "local service remember receipt unavailable"},
            )
        if receipt_status == "committed":
            assert remembered is not None
            if remembered.get("receipt_kind") == "mutation_response":
                stored_response = remembered.get("response")
                if not isinstance(stored_response, dict):
                    return _dispatch_envelope(
                        identity,
                        request_id,
                        {"ok": False, "error": "local service mutation receipt unavailable"},
                    )
                remembered = dict(stored_response)
            return _dispatch_envelope(
                identity,
                request_id,
                {**remembered, "replayed": True},
            )
        recovered_pending_receipt = receipt_status == "pending"
    profile_run_id = str(
        receipt_arguments.get("run_id")
        or (trusted_scope_lineage.session_id if trusted_scope_lineage is not None else "")
    )
    profile_user_id = trusted_scope_lineage.user_id if trusted_scope_lineage is not None else ""
    profile_chat_id = trusted_scope_lineage.session_id if trusted_scope_lineage is not None else ""
    profile_id = str(getattr(client, "profile_id", "") or "")
    with profile_memory_operation(
        identity.repo_root,
        operation=tool,
        component="local_service.core",
        workspace_id=identity.workspace_id,
        profile_id=profile_id,
        user_id=profile_user_id,
        chat_id=profile_chat_id,
        run_id=profile_run_id,
        backend=identity.backend,
        input_value=receipt_arguments,
    ) as profile:
        if tool == "operation_status" and isinstance(client, MemoryClient):
            unknown = set(receipt_arguments).difference(
                {"operation_id", "workspace_id", "backend", "repo_root"}
            )
            if unknown:
                payload = {"ok": False, "error": "unknown operation_status argument"}
            else:
                try:
                    operation_id = receipt_arguments.get("operation_id")
                    if not isinstance(operation_id, str):
                        raise ValueError("operation_status requires operation_id")
                    status = client._operation_status_for_receipt(
                        operation_id,
                        caller_role=identity.caller_role,
                        protocol_version=identity.protocol_version,
                    )
                    payload = {
                        **status,
                        "workspace_id": identity.workspace_id,
                        "backend": identity.backend,
                    }
                except Exception as exc:
                    payload = {"ok": False, "error": type(exc).__name__}
        elif isinstance(client, MemoryClient):
            with bound_mcp_client(client):
                payload = _project_local_service_payload(call_tool(tool, receipt_arguments))
        else:
            payload = _project_local_service_payload(call_tool(tool, receipt_arguments))
        profile.set_output(payload)
        profile.set_result(
            status="ok" if payload.get("ok") is True else "failed",
            outcome=str(
                payload.get("outcome")
                or payload.get("route_outcome")
                or ("returned" if payload.get("ok") is True else "rejected")
            ),
            reason=str(
                payload.get("reason_code")
                or payload.get("reason")
                or payload.get("native_path")
                or ""
            ),
        )
    if tool in _LOCAL_SERVICE_MUTATION_TOOLS and payload.get("ok") is True:
        try:
            complete_remember_receipt(
                identity.repo_root,
                workspace_id=identity.workspace_id,
                profile_id=identity.profile_id,
                backend=identity.backend,
                caller_role=identity.caller_role,
                protocol_version=identity.protocol_version,
                mutation_correlation_id=mutation_correlation_id,
                arguments=receipt_fingerprint_arguments,
                response=(
                    _compact_observe_receipt(payload)
                    if tool == "observe"
                    else (
                        _compact_remember_receipt(payload)
                        if tool == "remember"
                        else _mutation_receipt(tool, payload)
                    )
                ),
            )
        except (OSError, sqlite3.Error, RuntimeError, TypeError, ValueError):
            return _dispatch_envelope(
                identity,
                request_id,
                {"ok": False, "error": "local service remember receipt unavailable"},
            )
    elif tool == "remember" and payload.get("route_rejected") is True:
        try:
            payload = _compact_remember_receipt(payload)
            complete_remember_receipt(
                identity.repo_root,
                workspace_id=identity.workspace_id,
                profile_id=identity.profile_id,
                backend=identity.backend,
                caller_role=identity.caller_role,
                protocol_version=identity.protocol_version,
                mutation_correlation_id=mutation_correlation_id,
                arguments=effective_arguments,
                response=payload,
            )
        except (OSError, sqlite3.Error, RuntimeError, TypeError, ValueError):
            return _dispatch_envelope(
                identity,
                request_id,
                {"ok": False, "error": "local service remember receipt unavailable"},
            )
    elif tool in _LOCAL_SERVICE_MUTATION_TOOLS and payload.get("ok") is False:
        try:
            complete_remember_receipt(
                identity.repo_root,
                workspace_id=identity.workspace_id,
                profile_id=identity.profile_id,
                backend=identity.backend,
                caller_role=identity.caller_role,
                protocol_version=identity.protocol_version,
                mutation_correlation_id=mutation_correlation_id,
                arguments=receipt_fingerprint_arguments,
                outcome="failed",
                response={
                    "ok": False,
                    "error_type": str(payload.get("error_type") or ""),
                    "reason_code": str(payload.get("reason_code") or ""),
                    "error_sha256": hashlib.sha256(
                        str(payload.get("error") or "").encode("utf-8")
                    ).hexdigest(),
                },
            )
        except (OSError, sqlite3.Error, RuntimeError, TypeError, ValueError):
            return _dispatch_envelope(
                identity,
                request_id,
                {"ok": False, "error": "local service remember receipt unavailable"},
            )
    if recovered_pending_receipt:
        payload = {**payload, "replayed": True, "recovered": True}
    return _dispatch_envelope(identity, request_id, payload)


def _project_local_service_payload(payload: dict[str, Any]) -> dict[str, object]:
    projected: dict[str, object] = {}
    for key, value in payload.items():
        if key not in _LOCAL_SERVICE_HOST_PATH_FIELDS:
            projected[key] = value
    return projected


def _remember_mutation_correlation_id(
    request: dict[str, object],
    request_id: str,
) -> str:
    value = request.get("mutation_correlation_id")
    if value is None:
        return request_id
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("local service mutation correlation invalid")
    return value


def _compact_remember_receipt(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("ok") is True and payload.get("saved") is True:
        raw = payload.get("raw")
        if payload.get("operation") == "correct":
            raw = payload
        elif not isinstance(raw, dict):
            raise ValueError("local service remember receipt response invalid")
        assert isinstance(raw, dict)
        response = {
            "ok": True,
            "saved": True,
            "canonical_reference": payload.get("memory_id"),
            "canonical_key": raw.get("canonical_key"),
            "revision_id": raw.get("revision_id"),
            "recorded_at": raw.get("recorded_at"),
            "native_reference": _native_reference(raw),
            "backend": payload.get("backend"),
            "workspace_id": payload.get("workspace_id"),
        }
        if any(
            not isinstance(value, str) or not value
            for key, value in response.items()
            if key not in {"ok", "saved", "native_reference"}
        ):
            raise ValueError("local service remember receipt response invalid")
        return response
    if payload.get("route_rejected") is True:
        response = {
            "ok": False,
            "saved": False,
            "route_rejected": True,
            "reason": payload.get("reason"),
            "backend": payload.get("backend"),
            "workspace_id": payload.get("workspace_id"),
        }
        route_decision = payload.get("route_decision")
        if route_decision is not None:
            if not isinstance(route_decision, dict):
                raise ValueError("local service routed receipt response invalid")
            response["route_decision"] = dict(route_decision)
        policy_issues = payload.get("policy_issues")
        if policy_issues is not None:
            if not isinstance(policy_issues, list) or any(
                not isinstance(issue, str) or not issue for issue in policy_issues
            ):
                raise ValueError("local service routed receipt response invalid")
            response["policy_issues"] = list(policy_issues)
        suggested_route = payload.get("suggested_route")
        if suggested_route is not None:
            if (
                not isinstance(suggested_route, dict)
                or set(suggested_route) != {"action", "requested_block_id"}
                or any(
                    not isinstance(suggested_route.get(key), str) or not suggested_route.get(key)
                    for key in ("action", "requested_block_id")
                )
            ):
                raise ValueError("local service routed receipt response invalid")
            response["suggested_route"] = dict(suggested_route)
        if any(
            not isinstance(value, str) or not value
            for key, value in response.items()
            if key
            not in {
                "ok",
                "saved",
                "route_rejected",
                "route_decision",
                "policy_issues",
                "suggested_route",
            }
        ):
            raise ValueError("local service routed receipt response invalid")
        return response
    raise ValueError("local service remember receipt response invalid")


def _compact_observe_receipt(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("ok") is not True or payload.get("saved") is not True:
        raise ValueError("local service observe receipt response invalid")
    response = {
        "ok": True,
        "saved": True,
        "witness_id": payload.get("witness_id"),
        "memory_id": payload.get("memory_id") or "",
        "requested_block_id": payload.get("requested_block_id") or "",
        "actual_block_id": payload.get("actual_block_id") or "",
        "route_outcome": payload.get("route_outcome"),
        "reason": payload.get("reason"),
        "status": payload.get("status") or "candidate",
        "authority": payload.get("authority") or "advisory",
        "verification_status": payload.get("verification_status") or "observed",
        "evidence_refs": payload.get("evidence_refs") or [],
        "backend": payload.get("backend"),
        "workspace_id": payload.get("workspace_id"),
    }
    verification_report = payload.get("verification_report")
    if isinstance(verification_report, dict):
        response["verification_report"] = verification_report
    if payload.get("source_id") is not None:
        source_id = payload.get("source_id")
        source_sha256 = payload.get("source_sha256")
        source_bytes = payload.get("source_bytes")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
            or type(source_bytes) is not int
            or source_bytes <= 0
        ):
            raise ValueError("local service observe source receipt invalid")
        response.update(
            source_id=source_id,
            source_sha256=source_sha256,
            source_bytes=source_bytes,
        )
    required = (
        "witness_id",
        "route_outcome",
        "reason",
        "backend",
        "workspace_id",
    )
    if any(not isinstance(response[key], str) or not response[key] for key in required):
        raise ValueError("local service observe receipt response invalid")
    return response


def _compact_generic_mutation_receipt(
    tool: str,
    payload: dict[str, object],
) -> dict[str, object]:
    if payload.get("ok") is not True or payload.get("saved") is not True:
        raise ValueError("local service mutation receipt response invalid")
    canonical_reference = payload.get("new_id") if tool == "amend" else payload.get("block_id")
    previous_revision = (
        payload.get("old_revision") if tool == "amend" else payload.get("previous_revision") or ""
    )
    revision = payload.get("new_revision") if tool == "amend" else payload.get("revision")
    response = {
        "ok": True,
        "saved": True,
        "mutation_tool": tool,
        "canonical_reference": canonical_reference,
        "previous_revision": previous_revision,
        "revision": revision,
        "archived": bool(payload.get("archived")),
        "backend": payload.get("backend"),
        "workspace_id": payload.get("workspace_id"),
    }
    required = (
        "mutation_tool",
        "canonical_reference",
        "previous_revision",
        "revision",
        "backend",
        "workspace_id",
    )
    if any(not isinstance(response[key], str) for key in required):
        raise ValueError("local service mutation receipt response invalid")
    if any(
        not response[key]
        for key in ("mutation_tool", "canonical_reference", "revision", "backend", "workspace_id")
    ):
        raise ValueError("local service mutation receipt response invalid")
    return response


def _mutation_receipt(tool: str, payload: dict[str, object]) -> dict[str, object]:
    """Preserve an idempotent replay for every typed service mutation."""

    if tool == "amend":
        return _compact_generic_mutation_receipt(tool, payload)
    if payload.get("ok") is not True:
        raise ValueError("local service mutation receipt response invalid")
    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "receipt_kind": "mutation_response",
        "mutation_tool": tool,
        "response": dict(payload),
    }


def _receipt_remember_arguments(
    arguments: dict[str, object],
    *,
    recorded_at: str,
    operation_id: str = "",
) -> dict[str, object]:
    requested_memory_id = arguments.get("memory_id")
    operation = str(arguments.get("operation") or "add").strip().lower()
    if requested_memory_id not in {None, ""} and operation != "correct":
        raise ValueError("local service remember does not accept caller memory_id")
    requested_recorded_at = arguments.get("_recorded_at")
    if requested_recorded_at not in {None, "", recorded_at}:
        raise ValueError("local service remember receipt timestamp conflict")
    return {
        **arguments,
        "_recorded_at": recorded_at,
        **({"operation_id": operation_id} if operation_id else {}),
    }


def _native_reference(raw: dict[str, object]) -> str | None:
    for key in ("native_reference", "native_id"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    if raw.get("backend") == "mnemosyne":
        value = raw.get("memory_id")
        if isinstance(value, str) and value:
            return value
    if raw.get("backend") == "hindsight":
        value = raw.get("id")
        if isinstance(value, str) and value:
            return value
    return None


def _optional_revision(payload: object, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _dispatch_envelope(
    identity: LocalServiceIdentity,
    request_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "protocol_version": identity.protocol_version,
        "request_id": request_id,
        "payload": payload,
    }


@dataclass(frozen=True)
class _ListenerLease:
    token: str
    identity: LocalServiceIdentity
    session_id: str
    scope_lineage: ScopeLineage
    expires_at: float
    policy_bundle_revision: str
    catalog_revision: str
    catalog_snapshot: CatalogSnapshot


@dataclass
class _DispatchItem:
    lease: _ListenerLease
    request: dict[str, object]
    completed: threading.Event
    result: dict[str, object]
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    commit_started: bool = False
    cancelled: bool = False


def _lease_scope_lineage(
    identity: LocalServiceIdentity,
    *,
    session_id: str,
    payload: object,
) -> ScopeLineage:
    if payload is None:
        return ScopeLineage(
            workspace_id=identity.workspace_id,
            session_id=session_id,
        )
    if not isinstance(payload, dict):
        raise ValueError("local service scope lineage rejected")
    allowed = {
        "workspace_id",
        "project_id",
        "session_id",
        "task_id",
        "subtask_id",
        "user_id",
    }
    if any(key not in allowed for key in payload):
        raise ValueError("local service scope lineage rejected")
    if any(not isinstance(value, str) for value in payload.values()):
        raise ValueError("local service scope lineage rejected")
    lineage = ScopeLineage(
        workspace_id=str(payload.get("workspace_id") or ""),
        project_id=str(payload.get("project_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        task_id=str(payload.get("task_id") or ""),
        subtask_id=str(payload.get("subtask_id") or ""),
        user_id=str(payload.get("user_id") or ""),
    )
    if lineage.workspace_id != identity.workspace_id or lineage.session_id != session_id:
        raise ValueError("local service scope lineage rejected")
    return lineage


def _cache_revisions(
    lease: _ListenerLease,
    health: dict[str, object],
) -> dict[str, object] | None:
    """Bind a read result to every revision that can change its visibility."""

    try:
        live_block_snapshot = catalog_snapshot_from_blocks(
            load_memory_blocks(
                lease.identity.repo_root,
                lease.identity.workspace_id,
                profile_id=lease.identity.profile_id,
                strict=True,
            )
        )
    except (OSError, TypeError, ValueError):
        return None
    revisions = {
        "acl_revision": 1,
        "canonical_snapshot_sha256": health.get("canonical_snapshot_sha256"),
        "catalog_revision": lease.catalog_revision,
        "policy_revision": lease.policy_bundle_revision,
        "block_revision": block_revision_id_for_snapshot(live_block_snapshot),
        "index_revision": health.get("index_revision"),
    }
    if any(
        not isinstance(revisions[field_name], str) or not revisions[field_name]
        for field_name in (
            "canonical_snapshot_sha256",
            "catalog_revision",
            "policy_revision",
            "block_revision",
            "index_revision",
        )
    ):
        return None
    return revisions


class _LocalServiceListenerRuntime:
    """One serialized selected-backend dispatcher behind authenticated loopback HTTP."""

    def __init__(
        self,
        identity: LocalServiceIdentity,
        client: MemoryClient,
        *,
        idle_timeout_seconds: int | None = None,
    ) -> None:
        self.identity = identity
        self.client = client
        self._policy_store = ServicePolicyBundleStore(identity)
        self._initialize_policy_bundle()
        self._leases: dict[str, _ListenerLease] = {}
        self._leases_lock = threading.RLock()
        self._requests: queue.Queue[_DispatchItem | None] = queue.Queue(
            maxsize=LOCAL_SERVICE_LISTENER_QUEUE_SIZE
        )
        self._dispatcher: threading.Thread | None = None
        self._dispatch_worker_count = max(1, min(4, LOCAL_SERVICE_LISTENER_QUEUE_SIZE))
        self._dispatch_slots = threading.BoundedSemaphore(self._dispatch_worker_count)
        self._lifecycle_lock = threading.RLock()
        self._client_lock = threading.RLock()
        self._native_health_lock = threading.Lock()
        self._native_ok: bool | None = None
        self._native_probed_at = 0.0
        self._native_revisions: dict[str, str | None] = {
            "schema_revision": None,
            "index_revision": None,
            "canonical_snapshot_sha256": None,
        }
        self._draining = False
        self._closed = False
        self._shutdown_error = ""
        self._scheduler_error = ""
        self._scheduler = ServiceScheduler(
            identity,
            planner=lambda event: _host_state_scheduler_plan(identity, event, client=client),
        )
        self._work_queue = DurableServiceWorkQueue(identity, recover_abandoned=True)
        self._work_queue_error = ""
        self._work_worker = ServiceQueueWorker(
            self._work_queue,
            {
                "indexing": self._run_indexing_work,
                "working_continuity": self._run_native_validation_work,
                "prospective_scheduler": self._run_scheduler_work,
                "consolidation_transition": self._run_scheduler_work,
                "policy_shadow_qualification": self._run_policy_validation_work,
            },
        )
        self._scheduler_stopped = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._active_requests = 0
        self._idle_timeout_seconds = (
            idle_timeout_seconds
            if isinstance(idle_timeout_seconds, int) and idle_timeout_seconds > 0
            else _local_service_idle_timeout_seconds()
        )
        self._last_work_at = time.monotonic()
        self._successful_calls = 0
        self._failed_calls = 0
        self._metrics_lock = threading.Lock()
        from um_mem.memory.dashboard import Dashboard

        self.dashboard = Dashboard(self)

    def _initialize_policy_bundle(self) -> None:
        """Create one strict baseline once; later restarts use the persisted pin."""

        if not self._policy_store.state_path.exists():
            blocks = load_memory_blocks(
                self.identity.repo_root,
                self.identity.workspace_id,
                profile_id=self.identity.profile_id,
                strict=True,
            )
            snapshot = catalog_snapshot_from_blocks(blocks)
            self._policy_store.ensure_initialized(
                baseline_memory_policy_bundle(snapshot),
                snapshot,
            )
        bundle = self._policy_store.active_bundle()
        self._policy_store.snapshot_for(bundle)

    def start(self) -> None:
        self._native_backend_is_healthy(force=True)
        context = copy_context()
        self._dispatcher = threading.Thread(
            target=context.run,
            args=(self._dispatch,),
            daemon=True,
        )
        self._dispatcher.start()
        self._work_worker.start()
        self._enqueue_due_scheduler_work()
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True,
        )
        self._scheduler_thread.start()

    def close(self) -> None:
        self.begin_shutdown()
        self.dashboard.pool.close()
        with self.dashboard.feed.condition:
            self.dashboard.feed.condition.notify_all()
        self._scheduler_stopped.set()
        scheduler_thread = self._scheduler_thread
        if scheduler_thread is not None:
            scheduler_thread.join(timeout=5)
        self._work_worker.close()
        with self._lifecycle_lock:
            if self._closed:
                dispatcher = self._dispatcher
                already_closed = True
            else:
                self._closed = True
                dispatcher = self._dispatcher
                already_closed = False
        if dispatcher is None:
            self._scheduler.close()
            return
        if already_closed:
            dispatcher.join(timeout=5)
            self._scheduler.close()
            return

        deadline = time.monotonic() + 5
        sentinel_enqueued = False
        while dispatcher.is_alive() and time.monotonic() < deadline:
            try:
                self._requests.put(
                    None,
                    timeout=min(0.1, max(0.01, deadline - time.monotonic())),
                )
                sentinel_enqueued = True
                break
            except queue.Full:
                continue
        if not sentinel_enqueued:
            with self._lifecycle_lock:
                self._shutdown_error = "local service dispatcher drain timed out"
            self._scheduler.close()
            return
        dispatcher.join(timeout=5)
        if dispatcher.is_alive():
            with self._lifecycle_lock:
                self._shutdown_error = "local service dispatcher shutdown timed out"
        self._scheduler.close()

    def begin_shutdown(self) -> None:
        """Reject new work while the existing bounded queue drains in order."""

        with self._lifecycle_lock:
            self._draining = True

    def is_idle(self) -> bool:
        with self._lifecycle_lock:
            idle = (
                not self._draining
                and not self._closed
                and self._active_requests == 0
                and time.monotonic() - self._last_work_at >= self._idle_timeout_seconds
            )
        if not idle:
            return False
        try:
            queue_health = self._work_queue.health()
            queue_active = any(lane.pending or lane.running for lane in queue_health.lanes)
            return not self._scheduler.has_pending() and not queue_active
        except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
            return False

    def _touch_work(self) -> None:
        with self._lifecycle_lock:
            self._last_work_at = time.monotonic()

    def _run_scheduler(self) -> None:
        while not self._scheduler_stopped.wait(0.25):
            try:
                submissions = self._enqueue_due_scheduler_work()
            except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
                with self._lifecycle_lock:
                    self._scheduler_error = "local service scheduler unavailable"
                return
            if submissions:
                self._touch_work()

    def _enqueue_due_scheduler_work(self) -> int:
        submitted = 0
        for job in self._scheduler.recover_due():
            lane: WorkLane = (
                "consolidation_transition"
                if job.job_kind == "consolidation_assessment"
                else "prospective_scheduler"
            )
            receipt = self._work_queue.submit(
                lane,
                work_ref=job.job_id,
                work_revision=job.decision_revision,
            )
            if receipt.status == "backpressured":
                raise RuntimeError("local service scheduler work queue is backpressured")
            submitted += receipt.status == "queued"
        return submitted

    def _run_indexing_work(self, item: ServiceWorkItem) -> None:
        with self._client_lock:
            result = self.client.backend.refresh_memory_indexes(item.work_revision)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError("background memory index refresh failed")
        if self.dashboard.active:
            self.dashboard.feed.publish(
                {
                    "kind": "indexing",
                    "operation": "indexing",
                    "ok": True,
                    "record_id": item.work_revision,
                }
            )

    def _run_scheduler_work(self, _item: ServiceWorkItem) -> None:
        self._scheduler.fire_due()

    def _run_native_validation_work(self, _item: ServiceWorkItem) -> None:
        if not self._native_backend_is_healthy(force=True):
            raise RuntimeError("background native validation failed")

    def _run_policy_validation_work(self, _item: ServiceWorkItem) -> None:
        bundle = self._policy_store.active_bundle()
        self._policy_store.snapshot_for(bundle)

    def _background_queue_status(self) -> dict[str, object]:
        with self._lifecycle_lock:
            persistent_error = self._work_queue_error
        if persistent_error:
            return {
                "available": False,
                "emergency_read_only": True,
                "no_silent_drop": False,
                "lanes": [],
            }
        try:
            health = self._work_queue.health()
        except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
            with self._lifecycle_lock:
                self._work_queue_error = "local service background queue unavailable"
            return {
                "available": False,
                "emergency_read_only": True,
                "no_silent_drop": False,
                "lanes": [],
            }
        return {
            "available": True,
            "emergency_read_only": health.emergency_read_only,
            "no_silent_drop": health.no_silent_drop,
            "lanes": [
                {
                    "lane": lane.lane,
                    "capacity": lane.capacity,
                    "target_latency_ms": lane.target_latency_ms,
                    "pending": lane.pending,
                    "running": lane.running,
                    "completed": lane.completed,
                    "poisoned": lane.poisoned,
                    "backpressured": lane.backpressured,
                    "oldest_pending_age_ms": lane.oldest_pending_age_ms,
                    "slo_breached": lane.slo_breached,
                }
                for lane in health.lanes
            ],
        }

    def health(self, *, force_native_probe: bool = False) -> dict[str, object]:
        with self._metrics_lock:
            successful_calls = self._successful_calls
            failed_calls = self._failed_calls
        with self._lifecycle_lock:
            draining = self._draining
            closed = self._closed
            shutdown_error = self._shutdown_error
            scheduler_error = self._scheduler_error
            work_queue_error = self._work_queue_error
            dispatcher = self._dispatcher
        dispatcher_alive = dispatcher is not None and dispatcher.is_alive()
        liveness = "live" if dispatcher_alive and not closed and not shutdown_error else "not_live"
        base = {
            "ok": False,
            "liveness": liveness,
            "readiness": "not_ready",
            "native_health": "not_probed",
            "native_ok": False,
            "strict_ok": False,
            "effective_mode": "unavailable",
            "transport": LOCAL_SERVICE_LISTENER_TRANSPORT,
            "listener_pid": os.getpid(),
            "backend": self.identity.backend,
            "selected_backend": self.identity.backend,
            "workspace_id": self.identity.workspace_id,
            "profile_id": self.identity.profile_id,
            "acl_revision": 1,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "queue_capacity": LOCAL_SERVICE_LISTENER_QUEUE_SIZE,
            "queue_depth": self._requests.qsize(),
            "policy_bundle_revision": None,
            "service_schema_revision": f"local-service-schema:{LOCAL_SERVICE_SCHEMA_VERSION}",
            "slo_contract_revision": service_slo_contract_revision(),
            **self._native_revision_fields(),
            "catalog_revision": None,
            "block_revision": None,
            "projection_revision": None,
            "shutdown_error": shutdown_error or None,
            "scheduler_health": "unavailable" if scheduler_error else "healthy",
            "scheduler_error": scheduler_error or None,
            "background_queue": self._background_queue_status(),
            "background_queue_error": work_queue_error or None,
            "dispatcher_alive": dispatcher_alive,
        }
        if liveness != "live":
            return {
                **base,
                "state": "closed" if closed else "not_ready",
                "degraded_state": "listener_closed",
                "error": "local service dispatcher is unavailable",
            }
        try:
            bundle = self._policy_store.active_bundle()
            snapshot = self._policy_store.snapshot_for(bundle)
        except (OSError, RuntimeError, TypeError, ValueError):
            return {
                **base,
                "state": "not_ready",
                "degraded_state": "policy_bundle_unavailable",
                "error": "local service policy bundle unavailable",
            }
        base["policy_bundle_revision"] = bundle.revision_id
        base["catalog_revision"] = bundle.catalog_revision_id
        base["block_revision"] = block_revision_id_for_snapshot(snapshot)
        base["projection_revision"] = projection_revision_id_for_snapshot(snapshot)
        native_ok = self._native_backend_is_healthy(force=force_native_probe)
        base.update(self._native_revision_fields())
        if not native_ok:
            return {
                **base,
                "state": "not_ready",
                "native_health": "unavailable",
                "degraded_state": "native_memory_unavailable",
                "error": "local service native memory unavailable",
            }
        native_fields = {
            "native_health": "healthy",
            "native_ok": True,
            "effective_mode": "selected_native",
        }
        if scheduler_error:
            return {
                **base,
                **native_fields,
                "state": "not_ready",
                "degraded_state": "scheduler_unavailable",
                "error": "local service scheduler unavailable",
            }
        if draining:
            return {
                **base,
                **native_fields,
                "state": "draining",
                "degraded_state": "listener_draining",
                "error": "local service listener is draining",
            }
        queue_status = self._background_queue_status()
        queue_unavailable = queue_status.get("available") is not True
        emergency_read_only = queue_status.get("emergency_read_only") is True
        if queue_unavailable or emergency_read_only:
            return {
                **base,
                **native_fields,
                "background_queue": queue_status,
                "background_queue_error": (
                    "local service background queue unavailable" if queue_unavailable else None
                ),
                "ok": True,
                "state": "ready_read_only",
                "readiness": "ready",
                "strict_ok": True,
                "degraded_state": (
                    "background_queue_unavailable" if queue_unavailable else "emergency_read_only"
                ),
                "error": ("local service mutations are disabled until background queue recovery"),
            }
        return {
            **base,
            **native_fields,
            "ok": True,
            "state": "ready",
            "readiness": "ready",
            "strict_ok": True,
            "degraded_state": None,
            "error": None,
        }

    def _native_backend_is_healthy(self, *, force: bool = False) -> bool:
        """Use a short-lived probe cache without racing the serialized dispatcher."""

        with self._native_health_lock:
            cached = self._native_ok
            stale = time.monotonic() - self._native_probed_at >= 1.0
        if cached is not None and not force and not stale:
            return cached
        if not self._client_lock.acquire(blocking=force):
            return bool(cached)
        payload: object = {}
        try:
            payload = self.client.backend.health()
            healthy = (
                isinstance(payload, dict)
                and payload.get("ok") is True
                and payload.get("native_ok") is True
                and payload.get("strict_ok") is True
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
            healthy = False
        finally:
            self._client_lock.release()
        with self._native_health_lock:
            self._native_ok = healthy
            self._native_probed_at = time.monotonic()
            self._native_revisions = {
                "schema_revision": _optional_revision(payload, "schema_revision"),
                "index_revision": _optional_revision(payload, "index_revision"),
                "canonical_snapshot_sha256": _optional_revision(
                    payload,
                    "canonical_snapshot_sha256",
                ),
            }
        return healthy

    def _native_revision_fields(self) -> dict[str, str | None]:
        with self._native_health_lock:
            return dict(self._native_revisions)

    def issue_lease(self, request: dict[str, object]) -> dict[str, object]:
        frame_error = self._frame_error(request)
        if frame_error is not None:
            return {"ok": False, "error": frame_error}
        role = request.get("caller_role")
        session_id = request.get("session_id")
        if not isinstance(role, str) or not local_service_role_allows(role, "health"):
            return {"ok": False, "error": "local service caller role is not allowed"}
        if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 256:
            return {"ok": False, "error": "local service session identity rejected"}
        try:
            lease_identity = build_local_service_identity(
                repo_root=self.identity.repo_root,
                workspace_id=self.identity.workspace_id,
                profile_id=self.identity.profile_id,
                backend=self.identity.backend,
                caller_role=role,
            )
        except ValueError:
            return {"ok": False, "error": "local service caller role is not allowed"}
        expected_capabilities = list(local_service_role_capabilities(role))
        if request.get("capabilities") != expected_capabilities:
            return {"ok": False, "error": "local service capabilities mismatch"}
        provided_secret = request.get("bootstrap_secret")
        if not isinstance(provided_secret, str) or not hmac.compare_digest(
            local_service_bootstrap_secret(lease_identity),
            provided_secret,
        ):
            return {"ok": False, "error": "local service bootstrap authentication rejected"}
        with self._lifecycle_lock:
            if self._draining or self._closed:
                return {"ok": False, "error": "local service listener is draining"}
        health = self.health(force_native_probe=True)
        if health.get("liveness") != "live":
            return {"ok": False, "error": "local service dispatcher is unavailable"}
        if health.get("degraded_state") == "policy_bundle_unavailable":
            return {"ok": False, "error": "local service policy bundle unavailable"}
        try:
            scope_lineage = _lease_scope_lineage(
                lease_identity,
                session_id=session_id.strip(),
                payload=request.get("scope_lineage"),
            )
        except (TypeError, ValueError):
            return {"ok": False, "error": "local service scope lineage rejected"}
        try:
            bundle = self._policy_store.pin_session(session_id.strip())
            catalog_snapshot = self._policy_store.snapshot_for(bundle)
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"ok": False, "error": "local service policy bundle unavailable"}
        token = secrets.token_urlsafe(32)
        lease = _ListenerLease(
            token=token,
            identity=lease_identity,
            session_id=session_id.strip(),
            scope_lineage=scope_lineage,
            expires_at=time.monotonic() + LOCAL_SERVICE_LEASE_TTL_SECONDS,
            policy_bundle_revision=bundle.revision_id,
            catalog_revision=bundle.catalog_revision_id,
            catalog_snapshot=catalog_snapshot,
        )
        with self._leases_lock:
            self._leases[token] = lease
            self._expire_leases()
        self._touch_work()
        return {
            "ok": True,
            "lease": token,
            "expires_in_seconds": LOCAL_SERVICE_LEASE_TTL_SECONDS,
            "capabilities": expected_capabilities,
            "workspace_id": lease.identity.workspace_id,
            "profile_id": lease.identity.profile_id,
            "selected_backend": lease.identity.backend,
            "scope_chain": list(bind_scope_chain(lease.scope_lineage, ())),
            "policy_bundle_revision": lease.policy_bundle_revision,
            "catalog_revision": lease.catalog_revision,
        }

    def call(self, request: dict[str, object]) -> dict[str, object]:
        frame_error = self._frame_error(request)
        if frame_error is not None:
            return {"ok": False, "error": frame_error}
        requested_deadline = request.get("deadline_ns")
        if requested_deadline is not None and (
            type(requested_deadline) is not int or requested_deadline <= 0
        ):
            return {"ok": False, "error": "local service deadline is invalid"}
        deadline_ns = min(
            requested_deadline
            if isinstance(requested_deadline, int)
            else time.monotonic_ns() + 120_000_000_000,
            time.monotonic_ns() + 120_000_000_000,
        )
        lease = self._lease_for(request)
        if lease is None:
            return {"ok": False, "error": "local service lease rejected"}
        if request.get("caller_role") != lease.identity.caller_role:
            return {"ok": False, "error": "local service caller role mismatch"}
        if request.get("session_id") != lease.session_id:
            return {"ok": False, "error": "local service session identity rejected"}
        tool = request.get("tool")
        if not isinstance(tool, str) or not local_service_role_allows(
            lease.identity.caller_role, tool
        ):
            return {"ok": False, "error": "local service caller role is not allowed for tool"}
        arguments = request.get("arguments")
        if not isinstance(arguments, dict) or not local_service_identity_is_authorized(
            lease.identity,
            arguments,
        ):
            return {
                "ok": False,
                "error": "local service request identity is not authorized",
            }
        if arguments.get("session_id") not in (None, ""):
            return {"ok": False, "error": "local service session claim is not authorized"}
        health = self.health(force_native_probe=tool == "health")
        if tool == "health":
            with self._metrics_lock:
                if health.get("ok") is True:
                    self._successful_calls += 1
                else:
                    self._failed_calls += 1
            self._touch_work()
            return health
        if health.get("readiness") != "ready":
            if health.get("degraded_state") == "native_memory_unavailable":
                return {
                    "ok": False,
                    "error": "local service native memory unavailable",
                    "reason_code": "native_backend_unavailable",
                    "retryable": True,
                }
            if health.get("degraded_state") == "listener_draining":
                return {"ok": False, "error": "local service listener is draining"}
            return {"ok": False, "error": "local service listener is not ready"}
        if tool in _LOCAL_SERVICE_MUTATION_TOOLS:
            queue_status = health.get("background_queue")
            if (
                not isinstance(queue_status, dict)
                or queue_status.get("available") is not True
                or queue_status.get("emergency_read_only") is True
            ):
                return {"ok": False, "error": "local service emergency read-only"}
        internal_request = {
            "protocol_version": lease.identity.protocol_version,
            "request_id": request.get("request_id"),
            "lease": lease.token,
            "caller_role": lease.identity.caller_role,
            "tool": tool,
            "arguments": arguments,
        }
        mutation_correlation_id = request.get("mutation_correlation_id")
        if mutation_correlation_id is not None:
            internal_request["mutation_correlation_id"] = mutation_correlation_id
        completed = threading.Event()
        result: dict[str, object] = {}
        item = _DispatchItem(lease, internal_request, completed, result)
        operation_id = (
            mutation_correlation_id or request["request_id"]
            if tool in _LOCAL_SERVICE_MUTATION_TOOLS
            else None
        )
        if time.monotonic_ns() >= deadline_ns:
            return {
                "ok": False,
                "error": "local service dispatcher timed out",
                "reason_code": "dispatch_timeout",
                "request_id": request["request_id"],
                "durable_status": "not_started",
                "retryable": True,
                **({"operation_id": operation_id} if operation_id is not None else {}),
            }
        with self._lifecycle_lock:
            if self._draining or self._closed:
                return {"ok": False, "error": "local service listener is draining"}
            self._active_requests += 1
            try:
                self._requests.put_nowait(item)
            except queue.Full:
                self._active_requests -= 1
                return {
                    "ok": False,
                    "error": "local service dispatch queue is saturated",
                    "reason_code": "dispatch_queue_saturated",
                    "retryable": True,
                }
        self._touch_work()
        remaining_seconds = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        if not completed.wait(timeout=remaining_seconds):
            with item.lock:
                item.cancelled = (
                    not item.commit_started
                    if tool in _LOCAL_SERVICE_MUTATION_TOOLS
                    else not item.started
                )
                durable_status = "not_started" if item.cancelled else "unknown"
            return {
                "ok": False,
                "error": "local service dispatcher timed out",
                "reason_code": "dispatch_timeout",
                "request_id": request["request_id"],
                "durable_status": durable_status,
                "retryable": item.cancelled,
                **({"operation_id": operation_id} if operation_id is not None else {}),
            }
        return result

    def submit_host_state(self, request: dict[str, object]) -> dict[str, object]:
        """Accept a content-free event through the least-privilege scheduler role."""

        frame_error = self._frame_error(request)
        if frame_error is not None:
            return {"ok": False, "error": frame_error}
        lease = self._lease_for(request)
        if lease is None:
            return {"ok": False, "error": "local service lease rejected"}
        if lease.identity.caller_role != "host_state":
            return {"ok": False, "error": "local service host-state role is required"}
        if request.get("caller_role") != lease.identity.caller_role:
            return {"ok": False, "error": "local service caller role mismatch"}
        if request.get("session_id") != lease.session_id:
            return {"ok": False, "error": "local service session identity rejected"}
        health = self.health()
        if health.get("degraded_state") == "scheduler_unavailable":
            return {"ok": False, "error": "local service scheduler unavailable"}
        if health.get("readiness") != "ready":
            return {"ok": False, "error": "local service listener is not ready"}
        with self._lifecycle_lock:
            if self._draining or self._closed:
                return {"ok": False, "error": "local service listener is draining"}
            self._active_requests += 1
        try:
            typed_event = _host_state_event_from_request(
                lease.identity,
                policy_bundle_revision=lease.policy_bundle_revision,
                event=request.get("event"),
            )
            submission = self._scheduler.submit(typed_event)
        except (TypeError, ValueError, OSError, RuntimeError, sqlite3.Error):
            return {"ok": False, "error": "local service host state rejected"}
        finally:
            with self._lifecycle_lock:
                self._active_requests = max(0, self._active_requests - 1)
                self._last_work_at = time.monotonic()
        return {
            "ok": True,
            "replayed": submission.replayed,
            "event_id_sha256": submission.event_id_sha256,
            "event_fingerprint_sha256": submission.event_fingerprint_sha256,
            "policy_bundle_revision": lease.policy_bundle_revision,
            "catalog_revision": lease.catalog_revision,
            "decision_revision": submission.plan.decision_revision,
            "scheduler_plan_sha256": submission.plan.plan_sha256,
            "matched_rule_sha256s": list(submission.plan.matched_rule_sha256s),
            "matched_action_sha256s": list(submission.plan.matched_action_sha256s),
            "deferred_trigger_owner_refs": list(submission.plan.deferred_trigger_owner_refs),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "kind": job.job_kind,
                    "status": job.status,
                    "reason_code": job.reason_code,
                }
                for job in submission.jobs
            ],
        }

    def _frame_error(self, request: dict[str, object]) -> str | None:
        if request.get("protocol_version") != self.identity.protocol_version:
            return "local service protocol version mismatch"
        if request.get("schema_version") != LOCAL_SERVICE_SCHEMA_VERSION:
            return "local service schema version mismatch"
        if request.get("api_version") != LOCAL_SERVICE_API_VERSION:
            return "local service API version mismatch"
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return "local service request_id is required"
        return None

    def _lease_for(self, request: dict[str, object]) -> _ListenerLease | None:
        token = request.get("lease")
        if not isinstance(token, str) or not token:
            return None
        with self._leases_lock:
            self._expire_leases()
            return self._leases.get(token)

    def _expire_leases(self) -> None:
        now = time.monotonic()
        for token, lease in tuple(self._leases.items()):
            if lease.expires_at <= now:
                self._leases.pop(token, None)

    def _dispatch(self) -> None:
        while True:
            self._dispatch_slots.acquire()
            item = None
            received = False
            submitted = False
            try:
                item = self._requests.get()
                received = True
                if item is None:
                    return
                future: Future[None] = Future()
                worker = threading.Thread(
                    target=self._run_dispatch_item,
                    args=(item, future),
                    name="um-mem-dispatch-worker",
                    daemon=True,
                )
                worker.start()
                submitted = True
                future.add_done_callback(self._observe_dispatch_future)
            except Exception:
                _LOG.exception("Local service dispatch submission failed")
                if item is not None and not submitted:
                    item.result.clear()
                    item.result.update(
                        {
                            "ok": False,
                            "error": "local service dispatcher failed",
                            "reason_code": "dispatcher_submission_error",
                            "request_id": item.request.get("request_id"),
                            "retryable": False,
                            "durable_status": "unknown",
                        }
                    )
                    with self._metrics_lock:
                        self._failed_calls += 1
                    with self._lifecycle_lock:
                        self._active_requests = max(0, self._active_requests - 1)
                        self._last_work_at = time.monotonic()
            finally:
                if received:
                    self._requests.task_done()
                if not submitted:
                    self._dispatch_slots.release()
                    if item is not None:
                        item.completed.set()

    def _run_dispatch_item(self, item: _DispatchItem, future: Future[None]) -> None:
        try:
            self._dispatch_item(item)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(None)
        finally:
            backend = getattr(self.client, "_backend", None)
            index = getattr(backend, "_index", None)
            close_thread_connection = getattr(index, "close_thread_connection", None)
            if callable(close_thread_connection):
                try:
                    close_thread_connection()
                except (OSError, RuntimeError, sqlite3.Error):
                    _LOG.exception("Local service dispatch worker connection cleanup failed")

    @staticmethod
    def _observe_dispatch_future(future: Future[None]) -> None:
        try:
            failure = future.exception()
        except CancelledError:
            _LOG.error("Local service dispatch worker was cancelled")
            return
        if failure is not None:
            _LOG.error(
                "Local service dispatch worker terminated unexpectedly",
                exc_info=(type(failure), failure, failure.__traceback__),
            )

    def _dispatch_item(
        self,
        item: _DispatchItem,
    ) -> None:
        lease, request, completed, result = (
            item.lease,
            item.request,
            item.completed,
            item.result,
        )

        def begin_durable_commit() -> bool:
            with item.lock:
                if item.cancelled:
                    return False
                item.commit_started = True
                return True

        try:
            with item.lock:
                if item.cancelled:
                    result.update(
                        {
                            "ok": False,
                            "error": "local service request cancelled before dispatch",
                            "reason_code": "dispatch_cancelled",
                            "request_id": request.get("request_id"),
                            "durable_status": "not_started",
                            "retryable": True,
                        }
                    )
                    return
                item.started = True
            assert isinstance(lease, _ListenerLease)
            dispatch_client = self.client
            if isinstance(self.client, MemoryClient):
                lineage = lease.scope_lineage
                dispatch_client = MemoryClient(
                    self.client.repo_root,
                    workspace_id=lineage.workspace_id,
                    profile_id=self.client.profile_id,
                    user_id=lineage.user_id,
                    project_id=lineage.project_id,
                    session_id=lineage.session_id,
                    task_id=lineage.task_id,
                    subtask_id=lineage.subtask_id,
                    backend=lease.identity.backend,
                    backend_instance=self.client.backend,
                )
            with use_catalog_snapshot(
                lease.identity.repo_root,
                lease.identity.workspace_id,
                lease.catalog_snapshot,
                profile_id=lease.identity.profile_id,
            ):
                response = _dispatch_local_service_request(
                    lease.identity,
                    json.dumps(request, ensure_ascii=False),
                    client=dispatch_client,
                    expected_lease=lease.token,
                    trusted_scope_lineage=lease.scope_lineage,
                    before_durable_commit=begin_durable_commit,
                )
            payload = response.get("payload")
            if isinstance(payload, dict):
                result.update(payload)
                if result.get("ok") is True and request.get("tool") in {
                    "observe",
                    "remember",
                    "amend",
                }:
                    self._enqueue_index_refresh(request, result)
                if result.get("ok") is True and request.get("tool") in {
                    "recall",
                    "build_context",
                }:
                    revisions = _cache_revisions(lease, self.health(force_native_probe=True))
                    if revisions is not None:
                        result["cache_revisions"] = revisions
            else:
                result.update({"ok": False, "error": "local service response rejected"})
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _LOG.exception(
                "Local service dispatch failed for request %s",
                request.get("request_id"),
            )
            if isinstance(exc, PermissionError):
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
                reason_code = "dispatcher_operation_error"
            result.clear()
            result.update(
                {
                    "ok": False,
                    "error": "local service dispatcher failed",
                    "error_type": type(exc).__name__,
                    "reason_code": reason_code,
                    "request_id": request.get("request_id"),
                    "durable_status": "unknown",
                    "retryable": False,
                }
            )
        except Exception:
            _LOG.exception(
                "Unexpected local service dispatch failure for request %s",
                request.get("request_id"),
            )
            result.clear()
            result.update(
                {
                    "ok": False,
                    "error": "local service dispatcher failed",
                    "reason_code": "dispatcher_unexpected_error",
                    "request_id": request.get("request_id"),
                    "retryable": False,
                    "durable_status": "unknown",
                }
            )
        finally:
            if not result:
                result.update(
                    {
                        "ok": False,
                        "error": "local service dispatcher failed",
                        "reason_code": "dispatcher_unexpected_error",
                        "request_id": request.get("request_id"),
                        "retryable": False,
                        "durable_status": "unknown",
                    }
                )
            with self._metrics_lock:
                if result.get("ok") is True:
                    self._successful_calls += 1
                else:
                    self._failed_calls += 1
            with self._lifecycle_lock:
                self._active_requests = max(0, self._active_requests - 1)
                self._last_work_at = time.monotonic()
            self._dispatch_slots.release()
            try:
                arguments_value = request.get("arguments")
                self.dashboard.observe(
                    str(request.get("tool") or ""),
                    arguments_value if isinstance(arguments_value, dict) else {},
                    result,
                    lease.scope_lineage,
                )
            except (TypeError, ValueError, AttributeError):
                _LOG.exception("Panel observation failed; memory receipt is preserved")
            finally:
                completed.set()

    def _enqueue_index_refresh(
        self,
        _request: dict[str, object],
        result: dict[str, object],
    ) -> None:
        memory_id = str(result.get("memory_id") or "").strip()
        if not memory_id:
            return
        try:
            receipt = self._work_queue.submit(
                "indexing",
                work_ref=memory_id,
                work_revision=memory_id,
            )
        except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
            with self._lifecycle_lock:
                self._work_queue_error = "local service background queue unavailable"
            result["background_work"] = {
                "lane": "indexing",
                "status": "admission_failed",
                "no_silent_drop": False,
            }
            return
        if receipt.status == "backpressured":
            result["background_work"] = {
                "lane": receipt.lane,
                "work_id": receipt.work_id,
                "admission_id": receipt.admission_id,
                "status": receipt.status,
                "reason": receipt.reason,
                "no_silent_drop": True,
            }


class _LocalServiceListenerServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _listener_handler(runtime: _LocalServiceListenerRuntime):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?", 1)[0].startswith("/panel"):
                runtime.dashboard.handle(self)
            else:
                self.send_error(404)

        def finish(self) -> None:
            try:
                super().finish()
            finally:
                backend = getattr(runtime.client, "_backend", None)
                index = getattr(backend, "_index", backend)
                close_thread_connection = getattr(index, "close_thread_connection", None)
                if callable(close_thread_connection):
                    try:
                        close_thread_connection()
                    except (OSError, RuntimeError, sqlite3.Error):
                        _LOG.exception("Local service HTTP handler connection cleanup failed")

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0].startswith("/panel/"):
                runtime.dashboard.handle(self)
                return
            request = self._request_payload()
            if request is None:
                return
            if self.path == "/health":
                self._send(runtime.health())
                return
            if self.path == "/lease":
                self._send(runtime.issue_lease(request))
                return
            if self.path == "/call":
                self._send(runtime.call(request))
                return
            if self.path == "/host-state":
                self._send(runtime.submit_host_state(request))
                return
            if self.path == "/shutdown":
                result = _listener_shutdown_request(runtime, request)
                self._send(result)
                if result.get("ok") is True:
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._send({"ok": False, "error": "local service endpoint is unknown"})

        def log_message(self, _format: str, *args: object) -> None:
            del args

        def _request_payload(self) -> dict[str, object] | None:
            content_length = self.headers.get("Content-Length", "")
            try:
                size = int(content_length)
            except ValueError:
                self._send({"ok": False, "error": "local service request length invalid"})
                return None
            if size < 0 or size > LOCAL_SERVICE_MAX_REQUEST_BYTES:
                self._send({"ok": False, "error": "local service request exceeds protocol limit"})
                return None
            body = self.rfile.read(size)
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send({"ok": False, "error": "local service request is invalid"})
                return None
            if not isinstance(payload, dict):
                self._send({"ok": False, "error": "local service request must be an object"})
                return None
            return payload

        def _send(self, payload: dict[str, object]) -> None:
            serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if len(serialized) > LOCAL_SERVICE_MAX_RESPONSE_BYTES:
                serialized = json.dumps(
                    {"ok": False, "error": "local service response exceeds protocol limit"}
                ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(serialized)))
            self.end_headers()
            self.wfile.write(serialized)

    return Handler


def _listener_shutdown_request(
    runtime: _LocalServiceListenerRuntime,
    request: dict[str, object],
) -> dict[str, object]:
    frame_error = runtime._frame_error(request)
    if frame_error is not None:
        return {"ok": False, "error": frame_error}
    if request.get("caller_role") != "operator":
        return {"ok": False, "error": "local service shutdown requires operator role"}
    operator_identity = build_local_service_identity(
        repo_root=runtime.identity.repo_root,
        workspace_id=runtime.identity.workspace_id,
        profile_id=runtime.identity.profile_id,
        backend=runtime.identity.backend,
        caller_role="operator",
    )
    secret = request.get("bootstrap_secret")
    if not isinstance(secret, str) or not hmac.compare_digest(
        local_service_bootstrap_secret(operator_identity),
        secret,
    ):
        return {"ok": False, "error": "local service bootstrap authentication rejected"}
    runtime.begin_shutdown()
    return {"ok": True, "state": "shutting_down"}


def _local_listener_configuration(args: list[str]) -> tuple[LocalServiceIdentity, int]:
    if len(args) != 11 or args[0] != "--local-service-listener":
        raise ValueError("invalid local service listener arguments")
    values = dict(zip(args[1::2], args[2::2], strict=True))
    if set(values) != {
        "--repo-root",
        "--workspace-id",
        "--profile-id",
        "--backend",
        "--owner-pid",
    }:
        raise ValueError("local service listener requires fixed identity arguments")
    try:
        owner_pid = int(values["--owner-pid"])
    except ValueError as exc:
        raise ValueError("local service listener owner pid is invalid") from exc
    if owner_pid <= 0:
        raise ValueError("local service listener owner pid is invalid")
    return (
        build_local_service_identity(
            repo_root=values["--repo-root"],
            workspace_id=values["--workspace-id"],
            profile_id=values["--profile-id"],
            backend=values["--backend"],
            caller_role="operator",
        ),
        owner_pid,
    )


def _listener_lifecycle_monitor(
    server: _LocalServiceListenerServer,
    runtime: _LocalServiceListenerRuntime,
    *,
    owner_pid: int,
    stopped: threading.Event,
) -> None:
    """Stop an orphaned or idle detached listener without touching active hosts."""

    while not stopped.wait(0.25):
        runtime.dashboard.expire_if_idle()
        if not _process_is_alive(owner_pid) or runtime.is_idle():
            runtime.begin_shutdown()
            server.shutdown()
            return


def _reap_idle_model() -> None:
    """Release an embedding model that nobody used for a while.

    The model is a few gigabytes of resident memory and is only useful while a
    question is being scored. A long-lived service keeps it loaded only
    between the questions that need it.
    """

    from um_mem.memory import semantic as semantic_tier

    while True:
        time.sleep(15)
        idle = semantic_tier.idle_release_seconds()
        if idle <= 0:
            continue
        for embedder in semantic_tier.known_embedders():
            try:
                embedder.release_if_idle(idle)
            except Exception:  # noqa: S112 - best-effort idle release
                continue


def _warm_local_tier(identity: Any) -> None:
    """Load the embedding model in the background for this listener.

    The local retrieval tier runs inside the listener, so the model is loaded
    once per listener instead of once per call. Loading it in the background
    keeps the first question from paying the whole cost.
    """

    def warm() -> None:
        from um_mem.memory import semantic as semantic_tier

        if not semantic_tier.semantic_enabled():
            return
        if not semantic_tier.semantic_warmup_enabled():
            return
        embedder = semantic_tier.get_embedder(
            cache_dir=semantic_tier.default_cache_dir(identity.repo_root)
        )
        if embedder is None:
            return
        try:
            embedder.embed(["um-mem listener warmup"], kind="query")
        except Exception:  # pragma: no cover - optional dependency
            return

    threading.Thread(
        target=warm,
        name="um-mem-local-tier-warmup",
        daemon=True,
    ).start()
    threading.Thread(
        target=_reap_idle_model,
        name="um-mem-local-tier-reaper",
        daemon=True,
    ).start()


def _run_local_service_listener(args: list[str]) -> int:
    try:
        identity, owner_pid = _local_listener_configuration(args)
    except ValueError:
        return 2
    try:
        with persistent_mcp_client(
            repo_root=identity.repo_root,
            workspace_id=identity.workspace_id,
            profile_id=identity.profile_id,
            backend=identity.backend,
        ) as client:
            _warm_local_tier(identity)
            runtime = _LocalServiceListenerRuntime(identity, client)
            server = _LocalServiceListenerServer(
                (LOCAL_SERVICE_LISTENER_HOST, 0),
                _listener_handler(runtime),
            )
            runtime.start()
            write_local_service_registry(
                identity,
                port=server.server_port,
                pid=os.getpid(),
                owner_pid=owner_pid,
            )
            monitor_stopped = threading.Event()
            monitor = threading.Thread(
                target=_listener_lifecycle_monitor,
                args=(server, runtime),
                kwargs={"owner_pid": owner_pid, "stopped": monitor_stopped},
                daemon=True,
            )
            monitor.start()
            try:
                server.serve_forever(poll_interval=0.1)
            finally:
                monitor_stopped.set()
                monitor.join(timeout=1)
                server.server_close()
                runtime.close()
    finally:
        clear_local_service_registry(identity, pid=os.getpid())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "--local-service-listener":
        return _run_local_service_listener(args)
    print(
        json.dumps(
            {
                "ok": False,
                "error": "mcp_call only supports the authenticated local service listener",
            }
        ),
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
