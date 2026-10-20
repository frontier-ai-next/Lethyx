"""Canonical content-free transport for native host lifecycle hooks."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lethyx.memory.local_service import (
    LocalServiceClientPool,
    LocalServiceIdentity,
    build_local_service_identity,
    local_service_scope_lineage,
)
from lethyx.memory.scopes import ScopeLineage


class HostHookClientPool(Protocol):
    def call(
        self,
        identity: LocalServiceIdentity,
        tool: str,
        arguments: Mapping[str, object],
        *,
        timeout_seconds: int,
        mutation_correlation_id: str = "",
        session_id: str = "",
        scope_lineage: ScopeLineage | None = None,
    ) -> dict[str, object]: ...

    def submit_host_state(
        self,
        identity: LocalServiceIdentity,
        event: Mapping[str, object],
        *,
        timeout_seconds: int,
        session_id: str = "",
    ) -> dict[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HostHookEvent:
    host_slug: str
    host_label: str
    adapter_id: str
    capability_code: str
    schema_version: int
    event_name: str
    session_id: str
    identity_fields: tuple[tuple[str, str], ...] = ()
    stage_kind: str = ""
    stage_id: str = ""
    subtask_kind: str = ""
    subtask_id: str = ""


def capture_host_hook(
    event: HostHookEvent,
    *,
    repo_root: Path,
    workspace_id: str | None,
    profile_id: str | None = None,
    backend: str | None,
    scheduler_event_type: str,
    timeout_seconds: int,
    pool: HostHookClientPool | None,
) -> dict[str, object]:
    """Persist one filtered lifecycle fact through exactly one local service writer."""

    try:
        identity = build_local_service_identity(
            repo_root=repo_root,
            workspace_id=workspace_id,
            profile_id=profile_id,
            backend=backend,
            caller_role="host_hook",
        )
        state_identity = build_local_service_identity(
            repo_root=identity.repo_root,
            workspace_id=identity.workspace_id,
            profile_id=identity.profile_id,
            backend=identity.backend,
            caller_role="host_state",
        )
    except ValueError:
        return {"ok": False, "error": "local service identity rejected"}

    event_digest = _event_digest(event)
    memory_arguments = _memory_arguments(event, event_digest)
    subtask_id = str(memory_arguments["subtask_id"])
    scope_lineage = local_service_scope_lineage(
        identity,
        session_id=event.session_id,
        task_id=(
            _scoped_identifier(event.host_slug, "session-task", event.session_id)
            if subtask_id
            else ""
        ),
        subtask_id=subtask_id,
    )
    client_pool = pool if pool is not None else LocalServiceClientPool()
    owns_pool = pool is None
    try:
        state_response = client_pool.submit_host_state(
            state_identity,
            _host_state_arguments(event, event_digest, scheduler_event_type),
            timeout_seconds=timeout_seconds,
            session_id=event.session_id,
        )
        if state_response.get("ok") is not True:
            return {"ok": False, "error": "local service host-state scheduling failed"}
        response = client_pool.call(
            identity,
            "remember",
            memory_arguments,
            timeout_seconds=timeout_seconds,
            mutation_correlation_id=f"{event.host_slug}-hook:{event_digest}",
            session_id=event.session_id,
            scope_lineage=scope_lineage,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return {"ok": False, "error": "local service host capture failed"}
    finally:
        if owns_pool:
            client_pool.close()
    if response.get("ok") is not True:
        return {"ok": False, "error": "local service host capture failed"}
    result: dict[str, object] = {
        "ok": True,
        "captured": True,
        "event": event.event_name,
    }
    memory_id = response.get("memory_id")
    if isinstance(memory_id, str) and memory_id:
        result["memory_id"] = memory_id
    return result


def git_root(working_directory: Path) -> Path:
    for candidate in (working_directory, *working_directory.parents):
        if (candidate / ".git").exists():
            return candidate
    return working_directory


def _event_digest(event: HostHookEvent) -> str:
    material = {
        "event_name": event.event_name,
        "identity_fields": event.identity_fields,
        "schema_version": event.schema_version,
        "session_id": event.session_id,
        "stage_id": event.stage_id,
        "subtask_id": event.subtask_id,
    }
    encoded = json.dumps(material, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _scoped_identifier(host_slug: str, kind: str, value: str) -> str:
    if not kind or not value:
        return ""
    return f"{host_slug}-{kind}:{_identifier_digest(value)}"


def _memory_arguments(event: HostHookEvent, event_digest: str) -> dict[str, object]:
    return {
        "content": f"Observed {event.host_label} lifecycle event: {event.event_name}.",
        "title": f"{event.host_label} {event.event_name}",
        "kind": "observation",
        "requested_block_id": "workspace_episodes",
        "tags": [event.host_slug, "host_hook", "observed_event", event.event_name.lower()],
        "source_id": f"{event.host_slug}-hook:{event.event_name}:{event_digest[:24]}",
        "run_id": f"{event.host_slug}-session:{_identifier_digest(event.session_id)}",
        "stage_id": _scoped_identifier(event.host_slug, event.stage_kind, event.stage_id),
        "subtask_id": _scoped_identifier(
            event.host_slug,
            event.subtask_kind,
            event.subtask_id,
        ),
        "metadata": {
            "host_adapter": event.adapter_id,
            "host_event_name": event.event_name,
            "host_event_schema_version": event.schema_version,
            "host_identity_sha256": event_digest,
        },
    }


def _host_state_arguments(
    event: HostHookEvent,
    event_digest: str,
    scheduler_event_type: str,
) -> dict[str, object]:
    stage_id = _scoped_identifier(event.host_slug, event.stage_kind, event.stage_id)
    return {
        "event_id": f"{event.host_slug}-host-state:{event_digest}",
        "event_type": scheduler_event_type,
        "run_id": f"{event.host_slug}-session:{_identifier_digest(event.session_id)}",
        "stage_id": stage_id or f"{event.host_slug}-session-boundary",
        "signal_codes": (
            f"{event.host_slug}_event:{event.event_name}",
            f"schema:{event.schema_version}",
        ),
        "capability_codes": (event.capability_code, "typed_host_state"),
        "retry_count": 0,
    }
