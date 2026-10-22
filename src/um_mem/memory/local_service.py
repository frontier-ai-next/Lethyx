"""Identity and transport ownership for the local UM-MEM service."""

import errno
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from um_mem.contracts.memory import normalize_identity_path
from um_mem.memory.backends.factory import selected_memory_backend_mode
from um_mem.memory.paths import normalize_workspace_id
from um_mem.memory.scopes import ScopeLineage

LOCAL_SERVICE_PROTOCOL_VERSION = 1
LOCAL_SERVICE_API_VERSION = 2
LOCAL_SERVICE_SCHEMA_VERSION = 2
LOCAL_SERVICE_MAX_REQUEST_BYTES = 1_048_576
LOCAL_SERVICE_MAX_RESPONSE_BYTES = 1_048_576
LOCAL_SERVICE_LISTENER_HOST = "127.0.0.1"
LOCAL_SERVICE_LISTENER_TRANSPORT = "loopback_http"
LOCAL_SERVICE_LISTENER_QUEUE_SIZE = 32
LOCAL_SERVICE_LEASE_TTL_SECONDS = 120
LOCAL_SERVICE_IDLE_TIMEOUT_SECONDS = 300
_LOCAL_SERVICE_HEALTH_PROBE_ATTEMPTS = 2
_LOCAL_SERVICE_HEALTH_RETRY_SECONDS = 0.05
_LOCAL_SERVICE_ENDPOINT_PROBE_TIMEOUT_SECONDS = 1
_LOCAL_SERVICE_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_SERVICE_GUI_BOOTSTRAP = (
    "import runpy, site, sys;"
    "site.addsitedir(sys.argv[1]);"
    "sys.path.insert(0, sys.argv[2]);"
    "module = sys.argv[3];"
    "sys.argv = [module, *sys.argv[4:]];"
    "runpy.run_module(module, run_name='__main__')"
)
_LOCAL_SERVICE_MUTATION_TOOLS = frozenset(
    {
        "observe",
        "remember",
        "amend",
        "promote",
        "create_block",
        "convert_block_to_index",
        "amend_block",
        "move_block",
        "archive_block",
        "revive_block",
        "delete_block",
        "split_block",
        "merge_blocks",
        "resume_topology_evolution",
        "abort_topology_evolution",
        "add_block_edge",
        "remove_block_edge",
        "set_active_mask",
        "amend_active_mask",
        "transition_active_mask",
        "observe_maintenance",
        "consolidate_observations",
    }
)
_LOCAL_SERVICE_ROLE_TOOLS: dict[str, frozenset[str]] = {
    "parent": frozenset(
        {
            "health",
            "block_map",
            "expand_block",
            "navigation_envelope",
            "create_block",
            "amend_block",
            "move_block",
            "archive_block",
            "revive_block",
            "delete_block",
            "split_block",
            "merge_blocks",
            "pending_topology_evolutions",
            "resume_topology_evolution",
            "abort_topology_evolution",
            "add_block_edge",
            "remove_block_edge",
            "get_active_mask",
            "set_active_mask",
            "amend_active_mask",
            "transition_active_mask",
            "resolve_active_mask",
            "observe_maintenance",
            "recall",
            "procedure",
            "memory_map",
            "read_working_set",
            "read_exact_record",
            "read_exact_records",
            "list_records",
            "operation_status",
            "projection_barrier",
            "build_context",
            "build_granted_context",
            "remember",
            "amend",
            "forget",
            "export_graph",
            "export_obsidian",
        }
    ),
    "supervisor": frozenset(
        {
            "health",
            "block_map",
            "expand_block",
            "navigation_envelope",
            "create_block",
            "amend_block",
            "move_block",
            "archive_block",
            "revive_block",
            "delete_block",
            "split_block",
            "merge_blocks",
            "pending_topology_evolutions",
            "resume_topology_evolution",
            "abort_topology_evolution",
            "add_block_edge",
            "remove_block_edge",
            "get_active_mask",
            "set_active_mask",
            "amend_active_mask",
            "transition_active_mask",
            "resolve_active_mask",
            "observe_maintenance",
            "consolidate_observations",
            "convert_block_to_index",
            "recall",
            "procedure",
            "memory_map",
            "recent",
            "read_working_set",
            "read_exact_record",
            "read_exact_records",
            "list_records",
            "operation_status",
            "projection_barrier",
            "build_context",
            "build_granted_context",
            "remember",
            "amend",
            "promote",
            "commit_working_checkpoint",
            "verification_subject",
            "forget",
            "add_edge",
            "rebuild_indexes",
        }
    ),
    "operator": frozenset(
        {
            "health",
            "block_map",
            "expand_block",
            "navigation_envelope",
            "create_block",
            "amend_block",
            "move_block",
            "archive_block",
            "revive_block",
            "delete_block",
            "split_block",
            "merge_blocks",
            "pending_topology_evolutions",
            "resume_topology_evolution",
            "abort_topology_evolution",
            "add_block_edge",
            "remove_block_edge",
            "get_active_mask",
            "set_active_mask",
            "amend_active_mask",
            "transition_active_mask",
            "resolve_active_mask",
            "observe_maintenance",
            "consolidate_observations",
            "convert_block_to_index",
            "remember",
            "amend",
            "recall",
            "procedure",
            "memory_map",
            "recent",
            "read_working_set",
            "read_exact_record",
            "read_exact_records",
            "list_records",
            "operation_status",
            "projection_barrier",
            "build_context",
            "build_granted_context",
            "forget",
            "add_edge",
            "export_graph",
            "export_obsidian",
        }
    ),
    "host_hook": frozenset({"health", "observe", "remember", "operation_status"}),
    "host_state": frozenset({"health"}),
}
_LOCAL_SERVICE_PRIVATE_CAPABILITIES: dict[str, frozenset[str]] = {
    "host_state": frozenset({"host_state"}),
}


@dataclass(frozen=True)
class LocalServiceIdentity:
    repo_root: Path
    workspace_id: str
    profile_id: str
    backend: str
    caller_role: str = "operator"
    protocol_version: int = LOCAL_SERVICE_PROTOCOL_VERSION

    def bound_arguments(self, arguments: Mapping[str, object]) -> dict[str, object]:
        payload = dict(arguments)
        payload.update(
            {
                "repo_root": str(self.repo_root),
                "workspace_id": self.workspace_id,
                "backend": self.backend,
            }
        )
        return payload


def local_service_scope_lineage(
    identity: LocalServiceIdentity,
    *,
    session_id: str,
    project_id: str = "",
    task_id: str = "",
    subtask_id: str = "",
    user_id: str = "",
) -> ScopeLineage:
    """Build trusted scope lineage for one authenticated service lease."""

    if not isinstance(identity, LocalServiceIdentity):
        raise TypeError("identity must be LocalServiceIdentity")
    bounded_session = _local_service_session_id(session_id)
    if bounded_session is None:
        raise ValueError("local service session identity is invalid")
    return ScopeLineage(
        workspace_id=identity.workspace_id,
        project_id=project_id,
        session_id=bounded_session,
        task_id=task_id,
        subtask_id=subtask_id,
        user_id=user_id,
    )


def _scope_lineage_payload(lineage: ScopeLineage) -> dict[str, str]:
    return {
        "workspace_id": lineage.workspace_id,
        "project_id": lineage.project_id,
        "session_id": lineage.session_id,
        "task_id": lineage.task_id,
        "subtask_id": lineage.subtask_id,
        "user_id": lineage.user_id,
    }


def build_local_service_identity(
    *,
    repo_root: str | Path | None,
    workspace_id: str | None,
    profile_id: str | None = None,
    backend: str | None,
    caller_role: str | None = None,
) -> LocalServiceIdentity:
    root = _resolve_repo_root(repo_root)
    workspace = _resolve_workspace_id(workspace_id)
    profile = _resolve_profile_id(profile_id)
    selected_backend = selected_memory_backend_mode(backend)
    role = _resolve_caller_role(caller_role)
    return LocalServiceIdentity(
        repo_root=root,
        workspace_id=workspace,
        profile_id=profile,
        backend=selected_backend,
        caller_role=role,
    )


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None and not isinstance(repo_root, (str, Path)):
        raise ValueError("repo_root must be a path string")
    raw = str(repo_root or os.getenv("UM_MEM_REPO_ROOT") or "").strip()
    return Path(normalize_identity_path(raw or Path.cwd(), field_name="repo_root"))


def _resolve_workspace_id(workspace_id: str | None) -> str:
    if workspace_id is not None and not isinstance(workspace_id, str):
        raise ValueError("workspace_id must be a string")
    raw = str(workspace_id or os.getenv("UM_MEM_WORKSPACE") or "default").strip()
    workspace = normalize_workspace_id(raw)
    if not workspace:
        raise ValueError("workspace_id is required")
    return workspace


def _resolve_profile_id(profile_id: str | None) -> str:
    if profile_id is not None and not isinstance(profile_id, str):
        raise ValueError("profile_id must be a string")
    profile = str(profile_id or os.getenv("UM_MEM_PROFILE") or "default").strip()
    if (
        not profile
        or profile in {".", ".."}
        or any(separator in profile for separator in ("/", "\\", "\x00"))
    ):
        raise ValueError("profile_id is invalid")
    return profile


def local_service_role_allows(caller_role: str, tool: str) -> bool:
    return str(tool or "") in _LOCAL_SERVICE_ROLE_TOOLS.get(str(caller_role or ""), ())


def local_service_role_capabilities(caller_role: str) -> tuple[str, ...]:
    """Return the endpoint capabilities an authenticated role must negotiate."""

    role = str(caller_role or "")
    return tuple(
        sorted(
            frozenset(_LOCAL_SERVICE_ROLE_TOOLS.get(role, ()))
            | _LOCAL_SERVICE_PRIVATE_CAPABILITIES.get(role, frozenset())
        )
    )


def _resolve_caller_role(caller_role: str | None) -> str:
    if caller_role is not None and not isinstance(caller_role, str):
        raise ValueError("caller_role must be a string")
    role = str(caller_role or "operator").strip().lower()
    if role not in _LOCAL_SERVICE_ROLE_TOOLS:
        raise ValueError("unknown local service caller role")
    return role


@dataclass(frozen=True)
class LocalServiceRuntimePaths:
    """Content-free local files that locate one listener for a workspace."""

    registry: Path
    bootstrap: Path
    startup_lock: Path


def _local_service_host_scope() -> tuple[str, ...]:
    if sys.platform == "linux":
        return (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
            os.readlink("/proc/self/ns/pid"),
            os.readlink("/proc/self/ns/net"),
        )
    return (sys.platform, socket.gethostname())


def local_service_runtime_paths(identity: LocalServiceIdentity) -> LocalServiceRuntimePaths:
    runtime_dir = identity.repo_root / ".UM-MEM" / "memory"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    identity_key = hashlib.sha256(
        json.dumps(
            [identity.workspace_id, identity.profile_id, _local_service_host_scope()],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()[:20]
    return LocalServiceRuntimePaths(
        registry=runtime_dir / f"local_service_{identity_key}.json",
        bootstrap=runtime_dir / f"local_service_{identity_key}.bootstrap.json",
        startup_lock=runtime_dir / f"local_service_{identity_key}.startup.lock",
    )


def local_service_bootstrap_secret(identity: LocalServiceIdentity) -> str:
    """Return the role-bound local bootstrap credential without exposing it elsewhere."""

    paths = local_service_runtime_paths(identity)
    credentials = _read_local_json(paths.bootstrap)
    role = identity.caller_role
    if isinstance(credentials, dict):
        secrets_by_role = credentials.get("secrets")
        if isinstance(secrets_by_role, dict):
            secret = secrets_by_role.get(role)
            if isinstance(secret, str) and secret:
                return secret
    secrets_by_role = {
        name: secrets.token_urlsafe(32) for name in sorted(_LOCAL_SERVICE_ROLE_TOOLS)
    }
    _write_local_json(
        paths.bootstrap,
        {
            "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
            "workspace_id": identity.workspace_id,
            "profile_id": identity.profile_id,
            "backend": identity.backend,
            "secrets": secrets_by_role,
        },
        private=True,
    )
    return secrets_by_role[role]


def local_service_registry(identity: LocalServiceIdentity) -> dict[str, object] | None:
    registry = _read_local_json(local_service_runtime_paths(identity).registry)
    if not isinstance(registry, dict):
        return None
    if (
        registry.get("workspace_id") != identity.workspace_id
        or registry.get("profile_id") != identity.profile_id
        or registry.get("transport") != LOCAL_SERVICE_LISTENER_TRANSPORT
        or registry.get("host") != LOCAL_SERVICE_LISTENER_HOST
        or not isinstance(registry.get("port"), int)
        or not isinstance(registry.get("pid"), int)
    ):
        return None
    pid = registry["pid"]
    assert isinstance(pid, int)
    if not _process_is_alive(pid):
        clear_local_service_registry(identity, pid=pid)
        return None
    return registry


def write_local_service_registry(
    identity: LocalServiceIdentity,
    *,
    port: int,
    pid: int,
    owner_pid: int | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
        "api_version": LOCAL_SERVICE_API_VERSION,
        "protocol_version": identity.protocol_version,
        "host": LOCAL_SERVICE_LISTENER_HOST,
        "port": port,
        "pid": pid,
        "workspace_id": identity.workspace_id,
        "profile_id": identity.profile_id,
        "backend": identity.backend,
        "transport": LOCAL_SERVICE_LISTENER_TRANSPORT,
    }
    if isinstance(owner_pid, int) and owner_pid > 0:
        payload["owner_pid"] = owner_pid
    _write_local_json(
        local_service_runtime_paths(identity).registry,
        payload,
        private=False,
    )


def clear_local_service_registry(
    identity: LocalServiceIdentity,
    *,
    pid: int,
    port: int | None = None,
) -> None:
    paths = local_service_runtime_paths(identity)
    registry = _read_local_json(paths.registry)
    if (
        isinstance(registry, dict)
        and registry.get("pid") == pid
        and (port is None or registry.get("port") == port)
    ):
        try:
            paths.registry.unlink()
        except FileNotFoundError:
            pass


def local_service_identity_is_authorized(
    identity: LocalServiceIdentity,
    arguments: Mapping[str, object],
) -> bool:
    """Reject client identity claims before a listener can dispatch a tool."""

    repo_root = arguments.get("repo_root")
    workspace_id = arguments.get("workspace_id")
    backend = arguments.get("backend")
    if repo_root is not None and not isinstance(repo_root, (str, Path)):
        return False
    if workspace_id is not None and not isinstance(workspace_id, str):
        return False
    if backend is not None and not isinstance(backend, str):
        return False
    try:
        claimed = build_local_service_identity(
            repo_root=repo_root or identity.repo_root,
            workspace_id=workspace_id or identity.workspace_id,
            profile_id=identity.profile_id,
            backend=backend or identity.backend,
            caller_role=identity.caller_role,
        )
    except ValueError:
        return False
    return claimed == identity


@dataclass
class LocalServiceClientPool:
    """Client-side connection cache for the shared authenticated listener.

    Closing the pool discards only client leases. It never tears down the
    listener, so a short-lived host hook cannot terminate a shared service.
    """

    python_executable: str = sys.executable
    environment: dict[str, str] | None = None
    _leases: dict[tuple[LocalServiceIdentity, ScopeLineage], str] = field(
        default_factory=dict,
        init=False,
    )
    _request_sequence: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        # A POSIX venv launcher is often a symlink to the base interpreter. Its
        # invoked path selects site-packages, so dereferencing loses dependencies.
        self.python_executable = str(Path(self.python_executable).expanduser().absolute())
        if self.environment is not None:
            self.environment = dict(self.environment)

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
    ) -> dict[str, object]:
        payload = dict(arguments)
        if not local_service_identity_is_authorized(identity, payload):
            return {
                "ok": False,
                "error": "local service request identity is not authorized",
            }
        if not local_service_role_allows(identity.caller_role, tool):
            return {
                "ok": False,
                "error": "local service caller role is not allowed for tool",
            }
        bounded_session = _local_service_session_id(session_id or payload.get("run_id"))
        if bounded_session is None:
            return {"ok": False, "error": "local service session identity is invalid"}
        if scope_lineage is None:
            lease_lineage = local_service_scope_lineage(
                identity,
                session_id=bounded_session,
            )
        elif (
            not isinstance(scope_lineage, ScopeLineage)
            or scope_lineage.workspace_id != identity.workspace_id
            or scope_lineage.session_id != bounded_session
        ):
            return {"ok": False, "error": "local service scope lineage is invalid"}
        else:
            lease_lineage = scope_lineage
        effective_correlation_id = mutation_correlation_id
        if tool in _LOCAL_SERVICE_MUTATION_TOOLS and not effective_correlation_id:
            effective_correlation_id = f"local-service:{secrets.token_urlsafe(24)}"
        request_id = self._next_request_id()
        deadline_ns = time.monotonic_ns() + max(1, timeout_seconds) * 1_000_000_000

        def remaining_seconds() -> int:
            return max(1, (deadline_ns - time.monotonic_ns() + 999_999_999) // 1_000_000_000)

        def deadline_error() -> dict[str, object]:
            return {
                "ok": False,
                "error": "local service transport timed out",
                "reason_code": "transport_timeout",
                "request_id": request_id,
                "durable_status": "not_started",
                "retryable": True,
                **({"operation_id": effective_correlation_id} if effective_correlation_id else {}),
            }

        for attempt in range(2):
            if time.monotonic_ns() >= deadline_ns:
                return deadline_error()
            endpoint, error = self._ensure_listener(
                identity,
                timeout_seconds=remaining_seconds(),
                # A live dispatcher returns typed advisory failures. Rejecting its
                # readiness here would erase retryability before dispatch.
                allow_nonready_status=True,
                deadline_ns=deadline_ns,
            )
            if time.monotonic_ns() >= deadline_ns:
                return deadline_error()
            if error is not None:
                return {"ok": False, "error": error}
            assert endpoint is not None
            lease, lease_error = self._lease_for(
                identity,
                lease_lineage,
                endpoint,
                timeout_seconds=max(0.001, (deadline_ns - time.monotonic_ns()) / 1_000_000_000),
            )
            if time.monotonic_ns() >= deadline_ns:
                return deadline_error()
            if lease_error is not None:
                return {"ok": False, "error": lease_error}
            assert lease is not None
            request = {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": request_id,
                "lease": lease,
                "caller_role": identity.caller_role,
                "session_id": bounded_session,
                "tool": tool,
                "arguments": payload,
                "deadline_ns": deadline_ns,
            }
            if effective_correlation_id:
                request["mutation_correlation_id"] = effective_correlation_id
            if time.monotonic_ns() >= deadline_ns:
                return deadline_error()
            response = _local_service_http_request(
                endpoint,
                "/call",
                request,
                timeout_seconds=max(0.001, (deadline_ns - time.monotonic_ns()) / 1_000_000_000),
            )
            if effective_correlation_id and response.get("reason_code") == "transport_timeout":
                response = {
                    **response,
                    "operation_id": effective_correlation_id,
                    "request_id": request_id,
                    "durable_status": "unknown",
                }
            response_error = response.get("error")
            if (
                response_error
                not in {
                    "local service lease rejected",
                    "local service listener unavailable",
                }
                or attempt
            ):
                return response
            with self._lock:
                self._leases.pop((identity, lease_lineage), None)
            if response_error == "local service listener unavailable":
                pid = endpoint.get("pid")
                if isinstance(pid, int):
                    clear_local_service_registry(identity, pid=pid)
        return {"ok": False, "error": "local service listener unavailable"}

    def service_status(self, identity: LocalServiceIdentity) -> dict[str, object]:
        endpoint = local_service_registry(identity)
        if endpoint is None:
            return _unavailable_service_status(
                identity,
                state="not_ready",
                degraded_state="listener_unavailable",
                error="local service listener unavailable",
            )
        if endpoint.get("backend") != identity.backend:
            return _unavailable_service_status(
                identity,
                state="backend_conflict",
                degraded_state="backend_conflict",
                error="workspace already has a different selected memory backend",
            )
        status = _listener_health(endpoint, identity, timeout_seconds=2)
        if status is None:
            return _unavailable_service_status(
                identity,
                state="not_ready",
                degraded_state="listener_unavailable",
                error="local service listener unavailable",
            )
        return status

    def submit_host_state(
        self,
        identity: LocalServiceIdentity,
        event: Mapping[str, object],
        *,
        timeout_seconds: int,
        session_id: str = "",
    ) -> dict[str, object]:
        """Submit only typed host state to the listener-owned advisory scheduler."""

        if identity.caller_role != "host_state":
            return {"ok": False, "error": "local service host-state role is required"}
        if not isinstance(event, Mapping):
            return {"ok": False, "error": "local service host state is invalid"}
        bounded_session = _local_service_session_id(session_id)
        if bounded_session is None:
            return {"ok": False, "error": "local service session identity is invalid"}
        endpoint, error = self._ensure_listener(identity, timeout_seconds=timeout_seconds)
        if error is not None:
            return {"ok": False, "error": error}
        assert endpoint is not None
        lease_lineage = local_service_scope_lineage(
            identity,
            session_id=bounded_session,
        )
        lease, lease_error = self._lease_for(
            identity,
            lease_lineage,
            endpoint,
            timeout_seconds=timeout_seconds,
        )
        if lease_error is not None:
            return {"ok": False, "error": lease_error}
        assert lease is not None
        return _local_service_http_request(
            endpoint,
            "/host-state",
            {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": self._next_request_id(),
                "lease": lease,
                "caller_role": identity.caller_role,
                "session_id": bounded_session,
                "event": dict(event),
            },
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        with self._lock:
            self._leases.clear()

    def shutdown(self, identity: LocalServiceIdentity) -> dict[str, object]:
        endpoint = local_service_registry(identity)
        if endpoint is None:
            return {"ok": True, "state": "not_running"}
        if endpoint.get("backend") != identity.backend:
            return {
                "ok": False,
                "error": "workspace already has a different selected memory backend",
            }
        if identity.caller_role != "operator":
            return {"ok": False, "error": "local service shutdown requires operator role"}
        return _local_service_http_request(
            endpoint,
            "/shutdown",
            {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": self._next_request_id(),
                "caller_role": identity.caller_role,
                "bootstrap_secret": local_service_bootstrap_secret(identity),
            },
            timeout_seconds=5,
        )

    def _ensure_listener(
        self,
        identity: LocalServiceIdentity,
        *,
        timeout_seconds: int,
        allow_nonready_status: bool = False,
        deadline_ns: int | None = None,
    ) -> tuple[dict[str, object] | None, str | None]:
        def probe(endpoint: dict[str, object] | None) -> dict[str, object] | None:
            return (
                _listener_health(endpoint, identity, timeout_seconds=1, deadline_ns=deadline_ns)
                if endpoint
                else None
            )

        endpoint = local_service_registry(identity)
        if endpoint is not None and endpoint.get("backend") != identity.backend:
            return None, "workspace already has a different selected memory backend"
        status = probe(endpoint)
        if _listener_is_dispatch_ready(status):
            return endpoint, None
        if status is not None:
            if (
                allow_nonready_status
                and status.get("liveness") == "live"
                and status.get("state") != "draining"
            ):
                return endpoint, None
            return None, _listener_status_error(status)
        if endpoint is not None:
            if not _retire_refused_listener_endpoint(identity, endpoint, deadline_ns=deadline_ns):
                # A live process or an open port that cannot answer health
                # probes is not proof that it is safe to launch another
                # listener. Preserve its registry and fail closed.
                return None, "local service listener unavailable"
            endpoint = None
        with ExitStack() as lock_stack:
            if deadline_ns is None:
                lock_stack.enter_context(_LOCAL_SERVICE_START_LOCK)
            else:
                remaining = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
                if not _LOCAL_SERVICE_START_LOCK.acquire(timeout=remaining):
                    return None, "local service listener startup lock timed out"
                lock_stack.callback(_LOCAL_SERVICE_START_LOCK.release)
            endpoint = local_service_registry(identity)
            if endpoint is not None and endpoint.get("backend") != identity.backend:
                return None, "workspace already has a different selected memory backend"
            status = probe(endpoint)
            if _listener_is_dispatch_ready(status):
                return endpoint, None
            if status is not None:
                if (
                    allow_nonready_status
                    and status.get("liveness") == "live"
                    and status.get("state") != "draining"
                ):
                    return endpoint, None
                return None, _listener_status_error(status)
            if endpoint is not None:
                if not _retire_refused_listener_endpoint(
                    identity, endpoint, deadline_ns=deadline_ns
                ):
                    return None, "local service listener unavailable"
                endpoint = None
            descriptor = _acquire_listener_startup_lock(
                identity, timeout_seconds, deadline_ns=deadline_ns
            )
            if descriptor is None:
                return None, "local service listener startup lock timed out"
            try:
                endpoint = local_service_registry(identity)
                if endpoint is not None and endpoint.get("backend") != identity.backend:
                    return None, "workspace already has a different selected memory backend"
                status = probe(endpoint)
                if _listener_is_dispatch_ready(status):
                    return endpoint, None
                if status is not None:
                    if (
                        allow_nonready_status
                        and status.get("liveness") == "live"
                        and status.get("state") != "draining"
                    ):
                        return endpoint, None
                    return None, _listener_status_error(status)
                if endpoint is not None:
                    if not _retire_refused_listener_endpoint(
                        identity, endpoint, deadline_ns=deadline_ns
                    ):
                        return None, "local service listener unavailable"
                    endpoint = None
                local_service_bootstrap_secret(identity)
                try:
                    process = self._launch_listener(identity)
                except OSError as exc:
                    return None, f"local service listener startup failed: {type(exc).__name__}"
                deadline = time.monotonic() + max(2, timeout_seconds)
                if deadline_ns is not None:
                    deadline = min(deadline, deadline_ns / 1_000_000_000)
                while time.monotonic() < deadline:
                    endpoint = local_service_registry(identity)
                    status = probe(endpoint)
                    if _listener_is_dispatch_ready(status):
                        return endpoint, None
                    if status is not None:
                        if (
                            allow_nonready_status
                            and status.get("liveness") == "live"
                            and status.get("state") != "draining"
                        ):
                            return endpoint, None
                        return None, _listener_status_error(status)
                    exit_code = process.poll()
                    if exit_code is not None:
                        return None, (
                            "local service listener exited before readiness "
                            f"(exit code {exit_code})"
                        )
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            finally:
                _release_listener_startup_lock(identity, descriptor)
        return None, "local service listener startup timed out"

    def _launch_listener(self, identity: LocalServiceIdentity) -> subprocess.Popen[str]:
        environment = dict(self.environment) if self.environment is not None else os.environ.copy()
        environment["UM_MEM_PROFILE"] = identity.profile_id
        # pythonw owns no console and this HTTP listener has no stdio.  Keep
        # CREATE_NO_WINDOW separate: Windows ignores it when DETACHED_PROCESS
        # is combined with it.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        return subprocess.Popen(
            self._listener_command(identity),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            shell=False,
            close_fds=True,
            creationflags=creationflags,
            env=environment,
        )

    def _listener_python_executable(self) -> str:
        """Use pythonw for a detached Windows listener with no stdio contract."""

        if os.name != "nt":
            return self.python_executable
        executable = Path(self.python_executable)
        candidates: list[Path] = []
        home = self._virtual_environment_home()
        if home is not None and self._virtual_environment_site_packages() is not None:
            candidates.append(home / "pythonw.exe")
        candidates.append(executable.with_name("pythonw.exe"))
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        return self.python_executable

    def _listener_command(self, identity: LocalServiceIdentity) -> list[str]:
        listener_args = [
            "--local-service-listener",
            "--repo-root",
            str(identity.repo_root),
            "--workspace-id",
            identity.workspace_id,
            "--profile-id",
            identity.profile_id,
            "--backend",
            identity.backend,
            "--owner-pid",
            str(_listener_owner_pid(self.environment)),
        ]
        executable = self._listener_python_executable()
        site_packages = self._virtual_environment_site_packages()
        if (
            os.name == "nt"
            and Path(executable).resolve() != Path(self.python_executable).resolve()
            and site_packages is not None
        ):
            return [
                executable,
                "-c",
                _LOCAL_SERVICE_GUI_BOOTSTRAP,
                str(site_packages),
                str(_LOCAL_SERVICE_SOURCE_ROOT),
                "um_mem.memory.mcp_call",
                *listener_args,
            ]
        return [executable, "-m", "um_mem.memory.mcp_call", *listener_args]

    def _virtual_environment_home(self) -> Path | None:
        config = Path(self.python_executable).parent.parent / "pyvenv.cfg"
        try:
            for line in config.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip().casefold() == "home" and value.strip():
                    return Path(value.strip()).expanduser()
        except OSError:
            return None
        return None

    def _virtual_environment_site_packages(self) -> Path | None:
        site_packages = Path(self.python_executable).parent.parent / "Lib" / "site-packages"
        return site_packages if site_packages.is_dir() else None

    def _lease_for(
        self,
        identity: LocalServiceIdentity,
        scope_lineage: ScopeLineage,
        endpoint: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> tuple[str | None, str | None]:
        key = (identity, scope_lineage)
        with self._lock:
            cached_lease = self._leases.get(key)
        if cached_lease:
            return cached_lease, None
        response = _local_service_http_request(
            endpoint,
            "/lease",
            {
                "protocol_version": identity.protocol_version,
                "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
                "api_version": LOCAL_SERVICE_API_VERSION,
                "request_id": self._next_request_id(),
                "caller_role": identity.caller_role,
                "session_id": scope_lineage.session_id,
                "scope_lineage": _scope_lineage_payload(scope_lineage),
                "capabilities": list(local_service_role_capabilities(identity.caller_role)),
                "bootstrap_secret": local_service_bootstrap_secret(identity),
            },
            timeout_seconds=timeout_seconds,
        )
        lease = response.get("lease")
        if response.get("ok") is not True or not isinstance(lease, str) or not lease:
            return None, str(response.get("error") or "local service lease rejected")
        with self._lock:
            self._leases[key] = lease
        return lease, None

    def _next_request_id(self) -> str:
        with self._lock:
            self._request_sequence += 1
            return str(self._request_sequence)


def shutdown_local_service_listener(identity: LocalServiceIdentity) -> dict[str, object]:
    """Explicit test/operator lifecycle control; ordinary client close never stops it."""

    pool = LocalServiceClientPool()
    try:
        result = pool.shutdown(identity)
        if result.get("ok") is True and result.get("state") == "shutting_down":
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if local_service_registry(identity) is None:
                    break
                time.sleep(0.05)
        return result
    finally:
        pool.close()


_LOCAL_SERVICE_START_LOCK = threading.Lock()


def _acquire_listener_startup_lock(
    identity: LocalServiceIdentity,
    timeout_seconds: int,
    *,
    deadline_ns: int | None = None,
) -> int | None:
    path = local_service_runtime_paths(identity).startup_lock
    deadline = time.monotonic() + max(2, timeout_seconds)
    if deadline_ns is not None:
        deadline = min(deadline, deadline_ns / 1_000_000_000)
    while time.monotonic() < deadline:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = json.dumps(
                {"pid": os.getpid(), "created_at": time.time()},
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            except OSError:
                os.close(descriptor)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                raise
            return descriptor
        except FileExistsError:
            owner_pid = _listener_startup_lock_owner(path)
            if owner_pid is not None and not _process_is_alive(owner_pid):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                stale = time.time() - path.stat().st_mtime > max(30, timeout_seconds)
            except FileNotFoundError:
                continue
            if stale:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            time.sleep(0.05)
    return None


def _listener_startup_lock_owner(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    pid = payload.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def _release_listener_startup_lock(identity: LocalServiceIdentity, descriptor: int) -> None:
    try:
        os.close(descriptor)
    finally:
        try:
            local_service_runtime_paths(identity).startup_lock.unlink()
        except FileNotFoundError:
            pass


def _retire_refused_listener_endpoint(
    identity: LocalServiceIdentity,
    endpoint: Mapping[str, object],
    *,
    deadline_ns: int | None = None,
) -> bool:
    """Retire a persisted endpoint only when TCP confirms nothing is listening."""

    if not _listener_endpoint_refuses_connections(endpoint, deadline_ns=deadline_ns):
        return False
    pid = endpoint.get("pid")
    port = endpoint.get("port")
    if not isinstance(pid, int) or not isinstance(port, int):
        return False
    clear_local_service_registry(identity, pid=pid, port=port)
    return True


def _listener_endpoint_refuses_connections(
    endpoint: Mapping[str, object], *, deadline_ns: int | None = None
) -> bool:
    host = endpoint.get("host")
    port = endpoint.get("port")
    if host != LOCAL_SERVICE_LISTENER_HOST or not isinstance(port, int):
        return False
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        return False
    timeout: float = _LOCAL_SERVICE_ENDPOINT_PROBE_TIMEOUT_SECONDS
    if deadline_ns is not None:
        timeout = min(timeout, max(0.001, (deadline_ns - time.monotonic_ns()) / 1_000_000_000))
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return False
    except ConnectionRefusedError:
        return True
    except OSError as exc:
        return exc.errno == errno.ECONNREFUSED


def _listener_is_dispatch_ready(status: Mapping[str, object] | None) -> bool:
    return bool(
        status
        and status.get("ok") is True
        and status.get("liveness") == "live"
        and status.get("readiness") == "ready"
        and status.get("native_health") == "healthy"
        and status.get("degraded_state")
        in {None, "emergency_read_only", "background_queue_unavailable"}
    )


def _listener_health(
    endpoint: Mapping[str, object],
    identity: LocalServiceIdentity,
    *,
    timeout_seconds: float,
    deadline_ns: int | None = None,
) -> dict[str, object] | None:
    """Read a valid listener status; dispatch readiness is checked separately."""

    for attempt in range(_LOCAL_SERVICE_HEALTH_PROBE_ATTEMPTS):
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            return None
        probe_timeout = (
            min(timeout_seconds, max(0.001, (deadline_ns - time.monotonic_ns()) / 1_000_000_000))
            if deadline_ns is not None
            else timeout_seconds
        )
        response = _local_service_http_request(
            endpoint,
            "/health",
            {},
            timeout_seconds=probe_timeout,
        )
        if (
            response.get("workspace_id") == identity.workspace_id
            and response.get("profile_id") == identity.profile_id
            and response.get("selected_backend") == identity.backend
            and response.get("transport") == LOCAL_SERVICE_LISTENER_TRANSPORT
        ):
            return response
        if attempt + 1 < _LOCAL_SERVICE_HEALTH_PROBE_ATTEMPTS:
            delay = _LOCAL_SERVICE_HEALTH_RETRY_SECONDS
            if deadline_ns is not None:
                delay = min(delay, max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000))
            time.sleep(delay)
    return None


def _listener_status_error(status: Mapping[str, object]) -> str:
    degraded_state = status.get("degraded_state")
    if status.get("state") == "draining" or degraded_state == "listener_draining":
        return "local service listener is draining"
    if degraded_state == "native_memory_unavailable":
        return "local service native memory unavailable"
    if degraded_state == "policy_bundle_unavailable":
        return "local service policy bundle unavailable"
    if degraded_state == "scheduler_unavailable":
        return "local service scheduler unavailable"
    if degraded_state == "listener_closed":
        return "local service dispatcher is unavailable"
    return "local service listener is not ready"


def _unavailable_service_status(
    identity: LocalServiceIdentity,
    *,
    state: str,
    degraded_state: str,
    error: str,
) -> dict[str, object]:
    """Return the public, content-free health shape when no endpoint can report."""

    return {
        "ok": False,
        "state": state,
        "liveness": "not_live",
        "readiness": "not_ready",
        "native_health": "not_probed",
        "native_ok": False,
        "strict_ok": False,
        "effective_mode": "unavailable",
        "degraded_state": degraded_state,
        "error": error,
        "transport": LOCAL_SERVICE_LISTENER_TRANSPORT,
        "listener_pid": None,
        "selected_backend": identity.backend,
        "workspace_id": identity.workspace_id,
        "profile_id": identity.profile_id,
        "acl_revision": 1,
        "successful_calls": 0,
        "failed_calls": 0,
        "queue_capacity": 0,
        "queue_depth": 0,
        "policy_bundle_revision": None,
        "service_schema_revision": f"local-service-schema:{LOCAL_SERVICE_SCHEMA_VERSION}",
        "slo_contract_revision": None,
        "schema_revision": None,
        "index_revision": None,
        "canonical_snapshot_sha256": None,
        "catalog_revision": None,
        "block_revision": None,
        "projection_revision": None,
        "background_queue": {
            "available": False,
            "emergency_read_only": True,
            "no_silent_drop": False,
            "lanes": [],
        },
        "background_queue_error": error,
        "shutdown_error": None,
        "dispatcher_alive": False,
    }


def _local_service_http_request(
    endpoint: Mapping[str, object],
    path: str,
    payload: Mapping[str, object],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    host = endpoint.get("host")
    port = endpoint.get("port")
    if host != LOCAL_SERVICE_LISTENER_HOST or not isinstance(port, int):
        return {"ok": False, "error": "local service endpoint is invalid"}
    serialized = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    if len(serialized) > LOCAL_SERVICE_MAX_REQUEST_BYTES:
        return {"ok": False, "error": "local service request exceeds protocol limit"}
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=serialized,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(0.001, timeout_seconds)) as response:
            body = response.read(LOCAL_SERVICE_MAX_RESPONSE_BYTES + 1)
    except TimeoutError:
        return {
            "ok": False,
            "error": "local service transport timed out",
            "reason_code": "transport_timeout",
        }
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return {
                "ok": False,
                "error": "local service transport timed out",
                "reason_code": "transport_timeout",
            }
        return {"ok": False, "error": "local service listener unavailable"}
    except OSError:
        return {"ok": False, "error": "local service listener unavailable"}
    if len(body) > LOCAL_SERVICE_MAX_RESPONSE_BYTES:
        return {"ok": False, "error": "local service response exceeds protocol limit"}
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"ok": False, "error": "local service listener returned invalid JSON"}
    return (
        decoded
        if isinstance(decoded, dict)
        else {"ok": False, "error": "local service listener returned a non-object response"}
    )


def _local_service_session_id(value: object) -> str | None:
    if value in (None, ""):
        return "default"
    if not isinstance(value, str):
        return None
    session_id = value.strip()
    if not session_id or len(session_id) > 256:
        return None
    return session_id


def _local_service_idle_timeout_seconds() -> int:
    try:
        configured = int(os.getenv("UM_MEM_LOCAL_SERVICE_IDLE_SECONDS", ""))
    except ValueError:
        configured = LOCAL_SERVICE_IDLE_TIMEOUT_SECONDS
    return max(1, configured) if configured else LOCAL_SERVICE_IDLE_TIMEOUT_SECONDS


def _listener_owner_pid(environment: Mapping[str, str] | None = None) -> int:
    """Resolve the host process that owns a detached listener's lifecycle."""

    source_environment = environment if environment is not None else os.environ
    configured = str(source_environment.get("UM_MEM_LOCAL_SERVICE_OWNER_PID", "")).strip()
    try:
        explicit_owner = int(configured) if configured else 0
    except ValueError:
        explicit_owner = 0
    if explicit_owner > 0 and _process_is_alive(explicit_owner):
        return explicit_owner
    parent_pid = os.getppid()
    if os.name == "nt":
        parent_image = _windows_process_image_path(parent_pid)
        if parent_image is not None:
            try:
                shim = Path(sys.executable).resolve()
                current = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
            except OSError:
                shim = None
                current = None
            if shim is not None and current is not None and parent_image == shim != current:
                grandparent_pid = _windows_parent_process_id(parent_pid)
                if grandparent_pid and _process_is_alive(grandparent_pid):
                    return grandparent_pid
    return parent_pid if parent_pid > 0 else os.getpid()


def _windows_process_image_path(pid: int) -> Path | None:
    if os.name != "nt" or pid <= 0:
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    query_image = kernel32.QueryFullProcessImageNameW
    query_image.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    query_image.restype = wintypes.BOOL
    process = open_process(0x1000, False, pid)
    if not process:
        return None
    try:
        buffer = ctypes.create_unicode_buffer(32_768)
        size = wintypes.DWORD(len(buffer))
        if not query_image(process, 0, buffer, ctypes.byref(size)):
            return None
        return Path(buffer.value).resolve()
    finally:
        close_handle(process)


def _windows_parent_process_id(pid: int) -> int | None:
    if os.name != "nt" or pid <= 0:
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    create_snapshot.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    first = kernel32.Process32FirstW
    first.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
    first.restype = wintypes.BOOL
    next_entry = kernel32.Process32NextW
    next_entry.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
    next_entry.restype = wintypes.BOOL
    snapshot = create_snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return None
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        has_entry = first(snapshot, ctypes.byref(entry))
        while has_entry:
            if entry.th32ProcessID == pid:
                return int(entry.th32ParentProcessID) or None
            entry.dwSize = ctypes.sizeof(ProcessEntry)
            has_entry = next_entry(snapshot, ctypes.byref(entry))
        return None
    finally:
        close_handle(snapshot)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        get_exit_code.restype = wintypes.BOOL
        process = open_process(0x1000, False, pid)
        if not process:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            return bool(get_exit_code(process, ctypes.byref(exit_code))) and exit_code.value == 259
        finally:
            close_handle(process)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform == "linux":
        try:
            # An exited child retains its PID until its parent collects it.
            # Signal zero succeeds for that zombie although it cannot serve HTTP.
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return False
        except PermissionError:
            return True
        return state not in {"Z", "X", "x"}
    return True


def _read_local_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_local_json(path: Path, payload: Mapping[str, object], *, private: bool) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    if private:
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
    temporary.replace(path)
