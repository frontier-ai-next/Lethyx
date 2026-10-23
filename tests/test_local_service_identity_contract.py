import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from um_mem import MemoryClient
from um_mem.contracts import MemoryEvent
from um_mem.memory import local_service, mcp_call, mcp_server
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.registry import load_memory_blocks, load_workspace_topology
from um_mem.memory.kernel.telemetry import read_memory_events
from um_mem.memory.local_service import (
    LOCAL_SERVICE_API_VERSION,
    LOCAL_SERVICE_LISTENER_HOST,
    LOCAL_SERVICE_MAX_REQUEST_BYTES,
    LOCAL_SERVICE_PROTOCOL_VERSION,
    LOCAL_SERVICE_SCHEMA_VERSION,
    LocalServiceClientPool,
    LocalServiceIdentity,
    build_local_service_identity,
    local_service_bootstrap_secret,
    local_service_role_allows,
    shutdown_local_service_listener,
)
from um_mem.memory.service_receipts import (
    claim_remember_receipt,
    complete_remember_receipt,
)


class _HealthyBackend:
    def health(self) -> dict[str, object]:
        return {"ok": True, "native_ok": True, "strict_ok": True}


class _HealthyClient:
    backend = _HealthyBackend()


class _UnhealthyBackend:
    def health(self) -> dict[str, object]:
        return {"ok": False}


class _UnhealthyClient:
    backend = _UnhealthyBackend()


class _InProcessListener:
    def __init__(self, identity: LocalServiceIdentity, client: object | None = None) -> None:
        self.identity = identity
        self.runtime = mcp_call._LocalServiceListenerRuntime(
            identity,
            client if client is not None else _HealthyClient(),
        )
        self.server = mcp_call._LocalServiceListenerServer(
            (LOCAL_SERVICE_LISTENER_HOST, 0),
            mcp_call._listener_handler(self.runtime),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.runtime.start()
        self.thread.start()

    @property
    def endpoint(self) -> dict[str, object]:
        return {
            "host": LOCAL_SERVICE_LISTENER_HOST,
            "port": self.server.server_port,
        }

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.runtime.close()


def test_http_health_handlers_release_thread_local_sqlite_connections(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="alpha")
    index.health = lambda: {
        "ok": True,
        "native_ok": True,
        "strict_ok": True,
        **index.operational_revisions(),
    }
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
    listener = _InProcessListener(
        build_local_service_identity(
            repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
        ),
        client,
    )
    try:
        for _ in range(6):
            listener.runtime._native_probed_at = 0
            request = urllib.request.Request(
                f"http://127.0.0.1:{listener.server.server_port}/health",
                data=b"{}",
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                assert json.loads(response.read())["readiness"] == "ready"
        deadline = time.monotonic() + 2
        while len(index._connections) > 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(index._connections) == 1
    finally:
        listener.close()
        index.close()


@pytest.fixture
def in_process_listener(tmp_path: Path) -> Iterator[_InProcessListener]:
    listener = _InProcessListener(
        build_local_service_identity(
            repo_root=tmp_path,
            workspace_id="alpha",
            backend="mempalace",
            caller_role="operator",
        )
    )
    try:
        yield listener
    finally:
        listener.close()


def _lease_request(
    identity: LocalServiceIdentity,
    *,
    request_id: str = "lease-1",
    session_id: str = "run-1",
) -> dict[str, object]:
    return {
        "protocol_version": identity.protocol_version,
        "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
        "api_version": LOCAL_SERVICE_API_VERSION,
        "request_id": request_id,
        "caller_role": identity.caller_role,
        "session_id": session_id,
        "capabilities": sorted(
            tool for tool in mcp_call.TOOLS if local_service_role_allows(identity.caller_role, tool)
        ),
        "bootstrap_secret": local_service_bootstrap_secret(identity),
    }


def _issue_lease(
    listener: _InProcessListener,
    *,
    session_id: str = "run-1",
) -> str:
    response = listener.runtime.issue_lease(
        _lease_request(listener.identity, session_id=session_id)
    )
    assert response["ok"] is True
    lease = response.get("lease")
    assert isinstance(lease, str)
    return lease


def test_parent_role_can_retract_memory_but_cannot_promote_it() -> None:
    assert local_service_role_allows("parent", "forget") is True
    assert local_service_role_allows("parent", "promote") is False
    assert local_service_role_allows("operator", "promote") is False
    assert local_service_role_allows("supervisor", "promote") is True
    assert local_service_role_allows("supervisor", "consolidate_observations") is True
    assert local_service_role_allows("parent", "consolidate_observations") is False


def test_supervisor_rebuild_uses_existing_client_operation(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, str]] = []

    class Client:
        def rebuild_indexes(self) -> dict[str, object]:
            return {"ok": True, "backend": "mempalace", "index_sha256": "a" * 64}

    @contextmanager
    def existing_client(**kwargs):
        calls.append({key: str(value) for key, value in kwargs.items()})
        yield Client()

    monkeypatch.setattr(mcp_call, "persistent_mcp_client", existing_client)

    result = mcp_call._local_service_rebuild_indexes(
        repo_root=str(tmp_path),
        workspace_id="alpha",
        backend="mempalace",
    )

    assert local_service_role_allows("supervisor", "rebuild_indexes") is True
    assert result["index_sha256"] == "a" * 64
    assert calls == [
        {
            "repo_root": str(tmp_path),
            "workspace_id": "alpha",
            "backend": "mempalace",
        }
    ]


def test_host_state_lease_advertises_private_scheduler_capability(
    in_process_listener: _InProcessListener,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="host_state",
    )
    request = {
        "protocol_version": identity.protocol_version,
        "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
        "api_version": LOCAL_SERVICE_API_VERSION,
        "request_id": "host-state-lease",
        "caller_role": identity.caller_role,
        "session_id": "host-state-session",
        "capabilities": ["health", "host_state"],
        "bootstrap_secret": local_service_bootstrap_secret(identity),
    }

    issued = in_process_listener.runtime.issue_lease(request)

    assert issued["ok"] is True
    assert issued["capabilities"] == ["health", "host_state"]


def _call_request(
    identity: LocalServiceIdentity,
    lease: str,
    *,
    request_id: str = "call-1",
    session_id: str = "run-1",
    tool: str = "recall",
    arguments: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "protocol_version": identity.protocol_version,
        "schema_version": LOCAL_SERVICE_SCHEMA_VERSION,
        "api_version": LOCAL_SERVICE_API_VERSION,
        "request_id": request_id,
        "lease": lease,
        "caller_role": identity.caller_role,
        "session_id": session_id,
        "tool": tool,
        "arguments": dict(arguments or {}),
    }


def _client_process(
    repo_root: Path,
    *,
    backend: str,
    tool: str,
    arguments: dict[str, object] | None = None,
    mutation_correlation_id: str = "",
) -> dict[str, object]:
    script = """
import json
import sys
from pathlib import Path

from um_mem.memory.local_service import LocalServiceClientPool, build_local_service_identity

identity = build_local_service_identity(
    repo_root=Path(sys.argv[1]),
    workspace_id="alpha",
    backend=sys.argv[2],
    caller_role="operator",
)
pool = LocalServiceClientPool()
try:
    result = pool.call(
        identity,
        sys.argv[3],
        json.loads(sys.argv[4]),
        timeout_seconds=20,
        mutation_correlation_id=sys.argv[5],
    )
    print(json.dumps({"result": result, "service": pool.service_status(identity)}))
finally:
    pool.close()
"""
    source_root = str(Path.cwd() / "src")
    python_path = os.pathsep.join(
        value for value in (source_root, os.environ.get("PYTHONPATH", "")) if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(repo_root),
            backend,
            tool,
            json.dumps(arguments or {}),
            mutation_correlation_id,
        ],
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="strict",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env={**os.environ, "PYTHONPATH": python_path},
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _stop_listener(identity: LocalServiceIdentity) -> None:
    shutdown_local_service_listener(identity)
    deadline = time.monotonic() + 10
    while (
        local_service.local_service_registry(identity) is not None and time.monotonic() < deadline
    ):
        time.sleep(0.05)


def test_local_service_identity_normalizes_selected_key(tmp_path: Path) -> None:
    first = build_local_service_identity(
        repo_root=str(tmp_path / "."),
        workspace_id="workspaces/alpha",
        backend="MemPalace",
    )
    same = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    other_workspace = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="beta",
        backend="mempalace",
    )
    other_backend = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="hindsight",
    )
    other_profile = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        profile_id="umbrella",
        backend="mempalace",
    )

    assert first == same
    assert first.protocol_version == LOCAL_SERVICE_PROTOCOL_VERSION
    assert first != other_workspace
    assert first != other_backend
    assert first != other_profile
    assert (
        local_service.local_service_runtime_paths(first).registry
        != local_service.local_service_runtime_paths(other_profile).registry
    )
    assert local_service.local_service_identity_is_authorized(first, {"query": "parser"})
    assert not local_service.local_service_identity_is_authorized(
        first,
        {"backend": "hindsight"},
    )
    with pytest.raises(ValueError, match="unknown memory backend"):
        build_local_service_identity(
            repo_root=tmp_path,
            workspace_id="alpha",
            backend="unknown-backend",
        )


@pytest.mark.parametrize(
    "claim",
    [
        {"repo_root": []},
        {"repo_root": 42},
        {"workspace_id": {"workspace": "alpha"}},
        {"backend": ["mempalace"]},
    ],
)
def test_identity_authorization_rejects_malformed_claims(
    tmp_path: Path, claim: dict[str, object]
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace"
    )
    assert local_service.local_service_identity_is_authorized(identity, claim) is False


def test_canonical_identity_paths_share_unicode_symlink_normalization(tmp_path: Path) -> None:
    target = tmp_path / "caf\u00e9"
    target.mkdir()
    alias = tmp_path / "checkout-alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(alias), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )

    event = MemoryEvent(
        content="canonical path binding",
        workspace_id="alpha",
        checkout=str(tmp_path / "cafe\u0301"),
        worktree=str(alias),
    )
    identity = build_local_service_identity(
        repo_root=alias,
        workspace_id="alpha",
        backend="mempalace",
    )

    assert event.checkout == event.worktree == str(identity.repo_root)


def test_mcp_server_normalizes_identity_before_listener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identities: list[object] = []
    monkeypatch.setenv("UM_MEM_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("UM_MEM_WORKSPACE", "alpha")
    monkeypatch.setenv("UM_MEM_BACKEND", "mempalace")

    def fake_call(identity: object, *_args: object, **_kwargs: object) -> dict[str, object]:
        identities.append(identity)
        return {"ok": True}

    monkeypatch.setattr(mcp_server._LOCAL_SERVICE_CLIENTS, "call", fake_call)

    result = mcp_server._call_tool(
        "health",
        {
            "repo_root": str(tmp_path / "."),
            "workspace_id": "workspaces/alpha",
            "backend": "MemPalace",
        },
    )

    assert result["ok"] is True
    [identity] = identities
    assert identity == build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )


def test_mcp_server_rejects_foreign_client_identity_before_listener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    trusted_root = tmp_path / "trusted"
    monkeypatch.setenv("UM_MEM_REPO_ROOT", str(trusted_root))
    monkeypatch.setenv("UM_MEM_WORKSPACE", "trusted-workspace")
    monkeypatch.setenv("UM_MEM_BACKEND", "mempalace")

    def fake_call(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(mcp_server._LOCAL_SERVICE_CLIENTS, "call", fake_call)

    result = mcp_server._call_tool(
        "remember",
        {
            "repo_root": str(tmp_path / "foreign"),
            "workspace_id": "foreign-workspace",
            "backend": "hindsight",
            "content": "must not reach the selected listener",
        },
    )

    assert result == {
        "ok": False,
        "error": "local service request identity is not authorized",
    }
    assert calls == []


def test_mcp_server_rejects_foreign_export_path_before_listener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    trusted_vault = tmp_path / "trusted-vault"
    monkeypatch.setenv("UM_MEM_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("UM_MEM_WORKSPACE", "alpha")
    monkeypatch.setenv("UM_MEM_BACKEND", "mempalace")
    monkeypatch.setenv("UM_MEM_OBSIDIAN_VAULT", str(trusted_vault))

    def fake_call(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(mcp_server._LOCAL_SERVICE_CLIENTS, "call", fake_call)

    result = mcp_server._call_tool(
        "export_obsidian",
        {
            "repo_root": str(tmp_path),
            "workspace_id": "alpha",
            "backend": "mempalace",
            "vault_dir": str(tmp_path / "foreign-vault"),
        },
    )

    assert result == {
        "ok": False,
        "error": "local service export path is not authorized",
    }
    assert calls == []


def test_listener_issues_role_scoped_lease_and_rejects_tampered_capabilities(
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="host_hook",
    )
    listener = _InProcessListener(identity)
    try:
        issued = listener.runtime.issue_lease(_lease_request(identity))
        tampered = _lease_request(identity, request_id="lease-2")
        tampered["capabilities"] = ["health"]
        rejected = listener.runtime.issue_lease(tampered)
    finally:
        listener.close()

    assert issued["ok"] is True
    assert issued["capabilities"] == [
        "build_context",
        "health",
        "observe",
        "operation_status",
        "remember",
        "resolve_active_mask",
    ]
    assert isinstance(issued.get("lease"), str)
    assert rejected == {"ok": False, "error": "local service capabilities mismatch"}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "protocol_version",
            LOCAL_SERVICE_PROTOCOL_VERSION + 1,
            "local service protocol version mismatch",
        ),
        (
            "schema_version",
            f"{LOCAL_SERVICE_SCHEMA_VERSION}-mismatch",
            "local service schema version mismatch",
        ),
        (
            "api_version",
            f"{LOCAL_SERVICE_API_VERSION}-mismatch",
            "local service API version mismatch",
        ),
    ],
)
def test_listener_rejects_incompatible_frame_before_dispatch(
    in_process_listener: _InProcessListener,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error: str,
) -> None:
    call_tool_calls: list[object] = []
    writer_calls: list[object] = []

    def fake_call_tool(*args: object, **kwargs: object) -> dict[str, object]:
        call_tool_calls.append((args, kwargs))
        return {"ok": True, "saved": True}

    def fake_writer(*args: object, **kwargs: object) -> dict[str, object]:
        writer_calls.append((args, kwargs))
        return {"ok": True, "saved": True}

    monkeypatch.setattr(mcp_call, "call_tool", fake_call_tool)
    monkeypatch.setitem(mcp_call.TOOLS, "remember", fake_writer)
    identity = in_process_listener.identity
    lease = _issue_lease(in_process_listener)
    request = _call_request(
        identity,
        lease,
        tool="remember",
        arguments={"content": "frame rejection must not write"},
    )
    request[field] = value

    assert in_process_listener.runtime.call(request) == {"ok": False, "error": error}
    assert call_tool_calls == []
    assert writer_calls == []


def test_listener_reports_native_failure_without_implicit_no_memory_fallback(
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    listener = _InProcessListener(identity, _UnhealthyClient())
    try:
        issued = listener.runtime.issue_lease(_lease_request(identity))
        assert issued["ok"] is True
        lease = issued.get("lease")
        assert isinstance(lease, str)
        health = listener.runtime.call(_call_request(identity, lease, tool="health"))
        recall = listener.runtime.call(
            _call_request(identity, lease, tool="recall", arguments={"query": "blocked"})
        )
    finally:
        listener.close()

    assert health["liveness"] == "live"
    assert health["readiness"] == "not_ready"
    assert health["native_health"] == "unavailable"
    assert health["native_ok"] is False
    assert health["strict_ok"] is False
    assert health["effective_mode"] == "unavailable"
    assert health["degraded_state"] == "native_memory_unavailable"
    assert recall == {
        "ok": False,
        "error": "local service native memory unavailable",
        "reason_code": "native_backend_unavailable",
        "retryable": True,
    }


def test_listener_rejects_generic_health_success_without_native_attestation(
    tmp_path: Path,
) -> None:
    class Backend:
        def health(self) -> dict[str, object]:
            return {
                "ok": True,
                "native_ok": False,
                "strict_ok": False,
                "effective_mode": "unavailable",
            }

    class Client:
        backend = Backend()

    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    listener = _InProcessListener(identity, Client())
    try:
        lease = _issue_lease(listener)
        health = listener.runtime.call(_call_request(identity, lease, tool="health"))
    finally:
        listener.close()

    assert health["liveness"] == "live"
    assert health["readiness"] == "not_ready"
    assert health["native_health"] == "unavailable"
    assert health["native_ok"] is False
    assert health["strict_ok"] is False
    assert health["effective_mode"] == "unavailable"
    assert health["degraded_state"] == "native_memory_unavailable"


def test_listener_rejects_health_without_native_attestation(tmp_path: Path) -> None:
    class Backend:
        def health(self) -> dict[str, object]:
            return {"ok": True}

    class Client:
        backend = Backend()

    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    listener = _InProcessListener(identity, Client())
    try:
        lease = _issue_lease(listener)
        health = listener.runtime.call(_call_request(identity, lease, tool="health"))
    finally:
        listener.close()

    assert health["liveness"] == "live"
    assert health["readiness"] == "not_ready"
    assert health["native_health"] == "unavailable"
    assert health["native_ok"] is False
    assert health["strict_ok"] is False
    assert health["effective_mode"] == "unavailable"
    assert health["degraded_state"] == "native_memory_unavailable"


def test_listener_degrades_when_native_health_probe_raises_database_error(
    tmp_path: Path,
) -> None:
    class Backend:
        def health(self) -> dict[str, object]:
            raise sqlite3.OperationalError("native index unavailable")

    class Client:
        backend = Backend()

    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    listener = _InProcessListener(identity, Client())
    try:
        lease = _issue_lease(listener)
        health = listener.runtime.call(_call_request(identity, lease, tool="health"))
    finally:
        listener.close()

    assert health["liveness"] == "live"
    assert health["readiness"] == "not_ready"
    assert health["native_health"] == "unavailable"
    assert health["native_ok"] is False
    assert health["strict_ok"] is False
    assert health["effective_mode"] == "unavailable"
    assert health["degraded_state"] == "native_memory_unavailable"


def test_listener_call_health_keeps_the_local_service_health_shape(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    listener = _InProcessListener(identity)
    try:
        lease = _issue_lease(listener)
        health = listener.runtime.call(_call_request(identity, lease, tool="health"))
    finally:
        listener.close()

    assert health["backend"] == "mempalace"
    assert health["selected_backend"] == "mempalace"
    assert health["state"] == "ready"
    assert health["liveness"] == "live"
    assert health["readiness"] == "ready"
    assert health["native_health"] == "healthy"
    assert health["degraded_state"] is None


def test_mcp_health_keeps_tools_only_host_profile_separate_from_native_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("UM_MEM_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("UM_MEM_WORKSPACE", "alpha")
    monkeypatch.setenv("UM_MEM_BACKEND", "mempalace")
    service = {
        "ok": False,
        "state": "not_ready",
        "liveness": "live",
        "readiness": "not_ready",
        "native_health": "unavailable",
        "degraded_state": "native_memory_unavailable",
    }
    monkeypatch.setattr(
        mcp_server._LOCAL_SERVICE_CLIENTS, "call", lambda *_args, **_kwargs: service
    )
    monkeypatch.setattr(
        mcp_server._LOCAL_SERVICE_CLIENTS,
        "service_status",
        lambda _identity: service,
    )

    result = mcp_server._call_tool("health", {})

    assert result["integration_mode"] == "mcp_tools_only"
    assert result["runtime"]["effective_profile"] == "mcp_tools_only"
    assert result["service"]["degraded_state"] == "native_memory_unavailable"


def test_listener_pins_catalog_bundle_and_binds_snapshot_through_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    blocks_path = tmp_path / "workspaces" / "alpha" / ".memory" / "blocks.yaml"
    blocks_path.parent.mkdir(parents=True)
    blocks_path.write_text(
        """blocks:
  - block_id: release_pin_probe
    block_type: release_probe
    temperature: warm
    scope: workspace
    authority: advisory
    delivery_policy: on_demand
    retention: durable
    producer: test_fixture
    consumers: [memory_agent]
    recall_path: hybrid_recall
    lifecycle_policy: {initial_status: candidate, mutable: true, terminal_statuses: [retracted]}
    isolation_policy: {dimension: workspace, cross_workspace: false}
""",
        encoding="utf-8",
    )
    observed: list[bool] = []

    def fake_call(_name: str, _arguments: dict[str, object]) -> dict[str, object]:
        observed.append(
            any(
                block.block_id == "release_pin_probe"
                for block in load_memory_blocks(tmp_path, "alpha")
            )
        )
        return {"ok": True}

    monkeypatch.setattr(mcp_call, "call_tool", fake_call)
    first = _InProcessListener(identity)
    try:
        issued = first.runtime.issue_lease(_lease_request(identity, session_id="stable-session"))
        assert issued["ok"] is True
        first_lease = issued.get("lease")
        assert isinstance(first_lease, str)
        first_bundle = issued.get("policy_bundle_revision")
        first_catalog = issued.get("catalog_revision")
        assert isinstance(first_bundle, str)
        assert isinstance(first_catalog, str)

        blocks_path.write_text("blocks: []\n", encoding="utf-8")
        result = first.runtime.call(
            _call_request(identity, first_lease, session_id="stable-session")
        )
    finally:
        first.close()

    assert result == {"ok": True}
    assert observed == [True]
    assert all(
        block.block_id != "release_pin_probe" for block in load_memory_blocks(tmp_path, "alpha")
    )

    restarted = _InProcessListener(identity)
    try:
        replayed = restarted.runtime.issue_lease(
            _lease_request(identity, request_id="lease-2", session_id="stable-session")
        )
    finally:
        restarted.close()

    assert replayed["ok"] is True
    assert replayed["policy_bundle_revision"] == first_bundle
    assert replayed["catalog_revision"] == first_catalog


def test_listener_lease_sees_workspace_block_created_in_same_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    listener = _InProcessListener(identity)
    try:
        lease = _issue_lease(listener)
        root = load_workspace_topology(tmp_path, "alpha")[0].block_id
        created = listener.runtime.call(
            _call_request(
                identity,
                lease,
                tool="create_block",
                arguments={
                    "name": "Live branch",
                    "synopsis": "Facts selected during this session.",
                    "parent_id": root,
                    "reason": "the agent selected a new stable area",
                },
            )
        )
        block_id = str(created.get("block_id") or "")
        assert created["ok"] is True
        assert block_id
        replayed = listener.runtime.call(
            _call_request(
                identity,
                lease,
                tool="create_block",
                arguments={
                    "name": "Live branch",
                    "synopsis": "Facts selected during this session.",
                    "parent_id": root,
                    "reason": "the agent selected a new stable area",
                },
            )
        )
        assert replayed["replayed"] is True
        assert replayed["block_id"] == block_id
        assert (
            sum(state.block_id == block_id for state in load_workspace_topology(tmp_path, "alpha"))
            == 1
        )

        visible: list[bool] = []

        def inspect_bound_catalog(name: str, _arguments: dict[str, object]) -> dict[str, object]:
            assert name == "recall"
            visible.append(
                any(
                    block.block_id == block_id
                    for block in load_memory_blocks(tmp_path, "alpha", strict=True)
                )
            )
            return {"ok": True, "hits": []}

        monkeypatch.setattr(mcp_call, "call_tool", inspect_bound_catalog)
        recalled = listener.runtime.call(
            _call_request(
                identity,
                lease,
                request_id="call-2",
                tool="recall",
                arguments={"query": "explicit", "logical_block_ids": [block_id]},
            )
        )
    finally:
        listener.close()

    assert recalled["ok"] is True
    assert visible == [True]


def test_listener_refuses_malformed_catalog_before_initialization(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    blocks_path = tmp_path / "workspaces" / "alpha" / ".memory" / "blocks.yaml"
    blocks_path.parent.mkdir(parents=True)
    blocks_path.write_text("blocks: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="memory block catalog is invalid"):
        _InProcessListener(identity)


def test_client_pool_fails_closed_for_invalid_session_before_listener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    pool = LocalServiceClientPool()

    def should_not_start(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid session must not reach listener startup")

    monkeypatch.setattr(pool, "_ensure_listener", should_not_start)

    assert pool.call(
        identity,
        "health",
        {},
        timeout_seconds=1,
        session_id="x" * 257,
    ) == {"ok": False, "error": "local service session identity is invalid"}


def test_client_pool_retries_lost_mutation_response_with_one_stable_correlation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    pool = LocalServiceClientPool()
    endpoint = {"host": LOCAL_SERVICE_LISTENER_HOST, "port": 12345, "pid": 42}
    requests: list[dict[str, object]] = []

    def fake_ensure_listener(*_args: object, **_kwargs: object) -> tuple[dict[str, object], None]:
        return endpoint, None

    monkeypatch.setattr(pool, "_ensure_listener", fake_ensure_listener)
    monkeypatch.setattr(pool, "_lease_for", lambda *_args, **_kwargs: ("lease", None))

    def fake_request(
        _endpoint: object,
        path: str,
        payload: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        assert path == "/call"
        requests.append(dict(payload))
        if len(requests) == 1:
            return {"ok": False, "error": "local service listener unavailable"}
        return {"ok": True, "saved": True, "replayed": True}

    monkeypatch.setattr(local_service, "_local_service_http_request", fake_request)

    result = pool.call(identity, "remember", {"content": "once"}, timeout_seconds=1)

    assert result == {"ok": True, "saved": True, "replayed": True}
    assert len(requests) == 2
    assert requests[0]["request_id"] == requests[1]["request_id"]
    assert requests[0]["mutation_correlation_id"] == requests[1]["mutation_correlation_id"]
    assert str(requests[0]["mutation_correlation_id"]).startswith("local-service:")


def test_client_pool_transport_timeout_returns_original_operation_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace"
    )
    pool = LocalServiceClientPool()
    endpoint = {"host": LOCAL_SERVICE_LISTENER_HOST, "port": 12345, "pid": 42}
    monkeypatch.setattr(pool, "_ensure_listener", lambda *_args, **_kwargs: (endpoint, None))
    monkeypatch.setattr(pool, "_lease_for", lambda *_args, **_kwargs: ("lease", None))
    requests: list[dict[str, object]] = []

    def fake_request(
        _endpoint: object, path: str, payload: dict[str, object], **_kwargs: object
    ) -> dict[str, object]:
        assert path == "/call"
        requests.append(dict(payload))
        if len(requests) == 1:
            return {
                "ok": False,
                "error": "local service transport timed out",
                "reason_code": "transport_timeout",
            }
        return {"ok": True, "saved": True, "replayed": True}

    monkeypatch.setattr(local_service, "_local_service_http_request", fake_request)
    timeout = pool.call(identity, "remember", {"content": "once"}, timeout_seconds=1)
    assert timeout["reason_code"] == "transport_timeout"
    assert timeout["durable_status"] == "unknown"
    operation_id = timeout["operation_id"]
    assert isinstance(operation_id, str) and operation_id
    assert len(requests) == 1
    assert isinstance(requests[0]["deadline_ns"], int)
    assert requests[0]["deadline_ns"] > 0

    replay = pool.call(
        identity,
        "remember",
        {"content": "once"},
        timeout_seconds=1,
        mutation_correlation_id=operation_id,
    )
    assert replay == {"ok": True, "saved": True, "replayed": True}
    assert requests[0]["mutation_correlation_id"] == requests[1]["mutation_correlation_id"]
    assert isinstance(requests[1]["deadline_ns"], int)


def test_timeout_after_durable_commit_replays_same_operation_without_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
    )
    arguments = {
        "content": "A committed result can be lost in transport.",
        "title": "postcommit timeout",
        "requested_block_id": "workspace_episodes",
    }
    original_request = local_service._local_service_http_request
    committed: list[dict[str, object]] = []

    def lose_first_response(endpoint, path, payload, *, timeout_seconds):
        response = original_request(endpoint, path, payload, timeout_seconds=timeout_seconds)
        if path == "/call" and not committed:
            assert response.get("ok") is True
            committed.append(response)
            return {
                "ok": False,
                "error": "local service transport timed out",
                "reason_code": "transport_timeout",
            }
        return response

    monkeypatch.setattr(local_service, "_local_service_http_request", lose_first_response)
    pool = LocalServiceClientPool()
    try:
        timeout = pool.call(
            identity,
            "remember",
            arguments,
            timeout_seconds=20,
            mutation_correlation_id="postcommit-operation-1",
        )
        replay = pool.call(
            identity,
            "remember",
            arguments,
            timeout_seconds=20,
            mutation_correlation_id=str(timeout["operation_id"]),
        )
    finally:
        pool.close()
        _stop_listener(identity)

    assert timeout["reason_code"] == "transport_timeout"
    assert timeout["durable_status"] == "unknown"
    assert timeout["operation_id"] == "postcommit-operation-1"
    assert replay["replayed"] is True
    assert replay["canonical_reference"] == committed[0]["memory_id"]
    captures = [
        event for event in read_memory_events(tmp_path) if event["event_type"] == "memory_capture"
    ]
    assert len(captures) == 1


def test_client_deadline_bounds_listener_startup_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
    )
    pool = LocalServiceClientPool()

    def unexpected_launch(_identity):
        raise AssertionError("listener launch must not start while startup lock is held")

    monkeypatch.setattr(pool, "_launch_listener", unexpected_launch)
    result: dict[str, object] = {}
    lock = local_service._LOCAL_SERVICE_START_LOCK
    lock.acquire()
    caller = threading.Thread(
        target=lambda: result.update(
            pool.call(
                identity,
                "remember",
                {"content": "fact"},
                timeout_seconds=1,
                mutation_correlation_id="startup-lock-operation-1",
            )
        ),
        daemon=True,
    )
    caller.start()
    try:
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert result["reason_code"] == "transport_timeout"
        assert result["durable_status"] == "not_started"
        assert result["operation_id"] == "startup-lock-operation-1"
    finally:
        lock.release()
        caller.join(timeout=5)
        pool.close()


def test_local_service_http_timeout_is_distinct_from_unavailable_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> object:
        raise TimeoutError("simulated slow response")

    monkeypatch.setattr(local_service.urllib.request, "urlopen", timeout)
    result = local_service._local_service_http_request(
        {"host": LOCAL_SERVICE_LISTENER_HOST, "port": 12345},
        "/call",
        {"request_id": "write-1"},
        timeout_seconds=1,
    )
    assert result == {
        "ok": False,
        "error": "local service transport timed out",
        "reason_code": "transport_timeout",
    }


def test_listener_drains_bounded_queue_before_dispatcher_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mcp_call, "LOCAL_SERVICE_LISTENER_QUEUE_SIZE", 1)
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    entered = threading.Event()
    release = threading.Event()

    def fake_call(_name: str, _arguments: dict[str, object]) -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=5)
        return {"ok": True}

    monkeypatch.setattr(mcp_call, "call_tool", fake_call)
    runtime = mcp_call._LocalServiceListenerRuntime(identity, _HealthyClient())
    runtime.start()
    try:
        issued = runtime.issue_lease(_lease_request(identity))
        assert issued["ok"] is True
        lease = issued.get("lease")
        assert isinstance(lease, str)
        first_result: dict[str, object] = {}
        second_result: dict[str, object] = {}
        first = threading.Thread(
            target=lambda: first_result.update(
                runtime.call(_call_request(identity, lease, request_id="call-1"))
            )
        )
        second = threading.Thread(
            target=lambda: second_result.update(
                runtime.call(_call_request(identity, lease, request_id="call-2"))
            )
        )
        first.start()
        assert entered.wait(timeout=5)
        second.start()
        deadline = time.monotonic() + 5
        while runtime._requests.qsize() != 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        saturated = runtime.call(_call_request(identity, lease, request_id="call-3"))
        runtime.begin_shutdown()
        draining = runtime.call(_call_request(identity, lease, request_id="call-4"))
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        release.set()
        runtime.close()

    assert saturated == {
        "ok": False,
        "error": "local service dispatch queue is saturated",
        "reason_code": "dispatch_queue_saturated",
        "retryable": True,
    }
    assert draining == {"ok": False, "error": "local service listener is draining"}
    assert first_result == {"ok": True}
    assert second_result == {"ok": True}
    assert runtime._dispatcher is not None
    assert not runtime._dispatcher.is_alive()


def test_listener_rejects_lease_role_session_and_backend_mismatch_before_dispatch(
    in_process_listener: _InProcessListener,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    def fake_call(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(mcp_call, "call_tool", fake_call)
    lease = _issue_lease(in_process_listener)
    identity = in_process_listener.identity

    rejected_lease = in_process_listener.runtime.call(_call_request(identity, "not-a-lease"))
    rejected_role = in_process_listener.runtime.call(
        {**_call_request(identity, lease, request_id="call-2"), "caller_role": "host_hook"}
    )
    rejected_session = in_process_listener.runtime.call(
        _call_request(identity, lease, request_id="call-3", session_id="other-run")
    )
    rejected_backend = in_process_listener.runtime.call(
        _call_request(
            identity,
            lease,
            request_id="call-4",
            arguments={"backend": "hindsight"},
        )
    )

    assert rejected_lease == {"ok": False, "error": "local service lease rejected"}
    assert rejected_role == {"ok": False, "error": "local service caller role mismatch"}
    assert rejected_session == {
        "ok": False,
        "error": "local service session identity rejected",
    }
    assert rejected_backend == {
        "ok": False,
        "error": "local service request identity is not authorized",
    }
    assert calls == []
    assert tmp_path.exists()


def test_listener_rejects_role_disallowed_mutation_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="host_hook",
    )
    listener = _InProcessListener(identity)
    calls: list[object] = []

    def fake_call(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(mcp_call, "call_tool", fake_call)
    try:
        lease = _issue_lease(listener)
        result = listener.runtime.call(
            _call_request(identity, lease, tool="promote", arguments={"memory_id": "blocked"})
        )
    finally:
        listener.close()

    assert result == {
        "ok": False,
        "error": "local service caller role is not allowed for tool",
    }
    assert calls == []


def test_listener_dispatch_projects_host_paths_from_common_dispatcher(
    in_process_listener: _InProcessListener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_marker = "HOST_INFRASTRUCTURE_SECRET"

    def fake_call(_name: str, _arguments: dict[str, object]) -> dict[str, object]:
        return {
            "ok": True,
            "backend": "mempalace",
            "workspace_id": "alpha",
            "memory_id": "memory-1",
            "source_path": "src/um_mem/memory/mcp_call.py",
            "repo_root": rf"C:\\private\\{host_marker}",
            "vault_dir": rf"C:\\private\\{host_marker}\\vault",
            "files_written": [rf"C:\\private\\{host_marker}\\memory.md"],
        }

    monkeypatch.setattr(mcp_call, "call_tool", fake_call)
    lease = _issue_lease(in_process_listener)
    payload = in_process_listener.runtime.call(_call_request(in_process_listener.identity, lease))

    assert payload["ok"] is True
    assert payload["backend"] == "mempalace"
    assert payload["source_path"] == "src/um_mem/memory/mcp_call.py"
    assert host_marker not in json.dumps(payload)
    assert not {"repo_root", "vault_dir", "files_written"}.intersection(payload)


def test_listener_rejects_oversized_request_at_client_and_listener(
    in_process_listener: _InProcessListener,
) -> None:
    oversized = {"content": "x" * LOCAL_SERVICE_MAX_REQUEST_BYTES}
    rejection = local_service._local_service_http_request(
        in_process_listener.endpoint,
        "/call",
        oversized,
        timeout_seconds=1,
    )

    assert rejection == {
        "ok": False,
        "error": "local service request exceeds protocol limit",
    }


def test_listener_replaces_oversized_common_dispatch_response(
    in_process_listener: _InProcessListener,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(_name: str, _arguments: dict[str, object]) -> dict[str, object]:
        return {
            "ok": True,
            "backend": "mempalace",
            "workspace_id": "alpha",
            "detail": "x" * LOCAL_SERVICE_MAX_REQUEST_BYTES,
        }

    monkeypatch.setattr(mcp_call, "call_tool", fake_call)
    lease = _issue_lease(in_process_listener)
    response = local_service._local_service_http_request(
        in_process_listener.endpoint,
        "/call",
        _call_request(in_process_listener.identity, lease),
        timeout_seconds=5,
    )

    assert response == {
        "ok": False,
        "error": "local service response exceeds protocol limit",
    }


def test_client_pool_reuses_one_real_listener_across_short_lived_processes(
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    try:
        first = _client_process(tmp_path, backend="mempalace", tool="health")
        second = _client_process(tmp_path, backend="mempalace", tool="health")
    finally:
        _stop_listener(identity)

    assert first["result"]["ok"] is True
    assert second["result"]["ok"] is True
    assert first["result"]["backend"] == "mempalace"
    assert second["result"]["backend"] == "mempalace"
    assert first["service"]["state"] == "ready"
    assert second["service"]["state"] == "ready"
    assert first["service"]["listener_pid"] == second["service"]["listener_pid"]
    assert second["service"]["successful_calls"] >= 2


def test_client_pool_rejects_selected_backend_conflict_without_replacing_listener(
    tmp_path: Path,
) -> None:
    selected = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    conflicting = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="hindsight",
        caller_role="operator",
    )
    pool = LocalServiceClientPool()
    try:
        first = pool.call(selected, "health", {}, timeout_seconds=20)
        before = pool.service_status(selected)
        conflict = pool.call(conflicting, "health", {}, timeout_seconds=20)
        after = pool.service_status(selected)
    finally:
        pool.close()
        _stop_listener(selected)
        _stop_listener(conflicting)

    assert first["ok"] is True
    assert before["state"] == "ready"
    assert conflict == {
        "ok": False,
        "error": "workspace already has a different selected memory backend",
    }
    assert after["state"] == "ready"
    assert after["listener_pid"] == before["listener_pid"]


def test_client_pool_replays_durable_mutation_across_fresh_leases(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    content = "The service receipt must prevent duplicate durable writes."
    title = "durable service receipt"
    mutation_correlation_id = "service-mutation-correlation-1"
    arguments = {
        "content": content,
        "title": title,
        "requested_block_id": "workspace_episodes",
    }
    try:
        first = _client_process(
            tmp_path,
            backend="mempalace",
            tool="remember",
            arguments=arguments,
            mutation_correlation_id=mutation_correlation_id,
        )
        second = _client_process(
            tmp_path,
            backend="mempalace",
            tool="remember",
            arguments=arguments,
            mutation_correlation_id=mutation_correlation_id,
        )
    finally:
        _stop_listener(identity)

    first_payload = first["result"]
    second_payload = second["result"]
    assert first_payload["ok"] is True
    assert first_payload["saved"] is True
    assert second_payload == {
        "ok": True,
        "saved": True,
        "canonical_reference": first_payload["memory_id"],
        "canonical_key": first_payload["raw"]["canonical_key"],
        "revision_id": first_payload["raw"]["revision_id"],
        "recorded_at": first_payload["raw"]["recorded_at"],
        "native_reference": first_payload["raw"].get("native_id"),
        "backend": "mempalace",
        "workspace_id": "alpha",
        "replayed": True,
    }
    receipt_path = tmp_path / ".UM-MEM" / "memory" / "service_receipts.sqlite"
    receipt_bytes = receipt_path.read_bytes()
    for value in (content, title, mutation_correlation_id):
        assert value.encode("utf-8") not in receipt_bytes
    captures = [
        event for event in read_memory_events(tmp_path) if event["event_type"] == "memory_capture"
    ]
    assert len(captures) == 1


def test_local_service_operation_status_reads_caller_scoped_receipt(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    pool = LocalServiceClientPool()
    try:
        saved = pool.call(
            identity,
            "remember",
            {
                "content": "Status reads the original scoped receipt.",
                "requested_block_id": "workspace_episodes",
            },
            timeout_seconds=20,
            mutation_correlation_id="status-op-1",
        )
        status = pool.call(
            identity,
            "operation_status",
            {"operation_id": "status-op-1"},
            timeout_seconds=20,
        )
        listing = pool.call(
            identity,
            "list_records",
            {"scope": "workspace", "scope_id": "alpha", "logical_block_id": "workspace_episodes"},
            timeout_seconds=20,
            session_id="another-session",
        )
    finally:
        pool.close()
        _stop_listener(identity)

    assert saved["ok"] is True
    assert status["ok"] is True
    assert status["record_id"] == saved["memory_id"]
    assert status["original_receipt"]["canonical_reference"] == saved["memory_id"]
    assert status["durable_status"] == "committed"
    assert listing["ok"] is True
    assert saved["memory_id"] in {row["record_id"] for row in listing["records"]}


def test_client_pool_replays_durable_mutation_after_listener_restart(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    arguments = {
        "content": "Restart replay must preserve the one durable capture.",
        "title": "restart replay",
        "requested_block_id": "workspace_episodes",
    }
    correlation = "restart-replay-correlation"
    first_pool = LocalServiceClientPool()
    try:
        first = first_pool.call(
            identity,
            "remember",
            arguments,
            timeout_seconds=20,
            mutation_correlation_id=correlation,
        )
        first_status = first_pool.service_status(identity)
    finally:
        first_pool.close()

    shutdown_local_service_listener(identity)
    second_pool = LocalServiceClientPool()
    try:
        second = second_pool.call(
            identity,
            "remember",
            arguments,
            timeout_seconds=20,
            mutation_correlation_id=correlation,
        )
        second_status = second_pool.service_status(identity)
    finally:
        second_pool.close()
        _stop_listener(identity)

    assert first["ok"] is True
    assert first["saved"] is True
    assert second == {
        "ok": True,
        "saved": True,
        "canonical_reference": first["memory_id"],
        "canonical_key": first["raw"]["canonical_key"],
        "revision_id": first["raw"]["revision_id"],
        "recorded_at": first["raw"]["recorded_at"],
        "native_reference": first["raw"].get("native_id"),
        "backend": "mempalace",
        "workspace_id": "alpha",
        "replayed": True,
    }
    assert first_status["listener_pid"] != second_status["listener_pid"]
    captures = [
        event for event in read_memory_events(tmp_path) if event["event_type"] == "memory_capture"
    ]
    assert len(captures) == 1


def test_host_observation_receipt_is_idempotent_and_keeps_routing_fields(
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="host_hook",
    )
    pool = LocalServiceClientPool()
    arguments = {
        "event_id": "host-test-result-1",
        "event_kind": "test_passed",
        "raw_evidence": "pytest output: 3 passed",
        "atomic_claim": "The targeted pytest run passed.",
        "source_path": "artifacts/pytest.txt",
        "artifact_hash": "sha256:" + ("a" * 64),
        "observed_at": "2026-07-30T10:00:00Z",
        "producer_id": "hermes:session-1",
    }
    try:
        first = pool.call(
            identity,
            "observe",
            arguments,
            timeout_seconds=20,
            mutation_correlation_id="host-observation-1",
            session_id="session-1",
        )
        second = pool.call(
            identity,
            "observe",
            {**arguments, "observed_at": "2026-07-30T10:01:00Z"},
            timeout_seconds=20,
            mutation_correlation_id="host-observation-1",
            session_id="session-1",
        )
        changed_evidence = pool.call(
            identity,
            "observe",
            {**arguments, "raw_evidence": "pytest output: 2 failed"},
            timeout_seconds=20,
            mutation_correlation_id="host-observation-1",
            session_id="session-1",
        )
    finally:
        pool.close()
        _stop_listener(identity)

    assert first["ok"] is True
    assert first["requested_block_id"] == ""
    assert first["actual_block_id"] == ""
    assert first["memory_id"] == ""
    assert first["route_outcome"] == "needs_agent_choice"
    assert second == {
        "ok": True,
        "saved": True,
        "witness_id": first["witness_id"],
        "memory_id": "",
        "requested_block_id": "",
        "actual_block_id": "",
        "route_outcome": "needs_agent_choice",
        "reason": "semantic_placement_requires_explicit_agent_choice",
        "status": "observed",
        "authority": "advisory",
        "verification_status": "observed",
        "evidence_refs": first["evidence_refs"],
        "backend": "mempalace",
        "workspace_id": "alpha",
        "replayed": True,
    }
    assert changed_evidence == {
        "ok": False,
        "error": "local service remember receipt conflict",
        "error_type": "ValueError",
        "reason_code": "mutation_receipt_conflict",
        "retryable": False,
    }


def test_host_conversation_source_replay_preserves_exact_binding_receipt(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="host_hook",
    )
    pool = LocalServiceClientPool()
    arguments = {
        "event_id": "source-turn-1",
        "event_kind": "conversation_turn",
        "raw_evidence": "  Ада 🧠\r\n",
        "observed_at": "2026-10-15T10:00:00Z",
        "producer_id": "trusted-host",
        "speaker": " Ада ",
        "source_occurred_at": "2026-10-15T09:00:00Z",
    }
    try:
        first = pool.call(
            identity,
            "observe",
            arguments,
            timeout_seconds=20,
            mutation_correlation_id="source-turn-1",
            session_id="session-1",
        )
        replay = pool.call(
            identity,
            "observe",
            {**arguments, "observed_at": "2026-10-15T11:00:00Z"},
            timeout_seconds=20,
            mutation_correlation_id="source-turn-1",
            session_id="session-1",
        )
    finally:
        pool.close()
        _stop_listener(identity)

    assert first["ok"] is True
    assert first["route_outcome"] == "raw_witness_only"
    assert first["source_id"] == "source-turn-1"
    assert first["source_sha256"]
    assert first["source_bytes"] == len(arguments["raw_evidence"].encode("utf-8"))
    assert replay["source_id"] == first["source_id"]
    assert replay["source_sha256"] == first["source_sha256"]
    assert replay["source_bytes"] == first["source_bytes"]
    assert replay["witness_id"] == first["witness_id"]
    assert replay["replayed"] is True


def test_compact_remember_receipt_separates_canonical_and_native_references() -> None:
    receipt = mcp_call._compact_remember_receipt(
        {
            "ok": True,
            "saved": True,
            "memory_id": "mem_canonical_reference",
            "backend": "mempalace",
            "workspace_id": "alpha",
            "raw": {
                "canonical_key": "alpha:observation:semantic-key",
                "revision_id": "rev_semantic",
                "recorded_at": "2026-07-14T09:00:00Z",
                "native_id": "native-backend-reference",
            },
        }
    )

    assert receipt["canonical_reference"] == "mem_canonical_reference"
    assert receipt["canonical_key"] == "alpha:observation:semantic-key"
    assert receipt["native_reference"] == "native-backend-reference"

    no_native_receipt = mcp_call._compact_remember_receipt(
        {
            "ok": True,
            "saved": True,
            "memory_id": "mem_canonical_reference",
            "backend": "mempalace",
            "workspace_id": "alpha",
            "raw": {
                "canonical_key": "alpha:observation:semantic-key",
                "revision_id": "rev_semantic",
                "recorded_at": "2026-07-14T09:00:00Z",
            },
        }
    )
    assert no_native_receipt["native_reference"] is None


def test_generic_mutation_receipt_replays_content_free_amend_identity(tmp_path: Path) -> None:
    arguments = {
        "memory_id": "mem_original",
        "expected_revision": "rev_original",
        "reason": "source changed",
    }
    status, response, _ = claim_remember_receipt(
        tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
        protocol_version=1,
        mutation_correlation_id="amend-1",
        arguments=arguments,
    )
    assert status == "new"
    assert response is None

    receipt = mcp_call._compact_generic_mutation_receipt(
        "amend",
        {
            "ok": True,
            "saved": True,
            "new_id": "mem_replacement",
            "old_revision": "rev_original",
            "new_revision": "rev_replacement",
            "backend": "mempalace",
            "workspace_id": "alpha",
        },
    )
    complete_remember_receipt(
        tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
        protocol_version=1,
        mutation_correlation_id="amend-1",
        arguments=arguments,
        response=receipt,
    )

    replay_status, replay, _ = claim_remember_receipt(
        tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
        protocol_version=1,
        mutation_correlation_id="amend-1",
        arguments=arguments,
    )
    assert replay_status == "committed"
    assert replay == receipt
    assert set(replay or {}) == {
        "ok",
        "saved",
        "mutation_tool",
        "canonical_reference",
        "previous_revision",
        "revision",
        "archived",
        "backend",
        "workspace_id",
    }


def test_local_service_receipt_preserves_explicit_correction_target() -> None:
    arguments = {
        "content": "corrected content",
        "operation": "correct",
        "memory_id": "mem_original",
        "requested_block_id": "decisions_and_contracts",
    }

    receipt_arguments = mcp_call._receipt_remember_arguments(
        arguments,
        recorded_at="2026-07-22T00:00:00Z",
    )
    receipt = mcp_call._compact_remember_receipt(
        {
            "ok": True,
            "saved": True,
            "operation": "correct",
            "memory_id": "mem_replacement",
            "canonical_key": "alpha:observation:decision",
            "revision_id": "rev_replacement",
            "recorded_at": "2026-07-22T00:00:00Z",
            "native_id": "mem_replacement",
            "backend": "mempalace",
            "workspace_id": "alpha",
        }
    )

    assert receipt_arguments["memory_id"] == "mem_original"
    assert receipt["canonical_reference"] == "mem_replacement"
    assert receipt["revision_id"] == "rev_replacement"


def test_local_service_remember_correction_supersedes_selected_target(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    pool = LocalServiceClientPool()
    try:
        initial = pool.call(
            identity,
            "remember",
            {
                "content": "The initial decision is OLD-VALUE.",
                "requested_block_id": "decisions_and_contracts",
                "evidence_kind": "agent_tool_claim",
                "source_id": "hermes:producer-1",
            },
            timeout_seconds=20,
            session_id="producer-1",
        )
        correction_arguments = {
            "content": "The corrected decision is CURRENT-VALUE; OLD-VALUE is rejected.",
            "operation": "correct",
            "memory_id": initial["memory_id"],
            "requested_block_id": "decisions_and_contracts",
        }
        corrected = pool.call(
            identity,
            "remember",
            correction_arguments,
            timeout_seconds=20,
            mutation_correlation_id="correction-receipt-1",
            session_id="producer-2",
        )
        replayed = pool.call(
            identity,
            "remember",
            correction_arguments,
            timeout_seconds=20,
            mutation_correlation_id="correction-receipt-1",
            session_id="producer-2",
        )
        projected = pool.call(
            identity,
            "projection_barrier",
            {"record_id": corrected["memory_id"], "timeout_seconds": 20},
            timeout_seconds=20,
            session_id="consumer",
        )
        scoped = pool.call(
            identity,
            "build_context",
            {
                "query": "CURRENT-VALUE corrected decision",
                "logical_block_ids": ["decisions_and_contracts"],
                "include_candidates": True,
                "search_all": False,
                "render_context": True,
            },
            timeout_seconds=20,
            session_id="consumer",
        )
    finally:
        pool.close()
        _stop_listener(identity)

    assert initial["ok"] is True
    assert corrected["ok"] is True, corrected
    assert corrected["operation"] == "correct"
    assert corrected["memory_id"] != initial["memory_id"]
    assert replayed["replayed"] is True
    assert replayed["canonical_reference"] == corrected["memory_id"]
    assert projected["ok"] is True, projected
    assert len(scoped["items"]) == 1
    native_trace = scoped["diagnostics"]["native_query_trace"]
    assert native_trace["entry_count"] == 1
    assert native_trace["candidate_record_count"] >= 1
    assert native_trace["selected_record_count"] == 1


def test_pending_receipt_replays_through_the_planner_without_canonical_metadata(
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    arguments = {
        "content": "A pending receipt must replay its routed mutation.",
        "title": "pending receipt replay",
        "requested_block_id": "workspace_episodes",
        "metadata": {"source_ref": "service-receipt:pending-replay"},
    }
    correlation = "pending-receipt-replay"
    status, _, _ = claim_remember_receipt(
        identity.repo_root,
        workspace_id=identity.workspace_id,
        backend=identity.backend,
        caller_role=identity.caller_role,
        protocol_version=identity.protocol_version,
        mutation_correlation_id=correlation,
        arguments={**identity.bound_arguments(arguments), "run_id": "default"},
    )
    assert status == "new"

    pool = LocalServiceClientPool()
    try:
        result = pool.call(
            identity,
            "remember",
            arguments,
            timeout_seconds=20,
            mutation_correlation_id=correlation,
        )
    finally:
        pool.close()
        _stop_listener(identity)

    assert result["ok"] is True
    assert result["saved"] is True
    assert result["replayed"] is True
    assert result["recovered"] is True
    assert "service_receipt_sha256" not in json.dumps(result, sort_keys=True)


def test_mcp_server_uses_shared_listener_for_selected_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = {
        "repo_root": str(tmp_path),
        "workspace_id": "alpha",
        "backend": "mempalace",
    }
    monkeypatch.setenv("UM_MEM_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("UM_MEM_WORKSPACE", "alpha")
    monkeypatch.setenv("UM_MEM_BACKEND", "mempalace")
    identity = build_local_service_identity(**arguments)
    mcp_server.close_local_service_clients()
    try:
        first = mcp_server._call_tool("health", arguments)
        second = mcp_server._call_tool("health", arguments)
    finally:
        mcp_server.close_local_service_clients()
        _stop_listener(identity)

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["service"]["state"] == "ready"
    assert second["service"]["listener_pid"] == first["service"]["listener_pid"]
    assert second["service"]["successful_calls"] >= 2
    assert "bootstrap_secret" not in second["service"]
    assert "lease" not in second["service"]


def test_routed_policy_rejection_replays_through_common_dispatcher(
    in_process_listener: _InProcessListener,
) -> None:
    lease = _issue_lease(in_process_listener)
    identity = in_process_listener.identity
    request = _call_request(
        identity,
        lease,
        tool="remember",
        arguments={
            "content": "A direct directive must remain rejected.",
            "title": "blocked directive",
            "kind": "bkb_rule",
            "requested_block_id": "workspace_core_directives",
        },
    )
    request["mutation_correlation_id"] = "routed-policy-rejection-1"
    first = in_process_listener.runtime.call(request)
    replayed = in_process_listener.runtime.call({**request, "request_id": "call-2"})

    expected = {
        "ok": False,
        "saved": False,
        "route_rejected": True,
        "reason": "agent_write_requires_typed_promotion_route",
        "suggested_route": {
            "action": "retry_remember",
            "requested_block_id": "decisions_and_contracts",
        },
        "backend": "mempalace",
        "workspace_id": "alpha",
    }
    route_decision = first.pop("route_decision")
    assert route_decision["accepted"] is False
    assert route_decision["reason"] == "agent_write_requires_typed_promotion_route"
    assert route_decision["suggested_block_id"] == "decisions_and_contracts"
    assert route_decision["policy_issues"]
    assert first == expected

    replayed_route_decision = replayed.pop("route_decision")
    assert replayed_route_decision == route_decision
    assert replayed == {**expected, "replayed": True}
    receipt_path = identity.repo_root / ".UM-MEM" / "memory" / "service_receipts.sqlite"
    with sqlite3.connect(receipt_path) as connection:
        statuses = connection.execute("SELECT status FROM remember_receipts").fetchall()
    assert sorted(statuses) == [("committed",), ("committed",)]
