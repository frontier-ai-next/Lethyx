import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import venv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from lethyx.memory import local_service, mcp_call
from lethyx.memory.local_service import (
    LocalServiceClientPool,
    build_local_service_identity,
    shutdown_local_service_listener,
)
from lethyx.memory.service_scheduler import HostStateEvent


@pytest.mark.skipif(os.name == "nt", reason="POSIX virtual-environment symlink contract")
def test_listener_launcher_retains_virtual_environment_dependencies(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    executable = environment / "bin" / "python"
    assert executable.is_symlink()
    install_path = subprocess.run(
        [str(executable), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    marker = Path(install_path) / "listener_environment_probe.py"
    marker.write_text("VALUE = 'venv-only-dependency'\n", encoding="utf-8")
    pool = LocalServiceClientPool(python_executable=str(executable))
    identity = build_local_service_identity(
        repo_root=tmp_path / "memory",
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )

    selected = pool._listener_command(identity)[0]
    result = subprocess.run(
        [selected, "-c", "import listener_environment_probe as probe; print(probe.VALUE)"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "venv-only-dependency"
    assert selected == str(executable.absolute())


def test_listener_launcher_makes_relative_path_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    pool = LocalServiceClientPool(python_executable="venv/bin/python")

    assert pool.python_executable == str(tmp_path / "venv" / "bin" / "python")


def test_listener_startup_reports_actual_child_exit_without_waiting_for_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pool = LocalServiceClientPool()
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    monkeypatch.setattr(
        pool,
        "_listener_command",
        lambda _identity: [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('PRIVATE_STARTUP_DETAIL'); sys.exit(7)",
        ],
    )

    started = time.monotonic()
    result = pool.call(identity, "health", {}, timeout_seconds=5, session_id="probe")
    elapsed = time.monotonic() - started
    pool.close()

    assert result == {
        "ok": False,
        "error": "local service listener exited before readiness (exit code 7)",
    }
    assert elapsed < 4


def test_ready_registry_takes_precedence_over_launched_child_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
    )
    pool = LocalServiceClientPool()
    endpoint = {"backend": "mempalace", "host": "127.0.0.1", "port": 12345, "pid": 42}
    launched = False

    def launch(_identity: object) -> subprocess.Popen[str]:
        nonlocal launched
        child = subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(7)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        child.wait(timeout=10)
        launched = True
        return child

    monkeypatch.setattr(pool, "_launch_listener", launch)
    monkeypatch.setattr(
        local_service, "local_service_registry", lambda _identity: endpoint if launched else None
    )
    monkeypatch.setattr(
        local_service,
        "_listener_health",
        lambda *_args, **_kwargs: {
            "ok": True,
            "liveness": "live",
            "readiness": "ready",
            "native_health": "healthy",
        },
    )

    resolved, error = pool._ensure_listener(identity, timeout_seconds=5)

    assert resolved == endpoint
    assert error is None


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher contract")
def test_listener_launcher_uses_hidden_pythonw_sibling_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv = tmp_path / "venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    site_packages = venv / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    python = scripts / "python.exe"
    shim_pythonw = scripts / "pythonw.exe"
    base_runtime = tmp_path / "base-runtime"
    base_runtime.mkdir()
    pythonw = base_runtime / "pythonw.exe"
    python.touch()
    shim_pythonw.touch()
    pythonw.touch()
    (venv / "pyvenv.cfg").write_text(f"home = {base_runtime}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_popen(arguments: list[str], **kwargs: object) -> object:
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(local_service.subprocess, "Popen", fake_popen)
    pool = LocalServiceClientPool(python_executable=str(python), environment={})
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )

    pool._launch_listener(identity)

    assert captured["arguments"][0] == str(pythonw.resolve())
    assert captured["arguments"][1] == "-c"
    assert captured["arguments"][3] == str(site_packages)
    assert "--owner-pid" in captured["arguments"]
    creationflags = captured["kwargs"]["creationflags"]
    assert creationflags & subprocess.CREATE_NO_WINDOW
    assert captured["kwargs"]["close_fds"] is True


@pytest.mark.skipif(os.name != "nt", reason="Windows owner-inference contract")
def test_listener_owner_uses_venv_shim_grandparent_without_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent_pid = 600
    grandparent_pid = 700
    shim = tmp_path / "venv" / "Scripts" / "python.exe"
    base = tmp_path / "base" / "python.exe"
    monkeypatch.setattr(local_service.os, "getppid", lambda: parent_pid)
    monkeypatch.setattr(
        local_service,
        "_windows_process_image_path",
        lambda pid: shim if pid == parent_pid else None,
    )
    monkeypatch.setattr(
        local_service,
        "_windows_parent_process_id",
        lambda pid: grandparent_pid if pid == parent_pid else None,
    )
    monkeypatch.setattr(local_service, "_process_is_alive", lambda pid: pid == grandparent_pid)
    monkeypatch.setattr(local_service.sys, "executable", str(shim))
    monkeypatch.setattr(local_service.sys, "_base_executable", str(base), raising=False)

    assert local_service._listener_owner_pid({}) == grandparent_pid


@pytest.mark.skipif(os.name != "nt", reason="Windows process liveness contract")
def test_windows_process_liveness_rejects_an_exited_child_with_an_open_handle() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    child.wait(timeout=10)

    assert local_service._process_is_alive(child.pid) is False


def test_unreachable_live_listener_is_not_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    endpoint = {
        "backend": "mempalace",
        "host": "127.0.0.1",
        "port": 12345,
        "pid": 42,
    }
    pool = LocalServiceClientPool()
    launches: list[bool] = []
    monkeypatch.setattr(local_service, "local_service_registry", lambda _identity: endpoint)
    monkeypatch.setattr(local_service, "_listener_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        local_service,
        "_listener_endpoint_refuses_connections",
        lambda _endpoint, **_kwargs: False,
    )
    monkeypatch.setattr(pool, "_launch_listener", lambda _identity: launches.append(True))

    resolved, error = pool._ensure_listener(identity, timeout_seconds=1)

    assert resolved is None
    assert error == "local service listener unavailable"
    assert launches == []


def test_listener_endpoint_is_retired_only_for_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = {"host": "127.0.0.1", "port": 12345}

    def refused(*_args, **_kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(local_service.socket, "create_connection", refused)
    assert local_service._listener_endpoint_refuses_connections(endpoint) is True

    def timed_out(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr(local_service.socket, "create_connection", timed_out)
    assert local_service._listener_endpoint_refuses_connections(endpoint) is False


def test_reused_pid_with_refused_registry_endpoint_is_replaced_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stale_socket,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as replacement_socket,
    ):
        stale_socket.bind(("127.0.0.1", 0))
        replacement_socket.bind(("127.0.0.1", 0))
        stale_port = stale_socket.getsockname()[1]
        replacement_port = replacement_socket.getsockname()[1]
    local_service.write_local_service_registry(
        identity,
        port=stale_port,
        pid=os.getpid(),
    )
    pool = LocalServiceClientPool()

    def health(endpoint, *_args, **_kwargs):
        if endpoint["port"] == stale_port:
            return None
        return {
            "ok": True,
            "liveness": "live",
            "readiness": "ready",
            "native_health": "healthy",
            "degraded_state": None,
        }

    def launch(listener_identity):
        local_service.write_local_service_registry(
            listener_identity,
            port=replacement_port,
            pid=os.getpid(),
        )
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(local_service, "_listener_health", health)
    monkeypatch.setattr(
        local_service,
        "_listener_endpoint_refuses_connections",
        lambda endpoint, **_kwargs: endpoint["port"] == stale_port,
    )
    monkeypatch.setattr(pool, "_launch_listener", launch)

    try:
        resolved, error = pool._ensure_listener(identity, timeout_seconds=2)

        assert error is None
        assert resolved is not None
        assert resolved["port"] == replacement_port
        assert local_service.local_service_registry(identity) == resolved
    finally:
        local_service.clear_local_service_registry(identity, pid=os.getpid())


def test_mcp_call_rejects_retired_direct_stdin_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        mcp_call,
        "call_tool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy dispatch")),
    )

    assert mcp_call.main(["remember"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "mcp_call only supports the authenticated local service listener"


def test_listener_registry_reaps_a_dead_process_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    local_service.write_local_service_registry(identity, port=12345, pid=424242)
    monkeypatch.setattr(local_service, "_process_is_alive", lambda _pid: False)

    assert local_service.local_service_registry(identity) is None
    assert not local_service.local_service_runtime_paths(identity).registry.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux unreaped-child lifecycle")
def test_listener_registry_reaps_an_exited_unreaped_child(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
    )
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])
    try:
        local_service.write_local_service_registry(identity, port=12345, pid=child.pid)
        deadline = time.monotonic() + 5
        stat_path = Path(f"/proc/{child.pid}/stat")
        while stat_path.read_text().rsplit(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline, "child did not exit"
            time.sleep(0.01)
        # Signal zero still succeeds until the parent collects the exit status.
        os.kill(child.pid, 0)
        assert local_service.local_service_registry(identity) is None
        assert not local_service.local_service_runtime_paths(identity).registry.exists()
    finally:
        child.wait(timeout=5)


def test_listener_discovery_cannot_reap_another_host_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
    )
    monkeypatch.setattr(
        local_service,
        "_local_service_host_scope",
        lambda: ("host", "pid-a", "net-a"),
        raising=False,
    )
    local_service.write_local_service_registry(identity, port=12345, pid=424242)
    first = local_service.local_service_runtime_paths(identity)
    first_secret = local_service.local_service_bootstrap_secret(identity)
    monkeypatch.setattr(local_service, "_process_is_alive", lambda _pid: False)
    monkeypatch.setattr(
        local_service,
        "_local_service_host_scope",
        lambda: ("host", "pid-b", "net-b"),
        raising=False,
    )

    assert local_service.local_service_registry(identity) is None
    second = local_service.local_service_runtime_paths(identity)
    assert first.registry.exists()
    assert first.registry != second.registry
    assert first.startup_lock != second.startup_lock
    assert first_secret != local_service.local_service_bootstrap_secret(identity)


def test_listener_discovery_shares_one_namespace_across_roles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        local_service,
        "_local_service_host_scope",
        lambda: ("host", "pid-a", "net-a"),
        raising=False,
    )
    operator = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="operator"
    )
    supervisor = build_local_service_identity(
        repo_root=tmp_path, workspace_id="alpha", backend="mempalace", caller_role="supervisor"
    )
    assert local_service.local_service_runtime_paths(
        operator
    ) == local_service.local_service_runtime_paths(supervisor)
    assert local_service.local_service_bootstrap_secret(
        operator
    ) != local_service.local_service_bootstrap_secret(supervisor)


def test_listener_startup_lock_reclaims_a_dead_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    lock_path = local_service.local_service_runtime_paths(identity).startup_lock
    lock_path.write_text(json.dumps({"pid": 424242, "created_at": time.time()}), encoding="ascii")
    monkeypatch.setattr(local_service, "_process_is_alive", lambda pid: pid == os.getpid())

    descriptor = local_service._acquire_listener_startup_lock(identity, 1)
    assert descriptor is not None
    try:
        payload = json.loads(lock_path.read_text(encoding="ascii"))
        assert payload["pid"] == os.getpid()
    finally:
        local_service._release_listener_startup_lock(identity, descriptor)

    assert not lock_path.exists()


def test_listener_startup_file_lock_honors_caller_deadline(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    lock_path = local_service.local_service_runtime_paths(identity).startup_lock
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "created_at": time.time()}), encoding="ascii"
    )
    started = time.monotonic()

    descriptor = local_service._acquire_listener_startup_lock(
        identity, 5, deadline_ns=time.monotonic_ns() + 150_000_000
    )

    assert descriptor is None
    assert time.monotonic() - started < 0.5
    assert lock_path.exists()


def test_listener_lifecycle_monitor_stops_when_owner_is_gone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    runtime = mcp_call._LocalServiceListenerRuntime(
        identity,
        object(),
        idle_timeout_seconds=600,
    )
    stopped = threading.Event()
    shutdowns: list[bool] = []

    class Server:
        def shutdown(self) -> None:
            shutdowns.append(True)

    monkeypatch.setattr(mcp_call, "_process_is_alive", lambda _pid: False)
    try:
        mcp_call._listener_lifecycle_monitor(
            Server(),  # type: ignore[arg-type]
            runtime,
            owner_pid=424242,
            stopped=stopped,
        )
    finally:
        runtime.close()

    assert shutdowns == [True]


def test_listener_reaps_after_bounded_idle_timeout(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    pool = LocalServiceClientPool(
        environment={**os.environ, "LETHYX_LOCAL_SERVICE_IDLE_SECONDS": "1"}
    )
    try:
        result = pool.call(identity, "health", {}, timeout_seconds=20)
        endpoint = local_service.local_service_registry(identity)
        assert endpoint is not None
        pid = endpoint["pid"]
        assert isinstance(pid, int)
        deadline = time.monotonic() + 8
        while local_service.local_service_registry(
            identity
        ) is not None or local_service._process_is_alive(pid):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    finally:
        pool.close()
        shutdown_local_service_listener(identity)

    assert result["ok"] is True
    assert local_service.local_service_registry(identity) is None
    assert local_service._process_is_alive(pid) is False


def test_idle_monitor_keeps_listener_alive_for_a_later_persisted_scheduler_timer(
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )

    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(
        identity,
        Client(),  # type: ignore[arg-type]
        idle_timeout_seconds=1,
    )
    try:
        runtime._scheduler.submit(
            HostStateEvent(
                event_id="future-scheduler-job",
                event_type="tool_result",
                workspace_id="alpha",
                run_id="run-1",
                stage_id="stage_2_5",
                state_revision="policy-bundle@1",
                occurred_at=datetime.now(UTC) + timedelta(minutes=10),
            )
        )
        runtime._last_work_at = time.monotonic() - 2

        assert runtime.is_idle() is False
    finally:
        runtime.close()


def test_scheduler_failure_is_reported_as_explicit_degraded_health(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )

    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(identity, Client())  # type: ignore[arg-type]
    runtime.start()
    try:

        def fail_recover_due(*_args: object, **_kwargs: object) -> tuple[object, ...]:
            raise sqlite3.OperationalError("scheduler database unavailable")

        monkeypatch.setattr(runtime._scheduler, "recover_due", fail_recover_due)
        deadline = time.monotonic() + 5
        status: dict[str, object] = {}
        while time.monotonic() < deadline:
            status = runtime.health()
            if status.get("degraded_state") == "scheduler_unavailable":
                break
            time.sleep(0.05)
    finally:
        runtime.close()

    assert status["liveness"] == "live"
    assert status["readiness"] == "not_ready"
    assert status["degraded_state"] == "scheduler_unavailable"
    assert status["error"] == "local service scheduler unavailable"


@pytest.mark.skipif(os.name != "nt", reason="Windows owner-death contract")
def test_listener_reaps_when_its_explicit_owner_exits(tmp_path: Path) -> None:
    owner = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    pool = LocalServiceClientPool(
        environment={**os.environ, "LETHYX_LOCAL_SERVICE_OWNER_PID": str(owner.pid)}
    )
    try:
        result = pool.call(identity, "health", {}, timeout_seconds=20)
        endpoint = local_service.local_service_registry(identity)
        assert endpoint is not None
        listener_pid = endpoint["pid"]
        assert isinstance(listener_pid, int)
        owner.wait(timeout=10)
        deadline = time.monotonic() + 8
        while local_service.local_service_registry(
            identity
        ) is not None or local_service._process_is_alive(listener_pid):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=10)
        pool.close()
        shutdown_local_service_listener(identity)

    assert result["ok"] is True
    assert local_service.local_service_registry(identity) is None
    assert local_service._process_is_alive(listener_pid) is False


def test_idle_monitor_does_not_stop_an_inflight_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )

    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(
        identity,
        Client(),  # type: ignore[arg-type]
        idle_timeout_seconds=1,
    )
    started = threading.Event()
    release = threading.Event()
    shutdown = threading.Event()

    class Server:
        def shutdown(self) -> None:
            shutdown.set()

    def blocking_call_tool(*_args: object, **_kwargs: object) -> dict[str, object]:
        started.set()
        assert release.wait(timeout=5)
        return {"ok": True, "workspace_id": "alpha", "backend": "mempalace"}

    monkeypatch.setattr(mcp_call, "call_tool", blocking_call_tool)
    monkeypatch.setattr(mcp_call, "_process_is_alive", lambda _pid: True)
    runtime.start()
    lease_response = runtime.issue_lease(
        {
            "protocol_version": identity.protocol_version,
            "schema_version": local_service.LOCAL_SERVICE_SCHEMA_VERSION,
            "api_version": local_service.LOCAL_SERVICE_API_VERSION,
            "request_id": "lease",
            "caller_role": identity.caller_role,
            "session_id": "idle-test",
            "capabilities": sorted(
                tool
                for tool in mcp_call.TOOLS
                if local_service.local_service_role_allows(identity.caller_role, tool)
            ),
            "bootstrap_secret": local_service.local_service_bootstrap_secret(identity),
        }
    )
    lease = lease_response.get("lease")
    assert isinstance(lease, str)
    result: dict[str, object] = {}
    worker = threading.Thread(
        target=lambda: result.update(
            runtime.call(
                {
                    "protocol_version": identity.protocol_version,
                    "schema_version": local_service.LOCAL_SERVICE_SCHEMA_VERSION,
                    "api_version": local_service.LOCAL_SERVICE_API_VERSION,
                    "request_id": "recall",
                    "lease": lease,
                    "caller_role": identity.caller_role,
                    "session_id": "idle-test",
                    "tool": "recall",
                    "arguments": {},
                }
            )
        )
    )
    stopped = threading.Event()
    monitor = threading.Thread(
        target=mcp_call._listener_lifecycle_monitor,
        args=(Server(), runtime),  # type: ignore[arg-type]
        kwargs={"owner_pid": 42, "stopped": stopped},
    )
    try:
        worker.start()
        assert started.wait(timeout=2)
        with runtime._lifecycle_lock:
            runtime._last_work_at = time.monotonic() - 10
        monitor.start()
        time.sleep(0.6)
        assert shutdown.is_set() is False

        release.set()
        worker.join(timeout=5)
        assert result["ok"] is True
        with runtime._lifecycle_lock:
            runtime._last_work_at = time.monotonic() - 10
        assert shutdown.wait(timeout=2)
    finally:
        release.set()
        worker.join(timeout=5)
        stopped.set()
        monitor.join(timeout=2)
        runtime.close()


def test_service_status_retries_one_transient_listener_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    endpoint = {"backend": "mempalace", "host": "127.0.0.1", "port": 12345}
    responses = iter(
        (
            {"ok": False, "error": "local service listener unavailable"},
            {
                "ok": True,
                "state": "ready",
                "listener_pid": 42,
                "selected_backend": "mempalace",
                "workspace_id": "alpha",
                "profile_id": "default",
                "transport": "loopback_http",
            },
        )
    )
    monkeypatch.setattr(local_service, "local_service_registry", lambda _identity: endpoint)
    monkeypatch.setattr(
        local_service,
        "_local_service_http_request",
        lambda *_args, **_kwargs: next(responses),
    )

    status = LocalServiceClientPool().service_status(identity)

    assert status["state"] == "ready"
    assert status["listener_pid"] == 42


def test_service_status_has_a_stable_shape_when_listener_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    monkeypatch.setattr(local_service, "local_service_registry", lambda _identity: None)

    status = LocalServiceClientPool().service_status(identity)

    assert status["state"] == "not_ready"
    assert status["liveness"] == "not_live"
    assert status["readiness"] == "not_ready"
    assert status["native_health"] == "not_probed"
    assert status["degraded_state"] == "listener_unavailable"
    assert status["effective_mode"] == "unavailable"


def test_draining_listener_is_reportable_but_not_replaced_for_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    endpoint = {
        "backend": "mempalace",
        "host": "127.0.0.1",
        "port": 12345,
        "pid": 42,
    }
    draining = {
        "ok": False,
        "state": "draining",
        "liveness": "live",
        "readiness": "not_ready",
        "native_health": "healthy",
        "native_ok": True,
        "strict_ok": False,
        "effective_mode": "selected_native",
        "degraded_state": "listener_draining",
        "selected_backend": "mempalace",
        "workspace_id": "alpha",
        "profile_id": "default",
        "transport": "loopback_http",
    }
    pool = LocalServiceClientPool()
    monkeypatch.setattr(local_service, "local_service_registry", lambda _identity: endpoint)
    monkeypatch.setattr(
        local_service,
        "_local_service_http_request",
        lambda *_args, **_kwargs: draining,
    )
    monkeypatch.setattr(
        pool,
        "_launch_listener",
        lambda _identity: (_ for _ in ()).throw(AssertionError("must not relaunch")),
    )

    endpoint_result, error = pool._ensure_listener(identity, timeout_seconds=1)

    assert endpoint_result is None
    assert error == "local service listener is draining"


def test_listener_is_reused_by_independent_client_pools(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    first_pool = LocalServiceClientPool()
    try:
        first = first_pool.call(identity, "health", {}, timeout_seconds=20)
        first_status = first_pool.service_status(identity)
    finally:
        first_pool.close()

    second_pool = LocalServiceClientPool()
    try:
        second = second_pool.call(identity, "health", {}, timeout_seconds=20)
        second_status = second_pool.service_status(identity)
    finally:
        second_pool.close()
        shutdown_local_service_listener(identity)

    assert first["ok"] is True
    assert second["ok"] is True
    assert first_status["state"] == "ready"
    assert first_status["transport"] == "loopback_http"
    assert first_status["listener_pid"] == second_status["listener_pid"]
    assert first_status["listener_pid"] > 0
    assert "capability_lease" not in first_status
    assert "bootstrap_secret" not in first_status


def test_listener_is_isolated_by_profile_in_same_workspace(tmp_path: Path) -> None:
    first_identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        profile_id="default",
        backend="mempalace",
        caller_role="operator",
    )
    second_identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        profile_id="umbrella",
        backend="mempalace",
        caller_role="operator",
    )
    pool = LocalServiceClientPool()
    try:
        first = pool.call(first_identity, "health", {}, timeout_seconds=20)
        second = pool.call(second_identity, "health", {}, timeout_seconds=20)
        first_status = pool.service_status(first_identity)
        second_status = pool.service_status(second_identity)
    finally:
        pool.close()
        shutdown_local_service_listener(first_identity)
        shutdown_local_service_listener(second_identity)

    assert first["ok"] is True
    assert second["ok"] is True
    assert first_status["profile_id"] == "default"
    assert second_status["profile_id"] == "umbrella"
    assert first_status["listener_pid"] != second_status["listener_pid"]


def test_listener_rejects_foreign_identity_before_dispatch(tmp_path: Path) -> None:
    identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    pool = LocalServiceClientPool()
    try:
        result = pool.call(
            identity,
            "health",
            {"workspace_id": "foreign-workspace"},
            timeout_seconds=20,
        )
    finally:
        pool.close()
        shutdown_local_service_listener(identity)

    assert result == {
        "ok": False,
        "error": "local service request identity is not authorized",
    }
