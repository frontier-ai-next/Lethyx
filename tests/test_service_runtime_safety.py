import json
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import Future

import pytest

from um_mem.api import MemoryClient
from um_mem.memory import mcp_call
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.erase_journal import open_erase_request, record_erase_request_outcome
from um_mem.memory.local_service import build_local_service_identity
from um_mem.memory.operational_slo import (
    CostSample,
    OperationalSloLedger,
    OperationSample,
    service_slo_contract_revision,
)
from um_mem.memory.service_receipts import claim_remember_receipt, complete_remember_receipt
from um_mem.memory.service_scheduler import ServiceScheduler
from um_mem.memory.service_work_queue import DurableServiceWorkQueue


def _identity(tmp_path, *, backend: str = "mempalace"):
    return build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="release",
        backend=backend,
        caller_role="operator",
    )


def test_ephemeral_dispatch_workers_release_sqlite_connections(tmp_path, monkeypatch) -> None:
    identity = _identity(tmp_path)
    backend = MemPalaceBackend(tmp_path, workspace_id="release")
    client = MemoryClient(tmp_path, workspace_id="release", backend_instance=backend)
    runtime = mcp_call._LocalServiceListenerRuntime(identity, client)
    try:
        baseline = len(backend._index._connections)
        monkeypatch.setattr(
            runtime,
            "_dispatch_item",
            lambda _item: backend._index._conn.execute("SELECT 1").fetchone(),
        )
        for _ in range(12):
            future: Future[None] = Future()
            worker = threading.Thread(
                target=runtime._run_dispatch_item,
                args=(object(), future),  # type: ignore[arg-type]
            )
            worker.start()
            worker.join(timeout=2)
            assert not worker.is_alive()
            future.result(timeout=0)
            assert len(backend._index._connections) == baseline
    finally:
        runtime.close()
        backend.close()


def test_service_database_operations_release_connections_without_garbage_collection(
    tmp_path, monkeypatch
) -> None:
    opened: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def track_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track_connection)
    try:
        identity = _identity(tmp_path)
        queue = DurableServiceWorkQueue(identity)
        scheduler = ServiceScheduler(identity)
        ledger = OperationalSloLedger(identity)
        for _ in range(3):
            assert queue.claim("indexing") is None
            assert queue.health().lanes
            assert scheduler.recover_due() == ()
            ledger.headline_cost_report()
        with pytest.raises(KeyError):
            queue.item("missing-work")
        receipt = {
            "workspace_id": "release",
            "backend": "mempalace",
            "caller_role": "operator",
            "protocol_version": 1,
            "mutation_correlation_id": "write-1",
            "arguments": {"content": "fact"},
        }
        assert claim_remember_receipt(tmp_path, **receipt)[0] == "new"
        complete_remember_receipt(
            tmp_path,
            **receipt,
            response={
                "ok": False,
                "saved": False,
                "route_rejected": True,
                "reason": "scope unavailable",
                "backend": "mempalace",
                "workspace_id": "release",
            },
        )
        assert claim_remember_receipt(tmp_path, **receipt)[0] == "committed"
        with pytest.raises(ValueError, match="conflict"):
            claim_remember_receipt(tmp_path, **(receipt | {"arguments": {"content": "changed"}}))
        erase, _ = open_erase_request(
            tmp_path,
            workspace_id_sha256="a" * 64,
            request_id_sha256="b" * 64,
            fingerprint_sha256="c" * 64,
            target_id_sha256="d" * 64,
            backend="mempalace",
        )
        record_erase_request_outcome(tmp_path, erase, status="completed", error="")
        assert opened
        for connection in opened:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
    finally:
        for connection in opened:
            connection.close()


def test_service_health_publishes_the_numeric_slo_contract_revision(tmp_path) -> None:
    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(  # type: ignore[arg-type]
        _identity(tmp_path),
        Client(),
    )
    runtime.start()
    try:
        status = runtime.health()
    finally:
        runtime.close()

    assert status["readiness"] == "ready"
    assert status["slo_contract_revision"] == service_slo_contract_revision()


@pytest.mark.parametrize(
    "failure", ["database", "response", "access", "transport", "lock", "cas", "timeout"]
)
def test_dispatch_worker_failure_completes_and_releases_slot(
    tmp_path, monkeypatch, caplog, failure: str
) -> None:
    monkeypatch.setattr(mcp_call, "LOCAL_SERVICE_LISTENER_QUEUE_SIZE", 1)
    identity = _identity(tmp_path)
    runtime = mcp_call._LocalServiceListenerRuntime(identity, object())  # type: ignore[arg-type]
    bundle = runtime._policy_store.active_bundle()
    lease = mcp_call._ListenerLease(
        token="lease",
        identity=identity,
        session_id="run-1",
        scope_lineage=mcp_call.ScopeLineage(workspace_id="release", session_id="run-1"),
        expires_at=time.monotonic() + 60,
        policy_bundle_revision=bundle.revision_id,
        catalog_revision=bundle.catalog_revision_id,
        catalog_snapshot=runtime._policy_store.snapshot_for(bundle),
    )

    def fail_dispatch(*_args, **_kwargs):
        if failure == "database":
            raise sqlite3.OperationalError("simulated database failure")
        if failure == "access":
            raise PermissionError("simulated store access denied")
        if failure == "transport":
            raise ConnectionError("simulated native connection failure")
        if failure == "lock":
            raise mcp_call.MemoryBusyError("simulated store lock")
        if failure == "cas":
            raise mcp_call.CanonicalHeadConflict("simulated stale revision")
        if failure == "timeout":
            raise TimeoutError("simulated transport timeout")

    monkeypatch.setattr(mcp_call, "_dispatch_local_service_request", fail_dispatch)
    request = {"request_id": "broken-1", "tool": "block_map"}
    completed = threading.Event()
    result: dict[str, object] = {}
    runtime._active_requests = 1
    assert runtime._dispatch_slots.acquire(blocking=False)
    try:
        runtime._dispatch_item(mcp_call._DispatchItem(lease, request, completed, result))
        assert completed.is_set()
        assert result["ok"] is False
        assert (
            result["reason_code"]
            == {
                "database": "dispatcher_unexpected_error",
                "response": "dispatcher_unexpected_error",
                "access": "access_denied",
                "transport": "native_transport_failure",
                "lock": "store_lock_unavailable",
                "cas": "canonical_revision_conflict",
                "timeout": "transport_timeout",
            }[failure]
        )
        assert result["request_id"] == "broken-1"
        assert result["durable_status"] == "unknown"
        assert runtime._active_requests == 0
        assert runtime._failed_calls == 1
        assert runtime._dispatch_slots.acquire(blocking=False)
        monkeypatch.setattr(
            mcp_call,
            "_dispatch_local_service_request",
            lambda *_args, **_kwargs: {"payload": {"ok": True}},
        )
        next_completed = threading.Event()
        next_result: dict[str, object] = {}
        runtime._active_requests = 1
        runtime._dispatch_item(mcp_call._DispatchItem(lease, request, next_completed, next_result))
        assert next_completed.is_set()
        assert next_result == {"ok": True}
        assert runtime._active_requests == 0
        assert runtime._successful_calls == 1
        assert {
            "database": "simulated database failure",
            "response": "AttributeError",
            "access": "simulated store access denied",
            "transport": "simulated native connection failure",
            "lock": "simulated store lock",
            "cas": "simulated stale revision",
            "timeout": "simulated transport timeout",
        }[failure] in caplog.text
    finally:
        if not completed.is_set():
            runtime._dispatch_slots.release()
        runtime.close()


@pytest.mark.parametrize("failure", ["submission", "future"])
def test_dispatch_failure_is_observed_and_next_request_runs(
    tmp_path, monkeypatch, caplog, failure: str
) -> None:
    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    identity = _identity(tmp_path)
    runtime = mcp_call._LocalServiceListenerRuntime(identity, Client())  # type: ignore[arg-type]
    runtime.start()
    bundle = runtime._policy_store.active_bundle()
    lease = mcp_call._ListenerLease(
        token="lease",
        identity=identity,
        session_id="run-1",
        scope_lineage=mcp_call.ScopeLineage(workspace_id="release", session_id="run-1"),
        expires_at=time.monotonic() + 60,
        policy_bundle_revision=bundle.revision_id,
        catalog_revision=bundle.catalog_revision_id,
        catalog_snapshot=runtime._policy_store.snapshot_for(bundle),
    )
    original_start = threading.Thread.start

    def fail_start(worker: threading.Thread) -> None:
        if worker.name == "um-mem-dispatch-worker":
            raise sqlite3.OperationalError("simulated submission failure")
        original_start(worker)

    def abort_worker(*_args, **_kwargs):
        raise SystemExit("simulated worker abort")

    if failure == "submission":
        monkeypatch.setattr(threading.Thread, "start", fail_start)
    else:
        monkeypatch.setattr(mcp_call, "_dispatch_local_service_request", abort_worker)
    failures = [(threading.Event(), {}) for _ in range(4)]
    try:
        runtime._active_requests = len(failures)
        for index, (completed, result) in enumerate(failures):
            runtime._requests.put(
                mcp_call._DispatchItem(
                    lease, {"request_id": f"broken-{index}", "tool": "block_map"}, completed, result
                )
            )
        for completed, result in failures:
            assert completed.wait(timeout=2)
            assert result["ok"] is False
            assert result["reason_code"] == (
                "dispatcher_submission_error"
                if failure == "submission"
                else "dispatcher_unexpected_error"
            )
        assert runtime._active_requests == 0
        assert runtime._failed_calls == 4

        monkeypatch.setattr(threading.Thread, "start", original_start)
        monkeypatch.setattr(
            mcp_call,
            "_dispatch_local_service_request",
            lambda *_args, **_kwargs: {"payload": {"ok": True}},
        )
        next_completed = threading.Event()
        next_result: dict[str, object] = {}
        runtime._active_requests = 1
        runtime._requests.put(
            mcp_call._DispatchItem(
                lease, {"request_id": "next-1", "tool": "block_map"}, next_completed, next_result
            )
        )
        assert next_completed.wait(timeout=2)
        assert next_result == {"ok": True}
        assert runtime._active_requests == 0
        assert runtime._successful_calls == 1
        expected_cause = (
            "simulated submission failure" if failure == "submission" else "simulated worker abort"
        )
        deadline = time.monotonic() + 2
        while expected_cause not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)
        assert expected_cause in caplog.text
    finally:
        monkeypatch.setattr(threading.Thread, "start", original_start)
        runtime.close()


def test_listener_close_returns_while_started_dispatch_finishes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    identity = _identity(tmp_path)
    runtime = mcp_call._LocalServiceListenerRuntime(identity, Client())  # type: ignore[arg-type]
    runtime.start()
    bundle = runtime._policy_store.active_bundle()
    lease = mcp_call._ListenerLease(
        token="lease",
        identity=identity,
        session_id="run-1",
        scope_lineage=mcp_call.ScopeLineage(workspace_id="release", session_id="run-1"),
        expires_at=time.monotonic() + 60,
        policy_bundle_revision=bundle.revision_id,
        catalog_revision=bundle.catalog_revision_id,
        catalog_snapshot=runtime._policy_store.snapshot_for(bundle),
    )
    entered = threading.Event()
    release = threading.Event()

    def slow_dispatch(*_args: object, **_kwargs: object) -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=10)
        return {"payload": {"ok": True, "saved": True}}

    monkeypatch.setattr(mcp_call, "_dispatch_local_service_request", slow_dispatch)
    completed = threading.Event()
    result: dict[str, object] = {}
    runtime._active_requests = 1
    runtime._requests.put(
        mcp_call._DispatchItem(
            lease, {"request_id": "slow-1", "tool": "block_map"}, completed, result
        )
    )
    assert entered.wait(timeout=2)
    closer = threading.Thread(target=runtime.close, daemon=True)
    closer.start()
    try:
        closer.join(timeout=2)
        assert not closer.is_alive()
        assert not completed.is_set()
    finally:
        release.set()
        closer.join(timeout=5)
    assert completed.wait(timeout=2)
    assert result == {"ok": True, "saved": True}
    assert runtime._active_requests == 0


@pytest.mark.parametrize("stage", ["queued", "precommit", "started"])
def test_listener_deadline_distinguishes_cancelled_queue_from_started_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    monkeypatch.setattr(mcp_call, "LOCAL_SERVICE_LISTENER_QUEUE_SIZE", 1)

    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    identity = _identity(tmp_path)
    runtime = mcp_call._LocalServiceListenerRuntime(identity, Client())  # type: ignore[arg-type]
    bundle = runtime._policy_store.active_bundle()
    lease = mcp_call._ListenerLease(
        token="lease",
        identity=identity,
        session_id="run-1",
        scope_lineage=mcp_call.ScopeLineage(workspace_id="release", session_id="run-1"),
        expires_at=time.monotonic() + 60,
        policy_bundle_revision=bundle.revision_id,
        catalog_revision=bundle.catalog_revision_id,
        catalog_snapshot=runtime._policy_store.snapshot_for(bundle),
    )
    runtime._leases[lease.token] = lease
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def dispatch(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append("started")
        entered.set()
        begin_commit = kwargs["before_durable_commit"]
        assert callable(begin_commit)
        if stage == "started":
            assert begin_commit() is True
        assert release.wait(timeout=5)
        if stage == "precommit":
            assert begin_commit() is False
            return {"payload": {"ok": False, "reason_code": "dispatch_cancelled"}}
        return {"payload": {"ok": True, "saved": True}}

    monkeypatch.setattr(mcp_call, "_dispatch_local_service_request", dispatch)
    slot_held = stage == "queued"
    if slot_held:
        assert runtime._dispatch_slots.acquire(blocking=False)
    runtime.start()
    request = {
        "protocol_version": identity.protocol_version,
        "schema_version": mcp_call.LOCAL_SERVICE_SCHEMA_VERSION,
        "api_version": mcp_call.LOCAL_SERVICE_API_VERSION,
        "request_id": "write-1",
        "lease": lease.token,
        "caller_role": identity.caller_role,
        "session_id": lease.session_id,
        "tool": "remember",
        "arguments": {"content": "fact"},
        "mutation_correlation_id": "operation-1",
        "deadline_ns": time.monotonic_ns() + 300_000_000,
    }
    response: dict[str, object] = {}
    caller = threading.Thread(target=lambda: response.update(runtime.call(request)), daemon=True)
    caller.start()
    try:
        if stage != "queued":
            assert entered.wait(timeout=2)
        caller.join(timeout=2)
        assert not caller.is_alive()
        assert response["reason_code"] == "dispatch_timeout"
        assert response["operation_id"] == "operation-1"
        assert response["durable_status"] == ("unknown" if stage == "started" else "not_started")
    finally:
        if slot_held:
            runtime._dispatch_slots.release()
        release.set()
        caller.join(timeout=5)
        runtime.close()
    assert calls == ([] if stage == "queued" else ["started"])
    assert runtime._active_requests == 0


def test_cancel_before_durable_claim_skips_receipt_and_native_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity(tmp_path)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("durable work must not start after cancellation")

    monkeypatch.setattr(mcp_call, "claim_remember_receipt", unexpected)
    monkeypatch.setattr(mcp_call, "call_tool", unexpected)
    response = mcp_call._dispatch_local_service_request(
        identity,
        json.dumps(
            {
                "protocol_version": identity.protocol_version,
                "request_id": "write-1",
                "lease": "lease",
                "caller_role": identity.caller_role,
                "tool": "remember",
                "arguments": {"content": "fact"},
                "mutation_correlation_id": "operation-1",
            }
        ),
        client=object(),  # type: ignore[arg-type]
        expected_lease="lease",
        before_durable_commit=lambda: False,
    )

    assert response["payload"]["reason_code"] == "dispatch_cancelled"
    assert response["payload"]["operation_id"] == "operation-1"


def test_listener_close_keeps_committed_write_during_native_projection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity(tmp_path)
    client = MemoryClient(tmp_path, workspace_id="release", backend="mempalace")
    backend = client.backend
    project = backend._project_durable_record
    entered = threading.Event()
    release = threading.Event()

    def slow_retain(record):
        entered.set()
        assert release.wait(timeout=10)
        return project(record)

    monkeypatch.setattr(backend._index, "_projection_callback", slow_retain)
    runtime = mcp_call._LocalServiceListenerRuntime(identity, client)
    bundle = runtime._policy_store.active_bundle()
    lease = mcp_call._ListenerLease(
        token="lease",
        identity=identity,
        session_id="run-1",
        scope_lineage=mcp_call.ScopeLineage(workspace_id="release", session_id="run-1"),
        expires_at=time.monotonic() + 60,
        policy_bundle_revision=bundle.revision_id,
        catalog_revision=bundle.catalog_revision_id,
        catalog_snapshot=runtime._policy_store.snapshot_for(bundle),
    )
    runtime._leases[lease.token] = lease
    runtime.start()
    response: dict[str, object] = {}
    caller = threading.Thread(
        target=lambda: response.update(
            runtime.call(
                {
                    "protocol_version": identity.protocol_version,
                    "schema_version": mcp_call.LOCAL_SERVICE_SCHEMA_VERSION,
                    "api_version": mcp_call.LOCAL_SERVICE_API_VERSION,
                    "request_id": "write-1",
                    "lease": lease.token,
                    "caller_role": identity.caller_role,
                    "session_id": lease.session_id,
                    "tool": "remember",
                    "arguments": {
                        "content": "Committed record survives listener close.",
                        "title": "shutdown projection",
                        "requested_block_id": "workspace_episodes",
                    },
                    "mutation_correlation_id": "shutdown-operation-1",
                    "deadline_ns": time.monotonic_ns() + 10_000_000_000,
                }
            )
        ),
        daemon=True,
    )
    caller.start()
    closer = threading.Thread(target=runtime.close, daemon=True)
    try:
        assert entered.wait(timeout=5)
        [pending] = backend._index.pending_write_outbox()
        assert backend._index.canonical_head_status(pending.block_id) == "committed"
        closer.start()
        closer.join(timeout=2)
        assert not closer.is_alive()
        assert backend.write_outbox_status(pending.block_id) == "pending"
        release.set()
        caller.join(timeout=5)
        assert response["ok"] is True
        assert response["saved"] is True
        deadline = time.monotonic() + 5
        while backend.write_outbox_status(pending.block_id) != "committed":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert backend.write_outbox_status(pending.block_id) == "committed"
    finally:
        release.set()
        caller.join(timeout=5)
        if closer.is_alive():
            closer.join(timeout=5)
        runtime.close()
        backend.close()


def test_listener_process_exits_after_close_with_stalled_worker(tmp_path) -> None:
    script = """
import sys
import threading
import time
from um_mem.memory import mcp_call
from um_mem.memory.local_service import build_local_service_identity

class Backend:
    def health(self):
        return {"ok": True, "native_ok": True, "strict_ok": True}

class Client:
    backend = Backend()

identity = build_local_service_identity(
    repo_root=sys.argv[1], workspace_id="release", backend="mempalace", caller_role="operator"
)
runtime = mcp_call._LocalServiceListenerRuntime(identity, Client())
bundle = runtime._policy_store.active_bundle()
lease = mcp_call._ListenerLease(
    token="lease", identity=identity, session_id="run-1",
    scope_lineage=mcp_call.ScopeLineage(workspace_id="release", session_id="run-1"),
    expires_at=time.monotonic() + 60,
    policy_bundle_revision=bundle.revision_id,
    catalog_revision=bundle.catalog_revision_id,
    catalog_snapshot=runtime._policy_store.snapshot_for(bundle),
)
entered = threading.Event()
def stalled(*_args, **_kwargs):
    entered.set()
    time.sleep(60)
    return {"payload": {"ok": True}}
mcp_call._dispatch_local_service_request = stalled
runtime.start()
runtime._active_requests = 1
runtime._requests.put(mcp_call._DispatchItem(
    lease, {"request_id": "slow-1", "tool": "block_map"}, threading.Event(), {}
))
assert entered.wait(5)
runtime.close()
print("closed", flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        pytest.fail("listener process remained alive after bounded close")
    assert process.returncode == 0, stderr
    assert stdout.strip() == "closed"


def test_listener_restart_recovers_write_interrupted_during_native_projection(tmp_path) -> None:
    script = """
import json
import sys
import threading
import time
from um_mem.api import MemoryClient
from um_mem.memory import mcp_call
from um_mem.memory.local_service import build_local_service_identity

identity = build_local_service_identity(
    repo_root=sys.argv[1], workspace_id="release", backend="mempalace", caller_role="operator"
)
client = MemoryClient(sys.argv[1], workspace_id="release", backend="mempalace")
backend = client.backend
entered = threading.Event()
def stalled_retain(_record):
    entered.set()
    time.sleep(60)
backend._index._projection_callback = stalled_retain
runtime = mcp_call._LocalServiceListenerRuntime(identity, client)
bundle = runtime._policy_store.active_bundle()
lease = mcp_call._ListenerLease(
    token="lease", identity=identity, session_id="run-1",
    scope_lineage=mcp_call.ScopeLineage(workspace_id="release", session_id="run-1"),
    expires_at=time.monotonic() + 60,
    policy_bundle_revision=bundle.revision_id,
    catalog_revision=bundle.catalog_revision_id,
    catalog_snapshot=runtime._policy_store.snapshot_for(bundle),
)
runtime._leases[lease.token] = lease
runtime.start()
threading.Thread(target=runtime.call, args=({
    "protocol_version": identity.protocol_version,
    "schema_version": mcp_call.LOCAL_SERVICE_SCHEMA_VERSION,
    "api_version": mcp_call.LOCAL_SERVICE_API_VERSION,
    "request_id": "write-1", "lease": lease.token,
    "caller_role": identity.caller_role, "session_id": lease.session_id,
    "tool": "remember",
    "arguments": {"content": "Recover this canonical record after process exit.",
                  "title": "interrupted projection",
                  "requested_block_id": "workspace_episodes"},
    "mutation_correlation_id": "interrupted-operation-1",
    "deadline_ns": time.monotonic_ns() + 30_000_000_000,
},), daemon=True).start()
assert entered.wait(5)
[pending] = backend._index.pending_write_outbox()
assert backend._index.canonical_head_status(pending.block_id) == "committed"
runtime.close()
print(json.dumps({"block_id": pending.block_id, "status": "pending"}), flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
        pytest.fail("listener process did not exit with a recoverable outbox")
    assert process.returncode == 0, stderr
    receipt = json.loads(stdout.strip())
    assert receipt["status"] == "pending"

    backend = MemoryClient(tmp_path, workspace_id="release", backend="mempalace").backend
    try:
        assert backend._index.canonical_head_status(receipt["block_id"]) == "committed"
        deadline = time.monotonic() + 5
        while backend.write_outbox_status(receipt["block_id"]) != "committed":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert backend.write_outbox_status(receipt["block_id"]) == "committed"
        assert backend.get_block(receipt["block_id"]) is not None
    finally:
        backend.close()


def test_background_admission_failure_preserves_commit_and_enters_read_only(
    monkeypatch,
    tmp_path,
) -> None:
    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(  # type: ignore[arg-type]
        _identity(tmp_path),
        Client(),
    )
    runtime.start()
    committed = {"ok": True, "saved": True, "memory_id": "canonical-record-1"}

    def fail_admission(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    try:
        monkeypatch.setattr(runtime._work_queue, "submit", fail_admission)
        runtime._enqueue_index_refresh({"request_id": "write-1"}, committed)
        status = runtime.health()
    finally:
        runtime.close()

    assert committed["ok"] is True
    assert committed["saved"] is True
    assert committed["memory_id"] == "canonical-record-1"
    assert committed["background_work"]["status"] == "admission_failed"
    assert status["readiness"] == "ready"
    assert status["state"] == "ready_read_only"
    assert status["degraded_state"] == "background_queue_unavailable"


def test_background_index_refresh_is_record_scoped_and_reuses_serialized_listener_backend(
    monkeypatch,
    tmp_path,
) -> None:
    refreshed = threading.Event()

    class Backend:
        def refresh_memory_indexes(self, block_id: str) -> dict[str, bool]:
            assert block_id == "record-1"
            refreshed.set()
            return {"ok": True}

        def rebuild_memory_indexes(self, *, catalog):
            raise AssertionError("a committed record must not trigger a full index rebuild")

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(  # type: ignore[arg-type]
        _identity(tmp_path),
        Client(),
    )
    monkeypatch.setattr(
        mcp_call,
        "MemoryClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("background rebuild must not open another backend")
        ),
    )
    runtime._work_queue.submit(
        "indexing",
        work_ref="record-1",
        work_revision="record-1",
    )
    item = runtime._work_queue.claim("indexing")
    assert item is not None
    runtime._client_lock.acquire()
    worker = threading.Thread(
        target=runtime._run_indexing_work,
        args=(item,),
    )
    try:
        worker.start()
        assert refreshed.wait(timeout=0.1) is False
    finally:
        runtime._client_lock.release()
    worker.join(timeout=5)
    runtime.close()

    assert worker.is_alive() is False
    assert refreshed.is_set()


def test_background_cost_cannot_be_hidden_behind_online_latency(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path))
    ledger.record_cost(
        CostSample(
            category="online_recall",
            latency_ms=1,
            compute_ms=1,
            tokens=1,
            bytes_written=0,
        )
    )
    incomplete = ledger.headline_cost_report()
    assert incomplete["complete"] is False
    assert "consolidation" in incomplete["missing_categories"]
    assert "rebuild" in incomplete["missing_categories"]


def test_sidecar_quality_cannot_mask_an_unavailable_native_backend(tmp_path) -> None:
    ledger = OperationalSloLedger(_identity(tmp_path, backend="mnemosyne"))
    ledger.record_operation(
        OperationSample(
            backend="mnemosyne",
            corpus_tier="1k",
            concurrency=1,
            query_mode="warm_exact",
            component="warm_recall",
            latency_ms=0,
            operation_count=1_000,
            window_ms=1,
            native_ok=False,
            effective_mode="sidecar_quality_pass",
        )
    )

    report = ledger.latency_report()
    assert report["groups"][0]["slo_passed"] is False
    assert report["all_native_qualified"] is False
