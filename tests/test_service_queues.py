import threading
import time
from datetime import UTC, datetime, timedelta

from um_mem.memory import mcp_call
from um_mem.memory.local_service import build_local_service_identity
from um_mem.memory.service_work_queue import (
    DEFAULT_LANE_POLICIES,
    DurableServiceWorkQueue,
    ServiceQueueWorker,
    WorkLanePolicy,
)


def _identity(tmp_path):
    return build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="release",
        backend="mempalace",
        caller_role="operator",
    )


def _policies(*, capacity: int = 2) -> dict[str, WorkLanePolicy]:
    return {
        lane: WorkLanePolicy(capacity=capacity, target_latency_ms=index + 1)
        for index, lane in enumerate(DEFAULT_LANE_POLICIES)
    }


def test_five_lanes_have_independent_capacity_slo_and_restart_state(tmp_path) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies())
    for lane in DEFAULT_LANE_POLICIES:
        receipt = queue.submit(lane, work_ref=f"work-{lane}", work_revision="revision-1")
        assert receipt.status == "queued"

    later = datetime.now(UTC) + timedelta(seconds=1)
    health = queue.health(now=later)
    assert [lane.lane for lane in health.lanes] == list(DEFAULT_LANE_POLICIES)
    assert [lane.target_latency_ms for lane in health.lanes] == [1, 2, 3, 4, 5]
    assert all(lane.pending == 1 and lane.slo_breached for lane in health.lanes)

    restarted = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies())
    assert all(lane.pending == 1 for lane in restarted.health().lanes)


def test_restart_recovers_work_abandoned_by_a_dead_listener(tmp_path) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies(capacity=1))
    receipt = queue.submit("indexing", work_ref="record-1", work_revision="revision-1")
    running = queue.claim("indexing")
    assert running is not None and running.state == "running"
    rejected = queue.submit("indexing", work_ref="record-2", work_revision="revision-2")
    assert rejected.status == "backpressured"
    assert queue.health().emergency_read_only is True

    restarted = DurableServiceWorkQueue(
        _identity(tmp_path),
        policies=_policies(capacity=1),
        recover_abandoned=True,
    )

    recovered = restarted.item(receipt.work_id)
    assert recovered.state == "pending"
    assert recovered.attempts == 1
    completed = restarted.process_one("indexing", lambda _item: None)
    assert completed is not None and completed.state == "completed"
    assert restarted.health().emergency_read_only is False


def test_restart_preserves_each_abandoned_record_index_refresh(tmp_path) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies(capacity=4))
    receipts = []
    for index in range(3):
        receipts.append(
            queue.submit(
                "indexing",
                work_ref=f"record-{index}",
                work_revision=f"record-{index}",
            )
        )
        assert queue.claim("indexing") is not None

    restarted = DurableServiceWorkQueue(
        _identity(tmp_path),
        policies=_policies(capacity=4),
        recover_abandoned=True,
    )

    indexing = restarted.health().lanes[0]
    assert indexing.pending == 3
    assert indexing.running == 0
    assert indexing.completed == 0
    assert restarted.health().emergency_read_only is False
    assert [restarted.item(receipt.work_id).attempts for receipt in receipts] == [1, 1, 1]

    completed = [restarted.process_one("indexing", lambda _item: None) for _ in receipts]
    assert all(item is not None and item.state == "completed" for item in completed)


def test_restart_preserves_poisoned_record_index_refresh_for_operator_action(
    tmp_path,
) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies(capacity=3))
    stale = queue.submit(
        "indexing",
        work_ref="stale-record",
        work_revision="stale-revision",
        max_attempts=1,
    )
    assert queue.claim("indexing") is not None
    assert queue.fail(stale.work_id, error_code="RuntimeError").state == "poisoned"
    latest = queue.submit(
        "indexing",
        work_ref="latest-record",
        work_revision="latest-revision",
    )
    assert latest.status == "queued"
    assert queue.health().emergency_read_only is True

    restarted = DurableServiceWorkQueue(
        _identity(tmp_path),
        policies=_policies(capacity=3),
        recover_abandoned=True,
    )

    assert restarted.item(stale.work_id).state == "poisoned"
    assert restarted.item(latest.work_id).state == "pending"
    assert restarted.health().emergency_read_only is True


def test_restart_poisons_record_index_refresh_after_repeated_owner_death(
    tmp_path,
) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies(capacity=3))
    receipt = queue.submit(
        "indexing",
        work_ref="record-1",
        work_revision="record-1",
    )

    for _ in range(3):
        assert queue.claim("indexing") is not None
        queue = DurableServiceWorkQueue(
            _identity(tmp_path),
            policies=_policies(capacity=3),
            recover_abandoned=True,
        )

    recovered = queue.item(receipt.work_id)
    assert recovered.state == "poisoned"
    assert recovered.attempts == 3
    assert recovered.error_code == "listener_restarted"
    assert queue.health().emergency_read_only is True


def test_critical_backpressure_is_durable_observable_and_recovers(tmp_path) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies(capacity=1))
    accepted = queue.submit("indexing", work_ref="record-1", work_revision="revision-1")
    rejected = queue.submit("indexing", work_ref="record-2", work_revision="revision-2")

    assert accepted.status == "queued"
    assert rejected.status == "backpressured"
    assert queue.admission_count(status="queued") == 1
    assert queue.admission_count(status="backpressured") == 1
    assert queue.health().emergency_read_only is True
    lane = queue.health().lanes[0]
    assert lane.backpressured == 1
    assert queue.health().no_silent_drop is True

    running = queue.claim("indexing")
    assert running is not None
    queue.complete(running.work_id)
    assert queue.health().emergency_read_only is False
    replay = queue.submit("indexing", work_ref="record-1", work_revision="revision-1")
    assert replay.status == "replayed"
    assert queue.admission_count(status="replayed") == 1


def test_poison_retries_are_receipted_and_operator_requeue_is_lossless(tmp_path) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies())
    receipt = queue.submit(
        "consolidation_transition",
        work_ref="transition-1",
        work_revision="revision-1",
        max_attempts=2,
    )

    def fail(_item) -> None:
        raise RuntimeError("deterministic failure")

    first = queue.process_one("consolidation_transition", fail)
    second = queue.process_one("consolidation_transition", fail)
    assert first is not None and first.state == "pending"
    assert second is not None and second.state == "poisoned"
    assert queue.item(receipt.work_id).attempts == 2
    assert queue.health().emergency_read_only is True
    assert (
        next(
            lane for lane in queue.health().lanes if lane.lane == "consolidation_transition"
        ).poisoned
        == 1
    )

    requeued = queue.requeue_poison(receipt.work_id)
    assert requeued.state == "pending"
    assert requeued.attempts == 0
    completed = queue.process_one("consolidation_transition", lambda _item: None)
    assert completed is not None and completed.state == "completed"
    assert queue.health().emergency_read_only is False


def test_background_worker_never_holds_the_hot_recall_path(tmp_path) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies())
    started = threading.Event()
    release = threading.Event()
    handlers = {lane: (lambda _item: None) for lane in DEFAULT_LANE_POLICIES}

    def blocked_indexing(_item) -> None:
        started.set()
        assert release.wait(timeout=2)

    handlers["indexing"] = blocked_indexing
    worker = ServiceQueueWorker(queue, handlers)
    queue.submit("indexing", work_ref="record-1", work_revision="revision-1")
    worker.start()
    try:
        assert started.wait(timeout=1)
        started_at = time.perf_counter()
        hot_result = "recall-ready"
        elapsed = time.perf_counter() - started_at
        assert hot_result == "recall-ready"
        assert elapsed < 0.05
    finally:
        release.set()
        worker.close()
    assert queue.health().lanes[0].completed == 1


def test_background_lanes_continue_while_indexing_is_blocked(tmp_path) -> None:
    queue = DurableServiceWorkQueue(_identity(tmp_path), policies=_policies())
    indexing_started = threading.Event()
    release_indexing = threading.Event()
    scheduler_completed = threading.Event()
    handlers = {lane: (lambda _item: None) for lane in DEFAULT_LANE_POLICIES}

    def blocked_indexing(_item) -> None:
        indexing_started.set()
        assert release_indexing.wait(timeout=2)

    def run_scheduler(_item) -> None:
        scheduler_completed.set()

    handlers["indexing"] = blocked_indexing
    handlers["prospective_scheduler"] = run_scheduler
    queue.submit("indexing", work_ref="slow-full-index", work_revision="revision-1")
    queue.submit(
        "prospective_scheduler",
        work_ref="session-memory",
        work_revision="revision-1",
    )
    worker = ServiceQueueWorker(queue, handlers)
    worker.start()
    try:
        assert indexing_started.wait(timeout=1)
        assert scheduler_completed.wait(timeout=0.5)
    finally:
        release_indexing.set()
        worker.close()

    health = queue.health()
    assert next(lane for lane in health.lanes if lane.lane == "indexing").completed == 1
    assert (
        next(lane for lane in health.lanes if lane.lane == "prospective_scheduler").completed == 1
    )


def test_listener_owner_recovers_abandoned_queue_work_before_serving(tmp_path) -> None:
    identity = _identity(tmp_path)
    queue = DurableServiceWorkQueue(identity)
    receipt = queue.submit("indexing", work_ref="record-1", work_revision="revision-1")
    assert queue.claim("indexing") is not None

    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(identity, Client())  # type: ignore[arg-type]

    recovered = runtime._work_queue.item(receipt.work_id)
    assert recovered.state == "pending"
    assert recovered.attempts == 1


def test_listener_mutation_enqueues_durable_index_refresh_without_api_noise(tmp_path) -> None:
    class Backend:
        def health(self) -> dict[str, bool]:
            return {"ok": True, "native_ok": True, "strict_ok": True}

    class Client:
        backend = Backend()

    runtime = mcp_call._LocalServiceListenerRuntime(  # type: ignore[arg-type]
        _identity(tmp_path),
        Client(),
    )
    result = {"ok": True, "memory_id": "record-1"}

    runtime._enqueue_index_refresh({"request_id": "mutation-1"}, result)

    indexing = runtime._background_queue_status()["lanes"][0]
    assert indexing["lane"] == "indexing"
    assert indexing["pending"] == 1
    pending = runtime._work_queue.claim("indexing")
    assert pending is not None
    assert pending.work_revision == "record-1"
    assert "background_work" not in result
