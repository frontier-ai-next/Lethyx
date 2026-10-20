import time
from pathlib import Path

from lethyx.memory.local_service import (
    LocalServiceClientPool,
    build_local_service_identity,
    shutdown_local_service_listener,
)
from lethyx.memory.service_scheduler import ServiceScheduler


def test_listener_owned_scheduler_accepts_only_typed_host_state_and_fires_once(
    tmp_path: Path,
) -> None:
    host_state_identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="host_state",
    )
    operator_identity = build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="operator",
    )
    event = {
        "event_id": "host-event-1",
        "event_type": "tool_result",
        "run_id": "run-1",
        "stage_id": "stage_2_5",
        "signal_codes": ["tool_succeeded"],
        "capability_codes": ["typed_host_state"],
        "retry_count": 0,
    }
    pool = LocalServiceClientPool()
    try:
        first = pool.submit_host_state(
            host_state_identity,
            event,
            timeout_seconds=20,
            session_id="host-state-session",
        )
        replay = pool.submit_host_state(
            host_state_identity,
            event,
            timeout_seconds=20,
            session_id="host-state-session",
        )
        changed = pool.submit_host_state(
            host_state_identity,
            {**event, "retry_count": 1},
            timeout_seconds=20,
            session_id="host-state-session",
        )
        raw_content = pool.submit_host_state(
            host_state_identity,
            {**event, "content": "must-not-reach-the-scheduler"},
            timeout_seconds=20,
            session_id="host-state-session",
        )
        foreign_workspace = pool.submit_host_state(
            host_state_identity,
            {**event, "workspace_id": "foreign"},
            timeout_seconds=20,
            session_id="host-state-session",
        )
        denied = pool.submit_host_state(
            operator_identity,
            event,
            timeout_seconds=20,
            session_id="host-state-session",
        )
        scheduler = ServiceScheduler(host_state_identity)
        deadline = time.monotonic() + 5
        jobs = ()
        while time.monotonic() < deadline:
            event_id = first.get("event_id_sha256")
            assert isinstance(event_id, str)
            jobs = scheduler.jobs_for(event_id)
            if {job.status for job in jobs} == {"recorded"}:
                break
            time.sleep(0.05)
    finally:
        pool.close()
        shutdown_local_service_listener(operator_identity)

    assert first["ok"] is True
    assert first["replayed"] is False
    assert replay["ok"] is True
    assert replay["replayed"] is True
    assert first["event_id_sha256"] == replay["event_id_sha256"]
    assert first["scheduler_plan_sha256"] == replay["scheduler_plan_sha256"]
    assert first["decision_revision"].startswith(first["policy_bundle_revision"])
    assert {job.status for job in jobs} == {"recorded"}
    assert changed == {"ok": False, "error": "local service host state rejected"}
    assert raw_content == {"ok": False, "error": "local service host state rejected"}
    assert foreign_workspace == {"ok": False, "error": "local service host state rejected"}
    assert denied == {"ok": False, "error": "local service host-state role is required"}
