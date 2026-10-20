import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import pytest

from lethyx.memory import mcp_call
from lethyx.memory.local_service import build_local_service_identity
from lethyx.memory.prospective import ProspectiveTriggerDecision
from lethyx.memory.service_scheduler import (
    AdvisoryDecisionPlan,
    HostStateEvent,
    ServiceScheduler,
)


_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def _identity(tmp_path):
    return build_local_service_identity(
        repo_root=tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        caller_role="host_hook",
    )


def _event(
    *,
    event_id: str = "evt-1",
    retry_count: int = 0,
    occurred_at: datetime = _NOW,
    agent_kind: str = "generic_agent",
    changed_paths: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
    previous_stage_id: str = "",
    failure_signatures: tuple[str, ...] = (),
    env_flags: tuple[str, ...] = (),
) -> HostStateEvent:
    return HostStateEvent(
        event_id=event_id,
        event_type="tool_result",
        workspace_id="alpha",
        run_id="run-1",
        stage_id="stage_2_5",
        state_revision="host-state@1",
        occurred_at=occurred_at,
        signal_codes=("tool_succeeded",),
        capability_codes=("typed_host_state",),
        retry_count=retry_count,
        agent_kind=agent_kind,
        changed_paths=changed_paths,
        artifacts=artifacts,
        previous_stage_id=previous_stage_id,
        failure_signatures=failure_signatures,
        env_flags=env_flags,
    )


def _plan(*, revision: str = "policy-bundle@1", due_at: datetime = _NOW) -> AdvisoryDecisionPlan:
    return AdvisoryDecisionPlan(
        decision_revision=revision,
        activation_due_at=due_at,
        activation_reason_code="tool_result_activation",
        consolidation_due_at=due_at,
        consolidation_reason_code="tool_result_consolidation",
    )


def test_submit_replays_one_content_free_event_without_replanning_or_duplicate_jobs(
    tmp_path,
) -> None:
    planner_calls: list[HostStateEvent] = []

    def planner(event: HostStateEvent) -> AdvisoryDecisionPlan:
        planner_calls.append(event)
        return _plan()

    scheduler = ServiceScheduler(_identity(tmp_path), planner=planner)
    event = _event(event_id="evt-secret-phrase")

    first = scheduler.submit(event)
    replay = scheduler.submit(event)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.plan == first.plan
    assert replay.jobs == first.jobs
    assert len(first.jobs) == 2
    assert len(planner_calls) == 1
    persisted = scheduler.path.read_bytes()
    assert b"evt-secret-phrase" not in persisted
    assert b"run-1" not in persisted


def test_replay_uses_typed_state_identity_not_ingress_delivery_time(tmp_path) -> None:
    scheduler = ServiceScheduler(_identity(tmp_path), planner=lambda event: _plan())
    first = scheduler.submit(_event())

    replay = scheduler.submit(_event(occurred_at=_NOW + timedelta(minutes=5)))

    assert replay.replayed is True
    assert replay.plan == first.plan
    assert replay.jobs == first.jobs


def test_submit_rejects_a_changed_event_with_the_same_id(tmp_path) -> None:
    scheduler = ServiceScheduler(_identity(tmp_path), planner=lambda event: _plan())
    scheduler.submit(_event())

    with pytest.raises(ValueError, match="event conflict"):
        scheduler.submit(_event(retry_count=1))


def test_restart_recovers_due_jobs_and_fires_one_receipt_per_job(tmp_path) -> None:
    due_at = _NOW + timedelta(minutes=1)
    first = ServiceScheduler(_identity(tmp_path), planner=lambda event: _plan(due_at=due_at))
    submitted = first.submit(_event())
    first.close()

    reloaded = ServiceScheduler(
        _identity(tmp_path),
        planner=lambda event: pytest.fail("replayed scheduler state must not re-plan"),
    )

    assert reloaded.recover_due(now=due_at - timedelta(seconds=1)) == ()
    assert reloaded.recover_due(now=due_at) == submitted.jobs

    receipts = reloaded.fire_due(now=due_at)

    assert {receipt.job_kind for receipt in receipts} == {
        "activation_opportunity",
        "consolidation_assessment",
    }
    assert {receipt.outcome for receipt in receipts} == {
        "activation_opportunity_recorded",
        "consolidation_assessment_recorded",
    }
    assert reloaded.fire_due(now=due_at + timedelta(seconds=1)) == ()
    assert all(reloaded.receipt_for(job.job_id) is not None for job in submitted.jobs)


def test_concurrent_submit_and_restart_fire_are_exactly_once(tmp_path) -> None:
    event = _event(event_id="concurrent-event")
    first = ServiceScheduler(_identity(tmp_path), planner=lambda _event: _plan())
    second = ServiceScheduler(_identity(tmp_path), planner=lambda _event: _plan())
    submit_barrier = threading.Barrier(2)

    def submit(scheduler: ServiceScheduler):
        submit_barrier.wait(timeout=5)
        return scheduler.submit(event)

    with ThreadPoolExecutor(max_workers=2) as executor:
        submissions = tuple(executor.map(submit, (first, second)))

    assert sum(not submission.replayed for submission in submissions) == 1
    assert {submission.event_id_sha256 for submission in submissions} == {
        submissions[0].event_id_sha256
    }
    assert submissions[0].jobs == submissions[1].jobs
    first.close()
    second.close()

    reloaded_first = ServiceScheduler(_identity(tmp_path), planner=lambda _event: _plan())
    reloaded_second = ServiceScheduler(_identity(tmp_path), planner=lambda _event: _plan())
    fire_barrier = threading.Barrier(2)

    def fire(scheduler: ServiceScheduler):
        fire_barrier.wait(timeout=5)
        return scheduler.fire_due(now=_NOW)

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipt_sets = tuple(executor.map(fire, (reloaded_first, reloaded_second)))

    receipts = tuple(receipt for receipt_set in receipt_sets for receipt in receipt_set)
    assert len(receipts) == 2
    assert {receipt.job_id for receipt in receipts} == {job.job_id for job in submissions[0].jobs}
    assert reloaded_first.fire_due(now=_NOW + timedelta(seconds=1)) == ()
    reloaded_first.close()
    reloaded_second.close()

    final = ServiceScheduler(_identity(tmp_path), planner=lambda _event: _plan())
    assert final.fire_due(now=_NOW + timedelta(seconds=2)) == ()
    assert all(final.receipt_for(job.job_id) is not None for job in submissions[0].jobs)


def test_typed_trigger_dimensions_persist_only_hashed_rule_and_action_evidence(
    tmp_path,
) -> None:
    cue = "stage_2_5"
    obligation_id = "prospective-scheduler-evidence"

    class _Client:
        def prospective_trigger_candidates(self, _context):
            return [
                ProspectiveTriggerDecision(
                    obligation_id=obligation_id,
                    matched=True,
                    match_kind="typed",
                    reason="typed_predicates_matched",
                    trigger_type="host_event",
                    candidate_class="release",
                    advertised=True,
                    delivery_eligible=True,
                    matched_cues=(cue,),
                )
            ]

    event = _event(
        event_id="typed-dimension-event",
        retry_count=1,
        agent_kind="codex",
        changed_paths=("src/lethyx/memory/service_scheduler.py",),
        artifacts=("reports/scheduler.json",),
        previous_stage_id="stage_2",
        failure_signatures=("retryable_failure",),
        env_flags=("LETHYX_TEST_MODE",),
    )

    plan = mcp_call._host_state_scheduler_plan(_identity(tmp_path), event, client=_Client())

    expected_rule_hash = hashlib.sha256(
        f"prospective-obligation:{obligation_id}".encode()
    ).hexdigest()
    expected_action_hash = hashlib.sha256(f"prospective-cue:{cue}".encode()).hexdigest()
    assert plan.matched_rule_sha256s == (expected_rule_hash,)
    assert plan.matched_action_sha256s == (expected_action_hash,)
    assert plan.deferred_trigger_owner_refs == ()

    scheduler = ServiceScheduler(_identity(tmp_path), planner=lambda _event: plan)
    submitted = scheduler.submit(event)
    assert submitted.replayed is False
    scheduler.close()
    reloaded = ServiceScheduler(_identity(tmp_path), planner=lambda _event: _plan())
    receipts = reloaded.fire_due(now=_NOW)

    assert all(receipt.matched_rule_sha256s == (expected_rule_hash,) for receipt in receipts)
    assert all(receipt.matched_action_sha256s == (expected_action_hash,) for receipt in receipts)
    assert all(receipt.deferred_trigger_owner_refs == () for receipt in receipts)
    persisted = reloaded.path.read_bytes()
    assert cue.encode() not in persisted
    assert obligation_id.encode() not in persisted
    assert {receipt.outcome for receipt in receipts} == {
        "activation_opportunity_recorded",
        "consolidation_assessment_recorded",
    }
    assert all(receipt.executed is False for receipt in receipts)


def test_captured_decision_plan_is_immutable_across_replay_and_restart(tmp_path) -> None:
    current_revision = "policy-bundle@1"

    def planner(event: HostStateEvent) -> AdvisoryDecisionPlan:
        return _plan(revision=current_revision)

    scheduler = ServiceScheduler(_identity(tmp_path), planner=planner)
    first = scheduler.submit(_event())
    current_revision = "policy-bundle@2"

    with pytest.raises(FrozenInstanceError):
        first.plan.decision_revision = current_revision  # type: ignore[misc]

    replay = scheduler.submit(_event())
    scheduler.close()
    reloaded = ServiceScheduler(_identity(tmp_path), planner=planner)
    restored = reloaded.recover_due(now=_NOW)

    assert replay.plan.decision_revision == "policy-bundle@1"
    assert all(job.decision_revision == "policy-bundle@1" for job in restored)
    assert first.plan.plan_sha256 == replay.plan.plan_sha256


def test_scheduler_records_bounded_assessment_without_authorizing_semantic_mutation(
    tmp_path,
) -> None:
    scheduler = ServiceScheduler(_identity(tmp_path), planner=lambda event: _plan())
    submitted = scheduler.submit(_event())

    receipts = scheduler.fire_due(now=_NOW)

    by_kind = {receipt.job_kind: receipt for receipt in receipts}
    assert by_kind["activation_opportunity"].status == "recorded"
    assert by_kind["activation_opportunity"].outcome == "activation_opportunity_recorded"
    assert by_kind["consolidation_assessment"].status == "recorded"
    assert by_kind["consolidation_assessment"].outcome == "consolidation_assessment_recorded"
    assert {job.status for job in scheduler.jobs_for(submitted.event_id_sha256)} == {
        "recorded",
        "recorded",
    }

    with pytest.raises(ValueError, match="disposition"):
        replace(_plan(), consolidation_disposition="execute")


def test_host_state_event_rejects_content_bearing_values(tmp_path) -> None:
    del tmp_path
    with pytest.raises(ValueError, match="content-free"):
        _event(event_id="event contains user text")
