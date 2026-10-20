"""Private, content-free advisory scheduling for one authenticated local service."""

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from lethyx.memory.local_service import LocalServiceIdentity, local_service_runtime_paths
from lethyx.memory.paths import normalize_workspace_id


_SCHEMA_VERSION = 1
_TOKEN_START = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_TOKEN_REMAINDER = _TOKEN_START | frozenset("._:@/-")
_HEX = frozenset("0123456789abcdef")
_EVENT_TYPES = frozenset(
    {
        "compaction",
        "high_risk_boundary",
        "plan_revision",
        "prospective_trigger",
        "session_boundary",
        "source_change",
        "subgoal_transition",
        "turn_completion",
        "tool_result",
        "verification_result",
    }
)
_ACTIVATION_JOB = "activation_opportunity"
_CONSOLIDATION_JOB = "consolidation_assessment"
_JOB_KINDS = (_ACTIVATION_JOB, _CONSOLIDATION_JOB)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduler_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS host_events (
    identity_sha256 TEXT NOT NULL,
    event_id_sha256 TEXT NOT NULL,
    event_fingerprint_sha256 TEXT NOT NULL,
    decision_plan_json TEXT NOT NULL,
    decision_plan_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (identity_sha256, event_id_sha256)
);
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id TEXT PRIMARY KEY,
    identity_sha256 TEXT NOT NULL,
    event_id_sha256 TEXT NOT NULL,
    event_fingerprint_sha256 TEXT NOT NULL,
    job_kind TEXT NOT NULL,
    due_at TEXT NOT NULL,
    decision_revision TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    FOREIGN KEY (identity_sha256, event_id_sha256)
        REFERENCES host_events(identity_sha256, event_id_sha256)
);
CREATE INDEX IF NOT EXISTS scheduled_jobs_due_idx
    ON scheduled_jobs(identity_sha256, status, due_at, job_kind);
CREATE TABLE IF NOT EXISTS scheduler_receipts (
    job_id TEXT PRIMARY KEY,
    identity_sha256 TEXT NOT NULL,
    event_id_sha256 TEXT NOT NULL,
    event_fingerprint_sha256 TEXT NOT NULL,
    job_kind TEXT NOT NULL,
    decision_revision TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES scheduled_jobs(job_id)
);
"""


@dataclass(frozen=True)
class HostStateEvent:
    """A bounded host-state transition, with no host content or action payload."""

    event_id: str
    event_type: str
    workspace_id: str
    run_id: str
    stage_id: str
    state_revision: str
    occurred_at: datetime
    signal_codes: tuple[str, ...] = ()
    capability_codes: tuple[str, ...] = ()
    retry_count: int = 0
    agent_kind: str = "generic_agent"
    changed_paths: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    previous_stage_id: str = ""
    failure_signatures: tuple[str, ...] = ()
    env_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        event_id = _token(self.event_id, "event_id")
        event_type = _token(self.event_type, "event_type")
        if event_type not in _EVENT_TYPES:
            raise ValueError("event_type is not a supported typed host-state event")
        workspace = normalize_workspace_id(_token(self.workspace_id, "workspace_id"))
        if not workspace:
            raise ValueError("workspace_id must be a content-free identifier")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "run_id", _token(self.run_id, "run_id"))
        object.__setattr__(self, "stage_id", _token(self.stage_id, "stage_id"))
        object.__setattr__(self, "state_revision", _token(self.state_revision, "state_revision"))
        object.__setattr__(self, "occurred_at", _timestamp(self.occurred_at, "occurred_at"))
        object.__setattr__(self, "signal_codes", _tokens(self.signal_codes, "signal_codes"))
        object.__setattr__(
            self, "capability_codes", _tokens(self.capability_codes, "capability_codes")
        )
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int):
            raise ValueError("retry_count must be a non-negative integer")
        if self.retry_count < 0 or self.retry_count > 1_000_000:
            raise ValueError("retry_count must be a bounded non-negative integer")
        object.__setattr__(self, "agent_kind", _token(self.agent_kind, "agent_kind"))
        object.__setattr__(self, "changed_paths", _tokens(self.changed_paths, "changed_paths"))
        object.__setattr__(self, "artifacts", _tokens(self.artifacts, "artifacts"))
        object.__setattr__(
            self,
            "previous_stage_id",
            _optional_token(self.previous_stage_id, "previous_stage_id"),
        )
        object.__setattr__(
            self,
            "failure_signatures",
            _tokens(self.failure_signatures, "failure_signatures"),
        )
        object.__setattr__(self, "env_flags", _tokens(self.env_flags, "env_flags"))

    @property
    def fingerprint_sha256(self) -> str:
        return _sha256(_canonical_json(self._fingerprint_record()))

    def _fingerprint_record(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "workspace_id": self.workspace_id,
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "state_revision": self.state_revision,
            "signal_codes": list(self.signal_codes),
            "capability_codes": list(self.capability_codes),
            "retry_count": self.retry_count,
            "agent_kind": self.agent_kind,
            "changed_paths": list(self.changed_paths),
            "artifacts": list(self.artifacts),
            "previous_stage_id": self.previous_stage_id,
            "failure_signatures": list(self.failure_signatures),
            "env_flags": list(self.env_flags),
        }


@dataclass(frozen=True)
class AdvisoryDecisionPlan:
    """The immutable, policy-revision-pinned result of planning one event."""

    decision_revision: str
    activation_due_at: datetime
    activation_reason_code: str
    consolidation_due_at: datetime
    consolidation_reason_code: str
    consolidation_disposition: str = "assess"
    matched_rule_sha256s: tuple[str, ...] = ()
    matched_action_sha256s: tuple[str, ...] = ()
    deferred_trigger_owner_refs: tuple[str, ...] = ()
    plan_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decision_revision", _token(self.decision_revision, "decision_revision")
        )
        object.__setattr__(
            self,
            "activation_due_at",
            _timestamp(self.activation_due_at, "activation_due_at"),
        )
        object.__setattr__(
            self,
            "activation_reason_code",
            _token(self.activation_reason_code, "activation_reason_code"),
        )
        object.__setattr__(
            self,
            "consolidation_due_at",
            _timestamp(self.consolidation_due_at, "consolidation_due_at"),
        )
        object.__setattr__(
            self,
            "consolidation_reason_code",
            _token(self.consolidation_reason_code, "consolidation_reason_code"),
        )
        if self.consolidation_disposition not in {"assess", "defer"}:
            raise ValueError("consolidation disposition is invalid")
        object.__setattr__(
            self,
            "matched_rule_sha256s",
            _digests(self.matched_rule_sha256s, "matched_rule_sha256s"),
        )
        object.__setattr__(
            self,
            "matched_action_sha256s",
            _digests(self.matched_action_sha256s, "matched_action_sha256s"),
        )
        object.__setattr__(
            self,
            "deferred_trigger_owner_refs",
            _tokens(self.deferred_trigger_owner_refs, "deferred_trigger_owner_refs"),
        )
        object.__setattr__(
            self, "plan_sha256", _sha256(_canonical_json(self._record_without_hash()))
        )

    def to_record(self) -> dict[str, object]:
        return {**self._record_without_hash(), "plan_sha256": self.plan_sha256}

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> "AdvisoryDecisionPlan":
        expected = {
            "decision_revision",
            "activation_due_at",
            "activation_reason_code",
            "consolidation_due_at",
            "consolidation_reason_code",
            "consolidation_disposition",
            "matched_rule_sha256s",
            "matched_action_sha256s",
            "deferred_trigger_owner_refs",
            "plan_sha256",
        }
        if set(value) != expected:
            raise RuntimeError("scheduler decision plan is invalid")
        plan = cls(
            decision_revision=_required_string(value, "decision_revision"),
            activation_due_at=_timestamp_from_string(value.get("activation_due_at")),
            activation_reason_code=_required_string(value, "activation_reason_code"),
            consolidation_due_at=_timestamp_from_string(value.get("consolidation_due_at")),
            consolidation_reason_code=_required_string(value, "consolidation_reason_code"),
            consolidation_disposition=_required_string(value, "consolidation_disposition"),
            matched_rule_sha256s=_required_strings(value, "matched_rule_sha256s"),
            matched_action_sha256s=_required_strings(value, "matched_action_sha256s"),
            deferred_trigger_owner_refs=_required_strings(value, "deferred_trigger_owner_refs"),
        )
        if _required_string(value, "plan_sha256") != plan.plan_sha256:
            raise RuntimeError("scheduler decision plan fingerprint is invalid")
        return plan

    def _record_without_hash(self) -> dict[str, object]:
        return {
            "decision_revision": self.decision_revision,
            "activation_due_at": _timestamp_string(self.activation_due_at),
            "activation_reason_code": self.activation_reason_code,
            "consolidation_due_at": _timestamp_string(self.consolidation_due_at),
            "consolidation_reason_code": self.consolidation_reason_code,
            "consolidation_disposition": self.consolidation_disposition,
            "matched_rule_sha256s": list(self.matched_rule_sha256s),
            "matched_action_sha256s": list(self.matched_action_sha256s),
            "deferred_trigger_owner_refs": list(self.deferred_trigger_owner_refs),
        }


@dataclass(frozen=True)
class ScheduledJob:
    job_id: str
    event_id_sha256: str
    event_fingerprint_sha256: str
    job_kind: str
    due_at: datetime
    decision_revision: str
    reason_code: str
    status: str

    def __post_init__(self) -> None:
        _digest(self.job_id, "job_id")
        _digest(self.event_id_sha256, "event_id_sha256")
        _digest(self.event_fingerprint_sha256, "event_fingerprint_sha256")
        if self.job_kind not in _JOB_KINDS:
            raise ValueError("scheduler job kind is invalid")
        if self.status not in {"pending", "recorded", "deferred"}:
            raise ValueError("scheduler job status is invalid")
        _timestamp(self.due_at, "due_at")
        _token(self.decision_revision, "decision_revision")
        _token(self.reason_code, "reason_code")


@dataclass(frozen=True)
class SchedulerReceipt:
    job_id: str
    event_id_sha256: str
    event_fingerprint_sha256: str
    job_kind: str
    decision_revision: str
    status: str
    outcome: str
    resolved_at: datetime
    matched_rule_sha256s: tuple[str, ...] = ()
    matched_action_sha256s: tuple[str, ...] = ()
    deferred_trigger_owner_refs: tuple[str, ...] = ()
    executed: bool = False

    def __post_init__(self) -> None:
        _digest(self.job_id, "job_id")
        _digest(self.event_id_sha256, "event_id_sha256")
        _digest(self.event_fingerprint_sha256, "event_fingerprint_sha256")
        if self.job_kind not in _JOB_KINDS:
            raise ValueError("scheduler receipt job kind is invalid")
        expected = {
            _ACTIVATION_JOB: {("recorded", "activation_opportunity_recorded")},
            _CONSOLIDATION_JOB: {
                ("recorded", "consolidation_assessment_recorded"),
                ("deferred", "consolidation_deferred"),
            },
        }
        if (self.status, self.outcome) not in expected[self.job_kind]:
            raise ValueError("scheduler receipt is not advisory-only")
        if self.executed is not False:
            raise ValueError("scheduler receipt must not claim execution")
        _token(self.decision_revision, "decision_revision")
        _timestamp(self.resolved_at, "resolved_at")
        _digests(self.matched_rule_sha256s, "matched_rule_sha256s")
        _digests(self.matched_action_sha256s, "matched_action_sha256s")
        _tokens(self.deferred_trigger_owner_refs, "deferred_trigger_owner_refs")


@dataclass(frozen=True)
class ScheduleSubmission:
    event_id_sha256: str
    event_fingerprint_sha256: str
    plan: AdvisoryDecisionPlan
    jobs: tuple[ScheduledJob, ...]
    replayed: bool


DecisionPlanner = Callable[[HostStateEvent], AdvisoryDecisionPlan]


def default_advisory_decision_plan(event: HostStateEvent) -> AdvisoryDecisionPlan:
    """Create a no-action default plan when a listener has no richer policy planner."""

    return AdvisoryDecisionPlan(
        decision_revision="service_scheduler.default@1",
        activation_due_at=event.occurred_at,
        activation_reason_code=f"{event.event_type}_activation",
        consolidation_due_at=event.occurred_at,
        consolidation_reason_code=f"{event.event_type}_consolidation",
    )


def local_service_scheduler_path(identity: LocalServiceIdentity) -> Path:
    """Return the role-independent SQLite state path for one local-service identity."""

    _require_identity(identity)
    registry = local_service_runtime_paths(identity).registry
    return registry.with_name(f"{registry.stem}.{_identity_sha256(identity)[:20]}.scheduler.sqlite")


class ServiceScheduler:
    """Persist and resolve advisory jobs without calling a host or mutating memory."""

    def __init__(
        self,
        identity: LocalServiceIdentity,
        *,
        planner: DecisionPlanner = default_advisory_decision_plan,
    ) -> None:
        _require_identity(identity)
        if not callable(planner):
            raise TypeError("planner must be callable")
        self.identity = identity
        self.path = local_service_scheduler_path(identity)
        self._identity_sha256 = _identity_sha256(identity)
        self._planner = planner
        self._initialize()

    def close(self) -> None:
        """The scheduler owns no persistent connection or background worker."""

    def submit(self, event: HostStateEvent) -> ScheduleSubmission:
        """Capture a plan once, or return its replay without re-planning it."""

        if not isinstance(event, HostStateEvent):
            raise TypeError("event must be a HostStateEvent")
        if event.workspace_id != self.identity.workspace_id:
            raise ValueError("host-state event workspace does not match local service identity")
        event_id_sha256 = _sha256(f"host-event-id:{event.event_id}")
        event_fingerprint_sha256 = event.fingerprint_sha256
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT event_fingerprint_sha256, decision_plan_json
                FROM host_events
                WHERE identity_sha256=? AND event_id_sha256=?
                """,
                (self._identity_sha256, event_id_sha256),
            ).fetchone()
            if row is not None:
                stored_fingerprint, raw_plan = (str(value) for value in row)
                if stored_fingerprint != event_fingerprint_sha256:
                    raise ValueError("scheduler host-state event conflict")
                plan = _decision_plan_from_json(raw_plan)
                return ScheduleSubmission(
                    event_id_sha256=event_id_sha256,
                    event_fingerprint_sha256=event_fingerprint_sha256,
                    plan=plan,
                    jobs=self._jobs_for_event(connection, event_id_sha256),
                    replayed=True,
                )

            plan = self._planner(event)
            if not isinstance(plan, AdvisoryDecisionPlan):
                raise TypeError("planner must return an AdvisoryDecisionPlan")
            now = _timestamp_string(datetime.now(UTC))
            plan_json = _canonical_json(plan.to_record())
            connection.execute(
                """
                INSERT INTO host_events(
                    identity_sha256, event_id_sha256, event_fingerprint_sha256,
                    decision_plan_json, decision_plan_sha256, created_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    self._identity_sha256,
                    event_id_sha256,
                    event_fingerprint_sha256,
                    plan_json,
                    plan.plan_sha256,
                    now,
                ),
            )
            jobs = self._new_jobs(event_id_sha256, event_fingerprint_sha256, plan)
            for job in jobs:
                connection.execute(
                    """
                    INSERT INTO scheduled_jobs(
                        job_id, identity_sha256, event_id_sha256, event_fingerprint_sha256,
                        job_kind, due_at, decision_revision, reason_code, status,
                        created_at, resolved_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        job.job_id,
                        self._identity_sha256,
                        job.event_id_sha256,
                        job.event_fingerprint_sha256,
                        job.job_kind,
                        _timestamp_string(job.due_at),
                        job.decision_revision,
                        job.reason_code,
                        job.status,
                        now,
                        "",
                    ),
                )
            return ScheduleSubmission(
                event_id_sha256=event_id_sha256,
                event_fingerprint_sha256=event_fingerprint_sha256,
                plan=plan,
                jobs=jobs,
                replayed=False,
            )

    def recover_due(self, *, now: datetime | None = None) -> tuple[ScheduledJob, ...]:
        """Read pending due work after startup without changing its terminal state."""

        due_at = _timestamp_string(_clock(now))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT job_id, event_id_sha256, event_fingerprint_sha256, job_kind,
                       due_at, decision_revision, reason_code, status
                FROM scheduled_jobs
                WHERE identity_sha256=? AND status='pending' AND due_at<=?
                ORDER BY due_at, job_kind, job_id
                """,
                (self._identity_sha256, due_at),
            ).fetchall()
        return tuple(_job_from_row(row) for row in rows)

    def fire_due(self, *, now: datetime | None = None) -> tuple[SchedulerReceipt, ...]:
        """Write exact-once advisory receipts for due work; it never executes work."""

        resolved_at = _timestamp_string(_clock(now))
        receipts: list[SchedulerReceipt] = []
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT job_id, event_id_sha256, event_fingerprint_sha256, job_kind,
                       due_at, decision_revision, reason_code, status
                FROM scheduled_jobs
                WHERE identity_sha256=? AND status='pending' AND due_at<=?
                ORDER BY due_at, job_kind, job_id
                """,
                (self._identity_sha256, resolved_at),
            ).fetchall()
            for row in rows:
                job = _job_from_row(row)
                plan = self._plan_for_event(connection, job.event_id_sha256)
                status, outcome = _advisory_outcome(
                    job.job_kind,
                    consolidation_disposition=plan.consolidation_disposition,
                )
                updated = connection.execute(
                    """
                    UPDATE scheduled_jobs
                    SET status=?, resolved_at=?
                    WHERE job_id=? AND status='pending'
                    """,
                    (status, resolved_at, job.job_id),
                ).rowcount
                if updated != 1:
                    raise RuntimeError("scheduler job state changed while resolving")
                receipt = SchedulerReceipt(
                    job_id=job.job_id,
                    event_id_sha256=job.event_id_sha256,
                    event_fingerprint_sha256=job.event_fingerprint_sha256,
                    job_kind=job.job_kind,
                    decision_revision=job.decision_revision,
                    status=status,
                    outcome=outcome,
                    resolved_at=_timestamp_from_string(resolved_at),
                    matched_rule_sha256s=plan.matched_rule_sha256s,
                    matched_action_sha256s=plan.matched_action_sha256s,
                    deferred_trigger_owner_refs=plan.deferred_trigger_owner_refs,
                )
                connection.execute(
                    """
                    INSERT INTO scheduler_receipts(
                        job_id, identity_sha256, event_id_sha256, event_fingerprint_sha256,
                        job_kind, decision_revision, status, outcome, resolved_at
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        receipt.job_id,
                        self._identity_sha256,
                        receipt.event_id_sha256,
                        receipt.event_fingerprint_sha256,
                        receipt.job_kind,
                        receipt.decision_revision,
                        receipt.status,
                        receipt.outcome,
                        resolved_at,
                    ),
                )
                receipts.append(receipt)
        return tuple(receipts)

    def jobs_for(self, event_id_sha256: str) -> tuple[ScheduledJob, ...]:
        """Return jobs by opaque event hash, never by the raw host event id."""

        _digest(event_id_sha256, "event_id_sha256")
        with self._connection() as connection:
            return self._jobs_for_event(connection, event_id_sha256)

    def receipt_for(self, job_id: str) -> SchedulerReceipt | None:
        _digest(job_id, "job_id")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT receipts.job_id, receipts.event_id_sha256,
                       receipts.event_fingerprint_sha256, receipts.job_kind,
                       receipts.decision_revision, receipts.status, receipts.outcome,
                       receipts.resolved_at, events.decision_plan_json
                FROM scheduler_receipts AS receipts
                JOIN host_events AS events
                  ON events.identity_sha256=receipts.identity_sha256
                 AND events.event_id_sha256=receipts.event_id_sha256
                WHERE receipts.identity_sha256=? AND receipts.job_id=?
                """,
                (self._identity_sha256, job_id),
            ).fetchone()
        return _receipt_from_row(row) if row is not None else None

    def has_pending(self) -> bool:
        """Return whether this service identity owns any unresolved timer."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM scheduled_jobs
                WHERE identity_sha256=? AND status='pending'
                LIMIT 1
                """,
                (self._identity_sha256,),
            ).fetchone()
        return row is not None

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT value FROM scheduler_metadata WHERE key='identity_sha256'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO scheduler_metadata(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
                connection.execute(
                    "INSERT INTO scheduler_metadata(key, value) VALUES('identity_sha256', ?)",
                    (self._identity_sha256,),
                )
            elif str(row[0]) != self._identity_sha256:
                raise RuntimeError("scheduler state path identity mismatch")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection

    def _jobs_for_event(
        self,
        connection: sqlite3.Connection,
        event_id_sha256: str,
    ) -> tuple[ScheduledJob, ...]:
        rows = connection.execute(
            """
            SELECT job_id, event_id_sha256, event_fingerprint_sha256, job_kind,
                   due_at, decision_revision, reason_code, status
            FROM scheduled_jobs
            WHERE identity_sha256=? AND event_id_sha256=?
            ORDER BY due_at, job_kind, job_id
            """,
            (self._identity_sha256, event_id_sha256),
        ).fetchall()
        jobs = tuple(_job_from_row(row) for row in rows)
        if len(jobs) != len(_JOB_KINDS):
            raise RuntimeError("scheduler event has an incomplete job set")
        return jobs

    def _plan_for_event(
        self,
        connection: sqlite3.Connection,
        event_id_sha256: str,
    ) -> AdvisoryDecisionPlan:
        row = connection.execute(
            """
            SELECT decision_plan_json
            FROM host_events
            WHERE identity_sha256=? AND event_id_sha256=?
            """,
            (self._identity_sha256, event_id_sha256),
        ).fetchone()
        if row is None or not isinstance(row[0], str):
            raise RuntimeError("scheduler event plan is unavailable")
        return _decision_plan_from_json(row[0])

    def _new_jobs(
        self,
        event_id_sha256: str,
        event_fingerprint_sha256: str,
        plan: AdvisoryDecisionPlan,
    ) -> tuple[ScheduledJob, ...]:
        details = (
            (_ACTIVATION_JOB, plan.activation_due_at, plan.activation_reason_code),
            (_CONSOLIDATION_JOB, plan.consolidation_due_at, plan.consolidation_reason_code),
        )
        return tuple(
            ScheduledJob(
                job_id=_sha256(f"scheduler-job:{event_fingerprint_sha256}:{job_kind}"),
                event_id_sha256=event_id_sha256,
                event_fingerprint_sha256=event_fingerprint_sha256,
                job_kind=job_kind,
                due_at=due_at,
                decision_revision=plan.decision_revision,
                reason_code=reason_code,
                status="pending",
            )
            for job_kind, due_at, reason_code in details
        )


def _job_from_row(row: tuple[object, ...]) -> ScheduledJob:
    if len(row) != 8:
        raise RuntimeError("scheduler job row is invalid")
    return ScheduledJob(
        job_id=_row_string(row, 0),
        event_id_sha256=_row_string(row, 1),
        event_fingerprint_sha256=_row_string(row, 2),
        job_kind=_row_string(row, 3),
        due_at=_timestamp_from_string(row[4]),
        decision_revision=_row_string(row, 5),
        reason_code=_row_string(row, 6),
        status=_row_string(row, 7),
    )


def _receipt_from_row(row: tuple[object, ...]) -> SchedulerReceipt:
    if len(row) != 9:
        raise RuntimeError("scheduler receipt row is invalid")
    plan = _decision_plan_from_json(_row_string(row, 8))
    return SchedulerReceipt(
        job_id=_row_string(row, 0),
        event_id_sha256=_row_string(row, 1),
        event_fingerprint_sha256=_row_string(row, 2),
        job_kind=_row_string(row, 3),
        decision_revision=_row_string(row, 4),
        status=_row_string(row, 5),
        outcome=_row_string(row, 6),
        resolved_at=_timestamp_from_string(row[7]),
        matched_rule_sha256s=plan.matched_rule_sha256s,
        matched_action_sha256s=plan.matched_action_sha256s,
        deferred_trigger_owner_refs=plan.deferred_trigger_owner_refs,
    )


def _decision_plan_from_json(raw: str) -> AdvisoryDecisionPlan:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("scheduler decision plan JSON is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("scheduler decision plan JSON is invalid")
    return AdvisoryDecisionPlan.from_record(value)


def _advisory_outcome(
    job_kind: str,
    *,
    consolidation_disposition: str,
) -> tuple[str, str]:
    if job_kind == _ACTIVATION_JOB:
        return "recorded", "activation_opportunity_recorded"
    if job_kind == _CONSOLIDATION_JOB:
        if consolidation_disposition == "assess":
            return "recorded", "consolidation_assessment_recorded"
        return "deferred", "consolidation_deferred"
    raise RuntimeError("scheduler job kind is invalid")


def _identity_sha256(identity: LocalServiceIdentity) -> str:
    return _sha256(
        _canonical_json(
            {
                "repo_root": str(identity.repo_root.resolve()),
                "workspace_id": identity.workspace_id,
                "profile_id": identity.profile_id,
                "backend": identity.backend,
                "protocol_version": identity.protocol_version,
            }
        )
    )


def _require_identity(identity: LocalServiceIdentity) -> None:
    if not isinstance(identity, LocalServiceIdentity):
        raise TypeError("identity must be a LocalServiceIdentity")


def _token(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or value[0] not in _TOKEN_START
        or any(character not in _TOKEN_REMAINDER for character in value[1:])
    ):
        raise ValueError(f"{name} must be a content-free identifier")
    return value


def _tokens(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 64:
        raise ValueError(f"{name} must be a bounded sequence of content-free identifiers")
    return tuple(sorted({_token(item, name) for item in value}))


def _digests(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 64:
        raise ValueError(f"{name} must be a bounded sequence of SHA-256 digests")
    return tuple(sorted({_digest(item, name) for item in value}))


def _optional_token(value: object, name: str) -> str:
    if value == "":
        return ""
    return _token(value, name)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be an aware timestamp")
    return value.astimezone(UTC)


def _timestamp_string(value: datetime) -> str:
    return _timestamp(value, "timestamp").isoformat()


def _timestamp_from_string(value: object) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError("scheduler timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeError("scheduler timestamp is invalid") from exc
    try:
        return _timestamp(parsed, "timestamp")
    except ValueError as exc:
        raise RuntimeError("scheduler timestamp is invalid") from exc


def _clock(value: datetime | None) -> datetime:
    return _timestamp(value, "now") if value is not None else datetime.now(UTC)


def _row_string(row: tuple[object, ...], index: int) -> str:
    value = row[index]
    if not isinstance(value, str):
        raise RuntimeError("scheduler row is invalid")
    return value


def _required_string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise RuntimeError("scheduler decision plan is invalid")
    return item


def _required_strings(value: Mapping[str, object], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list):
        raise RuntimeError("scheduler decision plan is invalid")
    return tuple(item)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
