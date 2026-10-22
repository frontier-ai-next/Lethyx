"""Bounded durable background work lanes for the persistent local service."""

import hashlib
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, get_args

from um_mem.memory.local_service import LocalServiceIdentity, local_service_runtime_paths

WorkLane = Literal[
    "indexing",
    "working_continuity",
    "prospective_scheduler",
    "consolidation_transition",
    "policy_shadow_qualification",
]
_LANES: tuple[WorkLane, ...] = get_args(WorkLane)
_ACTIVE_STATES = ("pending", "running")
_CRITICAL_LANES: frozenset[WorkLane] = frozenset({"indexing", "consolidation_transition"})
_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS work_items (
    work_id TEXT PRIMARY KEY,
    lane TEXT NOT NULL,
    work_ref_sha256 TEXT NOT NULL,
    work_revision TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    submitted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_code TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS work_items_lane_state
ON work_items(lane, state, submitted_at, work_id);
CREATE TABLE IF NOT EXISTS admission_receipts (
    admission_id TEXT PRIMARY KEY,
    work_id TEXT NOT NULL,
    lane TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempt_receipts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id TEXT NOT NULL,
    lane TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    error_code TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _required(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 256:
        raise ValueError(f"{field_name} must be a bounded non-empty identifier")
    return text


@dataclass(frozen=True)
class WorkLanePolicy:
    capacity: int
    target_latency_ms: int

    def __post_init__(self) -> None:
        for field_name in ("capacity", "target_latency_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")


DEFAULT_LANE_POLICIES: dict[WorkLane, WorkLanePolicy] = {
    "indexing": WorkLanePolicy(capacity=64, target_latency_ms=5_000),
    "working_continuity": WorkLanePolicy(capacity=32, target_latency_ms=1_000),
    "prospective_scheduler": WorkLanePolicy(capacity=64, target_latency_ms=1_000),
    "consolidation_transition": WorkLanePolicy(capacity=32, target_latency_ms=10_000),
    "policy_shadow_qualification": WorkLanePolicy(capacity=16, target_latency_ms=30_000),
}


@dataclass(frozen=True)
class WorkAdmissionReceipt:
    admission_id: str
    work_id: str
    lane: WorkLane
    status: Literal["queued", "replayed", "backpressured"]
    reason: str
    recorded_at: str


@dataclass(frozen=True)
class ServiceWorkItem:
    work_id: str
    lane: WorkLane
    work_ref_sha256: str
    work_revision: str
    state: Literal["pending", "running", "completed", "poisoned"]
    attempts: int
    max_attempts: int
    submitted_at: str
    updated_at: str
    error_code: str


@dataclass(frozen=True)
class LaneHealth:
    lane: WorkLane
    capacity: int
    target_latency_ms: int
    pending: int
    running: int
    completed: int
    poisoned: int
    backpressured: int
    oldest_pending_age_ms: int
    slo_breached: bool


@dataclass(frozen=True)
class ServiceQueueHealth:
    lanes: tuple[LaneHealth, ...]
    emergency_read_only: bool
    no_silent_drop: bool = True


class DurableServiceWorkQueue:
    """Five isolated SQLite lanes with durable admission and attempt receipts."""

    def __init__(
        self,
        identity: LocalServiceIdentity,
        *,
        policies: Mapping[WorkLane, WorkLanePolicy] | None = None,
        recover_abandoned: bool = False,
    ) -> None:
        if not isinstance(identity, LocalServiceIdentity):
            raise TypeError("identity must be a LocalServiceIdentity")
        self.identity = identity
        self.policies = dict(policies or DEFAULT_LANE_POLICIES)
        if set(self.policies) != set(_LANES):
            raise ValueError("queue policies must define every service work lane")
        if any(not isinstance(policy, WorkLanePolicy) for policy in self.policies.values()):
            raise TypeError("queue policies must be WorkLanePolicy values")
        registry = local_service_runtime_paths(identity).registry
        digest = _sha256(
            "|".join(
                (
                    str(identity.repo_root.resolve()),
                    identity.workspace_id,
                    identity.profile_id,
                    identity.backend,
                )
            )
        )[:20]
        self.path = registry.with_name(f"{registry.stem}.{digest}.work.sqlite")
        self._initialize(recover_abandoned=recover_abandoned)

    def close(self) -> None:
        """Connections are operation-scoped, so there is no live handle to close."""

    def submit(
        self,
        lane: WorkLane,
        *,
        work_ref: str,
        work_revision: str,
        max_attempts: int = 3,
    ) -> WorkAdmissionReceipt:
        lane = self._lane(lane)
        reference_hash = _sha256(_required(work_ref, field_name="work_ref"))
        revision = _required(work_revision, field_name="work_revision")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 10
        ):
            raise ValueError("max_attempts must be between one and ten")
        work_id = _sha256(f"service-work:{lane}:{reference_hash}:{revision}")
        recorded_at = _now()
        admission_id = _sha256(f"service-admission:{work_id}:{recorded_at}:{secrets.token_hex(8)}")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT state FROM work_items WHERE work_id=?",
                (work_id,),
            ).fetchone()
            if existing is not None:
                return self._record_admission(
                    connection,
                    admission_id=admission_id,
                    work_id=work_id,
                    lane=lane,
                    status="replayed",
                    reason=str(existing[0]),
                    recorded_at=recorded_at,
                )
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM work_items WHERE lane=? AND state IN ('pending','running')",
                    (lane,),
                ).fetchone()[0]
            )
            if active >= self.policies[lane].capacity:
                if lane in _CRITICAL_LANES:
                    connection.execute(
                        "INSERT OR REPLACE INTO queue_metadata(key,value) VALUES('emergency_read_only','1')"
                    )
                return self._record_admission(
                    connection,
                    admission_id=admission_id,
                    work_id=work_id,
                    lane=lane,
                    status="backpressured",
                    reason="lane_capacity_exhausted",
                    recorded_at=recorded_at,
                )
            connection.execute(
                """
                INSERT INTO work_items(
                    work_id,lane,work_ref_sha256,work_revision,state,attempts,max_attempts,
                    submitted_at,updated_at,error_code
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    work_id,
                    lane,
                    reference_hash,
                    revision,
                    "pending",
                    0,
                    max_attempts,
                    recorded_at,
                    recorded_at,
                    "",
                ),
            )
            return self._record_admission(
                connection,
                admission_id=admission_id,
                work_id=work_id,
                lane=lane,
                status="queued",
                reason="accepted",
                recorded_at=recorded_at,
            )

    def claim(self, lane: WorkLane) -> ServiceWorkItem | None:
        lane = self._lane(lane)
        updated_at = _now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT work_id,lane,work_ref_sha256,work_revision,state,attempts,max_attempts,
                       submitted_at,updated_at,error_code
                FROM work_items WHERE lane=? AND state='pending'
                ORDER BY submitted_at,work_id LIMIT 1
                """,
                (lane,),
            ).fetchone()
            if row is None:
                return None
            work_id = str(row[0])
            connection.execute(
                "UPDATE work_items SET state='running',updated_at=? WHERE work_id=? AND state='pending'",
                (updated_at, work_id),
            )
            return self.item(work_id, connection=connection)

    def complete(self, work_id: str) -> ServiceWorkItem:
        return self._resolve(work_id, error_code="")

    def fail(self, work_id: str, *, error_code: str) -> ServiceWorkItem:
        return self._resolve(work_id, error_code=_required(error_code, field_name="error_code"))

    def process_one(
        self,
        lane: WorkLane,
        handler: Callable[[ServiceWorkItem], None],
    ) -> ServiceWorkItem | None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        item = self.claim(lane)
        if item is None:
            return None
        try:
            handler(item)
        except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
            return self.fail(item.work_id, error_code=type(exc).__name__)
        return self.complete(item.work_id)

    def item(
        self,
        work_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ServiceWorkItem:
        with nullcontext(connection) if connection is not None else self._connection() as current:
            row = current.execute(
                """
                SELECT work_id,lane,work_ref_sha256,work_revision,state,attempts,max_attempts,
                       submitted_at,updated_at,error_code
                FROM work_items WHERE work_id=?
                """,
                (_required(work_id, field_name="work_id"),),
            ).fetchone()
        if row is None:
            raise KeyError("unknown service work item")
        return ServiceWorkItem(*row)

    def health(self, *, now: datetime | None = None) -> ServiceQueueHealth:
        clock = (now or datetime.now(UTC)).astimezone(UTC)
        with self._connection() as connection:
            lanes = tuple(self._lane_health(connection, lane, clock=clock) for lane in _LANES)
            read_only = self._metadata_bool(connection, "emergency_read_only")
        return ServiceQueueHealth(lanes=lanes, emergency_read_only=read_only)

    def requeue_poison(self, work_id: str) -> ServiceWorkItem:
        updated_at = _now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            item = self.item(work_id, connection=connection)
            if item.state != "poisoned":
                raise ValueError("only poisoned work may be requeued")
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM work_items WHERE lane=? AND state IN ('pending','running')",
                    (item.lane,),
                ).fetchone()[0]
            )
            if active >= self.policies[item.lane].capacity:
                raise RuntimeError("work lane remains backpressured")
            connection.execute(
                "UPDATE work_items SET state='pending',attempts=0,error_code='',updated_at=? WHERE work_id=?",
                (updated_at, item.work_id),
            )
            self._refresh_read_only(connection)
            return self.item(item.work_id, connection=connection)

    def admission_count(self, *, status: str) -> int:
        with self._connection() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM admission_receipts WHERE status=?",
                    (_required(status, field_name="status"),),
                ).fetchone()[0]
            )

    def _resolve(self, work_id: str, *, error_code: str) -> ServiceWorkItem:
        updated_at = _now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            item = self.item(work_id, connection=connection)
            if item.state != "running":
                raise ValueError("only running work may be resolved")
            attempt = item.attempts + 1
            if error_code:
                state = "poisoned" if attempt >= item.max_attempts else "pending"
                outcome = state
            else:
                state = "completed"
                outcome = "completed"
            connection.execute(
                "UPDATE work_items SET state=?,attempts=?,updated_at=?,error_code=? WHERE work_id=?",
                (state, attempt, updated_at, error_code, item.work_id),
            )
            connection.execute(
                "INSERT INTO attempt_receipts(work_id,lane,attempt,outcome,error_code,recorded_at) VALUES(?,?,?,?,?,?)",
                (item.work_id, item.lane, attempt, outcome, error_code, updated_at),
            )
            if state == "poisoned" and item.lane in _CRITICAL_LANES:
                connection.execute(
                    "INSERT OR REPLACE INTO queue_metadata(key,value) VALUES('emergency_read_only','1')"
                )
            self._refresh_read_only(connection)
            return self.item(item.work_id, connection=connection)

    def _record_admission(
        self,
        connection: sqlite3.Connection,
        *,
        admission_id: str,
        work_id: str,
        lane: WorkLane,
        status: Literal["queued", "replayed", "backpressured"],
        reason: str,
        recorded_at: str,
    ) -> WorkAdmissionReceipt:
        connection.execute(
            "INSERT INTO admission_receipts(admission_id,work_id,lane,status,reason,recorded_at) VALUES(?,?,?,?,?,?)",
            (admission_id, work_id, lane, status, reason, recorded_at),
        )
        return WorkAdmissionReceipt(admission_id, work_id, lane, status, reason, recorded_at)

    def _lane_health(
        self,
        connection: sqlite3.Connection,
        lane: WorkLane,
        *,
        clock: datetime,
    ) -> LaneHealth:
        counts = {
            str(state): int(count)
            for state, count in connection.execute(
                "SELECT state,COUNT(*) FROM work_items WHERE lane=? GROUP BY state",
                (lane,),
            ).fetchall()
        }
        oldest = connection.execute(
            "SELECT MIN(submitted_at) FROM work_items WHERE lane=? AND state='pending'",
            (lane,),
        ).fetchone()[0]
        age_ms = 0
        if isinstance(oldest, str) and oldest:
            age_ms = max(0, int((clock - datetime.fromisoformat(oldest)).total_seconds() * 1_000))
        backpressured = int(
            connection.execute(
                "SELECT COUNT(*) FROM admission_receipts WHERE lane=? AND status='backpressured'",
                (lane,),
            ).fetchone()[0]
        )
        policy = self.policies[lane]
        return LaneHealth(
            lane=lane,
            capacity=policy.capacity,
            target_latency_ms=policy.target_latency_ms,
            pending=counts.get("pending", 0),
            running=counts.get("running", 0),
            completed=counts.get("completed", 0),
            poisoned=counts.get("poisoned", 0),
            backpressured=backpressured,
            oldest_pending_age_ms=age_ms,
            slo_breached=age_ms > policy.target_latency_ms,
        )

    def _refresh_read_only(self, connection: sqlite3.Connection) -> None:
        critical_poison = int(
            connection.execute(
                "SELECT COUNT(*) FROM work_items WHERE lane IN ('indexing','consolidation_transition') AND state='poisoned'"
            ).fetchone()[0]
        )
        critical_backpressure = any(
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM work_items WHERE lane=? AND state IN ('pending','running')",
                    (lane,),
                ).fetchone()[0]
            )
            >= self.policies[lane].capacity
            for lane in _CRITICAL_LANES
        )
        connection.execute(
            "INSERT OR REPLACE INTO queue_metadata(key,value) VALUES('emergency_read_only',?)",
            ("1" if critical_poison or critical_backpressure else "0",),
        )

    @staticmethod
    def _metadata_bool(connection: sqlite3.Connection, key: str) -> bool:
        row = connection.execute("SELECT value FROM queue_metadata WHERE key=?", (key,)).fetchone()
        return row is not None and str(row[0]) == "1"

    def _initialize(self, *, recover_abandoned: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO queue_metadata(key,value) VALUES('emergency_read_only','0')"
            )
            if not recover_abandoned:
                return
            recovered_at = _now()
            abandoned = connection.execute(
                "SELECT work_id,lane,attempts,max_attempts FROM work_items WHERE state='running'"
            ).fetchall()
            for work_id, lane, attempts, max_attempts in abandoned:
                attempt = int(attempts) + 1
                state = "poisoned" if attempt >= int(max_attempts) else "pending"
                connection.execute(
                    "UPDATE work_items SET state=?,attempts=?,updated_at=?,error_code=? WHERE work_id=?",
                    (state, attempt, recovered_at, "listener_restarted", work_id),
                )
                connection.execute(
                    "INSERT INTO attempt_receipts(work_id,lane,attempt,outcome,error_code,recorded_at) VALUES(?,?,?,?,?,?)",
                    (
                        work_id,
                        lane,
                        attempt,
                        "poisoned" if state == "poisoned" else "requeued",
                        "listener_restarted",
                        recovered_at,
                    ),
                )
            self._refresh_read_only(connection)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("PRAGMA busy_timeout=5000")
            yield connection

    @staticmethod
    def _lane(value: WorkLane) -> WorkLane:
        if value not in _LANES:
            raise ValueError("unknown service work lane")
        return value


class ServiceQueueWorker:
    """Run queue handlers on a separate bounded background thread."""

    def __init__(
        self,
        queue: DurableServiceWorkQueue,
        handlers: Mapping[WorkLane, Callable[[ServiceWorkItem], None]],
    ) -> None:
        if not isinstance(queue, DurableServiceWorkQueue):
            raise TypeError("queue must be a DurableServiceWorkQueue")
        if set(handlers) != set(_LANES) or any(not callable(item) for item in handlers.values()):
            raise ValueError("worker handlers must cover every service work lane")
        self.queue = queue
        self.handlers = dict(handlers)
        self._stopped = threading.Event()
        self._threads: dict[WorkLane, threading.Thread] = {}

    def start(self) -> None:
        for lane in _LANES:
            current = self._threads.get(lane)
            if current is not None and current.is_alive():
                continue
            thread = threading.Thread(
                target=self._run_lane,
                args=(lane,),
                name=f"um-mem-service-work-{lane}",
                daemon=True,
            )
            self._threads[lane] = thread
            thread.start()

    def close(self) -> None:
        self._stopped.set()
        # An in-flight handler can wait on native projection. Its work is durable;
        # listener shutdown must not wait for every lane's full handler timeout.
        deadline = time.monotonic() + 0.75
        for thread in self._threads.values():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _run_lane(self, lane: WorkLane) -> None:
        while not self._stopped.is_set():
            item = self.queue.process_one(lane, self.handlers[lane])
            if item is None:
                self._stopped.wait(0.05)
