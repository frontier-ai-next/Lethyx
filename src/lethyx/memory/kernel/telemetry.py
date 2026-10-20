"""Memory subsystem telemetry — failures must be visible, not silent."""

import atexit
import hashlib
import json
import logging
import os
import threading
import time
import tracemalloc
from collections.abc import Mapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from filelock import FileLock, Timeout

from lethyx.llm_budget import estimate_tokens

log = logging.getLogger(__name__)

_MEMORY_EVENT_SCHEMA_VERSION = 1
_MEMORY_EVENT_FILENAME = "memory_events.jsonl"
_EVENT_IDENTIFIER_HASH_LENGTH = 16
_PROFILE_MAX_FIELDS = 32
_PROFILE_MAX_ITEMS = 16
_PROFILE_MAX_DEPTH = 3
_EVENT_BATCH_LIMIT = 32
_EVENT_MAX_PENDING_BYTES = 256 * 1024
_EVENT_ROTATE_BYTES = 8 * 1024 * 1024
_EVENT_ROTATIONS = 3
_EVENT_FLUSH_INTERVAL_SECONDS = 0.25
_EVENT_BUFFER_LOCK = threading.Lock()
_EVENT_FLUSH_LOCK = threading.Lock()
_EVENT_WAKE = threading.Event()
_EVENT_BUFFER: dict[Path, list[bytes]] = {}
_EVENT_BUFFER_BYTES: dict[Path, int] = {}
_EVENT_DROPPED: dict[Path, int] = {}
_EVENT_FLUSH_THREAD: threading.Thread | None = None
_PROFILE_ACTIVE_LOCK = threading.Lock()
_PROFILE_ACTIVE: dict[str, int] = {}
_PROFILE_PEAK: dict[str, int] = {}
_CURRENT_PROFILE_SPAN: ContextVar[str | None] = ContextVar(
    "lethyx_current_profile_span",
    default=None,
)
_CURRENT_PROFILE_CONTEXT: ContextVar[tuple[str, str, str, str]] = ContextVar(
    "lethyx_current_profile_context",
    default=("", "", "", ""),
)


def memory_event_trace_path(
    repo_root: Path | str | None,
    *,
    drive_root: Path | None = None,
) -> Path | None:
    """Return the append-only memory event trace path for a local runtime."""

    if drive_root is not None:
        return drive_root / "logs" / _MEMORY_EVENT_FILENAME
    if repo_root is not None:
        return Path(repo_root) / ".Lethyx" / "memory" / "telemetry" / _MEMORY_EVENT_FILENAME
    return None


def _event_payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _identifier_hashes(values: list[str] | None) -> list[str]:
    return [
        hashlib.sha256(value.encode("utf-8")).hexdigest()[:_EVENT_IDENTIFIER_HASH_LENGTH]
        for value in values or []
        if value
    ]


def _profile_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


def memory_profile_summary(value: object, *, _depth: int = 0) -> dict[str, Any]:
    """Return bounded, content-free size and digest metadata for one value."""

    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, (int, float)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, str):
        return {
            "type": "text",
            "chars": len(value),
            "bytes": len(value.encode("utf-8")),
            "estimated_tokens": estimate_tokens(value),
            "sha256": _profile_digest(value),
        }
    if isinstance(value, Mapping):
        fields = sorted(str(key) for key in value)[:_PROFILE_MAX_FIELDS]
        summary: dict[str, Any] = {
            "type": "object",
            "field_count": len(value),
            "fields": fields,
            "sha256": _profile_digest(value),
        }
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            serialized = repr(value)
        summary["bytes"] = len(serialized.encode("utf-8"))
        summary["estimated_tokens"] = estimate_tokens(serialized)
        if _depth < _PROFILE_MAX_DEPTH:
            summary["field_summaries"] = {
                key: memory_profile_summary(value.get(key), _depth=_depth + 1)
                for key in fields
                if key in value
            }
        return summary
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        summary = {
            "type": "array",
            "item_count": len(values),
            "item_types": sorted({type(item).__name__ for item in values}),
            "sha256": _profile_digest(values),
        }
        try:
            serialized = json.dumps(
                values,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            serialized = repr(values)
        summary["bytes"] = len(serialized.encode("utf-8"))
        summary["estimated_tokens"] = estimate_tokens(serialized)
        if _depth < _PROFILE_MAX_DEPTH:
            summary["items"] = [
                memory_profile_summary(item, _depth=_depth + 1)
                for item in values[:_PROFILE_MAX_ITEMS]
            ]
        return summary
    return {
        "type": type(value).__name__,
        "sha256": _profile_digest(str(value)),
    }


def current_memory_profile_span_id() -> str:
    """Return the current parent span for backend-level telemetry."""

    return _CURRENT_PROFILE_SPAN.get() or ""


def current_memory_profile_context() -> dict[str, str]:
    """Return correlation fields inherited by backend-level telemetry."""

    profile_id, user_id, chat_id, run_id = _CURRENT_PROFILE_CONTEXT.get()
    return {
        "profile_id": profile_id,
        "user_id": user_id,
        "chat_id": chat_id,
        "run_id": run_id,
    }


def _profile_resource_snapshot() -> dict[str, float | int]:
    resources: dict[str, float | int] = {
        "process_cpu_ms": round(time.process_time() * 1_000, 3),
        "thread_cpu_ms": round(time.thread_time() * 1_000, 3),
    }
    if tracemalloc.is_tracing():
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        resources["python_alloc_current_bytes"] = int(current_bytes)
        resources["python_alloc_peak_bytes"] = int(peak_bytes)
    return resources


def _flush_memory_events(log_path: Path | None = None) -> None:
    with _EVENT_FLUSH_LOCK:
        with _EVENT_BUFFER_LOCK:
            paths = [log_path] if log_path is not None else list(_EVENT_BUFFER)
            batches = {path: _EVENT_BUFFER.pop(path, []) for path in paths}
            for path in paths:
                _EVENT_BUFFER_BYTES.pop(path, None)
        for path, rows in batches.items():
            if not rows:
                continue
            encoded = b"".join(rows)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(str(path) + ".lock", timeout=1):
                    if path.exists() and path.stat().st_size + len(encoded) > _EVENT_ROTATE_BYTES:
                        oldest = path.with_name(path.name + f".{_EVENT_ROTATIONS}")
                        oldest.unlink(missing_ok=True)
                        for number in range(_EVENT_ROTATIONS - 1, 0, -1):
                            source = path.with_name(path.name + f".{number}")
                            if source.exists():
                                source.replace(path.with_name(path.name + f".{number + 1}"))
                        path.replace(path.with_name(path.name + ".1"))
                    with path.open("ab") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
            except (OSError, Timeout):
                with _EVENT_BUFFER_LOCK:
                    current = _EVENT_BUFFER_BYTES.get(path, 0)
                    if current + len(encoded) <= _EVENT_MAX_PENDING_BYTES:
                        _EVENT_BUFFER[path] = rows + _EVENT_BUFFER.get(path, [])
                        _EVENT_BUFFER_BYTES[path] = current + len(encoded)
                    else:
                        _EVENT_DROPPED[path] = _EVENT_DROPPED.get(path, 0) + len(rows)
                log.warning("Failed to flush memory event trace", exc_info=True)


def _event_flush_loop() -> None:
    while True:
        _EVENT_WAKE.wait(_EVENT_FLUSH_INTERVAL_SECONDS)
        _EVENT_WAKE.clear()
        _flush_memory_events()


def _append_memory_event(log_path: Path, payload: dict[str, Any]) -> None:
    global _EVENT_FLUSH_THREAD
    try:
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with _EVENT_BUFFER_LOCK:
            pending = _EVENT_BUFFER_BYTES.get(log_path, 0)
            if pending + len(encoded) > _EVENT_MAX_PENDING_BYTES:
                _EVENT_DROPPED[log_path] = _EVENT_DROPPED.get(log_path, 0) + 1
                return
            rows = _EVENT_BUFFER.setdefault(log_path, [])
            rows.append(encoded)
            _EVENT_BUFFER_BYTES[log_path] = pending + len(encoded)
            if _EVENT_FLUSH_THREAD is None or not _EVENT_FLUSH_THREAD.is_alive():
                _EVENT_FLUSH_THREAD = threading.Thread(
                    target=_event_flush_loop,
                    name="lethyx-telemetry-flush",
                    daemon=True,
                )
                _EVENT_FLUSH_THREAD.start()
            if len(rows) >= _EVENT_BATCH_LIMIT:
                _EVENT_WAKE.set()
    except (TypeError, ValueError):
        with _EVENT_BUFFER_LOCK:
            _EVENT_DROPPED[log_path] = _EVENT_DROPPED.get(log_path, 0) + 1
        log.warning("Failed to append memory event trace", exc_info=True)


def _read_memory_event_rows(log_path: Path, *, limit: int) -> tuple[list[dict[str, Any]], int]:
    _flush_memory_events(log_path)
    if limit <= 0:
        return [], 0
    rows: list[dict[str, Any]] = []
    invalid_row_count = 0
    paths = [
        log_path.with_name(log_path.name + f".{number}")
        for number in range(_EVENT_ROTATIONS, 0, -1)
    ] + [log_path]
    if not any(path.exists() for path in paths):
        return [], 0
    try:
        with FileLock(str(log_path) + ".lock", timeout=1):
            for path in paths:
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    for physical_line in handle:
                        if not physical_line.strip():
                            continue
                        try:
                            row = json.loads(physical_line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            invalid_row_count += 1
                            continue
                        if not isinstance(row, dict):
                            invalid_row_count += 1
                            continue
                        event_sha256 = row.pop("event_sha256", "")
                        if (
                            row.get("schema_version") != _MEMORY_EVENT_SCHEMA_VERSION
                            or not isinstance(event_sha256, str)
                            or event_sha256 != _event_payload_sha256(row)
                        ):
                            invalid_row_count += 1
                            continue
                        row["event_sha256"] = event_sha256
                        rows.append(row)
                        if len(rows) > limit:
                            del rows[: len(rows) - limit]
    except (OSError, Timeout):
        log.warning("Failed to read memory event trace", exc_info=True)
    return rows, invalid_row_count


atexit.register(_flush_memory_events)


def read_memory_events(
    repo_root: Path | str | None,
    *,
    drive_root: Path | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Read valid, hash-addressed memory event rows without stdout scraping."""

    log_path = memory_event_trace_path(repo_root, drive_root=drive_root)
    if log_path is None:
        return []
    rows, _ = _read_memory_event_rows(log_path, limit=max(0, limit))
    return rows


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(len(ordered) * percentile + 0.499999) - 1))
    return round(ordered[index], 3)


def _profile_report_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiled = [row for row in rows if row.get("event_type") == "memory_profile"]
    by_operation: dict[str, dict[str, Any]] = {}
    by_component: dict[str, int] = {}
    all_durations: list[float] = []
    all_process_cpu_ms = 0.0
    all_thread_cpu_ms = 0.0
    max_in_flight = 0
    for row in profiled:
        operation = str(row.get("operation") or "unknown")
        component = str(row.get("component") or "unknown")
        duration = row.get("duration_ms")
        duration_value = float(duration) if isinstance(duration, (int, float)) else 0.0
        stats = by_operation.setdefault(
            operation,
            {
                "count": 0,
                "failed": 0,
                "durations_ms": [],
                "input_estimated_tokens": 0,
                "output_estimated_tokens": 0,
                "process_cpu_ms": 0.0,
                "thread_cpu_ms": 0.0,
                "max_in_flight": 0,
                "components": {},
            },
        )
        stats["count"] += 1
        if row.get("status") in {"failed", "blocked", "unavailable"}:
            stats["failed"] += 1
        stats["durations_ms"].append(duration_value)
        data = row.get("data")
        if isinstance(data, dict):
            input_summary = data.get("input")
            output_summary = data.get("output")
            if isinstance(input_summary, dict):
                stats["input_estimated_tokens"] += int(input_summary.get("estimated_tokens") or 0)
            if isinstance(output_summary, dict):
                stats["output_estimated_tokens"] += int(output_summary.get("estimated_tokens") or 0)
        resources = row.get("resources")
        if isinstance(resources, dict):
            process_cpu_ms = float(resources.get("process_cpu_ms") or 0)
            thread_cpu_ms = float(resources.get("thread_cpu_ms") or 0)
            stats["process_cpu_ms"] += process_cpu_ms
            stats["thread_cpu_ms"] += thread_cpu_ms
            all_process_cpu_ms += process_cpu_ms
            all_thread_cpu_ms += thread_cpu_ms
        parallelism = row.get("parallelism")
        active = int(parallelism.get("max_in_flight") or 0) if isinstance(parallelism, dict) else 0
        stats["max_in_flight"] = max(stats["max_in_flight"], active)
        max_in_flight = max(max_in_flight, active)
        components = stats["components"]
        components[component] = int(components.get(component, 0)) + 1
        by_component[component] = by_component.get(component, 0) + 1
        all_durations.append(duration_value)

    operation_report: dict[str, Any] = {}
    for operation, stats in sorted(by_operation.items()):
        durations = [float(value) for value in stats.pop("durations_ms")]
        operation_report[operation] = {
            **stats,
            "total_ms": round(sum(durations), 3),
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "max_ms": round(max(durations), 3) if durations else 0.0,
        }
    slowest = sorted(
        (
            {
                "event_id": str(row.get("event_id") or ""),
                "operation": str(row.get("operation") or ""),
                "component": str(row.get("component") or ""),
                "backend": str(row.get("backend") or ""),
                "run_id": str(row.get("run_id") or ""),
                "user_id": str(row.get("user_id") or ""),
                "chat_id": str(row.get("chat_id") or ""),
                "duration_ms": row.get("duration_ms") or 0,
                "status": str(row.get("status") or ""),
                "reason": str(row.get("reason") or ""),
                "outcome": str(row.get("outcome") or ""),
            }
            for row in profiled
        ),
        key=lambda row: float(row["duration_ms"]),
        reverse=True,
    )[:10]
    return {
        "profiled_operation_count": len(profiled),
        "total_duration_ms": round(sum(all_durations), 3),
        "total_process_cpu_ms": round(all_process_cpu_ms, 3),
        "total_thread_cpu_ms": round(all_thread_cpu_ms, 3),
        "p50_ms": _percentile(all_durations, 0.50),
        "p95_ms": _percentile(all_durations, 0.95),
        "max_in_flight": max_in_flight,
        "parallelism_observed_count": sum(
            1
            for row in profiled
            if isinstance(row.get("parallelism"), dict)
            and int(row["parallelism"].get("max_in_flight") or 0) > 1
        ),
        "by_operation": operation_report,
        "by_component": dict(sorted(by_component.items())),
        "slowest": slowest,
    }


def memory_profile_report(
    repo_root: Path | str | None,
    *,
    drive_root: Path | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Summarize developer profile spans by operation and implementation side."""

    log_path = memory_event_trace_path(repo_root, drive_root=drive_root)
    rows, invalid_row_count = (
        _read_memory_event_rows(log_path, limit=max(0, limit)) if log_path is not None else ([], 0)
    )
    return {
        "schema_version": _MEMORY_EVENT_SCHEMA_VERSION,
        "invalid_event_count": invalid_row_count,
        "dropped_event_count": _EVENT_DROPPED.get(log_path, 0) if log_path is not None else 0,
        **_profile_report_from_rows(rows),
    }


class MemoryProfileSpan:
    """Fail-open timing span for content-free Lethyx developer telemetry."""

    def __init__(
        self,
        repo_root: Path | str | None,
        *,
        operation: str,
        component: str,
        workspace_id: str = "",
        profile_id: str = "",
        user_id: str = "",
        chat_id: str = "",
        run_id: str = "",
        stage_id: str = "",
        backend: str = "",
        input_value: object = None,
        input_summary: dict[str, Any] | None = None,
    ) -> None:
        self._repo_root = Path(repo_root) if repo_root is not None else None
        self._operation = operation
        self._component = component
        self._workspace_id = workspace_id
        self._profile_id = profile_id
        self._user_id = user_id
        self._chat_id = chat_id
        self._run_id = run_id
        self._stage_id = stage_id
        self._backend = backend
        self._input_summary = input_summary or memory_profile_summary(input_value)
        self._output_summary: dict[str, Any] = {"type": "unset"}
        self._stages: list[dict[str, Any]] = []
        self._status = "ok"
        self._decision = ""
        self._outcome = ""
        self._reason = ""
        self._started_at = datetime.now(UTC).isoformat()
        self._started_monotonic = time.perf_counter()
        self._span_id = uuid4().hex
        self._parent_span_id = ""
        self._concurrency_key = ""
        self._max_in_flight = 0
        self._token: Token[str | None] | None = None
        self._context_token: Token[tuple[str, str, str, str]] | None = None
        self._started_resources: dict[str, float | int] = {}

    def __enter__(self) -> "MemoryProfileSpan":
        self._parent_span_id = _CURRENT_PROFILE_SPAN.get() or ""
        parent_profile_id, parent_user_id, parent_chat_id, parent_run_id = (
            _CURRENT_PROFILE_CONTEXT.get()
        )
        self._profile_id = self._profile_id or parent_profile_id
        self._user_id = self._user_id or parent_user_id
        self._chat_id = self._chat_id or parent_chat_id
        self._run_id = self._run_id or parent_run_id
        self._concurrency_key = "|".join((str(self._workspace_id), str(self._backend)))
        with _PROFILE_ACTIVE_LOCK:
            active = _PROFILE_ACTIVE.get(self._concurrency_key, 0) + 1
            _PROFILE_ACTIVE[self._concurrency_key] = active
            _PROFILE_PEAK[self._concurrency_key] = max(
                active,
                _PROFILE_PEAK.get(self._concurrency_key, 0),
            )
            self._max_in_flight = _PROFILE_PEAK[self._concurrency_key]
        self._token = _CURRENT_PROFILE_SPAN.set(self._span_id)
        self._context_token = _CURRENT_PROFILE_CONTEXT.set(
            (self._profile_id, self._user_id, self._chat_id, self._run_id)
        )
        self._started_resources = _profile_resource_snapshot()
        return self

    def set_output(self, value: object) -> None:
        self._output_summary = memory_profile_summary(value)

    def add_stage(
        self,
        operation: str,
        *,
        duration_ms: float,
        status: str = "ok",
        details: Mapping[str, object] | None = None,
    ) -> None:
        stage: dict[str, Any] = {
            "operation": str(operation),
            "duration_ms": round(max(0.0, float(duration_ms)), 3),
            "status": str(status),
        }
        if details:
            stage["details"] = memory_profile_summary(details)
        self._stages.append(stage)

    def set_result(
        self,
        *,
        status: str = "ok",
        decision: str = "",
        outcome: str = "",
        reason: str = "",
    ) -> None:
        self._status = str(status or "ok")
        self._decision = str(decision or "")
        self._outcome = str(outcome or "")
        self._reason = str(reason or "")

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> Literal[False]:
        if exc is not None:
            self._status = "failed"
        duration_ms = round(max(0.0, time.perf_counter() - self._started_monotonic) * 1_000, 3)
        with _PROFILE_ACTIVE_LOCK:
            self._max_in_flight = max(
                self._max_in_flight,
                _PROFILE_PEAK.get(self._concurrency_key, 0),
            )
            active = max(0, _PROFILE_ACTIVE.get(self._concurrency_key, 1) - 1)
            if active:
                _PROFILE_ACTIVE[self._concurrency_key] = active
            else:
                _PROFILE_ACTIVE.pop(self._concurrency_key, None)
                _PROFILE_PEAK.pop(self._concurrency_key, None)
        if self._token is not None:
            _CURRENT_PROFILE_SPAN.reset(self._token)
        if self._context_token is not None:
            _CURRENT_PROFILE_CONTEXT.reset(self._context_token)
        error = type(exc).__name__ if exc is not None else ""
        ended_resources = _profile_resource_snapshot()
        resources: dict[str, float | int] = {}
        for key in ("process_cpu_ms", "thread_cpu_ms"):
            resources[key] = round(
                max(
                    0.0,
                    float(ended_resources.get(key, 0)) - float(self._started_resources.get(key, 0)),
                ),
                3,
            )
        if "python_alloc_current_bytes" in ended_resources:
            resources["python_alloc_current_bytes"] = int(
                ended_resources["python_alloc_current_bytes"]
            )
            resources["python_alloc_peak_bytes"] = int(ended_resources["python_alloc_peak_bytes"])
        record_memory_event(
            self._repo_root,
            event_type="memory_profile",
            workspace_id=self._workspace_id,
            profile_id=self._profile_id,
            user_id=self._user_id,
            chat_id=self._chat_id,
            run_id=self._run_id,
            stage_id=self._stage_id,
            backend=self._backend,
            status=self._status,
            error=error,
            operation=self._operation,
            component=self._component,
            decision=self._decision,
            outcome=self._outcome,
            reason=self._reason,
            duration_ms=duration_ms,
            parent_span_id=self._parent_span_id,
            parallelism={
                "concurrency_key": self._concurrency_key,
                "max_in_flight": self._max_in_flight,
            },
            resources=resources,
            data={
                "span_id": self._span_id,
                "input": self._input_summary,
                "output": self._output_summary,
                "stages": self._stages,
            },
        )
        return False


def profile_memory_operation(
    repo_root: Path | str | None,
    *,
    operation: str,
    component: str,
    workspace_id: str = "",
    profile_id: str = "",
    user_id: str = "",
    chat_id: str = "",
    run_id: str = "",
    stage_id: str = "",
    backend: str = "",
    input_value: object = None,
    input_summary: dict[str, Any] | None = None,
) -> MemoryProfileSpan:
    return MemoryProfileSpan(
        repo_root,
        operation=operation,
        component=component,
        workspace_id=workspace_id,
        profile_id=profile_id,
        user_id=user_id,
        chat_id=chat_id,
        run_id=run_id,
        stage_id=stage_id,
        backend=backend,
        input_value=input_value,
        input_summary=input_summary,
    )


def memory_event_report(
    repo_root: Path | str | None,
    *,
    drive_root: Path | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Summarize valid event rows directly from the append-only trace."""

    log_path = memory_event_trace_path(repo_root, drive_root=drive_root)
    if log_path is None:
        rows: list[dict[str, Any]] = []
        invalid_row_count = 0
    else:
        rows, invalid_row_count = _read_memory_event_rows(log_path, limit=max(0, limit))

    by_event_type: dict[str, int] = {}
    by_operation: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for row in rows:
        for field, counts in (
            ("event_type", by_event_type),
            ("operation", by_operation),
            ("decision", by_decision),
            ("outcome", by_outcome),
        ):
            value = row.get(field)
            if isinstance(value, str) and value:
                counts[value] = counts.get(value, 0) + 1
    return {
        "schema_version": _MEMORY_EVENT_SCHEMA_VERSION,
        "event_count": len(rows),
        "invalid_event_count": invalid_row_count,
        "dropped_event_count": _EVENT_DROPPED.get(log_path, 0) if log_path is not None else 0,
        "by_event_type": dict(sorted(by_event_type.items())),
        "by_operation": dict(sorted(by_operation.items())),
        "by_decision": dict(sorted(by_decision.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "profile": _profile_report_from_rows(rows),
    }


def record_memory_event(
    repo_root: Path | None,
    *,
    event_type: str,
    workspace_id: str = "",
    run_id: str = "",
    stage_id: str = "",
    backend: str = "",
    status: str = "",
    error: str = "",
    data: dict[str, Any] | None = None,
    drive_root: Path | None = None,
    turn_index: int | None = None,
    operation: str = "",
    decision: str = "",
    outcome: str = "",
    reason: str = "",
    candidate_count: int | None = None,
    selected_ids: list[str] | None = None,
    profile_id: str = "",
    user_id: str = "",
    chat_id: str = "",
    duration_ms: float | None = None,
    component: str = "",
    parent_event_id: str = "",
    parent_span_id: str = "",
    parallelism: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": _MEMORY_EVENT_SCHEMA_VERSION,
        "event_id": uuid4().hex,
        "recorded_at": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "workspace_id": workspace_id,
        "run_id": run_id,
        "stage_id": stage_id,
        "backend": backend,
        "status": status,
        "error": error,
        "data": data or {},
        "turn_index": turn_index,
        "operation": operation,
        "decision": decision,
        "outcome": outcome,
        "reason": reason,
        "candidate_count": candidate_count,
        "selected_id_hashes": _identifier_hashes(selected_ids),
    }
    optional_fields = {
        "profile_id": profile_id,
        "user_id": user_id,
        "chat_id": chat_id,
        "duration_ms": (
            round(max(0.0, float(duration_ms)), 3) if duration_ms is not None else None
        ),
        "component": component,
        "parent_event_id": parent_event_id,
        "parent_span_id": parent_span_id,
        "parallelism": parallelism,
        "resources": resources,
    }
    payload.update(
        {key: value for key, value in optional_fields.items() if value not in (None, "", {})}
    )
    payload["event_sha256"] = _event_payload_sha256(payload)
    if error or status in {"failed", "blocked", "unavailable"}:
        log.warning("memory_event %s: %s", event_type, error or status)
    else:
        log.debug("memory_event %s status=%s", event_type, status)

    log_path = memory_event_trace_path(repo_root, drive_root=drive_root)
    if log_path is None:
        return
    _append_memory_event(log_path, payload)
