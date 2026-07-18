"""Memory subsystem telemetry — failures must be visible, not silent."""

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def record_memory_event(
    repo_root: Path | None,
    *,
    event_type: str,
    workspace_id: str = "",
    run_id: str = "",
    phase_id: str = "",
    backend: str = "",
    status: str = "",
    error: str = "",
    data: dict[str, Any] | None = None,
    drive_root: Path | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event_type": event_type,
        "workspace_id": workspace_id,
        "run_id": run_id,
        "phase_id": phase_id,
        "backend": backend,
        "status": status,
        "error": error,
        "data": data or {},
    }
    if error or status in {"failed", "blocked", "unavailable"}:
        log.warning("memory_event %s: %s", event_type, error or status)
    else:
        log.debug("memory_event %s status=%s", event_type, status)

    try:
        from lethyx.telemetry.events import EventType, TelemetryEvent
        from lethyx.telemetry.store import emit_event as emit_telemetry_event

        LETHYX_type = (
            EventType.ERROR_OCCURRED
            if status in {"failed", "blocked", "unavailable"} or error
            else EventType.MEMORY_WRITE
        )
        emit_telemetry_event(
            TelemetryEvent(
                event_type=LETHYX_type,
                workspace_id=workspace_id,
                run_id=run_id,
                data=payload,
                source="memory_kernel",
                level="warning" if error else "info",
            )
        )
    except Exception:
        log.debug("Lethyx telemetry emit skipped", exc_info=True)

    if drive_root is not None:
        log_path = drive_root / "logs" / "memory_events.jsonl"
    elif repo_root is not None:
        log_path = repo_root / ".Lethyx" / "memory" / "telemetry" / "memory_events.jsonl"
    else:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        log.debug("Failed to append memory_events.jsonl", exc_info=True)
