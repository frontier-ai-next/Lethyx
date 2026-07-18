"""Small append-only evidence ledger for memory promotion tests and hooks."""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    event_hash: str
    path: Path


def append_supervisor_ledger_event(
    *,
    repo_root: Path,
    workspace_id: str,
    actor: str,
    phase: str,
    tool: str,
    result: dict[str, Any],
) -> LedgerEvent:
    """Append a typed verification event and return its evidence reference."""

    ledger_path = _ledger_path(repo_root, workspace_id)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    event_id = f"ledger-{uuid.uuid4().hex[:12]}"
    payload = {
        "event_id": event_id,
        "ts": time.time(),
        "workspace_id": workspace_id,
        "actor": actor,
        "phase": phase,
        "tool": tool,
        "result": result,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["event_hash"] = event_hash
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return LedgerEvent(event_id=event_id, event_hash=event_hash, path=ledger_path)


def read_supervisor_ledger_events(*, repo_root: Path, workspace_id: str) -> list[dict[str, Any]]:
    """Read supervisor ledger rows for a workspace."""

    path = _ledger_path(repo_root, workspace_id)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def find_supervisor_ledger_event(
    *,
    repo_root: Path,
    workspace_id: str,
    event_id: str,
) -> dict[str, Any] | None:
    """Find a ledger event by id or hash."""

    needle = str(event_id or "")
    for row in read_supervisor_ledger_events(repo_root=repo_root, workspace_id=workspace_id):
        if needle in {str(row.get("event_id") or ""), str(row.get("event_hash") or "")}:
            return row
    return None


def _ledger_path(repo_root: Path, workspace_id: str) -> Path:
    return (
        Path(repo_root)
        / "workspaces"
        / str(workspace_id or "default")
        / ".memory"
        / "drive"
        / "logs"
        / "supervisor_ledger.jsonl"
    )
