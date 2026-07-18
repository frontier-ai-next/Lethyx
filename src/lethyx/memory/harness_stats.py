"""Learning side of harness mode: aggregate per-phase candidate selections.

The harness module WRITES ``workspaces/<id>/.memory/drive/state/harness_selection.jsonl``
rows shaped like::

    {"ts": ..., "run_id": ..., "phase_id": ..., "winner_id": ..., "winner_head": ...,
     "winner_mutation": ..., "candidates": [{"id", "head", "mutation_id", "status",
     "score", "completion_failed"}]}

This module only CONSUMES that file: it folds selections into the manager-level
``.Lethyx/memory/harness_stats.json`` (read back by the harness for candidate
weighting; schema is a fixed contract) and records per-workspace lessons so the
selection outcome becomes recallable workspace memory.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from lethyx.memory.lessons import record_workspace_lesson
from lethyx.memory.paths import get_workspace_store, manager_memory_root

log = logging.getLogger(__name__)

HARNESS_STATS_FILENAME = "harness_stats.json"
SELECTION_FILENAME = "harness_selection.jsonl"
PROCESSED_SIDECAR_FILENAME = "harness_selection.processed.json"


def harness_stats_path(repo_root: Path) -> Path:
    """``.Lethyx/memory/harness_stats.json`` — contract file read by the harness."""
    return manager_memory_root(repo_root) / HARNESS_STATS_FILENAME


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            log.debug("Failed to remove tmp file %s", tmp_name, exc_info=True)
        raise


def _load_existing_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"phase_head_wins": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.debug("Corrupt harness stats at %s; starting fresh", path, exc_info=True)
        return {"phase_head_wins": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("phase_head_wins"), dict):
        log.debug("Unexpected harness stats shape at %s; starting fresh", path)
        return {"phase_head_wins": {}}
    return payload


def _stats_key(phase_id: str, head: str, mutation_id: str) -> str:
    return f"{phase_id}|{head}|{mutation_id}"


def _winner_key_for_row(row: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    phase_id = str(row.get("phase_id") or "")
    head = str(row.get("winner_head") or "")
    mutation = str(row.get("winner_mutation") or "")
    if not head:
        winner_id = str(row.get("winner_id") or "")
        for candidate in candidates:
            if str(candidate.get("id") or "") == winner_id:
                head = str(candidate.get("head") or "")
                mutation = mutation or str(candidate.get("mutation_id") or "")
                break
    if not head:
        return ""
    return _stats_key(phase_id, head, mutation)


def update_harness_stats(repo_root: Path, selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ``selection_rows`` into ``.Lethyx/memory/harness_stats.json``.

    Schema (fixed contract; the harness reads this file):
    ``{"phase_head_wins": {"<phase_id>|<head>|<mutation_id>": {"wins": int, "candidates": int}}}``.
    The write is atomic (tmp file + ``os.replace``) and a corrupt existing file
    is tolerated by starting from empty stats.
    """
    path = harness_stats_path(Path(repo_root))
    stats = _load_existing_stats(path)
    wins: dict[str, Any] = stats.setdefault("phase_head_wins", {})

    def _entry(key: str) -> dict[str, int]:
        entry = wins.get(key)
        if not isinstance(entry, dict):
            entry = {"wins": 0, "candidates": 0}
            wins[key] = entry
        entry.setdefault("wins", 0)
        entry.setdefault("candidates", 0)
        return entry

    for row in selection_rows:
        if not isinstance(row, dict):
            continue
        phase_id = str(row.get("phase_id") or "")
        candidates = [c for c in (row.get("candidates") or []) if isinstance(c, dict)]
        for candidate in candidates:
            head = str(candidate.get("head") or "")
            if not head:
                continue
            key = _stats_key(phase_id, head, str(candidate.get("mutation_id") or ""))
            entry = _entry(key)
            entry["candidates"] = int(entry["candidates"]) + 1
        winner_key = _winner_key_for_row(row, candidates)
        if winner_key:
            entry = _entry(winner_key)
            entry["wins"] = int(entry["wins"]) + 1

    _atomic_write_json(path, stats)
    return stats


def _row_id(raw_line: str) -> str:
    return hashlib.sha256(raw_line.strip().encode("utf-8", errors="replace")).hexdigest()


def _load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.debug("Corrupt processed sidecar at %s; reprocessing", path, exc_info=True)
        return set()
    if isinstance(payload, dict) and isinstance(payload.get("processed"), list):
        return {str(item) for item in payload["processed"]}
    return set()


def _record_selection_lesson(
    *,
    repo_root: Path,
    workspace_id: str,
    row: dict[str, Any],
) -> None:
    candidates = [c for c in (row.get("candidates") or []) if isinstance(c, dict)]
    phase_id = str(row.get("phase_id") or "")
    run_id = str(row.get("run_id") or "")
    winner_id = str(row.get("winner_id") or "")
    head = str(row.get("winner_head") or "")
    mutation = str(row.get("winner_mutation") or "")
    score: Any = None
    for candidate in candidates:
        if str(candidate.get("id") or "") == winner_id:
            head = head or str(candidate.get("head") or "")
            mutation = mutation or str(candidate.get("mutation_id") or "")
            score = candidate.get("score")
            break
    score_text = f" (score {score})" if score is not None else ""
    summary = (
        f"harness: head={head or 'unknown'} mutation={mutation or 'none'} "
        f"won phase {phase_id or 'unknown'}{score_text}"
    )
    store = get_workspace_store(Path(repo_root), workspace_id)
    record_workspace_lesson(
        store=store,
        task_id=f"{run_id}:{phase_id}" if run_id else phase_id,
        workspace_id=workspace_id,
        change_summary=summary,
        expected_effect="Prefer winning head/mutation combos for similar phases",
        observed_effect=(f"{len(candidates)} candidates ran; winner={winner_id or 'unknown'}"),
        conclusion=summary,
        evidence_summary=json.dumps(
            {"run_id": run_id, "phase_id": phase_id, "winner_id": winner_id},
            ensure_ascii=False,
            sort_keys=True,
        ),
        repeat_tags=[tag for tag in (head, mutation) if tag],
        avoid_tags=[],
        priority=5,
        tags={"harness", "self_improvement"},
        metadata={
            "source": "harness_selection",
            "trust_level": "agent_claim",
            "run_id": run_id,
            "phase_id": phase_id,
        },
    )


def ingest_run_harness_selections(repo_root: Path, workspace_id: str, drive_root: Path) -> int:
    """Fold new harness selections into stats + workspace lessons.

    Reads ``<drive_root>/state/harness_selection.jsonl``, updates
    ``.Lethyx/memory/harness_stats.json`` and records one workspace lesson per
    newly seen selection row. Processed row ids (sha256 of the raw line) are
    tracked in ``harness_selection.processed.json`` so re-ingest is idempotent.
    Returns the number of newly processed rows.
    """
    state_dir = Path(drive_root) / "state"
    selection_path = state_dir / SELECTION_FILENAME
    if not selection_path.exists():
        return 0
    sidecar_path = state_dir / PROCESSED_SIDECAR_FILENAME
    processed = _load_processed_ids(sidecar_path)

    new_rows: list[dict[str, Any]] = []
    new_ids: list[str] = []
    try:
        raw_lines = selection_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        log.debug("Failed to read %s", selection_path, exc_info=True)
        return 0
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        row_id = _row_id(line)
        if row_id in processed:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.debug("Skipping malformed harness selection row", exc_info=True)
            processed.add(row_id)
            new_ids.append(row_id)
            continue
        if not isinstance(row, dict):
            processed.add(row_id)
            new_ids.append(row_id)
            continue
        new_rows.append(row)
        processed.add(row_id)
        new_ids.append(row_id)

    if new_rows:
        update_harness_stats(Path(repo_root), new_rows)
        for row in new_rows:
            try:
                _record_selection_lesson(
                    repo_root=Path(repo_root), workspace_id=workspace_id, row=row
                )
            except Exception:
                log.debug("Failed to record harness selection lesson", exc_info=True)

    if new_ids:
        _atomic_write_json(sidecar_path, {"processed": sorted(processed)})
    return len(new_rows)


__all__ = [
    "harness_stats_path",
    "ingest_run_harness_selections",
    "update_harness_stats",
]
