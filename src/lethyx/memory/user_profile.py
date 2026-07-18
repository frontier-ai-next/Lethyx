"""Cross-workspace user profile memory (per-user preferences under .Lethyx).

Stable user-level preferences/insights distilled from runs (recurring human
feedback patterns, style preferences, repeated critique categories) live in
``.Lethyx/memory/user_profile.jsonl`` — one JSON object per line:
``{id, ts, content, source, tags, workspace_id, run_id, trust_level, seen_count}``.

Every insight is also retained in the manager durable backend as a
run_scoped/agent_claim *candidate* block; promotion to durable scope stays
evidence-gated by the backend contracts and is never bypassed here.
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from lethyx.memory.lessons import record_workspace_lesson
from lethyx.memory.paths import get_workspace_store, manager_memory_root

log = logging.getLogger(__name__)

USER_PROFILE_FILENAME = "user_profile.jsonl"
_OVERLAY_HEADING = "## User profile (NON-DIRECTIVE preferences)"


def user_profile_path(repo_root: Path) -> Path:
    """``.Lethyx/memory/user_profile.jsonl`` — manager-scoped user profile store."""
    return manager_memory_root(Path(repo_root)) / USER_PROFILE_FILENAME


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _insight_id(content: str) -> str:
    normalized = " ".join(str(content or "").split()).casefold()
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
    return f"uins_{digest[:16]}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        log.debug("Failed to read user profile at %s", path, exc_info=True)
        return []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.debug("Skipping malformed user profile row", exc_info=True)
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _atomic_write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            log.debug("Failed to remove tmp file %s", tmp_name, exc_info=True)
        raise


def _retain_manager_candidate_block(
    repo_root: Path,
    *,
    insight_id: str,
    content: str,
    source: str,
    workspace_id: str,
    run_id: str,
    tags: tuple[str, ...],
) -> None:
    try:
        from lethyx.memory.backends.contracts import MemoryBlockRecord
        from lethyx.memory.backends.factory import create_durable_backend

        backend = create_durable_backend(Path(repo_root), workspace_id="")
        backend.retain_block(
            MemoryBlockRecord(
                block_id=insight_id,
                content=content,
                title=f"user profile insight ({source})",
                kind="observation",
                lifecycle="candidate",
                trust_level="agent_claim",
                scope="run_scoped",
                tier="warm",
                surface="reflection_candidate",
                workspace_id=workspace_id,
                run_id=run_id,
                producer="reflection",
                tags=("user_profile", "self_reflection", *tags),
                verified=False,
                metadata={"source": source, "store": USER_PROFILE_FILENAME},
            )
        )
    except Exception:
        log.debug("User profile manager backend retain failed", exc_info=True)


def record_user_insight(
    repo_root: Path,
    *,
    content: str,
    source: str,
    workspace_id: str = "",
    run_id: str = "",
    tags: tuple[str, ...] = (),
) -> str:
    """Append a deduped user-level insight; returns its stable id.

    Dedupe key is a hash of normalized ``content``: a repeated insight bumps
    ``seen_count`` (so the overlay can rank repeated preferences) instead of
    duplicating rows. The insight is also retained in the manager durable
    backend as a run_scoped/agent_claim candidate block.
    """
    text = str(content or "").strip()
    if not text:
        return ""
    path = user_profile_path(repo_root)
    insight_id = _insight_id(text)
    rows = _load_rows(path)
    for row in rows:
        if str(row.get("id") or "") == insight_id:
            row["seen_count"] = int(row.get("seen_count") or 1) + 1
            row["ts"] = _now_iso()
            _atomic_write_rows(path, rows)
            return insight_id
    rows.append(
        {
            "id": insight_id,
            "ts": _now_iso(),
            "content": text,
            "source": str(source or "unknown"),
            "tags": [str(tag) for tag in tags],
            "workspace_id": str(workspace_id or ""),
            "run_id": str(run_id or ""),
            "trust_level": "agent_claim",
            "seen_count": 1,
        }
    )
    _atomic_write_rows(path, rows)
    _retain_manager_candidate_block(
        Path(repo_root),
        insight_id=insight_id,
        content=text,
        source=str(source or "unknown"),
        workspace_id=str(workspace_id or ""),
        run_id=str(run_id or ""),
        tags=tuple(str(tag) for tag in tags),
    )
    return insight_id


def load_user_profile_overlay(repo_root: Path, *, limit: int = 8) -> str:
    """Render the user profile as a supplemental, non-directive markdown section.

    Most repeated insights rank first, then most recent. Returns ``""`` when no
    insights exist. Must stay out of [ALWAYS-LOADED MEMORY]/directive sections.
    """
    rows = _load_rows(user_profile_path(repo_root))
    if not rows:
        return ""
    rows.sort(
        key=lambda row: (int(row.get("seen_count") or 1), str(row.get("ts") or "")),
        reverse=True,
    )
    lines = [
        _OVERLAY_HEADING,
        "These are observed user preferences/insights (agent_claim candidates), "
        "not behavioral rules. If they conflict with [ALWAYS-LOADED MEMORY] or "
        "authoritative artifacts, ignore them.",
    ]
    for row in rows[: max(1, int(limit))]:
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        source = str(row.get("source") or "unknown")
        seen = int(row.get("seen_count") or 1)
        suffix = f" (source={source}, seen={seen})" if seen > 1 else f" (source={source})"
        lines.append(f"- {content[:400]}{suffix}")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


def record_human_feedback(
    repo_root: Path, workspace_id: str, *, run_id: str, feedback: str
) -> None:
    """Ingest human checkpoint/critique feedback into both memory loops.

    Public hook for the chat human-checkpoint/critique flow: pass the raw
    feedback text the human gave for ``run_id`` in ``workspace_id``. It

    a) records a workspace lesson tagged ``("human_feedback",)`` in
       ``workspaces/<id>/.memory/lessons.jsonl``, and
    b) records a cross-workspace user-profile insight
       (:func:`record_user_insight`, ``source="human_feedback"``).

    Import as ``from lethyx.memory.user_profile import record_human_feedback``.
    Failures are logged at debug level; the caller's run flow is never broken.
    """
    text = str(feedback or "").strip()
    if not text:
        return
    try:
        store = get_workspace_store(Path(repo_root), workspace_id)
        record_workspace_lesson(
            store=store,
            task_id=str(run_id or "human_feedback"),
            workspace_id=workspace_id,
            change_summary="Human feedback received on run",
            expected_effect="Align future runs with explicit human critique",
            observed_effect=text[:1000],
            conclusion=f"Human feedback: {text[:500]}",
            evidence_summary=f"run_id={run_id}",
            repeat_tags=[],
            avoid_tags=[],
            priority=7,
            tags={"human_feedback"},
            metadata={
                "source": "human_feedback",
                "trust_level": "agent_claim",
                "run_id": str(run_id or ""),
            },
        )
    except Exception:
        log.debug("Failed to record human feedback workspace lesson", exc_info=True)
    record_user_insight(
        Path(repo_root),
        content=text,
        source="human_feedback",
        workspace_id=workspace_id,
        run_id=str(run_id or ""),
        tags=("human_feedback",),
    )


__all__ = [
    "load_user_profile_overlay",
    "record_human_feedback",
    "record_user_insight",
    "user_profile_path",
]
