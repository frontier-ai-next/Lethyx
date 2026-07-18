"""MemPalace helpers and write-event recording for the loop.

Why this module exists
----------------------
Lethyx ships a real semantic-memory system (MemPalace, ChromaDB-backed,
wing/hall/room/drawer hierarchy) plus a structured-lessons store with
decay scoring. In the JKX run we shipped that material as *tools* the
agent could call (``get_LETHYX_memory``, ``save_LETHYX_memory``,
``record_workspace_event``, ``save_LETHYX_lesson``), and the agent
called them ~1 time across 200 rounds.

That isn't a tool problem; it's a wiring problem. Memory has to be
*pushed*, not pulled — at least at the boundaries where it pays off:

1. **Task start** — before the first LLM round, recall recent + relevant
   memory for the workspace and inject it as a ``[MEMORY_RECALL]``
   system message. Makes the agent aware that the palace isn't empty.
2. **Periodic refresh** — every ``RECALL_INTERVAL`` rounds, do a fresh
   recall scoped to recent activity. This counters context drift in
   long runs (the agent has forgotten what it learned 60 rounds ago).
3. **Auto-record writes** — every successful write-style tool call is
   retained by the selected durable backend, so the next periodic recall
   can find it. The agent does not pay an LLM round for the write; the
   loop handles it.

Adaptive scope
--------------
Per the user's preference, recall scope grows with round count. Early
rounds get a *minimal* bundle (recent palace drawers + a small targeted
search) to keep the prompt light. Past round 50 the bundle becomes
*rich*, also including structured lessons, competency gaps, and the most
recent verify-run records. Rationale: by round 50 the context is already
large; the marginal cost of a richer recall is low and the marginal
value (relinking forgotten earlier learnings) is high.

Layering
--------
This module imports from ``lethyx.*`` lazily and wraps every import
in ``try/except`` so a standalone agent loop (without Lethyx installed)
doesn't crash. The same pattern is used in ``LETHYX_tools.py``.
"""

import logging
import os
import re
import shlex
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from lethyx.contracts import EvidenceRef
from lethyx.limits import (
    RECALL_BLOCK_MAX_CHARS,
    RECALL_DRAWER_PREVIEW_CHARS,
    RECALL_LESSON_PREVIEW_CHARS,
    TASK_BRIEF_MAX_CHARS,
)
from lethyx.memory.backends.contracts import MemoryBlockRecord

log = logging.getLogger(__name__)


def _int_env(neutral_name: str, default: int) -> int:
    raw = os.environ.get(neutral_name)
    try:
        return int(raw if raw is not None else default)
    except (TypeError, ValueError):
        return int(default)


# How often to inject a fresh recall block. Configurable via env var so
# we can dial it without code changes during runs.
RECALL_INTERVAL: int = max(1, _int_env("LETHYX_RECALL_INTERVAL", 30))

# Round count past which recall switches from minimal to rich bundle.
# Picked for long agent runs with a ratio of ~25% before the switch
# (plenty of warm-up, but the agent has
# real activity to summarize before we pay for the richer bundle).
RICH_RECALL_THRESHOLD: int = _int_env("LETHYX_RICH_RECALL_AT", 50)

# Recall result limits. Conservative defaults so the system message stays
# readable; override via env if a workspace has unusually rich history.
RECENT_LIMIT: int = _int_env("LETHYX_RECALL_RECENT", 10)
SEARCH_LIMIT: int = _int_env("LETHYX_RECALL_SEARCH", 5)
LESSONS_LIMIT: int = _int_env("LETHYX_RECALL_LESSONS", 5)

DRAWER_PREVIEW_CHARS = RECALL_DRAWER_PREVIEW_CHARS
LESSON_PREVIEW_CHARS = RECALL_LESSON_PREVIEW_CHARS


def _safe_palace(repo_root: Path, workspace_id: str = "") -> Any | None:
    """Return the selected memory backend for a manager/workspace recall path."""
    try:
        from lethyx.memory.backends.factory import create_durable_backend

        return create_durable_backend(Path(repo_root), workspace_id=workspace_id)
    except Exception as exc:
        log.warning(
            "Selected memory backend unavailable for recall (workspace_id=%r): %s",
            workspace_id or "<manager>",
            exc,
        )
        return None


def _safe_store(repo_root: Path, workspace_id: str = "") -> Any | None:
    """Return Lethyx's ``MemoryStore`` (manager or workspace shard)."""
    try:
        from lethyx.memory.paths import get_workspace_store
    except Exception:
        return None
    try:
        return get_workspace_store(Path(repo_root), workspace_id)
    except Exception as exc:
        log.warning("MemoryStore unavailable for recall: %s", exc)
        return None


def _new_block_id(prefix: str, workspace_id: str) -> str:
    ws = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(workspace_id or "root")).strip("_") or "root"
    return f"{prefix}-{ws}-{uuid.uuid4().hex[:12]}"


def _block_to_recall_hit(block: Any) -> dict[str, Any]:
    """Render a backend-neutral block in the legacy recall-hit shape.

    The recall prompt renderer is intentionally simple and still consumes the
    old ``hall/room/content`` dict shape. This adapter preserves each selected
    backend's own retain/recall implementation while normalizing only the UI
    shape needed by the prompt text.
    """
    if isinstance(block, dict):
        return dict(block)
    metadata = getattr(block, "metadata", None)
    meta = metadata if isinstance(metadata, dict) else {}
    tags = list(getattr(block, "tags", ()) or ())
    return {
        "id": getattr(block, "block_id", ""),
        "block_id": getattr(block, "block_id", ""),
        "title": getattr(block, "title", ""),
        "content": getattr(block, "content", ""),
        "hall": meta.get("hall") or getattr(block, "source_backend", "") or "memory",
        "room": meta.get("room") or getattr(block, "kind", "") or "memory",
        "kind": getattr(block, "kind", ""),
        "scope": getattr(block, "scope", ""),
        "surface": getattr(block, "surface", ""),
        "run_id": getattr(block, "run_id", ""),
        "task_id": meta.get("task_id") or "",
        "tags": tags,
        "verified": bool(getattr(block, "verified", False)),
        "metadata": {
            **meta,
            "block_id": getattr(block, "block_id", ""),
            "source_backend": getattr(block, "source_backend", ""),
            "lifecycle": getattr(block, "lifecycle", ""),
            "trust_level": getattr(block, "trust_level", ""),
            "surface": getattr(block, "surface", ""),
            "workspace_id": getattr(block, "workspace_id", ""),
            "run_id": getattr(block, "run_id", ""),
            "tags": tags,
        },
    }


def _selected_backend_recall(
    backend: Any,
    *,
    workspace_id: str,
    limit: int,
    query: str = "",
    room: str = "",
    current_run_id: str = "",
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {"workspace_id": workspace_id, "limit": limit}
    if query:
        payload["query"] = query
    if room:
        payload["room"] = room
    if current_run_id:
        payload["current_run_id"] = current_run_id
    rows = backend.recall(payload) or []
    hits = [_block_to_recall_hit(row) for row in rows]
    if room:
        room_l = room.strip().lower()
        hits = [hit for hit in hits if _memory_hit_room(hit) == room_l]
    return hits[:limit]


def _retain_selected_block(backend: Any, record: MemoryBlockRecord) -> dict[str, Any]:
    return backend.retain_block(record)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_truthy_any(*names: str) -> bool:
    return any(_env_truthy(name) for name in names)


def _context_overlays(tools_ctx: Any | None) -> dict[str, Any]:
    if tools_ctx is None:
        return {}
    overlays = getattr(tools_ctx, "context_overlays", None)
    return overlays if isinstance(overlays, dict) else {}


def _LETHYX_overlay_already_compiled(tools_ctx: Any | None) -> bool:
    overlays = _context_overlays(tools_ctx)
    contract = overlays.get("memory_injection_contract")
    if isinstance(contract, dict):
        if contract.get("mode") == "LETHYX_owned" and contract.get("proactive_overlay_injected"):
            return True
    return bool(
        overlays.get("prevent_host_auto_core_overlay")
        or overlays.get("proactive_memory_rendered_in_task_input")
        or overlays.get("memory_overlay_origin")
        or overlays.get("proactive_memory")
    )


def _is_LETHYX_managed(tools_ctx: Any) -> bool:
    if getattr(tools_ctx, "LETHYX_managed", False):
        return True
    overlays = _context_overlays(tools_ctx)
    if overlays.get("LETHYX_managed") or overlays.get("phase_manifest"):
        return True
    if overlays.get("proactive_memory") or overlays.get("llm_input_bundle"):
        return True
    phase = str(getattr(tools_ctx, "LETHYX_phase_id", "") or "").strip()
    if phase:
        return True
    manifest = getattr(tools_ctx, "phase_manifest", None)
    return manifest is not None


_WORKSPACE_SKIP_NAMES = {"", "auto", "registry", "_template"}
_WORKSPACE_LABELS = frozenset({"workspace", "workspace path"})
_WORKSPACE_ID_LABELS = frozenset({"workspace_id"})


def _workspace_candidate(value: Any) -> str:
    raw = str(value or "").strip().strip("`\"'.,:;)]}")
    if not raw:
        return ""
    try:
        from lethyx.memory.paths import normalize_workspace_id

        candidate = normalize_workspace_id(raw)
    except Exception:
        parts = PurePosixPath(raw.replace("\\", "/").strip("/")).parts
        if parts and parts[0].casefold() == "workspaces":
            parts = parts[1:]
        candidate = parts[0] if parts else ""
    candidate = candidate.strip().strip("`\"'.,:;)]}")
    if (
        not candidate
        or candidate.casefold() in _WORKSPACE_SKIP_NAMES
        or Path(candidate).suffix
        or ":" in candidate
    ):
        return ""
    return candidate


def _workspace_after_workspaces(parts: tuple[str, ...]) -> str:
    folded = [part.casefold() for part in parts]
    try:
        idx = folded.index("workspaces")
    except ValueError:
        return ""
    if idx + 1 >= len(parts):
        return ""
    return _workspace_candidate(parts[idx + 1])


def _workspace_from_drive_root(tools_ctx: Any | None) -> str:
    try:
        drive_root = Path(str(getattr(tools_ctx, "drive_root", "") or "")).resolve()
    except Exception:
        return ""
    return _workspace_after_workspaces(drive_root.parts)


def _workspace_from_context(tools_ctx: Any | None) -> str:
    overlays = _context_overlays(tools_ctx)
    for source in (
        overlays.get("memory_injection_contract"),
        overlays.get("llm_input_bundle"),
        overlays,
    ):
        if isinstance(source, dict):
            candidate = _workspace_candidate(source.get("workspace_id"))
            if candidate:
                return candidate
    for attr in ("workspace_id", "LETHYX_workspace_id"):
        candidate = _workspace_candidate(getattr(tools_ctx, attr, ""))
        if candidate:
            return candidate
    drive_candidate = _workspace_from_drive_root(tools_ctx)
    if drive_candidate:
        return drive_candidate
    return _workspace_candidate(getattr(tools_ctx, "active_workspace_id", ""))


def _auto_recall_enabled(kind: str, *, tools_ctx: Any | None = None) -> bool:
    """Return whether the loop should inject recall without an explicit tool call.

    Activation policy:
    - Lethyx-managed runs: task_start core overlay ON by default.
    - Standalone host agents: task_start OFF unless ``LETHYX_TASK_START_RECALL=1``.
    - Periodic recall still requires explicit env opt-in.
    """
    if _env_truthy_any("LETHYX_AUTO_MEMORY_RECALL"):
        return True
    if kind == "task_start":
        if tools_ctx is not None and _LETHYX_overlay_already_compiled(tools_ctx):
            return False
        if tools_ctx is not None and _is_LETHYX_managed(tools_ctx):
            return True
        raw = str(os.environ.get("LETHYX_TASK_START_RECALL") or "").strip().lower()
        return raw in {"1", "true", "yes", "on"}
    if kind == "periodic":
        return _env_truthy_any(
            "LETHYX_PERIODIC_RECALL",
            "LETHYX_ENABLE_PERIODIC_RECALL",
        )
    return False


def _format_drawer(hit: dict[str, Any]) -> str:
    room = hit.get("room", "?") or "?"
    hall = hit.get("hall", "?") or "?"
    content = (hit.get("content") or "").strip().replace("\n", " ")
    if len(content) > DRAWER_PREVIEW_CHARS:
        content = content[:DRAWER_PREVIEW_CHARS].rstrip() + "…"
    distance = hit.get("distance")
    if isinstance(distance, (int, float)):
        return f"  - [{hall}/{room}] (d={distance:.2f}) {content}"
    return f"  - [{hall}/{room}] {content}"


def _format_lesson(lesson: Any) -> str:
    summary = (getattr(lesson, "change_summary", "") or "").strip()
    expected = (getattr(lesson, "expected_effect", "") or "").strip()
    observed = (getattr(lesson, "observed_effect", "") or "").strip()
    body = f"{summary} | expected={expected} | observed={observed}"
    if len(body) > LESSON_PREVIEW_CHARS:
        body = body[:LESSON_PREVIEW_CHARS].rstrip() + "…"
    return f"  - {body}"


_PHASE_TAGS: dict[str, set[str]] = {
    "planner": {"design", "architecture", "discovery", "prior_art"},
    "subtask": {"implementation", "skill", "gmas", "code_pattern", "subtask"},
    "implement": {"implementation", "skill", "gmas", "code_pattern", "subtask"},
    "review": {"review", "defect_pattern"},
    "remediation": {"verification_failure", "lesson", "bug_fix", "verify", "fail"},
}


def _phase_tags(phase: str | None) -> set[str]:
    normalized = str(phase or "").lower()
    tags: set[str] = set()
    for key, values in _PHASE_TAGS.items():
        if key in normalized:
            tags.update(values)
    return tags


def _hit_tags(hit: Any) -> set[str]:
    if isinstance(hit, dict):
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        raw = hit.get("tags") or metadata.get("tags")
        room = str(hit.get("room") or "")
        hall = str(hit.get("hall") or "")
        text = f"{room} {hall} {hit.get('content') or ''}".lower()
    else:
        raw = getattr(hit, "tags", None)
        text = " ".join(
            str(getattr(hit, attr, "") or "")
            for attr in (
                "change_summary",
                "expected_effect",
                "observed_effect",
                "context",
            )
        ).lower()
    tags: set[str] = set()
    if isinstance(raw, str):
        tags.update(
            part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()
        )
    elif isinstance(raw, (list, tuple, set)):
        tags.update(str(part).strip().lower() for part in raw if str(part).strip())
    for marker in (
        "gmas",
        "verification_failure",
        "bug_fix",
        "implementation",
        "cleanup",
        "hygiene",
        "review",
    ):
        if marker in text:
            tags.add(marker)
    return tags


def _phase_rerank_hits(items: list[Any], phase: str | None) -> list[Any]:
    preferred = _phase_tags(phase)
    if not preferred or not items:
        return items
    scored = [(bool(_hit_tags(item) & preferred), index, item) for index, item in enumerate(items)]
    tagged = [item for matched, _index, item in scored if matched]
    if len(tagged) >= 3:
        return tagged
    return [item for _matched, _index, item in sorted(scored, key=lambda row: (not row[0], row[1]))]


def _build_recall_text(
    *,
    workspace_id: str,
    title: str,
    workspace_recent: list[dict[str, Any]],
    workspace_semantic: list[dict[str, Any]],
    manager_recent: list[dict[str, Any]],
    manager_semantic: list[dict[str, Any]],
    lessons: list[Any] | None = None,
    gaps: list[Any] | None = None,
    verify_runs: list[dict[str, Any]] | None = None,
    phase: str | None = None,
) -> str:
    """Render the recall blob with explicit workspace vs manager blocks."""
    workspace_recent = _phase_rerank_hits(workspace_recent, phase)
    workspace_semantic = _phase_rerank_hits(workspace_semantic, phase)
    manager_recent = _phase_rerank_hits(manager_recent, phase)
    manager_semantic = _phase_rerank_hits(manager_semantic, phase)
    lessons = _phase_rerank_hits(list(lessons or []), phase)
    parts: list[str] = [f"[MEMORY_RECALL] {title} (workspace={workspace_id or '<root>'})"]

    ws_lines: list[str] = []
    if workspace_recent:
        ws_lines.append(f"Recent ({len(workspace_recent)}):")
        ws_lines.extend(_format_drawer(h) for h in workspace_recent)
    if workspace_semantic:
        ws_lines.append(f"Semantic matches ({len(workspace_semantic)}):")
        ws_lines.extend(_format_drawer(h) for h in workspace_semantic)
    if ws_lines:
        parts.append("[WORKSPACE MEMORY]")
        parts.extend(ws_lines)

    mgr_lines: list[str] = []
    if manager_recent:
        mgr_lines.append(f"Recent ({len(manager_recent)}):")
        mgr_lines.extend(_format_drawer(h) for h in manager_recent)
    if manager_semantic:
        mgr_lines.append(f"Semantic matches ({len(manager_semantic)}):")
        mgr_lines.extend(_format_drawer(h) for h in manager_semantic)
    if mgr_lines:
        parts.append("[MANAGER MEMORY]")
        parts.extend(mgr_lines)

    if verify_runs:
        parts.append(f"Recent verify runs ({len(verify_runs)}):")
        parts.extend(_format_drawer(h) for h in verify_runs)
    if lessons:
        parts.append(f"Structured lessons ({len(lessons)}):")
        parts.extend(_format_lesson(lesson) for lesson in lessons)
    if gaps:
        parts.append(f"Open competency gaps ({len(gaps)}):")
        for gap in gaps:
            desc = (getattr(gap, "description", "") or "").strip()
            sev = getattr(gap, "severity", "?")
            area = getattr(gap, "capability_area", "?")
            parts.append(f"  - [{sev}/{area}] {desc}")

    if len(parts) == 1:
        return ""

    parts.append(
        "Use `get_LETHYX_memory(query=..., workspace_id=...)` to dig deeper "
        "before duplicating work or re-deriving findings."
    )
    return "\n".join(parts)


def _lexical_fallback_recall(
    *,
    repo_root: Path,
    workspace_id: str,
    task_input: str,
) -> str:
    """When Chroma is unavailable, surface ideas.jsonl matches (lexical keyword scan)."""
    import json as _json

    try:
        from lethyx.memory.paths import workspace_memory_root
    except Exception:
        return ""
    try:
        ws_root = workspace_memory_root(Path(repo_root), workspace_id)
        ideas_path = ws_root / "ideas.jsonl"
        if not ideas_path.exists():
            return ""
        query_lower = task_input[:1200].lower()
        rows = []
        with ideas_path.open(encoding="utf-8", errors="replace") as _fh:
            for line in _fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                content = str(rec.get("content") or rec.get("body") or "")
                title = str(rec.get("title") or "")
                if query_lower and not (
                    query_lower in content.lower() or query_lower in title.lower()
                ):
                    continue
                rows.append(rec)
                if len(rows) >= min(RECENT_LIMIT, 8):
                    break
    except Exception as exc:
        log.debug("Lexical memory fallback failed: %s", exc)
        return ""
    if not rows:
        return ""
    lines = [
        f"[MEMORY_RECALL] Task-start recall (lexical fallback; workspace={workspace_id})",
        "[WORKSPACE MEMORY]",
        "Ideas (from ideas.jsonl):",
    ]
    for r in rows:
        snippet = str(r.get("content") or r.get("body") or "").strip().replace("\n", " ")
        if len(snippet) > DRAWER_PREVIEW_CHARS:
            snippet = snippet[:DRAWER_PREVIEW_CHARS].rstrip() + "…"
        path = r.get("palace_path") or r.get("kind") or "idea"
        lines.append(f"  - [{path}] {r.get('title', '')}: {snippet}")
    lines.append("Use `get_LETHYX_memory` / `list_memory_tree` once MemPalace is available.")
    return "\n".join(lines)


def recall_for_task_start(
    *,
    workspace_id: str,
    task_input: str,
    repo_root: Path,
    phase: str | None = None,
) -> str:
    """Build the initial recall block for the very first LLM round.

    Always uses the *minimal* bundle (recent + small semantic search).
    The richer bundle is reserved for periodic recall after the agent
    has produced its own activity to relate against.
    Returns ``""`` if there's nothing in memory to recall — caller MUST
    treat empty string as "do not inject".
    """
    log.debug(
        "recall_for_task_start workspace_id=%r repo_root=%s",
        workspace_id,
        repo_root,
    )
    ws_palace = _safe_palace(repo_root, workspace_id) if workspace_id else None
    mgr_palace = _safe_palace(repo_root, "")

    if not workspace_id:
        return ""

    if ws_palace is None and mgr_palace is None:
        return _lexical_fallback_recall(
            repo_root=repo_root, workspace_id=workspace_id, task_input=task_input
        )

    recent_ws: list[dict[str, Any]] = []
    semantic_ws: list[dict[str, Any]] = []
    if ws_palace is not None:
        try:
            recent_ws = _selected_backend_recall(
                ws_palace,
                workspace_id=workspace_id,
                limit=RECENT_LIMIT,
            )
        except Exception as exc:
            log.warning("Selected backend recall failed at task start (workspace): %s", exc)
        if task_input.strip():
            try:
                semantic_ws = _selected_backend_recall(
                    ws_palace,
                    workspace_id=workspace_id,
                    query=task_input[:1000],
                    limit=SEARCH_LIMIT,
                )
            except Exception as exc:
                log.warning(
                    "Selected backend semantic recall failed at task start (workspace): %s", exc
                )

    recent_mgr: list[dict[str, Any]] = []
    semantic_mgr: list[dict[str, Any]] = []
    if mgr_palace is not None:
        try:
            recent_mgr = _selected_backend_recall(
                mgr_palace,
                workspace_id="",
                limit=min(5, RECENT_LIMIT),
            )
        except Exception as exc:
            log.debug("Selected backend recall failed at task start (manager): %s", exc)
        if task_input.strip():
            try:
                semantic_mgr = _selected_backend_recall(
                    mgr_palace,
                    workspace_id="",
                    query=task_input[:1000],
                    limit=min(3, SEARCH_LIMIT),
                )
            except Exception as exc:
                log.debug(
                    "Selected backend semantic recall failed at task start (manager): %s", exc
                )

    recall = _build_recall_text(
        workspace_id=workspace_id,
        title="Task-start recall",
        workspace_recent=recent_ws,
        workspace_semantic=semantic_ws,
        manager_recent=recent_mgr,
        manager_semantic=semantic_mgr,
        phase=phase,
    )
    if not recall and ws_palace is None:
        return _lexical_fallback_recall(
            repo_root=repo_root, workspace_id=workspace_id, task_input=task_input
        )
    return recall


_RUN_SCOPED_RECALL_ROOMS = {"changes", "errors"}


def _run_id_from_task_id(task_id: str) -> str:
    value = str(task_id or "").strip()
    return value.split(":", 1)[0] if ":" in value else value


def _memory_hit_metadata(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("metadata")
    return meta if isinstance(meta, dict) else hit


def _memory_hit_room(hit: dict[str, Any]) -> str:
    meta = _memory_hit_metadata(hit)
    return str(meta.get("room") or meta.get("event_type") or "").strip().lower()


def _memory_hit_run_id(hit: dict[str, Any]) -> str:
    meta = _memory_hit_metadata(hit)
    run_id = str(meta.get("run_id") or "").strip()
    if run_id:
        return run_id
    task_id = str(meta.get("task_id") or "").strip()
    return _run_id_from_task_id(task_id)


def _filter_run_scoped_recall_hits(
    hits: list[dict[str, Any]], *, task_id: str = ""
) -> list[dict[str, Any]]:
    current_run_id = _run_id_from_task_id(task_id)
    if not current_run_id:
        return hits
    filtered: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        if _memory_hit_room(hit) in _RUN_SCOPED_RECALL_ROOMS:
            hit_run_id = _memory_hit_run_id(hit)
            if not hit_run_id or hit_run_id != current_run_id:
                continue
        filtered.append(hit)
    return filtered


def recall_periodic(
    *,
    workspace_id: str,
    round_idx: int,
    recent_actions_summary: str,
    repo_root: Path,
    phase: str | None = None,
    task_id: str = "",
) -> str:
    """Build a fresh recall block for the periodic mid-loop injection.

    Adaptivity: rounds <= ``RICH_RECALL_THRESHOLD`` get the minimal
    bundle (recent + targeted search). Past the threshold we also pull
    structured lessons, competency gaps, and recent ``verify_runs``
    drawers. The intuition: by then the agent has accumulated enough
    activity that a richer recall is worth its prompt cost, and it's
    started to forget early-run insights.

    Returns ``""`` if no data — caller MUST skip injection in that case.
    """
    log.debug(
        "recall_periodic round=%s workspace_id=%r repo_root=%s",
        round_idx,
        workspace_id,
        repo_root,
    )
    if not workspace_id:
        return ""

    ws_palace = _safe_palace(repo_root, workspace_id)
    mgr_palace = _safe_palace(repo_root, "")
    if ws_palace is None and mgr_palace is None:
        return ""

    recent_ws: list[dict[str, Any]] = []
    semantic_ws: list[dict[str, Any]] = []
    if ws_palace is not None:
        try:
            recent_ws = _selected_backend_recall(
                ws_palace,
                workspace_id=workspace_id,
                limit=RECENT_LIMIT,
                current_run_id=_run_id_from_task_id(task_id),
            )
            recent_ws = _filter_run_scoped_recall_hits(recent_ws, task_id=task_id)
        except Exception as exc:
            log.warning(
                "Selected backend recall failed at round %d (workspace): %s", round_idx, exc
            )
        query = (recent_actions_summary or "").strip()
        if query:
            try:
                semantic_ws = _selected_backend_recall(
                    ws_palace,
                    workspace_id=workspace_id,
                    query=query[:1000],
                    limit=SEARCH_LIMIT,
                    current_run_id=_run_id_from_task_id(task_id),
                )
                semantic_ws = _filter_run_scoped_recall_hits(
                    semantic_ws,
                    task_id=task_id,
                )
            except Exception as exc:
                log.warning(
                    "Selected backend semantic recall failed at round %d (workspace): %s",
                    round_idx,
                    exc,
                )

    recent_mgr: list[dict[str, Any]] = []
    semantic_mgr: list[dict[str, Any]] = []
    query = (recent_actions_summary or "").strip()
    if mgr_palace is not None:
        try:
            recent_mgr = _selected_backend_recall(
                mgr_palace,
                workspace_id="",
                limit=min(5, RECENT_LIMIT),
            )
        except Exception as exc:
            log.debug("Selected backend recall failed at round %d (manager): %s", round_idx, exc)
        if query:
            try:
                semantic_mgr = _selected_backend_recall(
                    mgr_palace,
                    workspace_id="",
                    query=query[:1000],
                    limit=min(3, SEARCH_LIMIT),
                )
            except Exception as exc:
                log.debug(
                    "Selected backend semantic recall failed at round %d (manager): %s",
                    round_idx,
                    exc,
                )

    lessons: list[Any] = []
    gaps: list[Any] = []
    verify_runs: list[dict[str, Any]] = []
    if round_idx > RICH_RECALL_THRESHOLD:
        store_ws = _safe_store(repo_root, workspace_id)
        store_mgr = _safe_store(repo_root, "")
        if store_ws is not None:
            try:
                from lethyx.memory.models import MemoryQuery

                mq = MemoryQuery(limit=LESSONS_LIMIT * 4, include_stale=False)
                mq.workspace_id = workspace_id
                lessons.extend(store_ws.query_lessons(mq)[:LESSONS_LIMIT])
            except Exception as exc:
                log.debug("Lesson recall failed (workspace): %s", exc)
        if store_mgr is not None:
            try:
                from lethyx.memory.models import LessonType, MemoryQuery

                mq_mgr = MemoryQuery(
                    limit=LESSONS_LIMIT * 2,
                    include_stale=False,
                    lesson_type=LessonType.MANAGER,
                )
                lessons.extend(store_mgr.query_lessons(mq_mgr)[:LESSONS_LIMIT])
            except Exception as exc:
                log.debug("Lesson recall failed (manager): %s", exc)
            try:
                if hasattr(store_mgr, "get_active_gaps"):
                    gaps = list(store_mgr.get_active_gaps()[:5])
                elif hasattr(store_mgr, "query_gaps"):
                    gaps = list(store_mgr.query_gaps(limit=5))
            except Exception as exc:
                log.debug("Gap recall failed: %s", exc)

        target = ws_palace or mgr_palace
        if target is not None:
            try:
                verify_runs = _selected_backend_recall(
                    target,
                    workspace_id=workspace_id,
                    query="verify run results",
                    room="verify_runs",
                    limit=3,
                )
            except Exception as exc:
                log.debug("verify_runs recall failed: %s", exc)

    title = f"Periodic recall @ round {round_idx}"
    if round_idx > RICH_RECALL_THRESHOLD:
        title += " (rich)"

    return _build_recall_text(
        workspace_id=workspace_id,
        title=title,
        workspace_recent=recent_ws,
        workspace_semantic=semantic_ws,
        manager_recent=recent_mgr,
        manager_semantic=semantic_mgr,
        lessons=lessons[: LESSONS_LIMIT * 2],
        gaps=gaps,
        verify_runs=verify_runs,
        phase=phase,
    )


def record_workspace_change(
    *,
    workspace_id: str,
    tool_name: str,
    args_summary: str,
    result_summary: str,
    repo_root: Path,
    success: bool = True,
    task_id: str = "",
) -> None:
    """Fire-and-forget palace write after a successful write-style tool call.

    Stored under ``room=changes`` so periodic recall (which can scope
    by room via the palace tools) can find it cleanly. Failures here
    must never break the loop — we log at debug and move on.
    """
    if not workspace_id:
        return
    palace = _safe_palace(repo_root, workspace_id)
    if palace is None:
        return
    try:
        title = f"{tool_name}: {args_summary[:120]}"
        body = f"args: {args_summary}\nresult: {result_summary[:600]}\nsuccess: {success}"
        run_id = _run_id_from_task_id(task_id)
        room = "changes" if success else "errors"
        _retain_selected_block(
            palace,
            MemoryBlockRecord(
                block_id=_new_block_id("change" if success else "error", workspace_id),
                workspace_id=workspace_id,
                run_id=run_id,
                title=title,
                content=body,
                kind="change" if success else "error",
                lifecycle="active",
                trust_level="agent_claim",
                scope="run_scoped",
                tier="hot",
                surface="current_run_state",
                tags=(tool_name, "auto-recorded", room),
                metadata={
                    "room": room,
                    "hall": "workspace",
                    "tool_name": tool_name,
                    "task_id": task_id,
                    "run_id": run_id,
                    "success": bool(success),
                },
            ),
        )
    except Exception as exc:
        log.debug("Auto-record failed for %s: %s", tool_name, exc)


def mirror_subtask_to_memory(
    *,
    plan: Any,
    subtask: Any,
    repo_root: Path,
    workspace_id: str,
) -> None:
    """Persist a finished planner subtask into MemPalace.

    The plan file remains the canonical state — this mirror only serves
    cross-task recall (other workspaces, future runs, periodic recall
    queries) so the agent can find prior progress through the same
    semantic interface as any other memory record.

    Failure is non-fatal: memory backends may be unavailable in CI or
    in standalone host-agent builds; we never break the planner loop.
    """
    if subtask is None or plan is None:
        return
    try:
        backend = _safe_palace(repo_root, workspace_id)
        if backend is None:
            return
        subtask_id = getattr(subtask, "id", "") or str(getattr(plan, "cursor", ""))
        goal = (getattr(subtask, "title", "") or getattr(subtask, "description", "") or "")[:400]
        content = f"subtask:{subtask_id} goal:{goal}"
        _retain_selected_block(
            backend,
            MemoryBlockRecord(
                block_id=_new_block_id("subtask", workspace_id),
                workspace_id=workspace_id,
                run_id=str(getattr(plan, "run_id", "") or ""),
                phase_id="execute",
                subtask_id=subtask_id,
                title=f"subtask {subtask_id}".strip(),
                content=content,
                kind="subtask_card",
                lifecycle="active",
                trust_level="agent_claim",
                scope="subtask_scoped",
                tier="hot",
                surface="current_run_state",
                tags=("subtask",),
                metadata={
                    "room": "subtask",
                    "hall": "workspace",
                    "task_id": str(getattr(plan, "task_id", "") or ""),
                    "status": str(getattr(subtask, "status", "") or ""),
                    "palace_store": "palace.subtask",
                },
            ),
        )
    except Exception:
        log.debug("subtask palace mirror failed", exc_info=True)


def _subtask_has_verified_evidence(evidence: list[Any]) -> bool:
    joined = "\n".join(str(item) for item in evidence).lower()
    if not joined:
        return False
    has_verify = "run_workspace_verify" in joined or "verify_run_id=" in joined
    has_pass = any(marker in joined for marker in ("pass", "passed", "exit_code=0", "exit 0"))
    has_fail = any(marker in joined for marker in ("fail", "failed", "exit_code=1", "traceback"))
    return has_verify and has_pass and not has_fail


def init_loop_memory(
    messages: list[dict[str, Any]],
    tools_ctx: Any,
) -> tuple[Path, str]:
    """Set up memory state for a fresh loop.

    Resolves ``repo_root`` from the tool context and tries to figure out
    which workspace the loop is about. If we found a workspace and
    MemPalace can answer, the initial recall block is appended directly
    to ``messages`` so the very first LLM call sees it.

    Returns ``(repo_root, initial_workspace_id)``. Both fields are
    safe to use even if memory is unavailable: ``initial_workspace_id``
    is ``""`` when nothing was detected, and the rest of the loop must
    treat that as "no recall scope yet — wait for the agent to name a
    workspace via tool args".

    Imported lazily from loop.py to keep run_llm_loop short.
    """
    repo_root = Path(
        getattr(tools_ctx, "host_repo_root", None)
        or getattr(tools_ctx, "repo_dir", None)
        or Path.cwd()
    )
    initial_ws = _workspace_from_context(tools_ctx) or _guess_initial_workspace(messages)
    if _LETHYX_overlay_already_compiled(tools_ctx):
        try:
            from lethyx.memory.kernel.telemetry import record_memory_event

            record_memory_event(
                repo_root,
                event_type="memory_overlay_duplicate_suppressed",
                workspace_id=initial_ws,
                status="ok",
                data={"origin": _context_overlays(tools_ctx).get("memory_overlay_origin", "")},
            )
        except Exception:
            log.debug("memory overlay suppression telemetry skipped", exc_info=True)
        return repo_root, initial_ws
    if _auto_recall_enabled("task_start", tools_ctx=tools_ctx):
        try:
            if _is_LETHYX_managed(tools_ctx):
                recall_text = recall_core_overlay_for_task_start(
                    workspace_id=initial_ws,
                    repo_root=repo_root,
                    tools_ctx=tools_ctx,
                )
            elif initial_ws:
                recall_text = recall_for_task_start(
                    workspace_id=initial_ws,
                    task_input=_extract_task_brief(messages),
                    repo_root=repo_root,
                )
            else:
                recall_text = recall_core_overlay_for_task_start(
                    workspace_id="",
                    repo_root=repo_root,
                    tools_ctx=tools_ctx,
                )
            if recall_text:
                messages.append({"role": "system", "content": recall_text})
        except Exception:
            log.debug("Initial memory recall failed", exc_info=True)
    return repo_root, initial_ws


def recall_core_overlay_for_task_start(
    *,
    workspace_id: str,
    repo_root: Path,
    tools_ctx: Any | None = None,
) -> str:
    """Inject proactive core overlay (not legacy palace semantic search)."""
    try:
        from lethyx.memory.proactive.compiler import ProactiveMemoryCompiler
    except Exception:
        return ""
    phase_id = ""
    if tools_ctx is not None:
        phase_id = str(getattr(tools_ctx, "LETHYX_phase_id", "") or "")
    compiler = ProactiveMemoryCompiler()
    overlay = compiler.build_minimal_overlay(
        repo_root=repo_root,
        workspace_id=workspace_id,
        phase_id=phase_id or "task_start",
    )
    if not overlay.sections:
        return ""
    return overlay.render_markdown()


def _guess_initial_workspace(messages: list[dict[str, Any]]) -> str:
    """Mirror of ``loop._guess_initial_workspace``; kept here so the
    initialization helper is self-contained and unit-testable without
    pulling the loop module.

    We prefer (in order):
      1. ``Workspace: workspaces/<name>`` markers from the launcher
         task brief,
      2. ``workspace_id="<name>"`` literals — but only when seen in
         non-system messages, because the system prompt embeds these as
         code examples (e.g. ``run_manager_task(workspace_id="agent_research")``)
         that would otherwise hijack initial recall.
      3. ``workspaces/<name>/`` style file paths. These are useful fallbacks,
         but weaker than the launcher marker because environment snapshots may
         include unrelated workspace examples.
    """
    all_messages = [msg for msg in messages[:10] if isinstance(msg, dict)]
    non_system = [msg for msg in all_messages if msg.get("role") != "system"]

    for msg in all_messages:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            candidate = _workspace_from_labeled_line(line)
            if candidate:
                return candidate

    for msg in non_system:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        match = re.search(
            r"(?<![\w.-])workspace_id\s*=\s*[`'\"]?([A-Za-z0-9_.-]+)",
            content,
        )
        if match:
            return _workspace_candidate(match.group(1))
        for line in content.splitlines():
            candidate = _workspace_from_labeled_line(line, labels=_WORKSPACE_ID_LABELS)
            if candidate:
                return candidate

    for msg in all_messages:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for token in _shell_tokens(content):
            candidate = _workspace_after_workspaces(
                PurePosixPath(token.replace("\\", "/").strip("`\"'.,:;)]}")).parts
            )
            if candidate:
                return candidate
    return ""


def _workspace_from_labeled_line(
    line: str,
    *,
    labels: set[str] | frozenset[str] = _WORKSPACE_LABELS,
) -> str:
    for sep in (":", "="):
        key, found, value = line.partition(sep)
        if found and key.strip().strip("`\"'").casefold() in labels:
            return _workspace_candidate(value)
    return ""


def _shell_tokens(text: str) -> list[str]:
    try:
        return shlex.split(text, posix=False)
    except ValueError:
        return text.split()


def _extract_task_brief(messages: list[dict[str, Any]]) -> str:
    # Prefer user/task messages. The first system prompt can be a large
    # persona/constitution block, and persisting that as the plan objective
    # poisons the planner state.
    preferred = [
        msg for msg in messages[:12] if isinstance(msg, dict) and msg.get("role") != "system"
    ]
    candidates = preferred or [msg for msg in messages[:6] if isinstance(msg, dict)]

    pieces: list[str] = []
    for msg in candidates:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str) and content.strip():
            pieces.append(content.strip())
    return "\n".join(pieces)[:TASK_BRIEF_MAX_CHARS]


def maybe_inject_periodic_recall(
    *,
    workspace_id: str,
    round_idx: int,
    last_recall_round: int,
    recent_actions_summary: str,
    repo_root: Path,
    messages: list[dict[str, Any]],
    phase: str | None = None,
    task_id: str = "",
) -> int:
    """If it's time, inject a periodic recall block into ``messages``.

    Returns the (possibly updated) ``last_recall_round`` so the caller
    can store it back without us reaching into its scope. The check on
    ``round_idx > 1`` mirrors the loop's "skip first round" rule because
    the initial recall already covers it.
    """
    if not workspace_id or round_idx <= 1:
        return last_recall_round
    if not _auto_recall_enabled("periodic"):
        return last_recall_round
    if round_idx - last_recall_round < RECALL_INTERVAL:
        return last_recall_round
    try:
        recall_text = recall_periodic(
            workspace_id=workspace_id,
            round_idx=round_idx,
            recent_actions_summary=recent_actions_summary,
            repo_root=repo_root,
            phase=phase,
            task_id=task_id,
        )
    except Exception:
        log.debug("Periodic memory recall failed", exc_info=True)
        return last_recall_round
    if recall_text:
        messages.append({"role": "system", "content": recall_text})
        return round_idx
    return last_recall_round


def observe_tool_calls(
    *,
    tool_calls: list[dict[str, Any]],
    recent_tool_results: list[dict[str, Any]] | None = None,
    write_tool_names: frozenset[str],
    verify_gate: Any,
    repo_root: Path,
    current_workspace_id: str,
    task_id: str = "",
) -> str:
    """Process the tool_calls list for the freshly executed round.

    Side effects:
      * Bumps the verify-gate counter for each call (write tools count
        toward the threshold; verify tools reset it).
      * For successful write tools, mirrors a structured event into
        MemPalace via ``record_workspace_change``.

    Returns the new ``current_workspace_id`` (may have advanced if the
    agent named a different workspace in this round).
    """
    import json as _json

    new_ws = current_workspace_id
    recent_tool_results = list(recent_tool_results or [])
    for idx, tc in enumerate(tool_calls):
        fn_name = tc.get("function", {}).get("name", "") if isinstance(tc, dict) else ""
        try:
            args = _json.loads(tc.get("function", {}).get("arguments") or "{}")
        except Exception:
            args = {}
        ws = ""
        if isinstance(args, dict):
            raw = args.get("workspace_id")
            if isinstance(raw, str) and raw.strip():
                ws = raw.strip()
                new_ws = ws
        try:
            verify_gate.observe(fn_name, workspace_id=ws)
        except Exception:
            log.debug("VerifyGate.observe failed for %s", fn_name, exc_info=True)
        if fn_name in write_tool_names and ws:
            try:
                trace = recent_tool_results[idx] if idx < len(recent_tool_results) else {}
                success = not bool(trace.get("is_error")) if isinstance(trace, dict) else True
                result_summary = str(trace.get("result") or "") if isinstance(trace, dict) else ""
                record_workspace_change(
                    workspace_id=ws,
                    tool_name=fn_name,
                    args_summary=_short_args_repr(args),
                    result_summary=result_summary,
                    repo_root=repo_root,
                    success=success,
                    task_id=task_id,
                )
            except Exception:
                log.debug("Auto-record of %s failed", fn_name, exc_info=True)
    return new_ws


def _short_args_repr(args: Any) -> str:
    import json as _json

    if not isinstance(args, dict):
        return str(args)[:200]
    parts: list[str] = []
    for key in ("workspace_id", "file_path", "commit_message", "validation_summary"):
        val = args.get(key)
        if val:
            parts.append(f"{key}={str(val)[:80]}")
    if not parts:
        try:
            return _json.dumps(args, ensure_ascii=False)[:200]
        except Exception:
            return str(args)[:200]
    return ", ".join(parts)


def record_verify_outcome(
    *,
    workspace_id: str,
    passed: bool,
    pass_rate: float,
    summary: str,
    details: str,
    repo_root: Path,
    failed_step_count: int = 0,
) -> str:
    """Persist a verify-run outcome under ``room=verify_runs``.

    This is what makes the rich periodic recall actually useful — without
    it, the ``verify_runs`` lookup returns empty even after the agent
    runs verify many times. Called from the ``run_workspace_verify``
    tool wrapper, not from the loop directly.

    Tier 2.4 — returns a stable ``verify_run_id`` derived from the
    timestamp+workspace, so callers (the tool wrapper, completion gates)
    can correlate this verify with downstream lessons. Returns an empty
    string when the palace is unavailable; callers should treat that as
    "no id" and degrade gracefully.
    """
    palace = _safe_palace(repo_root, workspace_id)
    if palace is None:
        return ""
    verify_run_id = _new_verify_run_id(workspace_id)
    try:
        title = f"verify {'PASS' if passed else 'FAIL'} (rate={pass_rate:.1%}): {summary[:120]}"
        evidence = EvidenceRef(
            ref_type="test_run",
            ref_id=verify_run_id,
            produced_by="verifier",
            phase="verify",
        )
        _retain_selected_block(
            palace,
            MemoryBlockRecord(
                block_id=verify_run_id,
                workspace_id=workspace_id,
                title=title,
                content=details[:RECALL_BLOCK_MAX_CHARS],
                kind="verification_report",
                lifecycle="active",
                trust_level="workspace_verified",
                scope="cross_run_durable",
                tier="hot",
                surface="supplemental_evidence",
                tags=("verify", "verify_runs", "pass" if passed else "fail"),
                evidence_refs=(evidence,),
                verified=True,
                metadata={
                    "room": "verify_runs",
                    "hall": "workspace",
                    "verify_run_id": verify_run_id,
                    "pass_rate": float(pass_rate),
                    "failed_step_count": int(failed_step_count or 0),
                    "passed": bool(passed),
                },
            ),
        )
    except Exception as exc:
        log.debug("Verify outcome recording failed: %s", exc)
        return ""
    return verify_run_id


def _new_verify_run_id(workspace_id: str) -> str:
    """Mint a stable id for a verify run. Format: ``verify-<ws>-<ts_ms>``.

    Using millisecond resolution avoids collisions when the loop hits
    verify twice in the same second, which can happen in tight
    remediation cycles.
    """

    import time as _time

    ts_ms = int(_time.time() * 1000)
    ws = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(workspace_id or "unknown")).strip("_") or "unknown"
    return f"verify-{ws}-{ts_ms}"


@runtime_checkable
class MemoryHooks(Protocol):
    """Protocol for injecting memory behavior into an agent loop."""

    def init_loop_memory(self, *args, **kwargs): ...

    def maybe_inject_periodic_recall(self, *args, **kwargs): ...

    def observe_tool_calls(self, *args, **kwargs): ...

    def mirror_subtask_to_memory(self, *args, **kwargs): ...

    def record_verify_outcome(self, *args, **kwargs): ...


class LethyxMemoryHooks:
    """Default implementation backed by Lethyx's own memory hook functions."""

    def init_loop_memory(self, *args, **kwargs):
        return init_loop_memory(*args, **kwargs)

    def maybe_inject_periodic_recall(self, *args, **kwargs):
        return maybe_inject_periodic_recall(*args, **kwargs)

    def observe_tool_calls(self, *args, **kwargs):
        return observe_tool_calls(*args, **kwargs)

    def mirror_subtask_to_memory(self, *args, **kwargs):
        return mirror_subtask_to_memory(*args, **kwargs)

    def record_verify_outcome(self, *args, **kwargs):
        return record_verify_outcome(*args, **kwargs)


DEFAULT_MEMORY_HOOKS = LethyxMemoryHooks()
