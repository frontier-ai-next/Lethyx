"""Smart Hermes memory migration helpers for Lethyx.

The migrator treats legacy sources as evidence.  It never copies raw tool logs or
full transcripts into prompt memory; it creates scoped, typed summary blocks with
back-pointers to the source session/message/file identifiers.
"""

import html
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from lethyx import MemoryClient
from lethyx.contracts import EvidenceRef

VALID_BACKENDS = {"mempalace", "hindsight", "mnemosyne", "composite"}
_CONTROL_PROFILE = "umb" + "rella"
PROFILE_WORKSPACES = {
    "default": "admin",
    "admin": "admin",
    _CONTROL_PROFILE: _CONTROL_PROFILE,
    "lethyx": "lethyx",
    "research": "research",
    "ops": "ops",
}
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9._-]{6,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.I),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]{4,}"),
    re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"),
]
_PROJECT_HINTS = {
    "lethyx": ["lethyx", "lethyx", "memory_bench", "hindsight", "mnemosyne", "mempalace", "mim"],
    "research": ["arxiv", "paper", "литератур", "research", "citation", "andrew"],
    "ops": ["gateway", "docker", "service", "telegram", "process", "ops", "cc"],
}
_IMPORTANT_TERMS = {
    "memory",
    "память",
    "mcp",
    "skill",
    "provider",
    "gateway",
    "profile",
    "setup",
    "настрой",
    "install",
    "migration",
    "миграц",
    "architecture",
    "архитект",
    "release",
    "benchmark",
    "runtime",
    "control",
    "agent",
    "project",
    "docs",
    "obsidian",
}
_NOISY_TERMS = {"log", "tail", "debug", "temporary", "tmp", "process output", "raw"}
_PROJECT_DOC_PATTERNS = [
    ".hermes.md",
    "HERMES.md",
    "AGENTS.md",
    "agents.md",
    "CLAUDE.md",
    "claude.md",
    "README.md",
    "README*.md",
    "docs/**/*.md",
    ".hermes/**/*.md",
    ".codex/**/*.md",
    "paper/**/*.md",
    "paper/**/*.tex",
]
_PROJECT_DOC_EXCLUDE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    "vendor",
}


@dataclass(frozen=True)
class SessionSummary:
    memory_id: str
    workspace_id: str
    title: str
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    evidence_refs: list[EvidenceRef]
    tier: str
    lifecycle: str
    surface: str
    memory_activity: str
    trust_level: str = "workspace_verified"
    kind: str = "session_summary"

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_refs"] = [asdict(ref) for ref in self.evidence_refs]
        return payload


@dataclass(frozen=True)
class MigrationResult:
    ok: bool
    count: int = 0
    saved: int = 0
    skipped: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    output_path: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def redact_sensitive_text(text: str, max_chars: int = 1200) -> str:
    out = str(text or "")
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted-secret]", out)
    out = out.replace("\x00", " ")
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _content_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped and stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except Exception:
                return raw
            return _content_text(parsed)
        return raw
    if isinstance(raw, dict):
        parts: list[str] = []
        for key in ("text", "content", "value"):
            if key in raw:
                parts.append(_content_text(raw.get(key)))
        if not parts:
            parts.extend(_content_text(value) for value in raw.values())
        return " ".join(part for part in parts if part)
    if isinstance(raw, (list, tuple)):
        return " ".join(_content_text(item) for item in raw)
    return str(raw)


def _slug(value: str, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return (cleaned or "item")[:limit]


def _iso(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except Exception:
        return ""


def detect_workspace(profile: str, cwd: str = "", title: str = "") -> str:
    profile_key = PROFILE_WORKSPACES.get(str(profile or "").strip(), str(profile or "").strip())
    haystack = f"{cwd} {title}".lower().replace("\\", "/")
    if "documents/lethyx" in haystack or "lethyx" in haystack or "lethyx" in haystack:
        return "lethyx"
    cwd_name = _slug(Path(str(cwd or "").replace("\\", "/")).name).lower().replace("-", "_")
    if profile_key in {"", "default", "admin"} and cwd_name not in {
        "",
        "item",
        "documents",
        "admin",
    }:
        return cwd_name
    for workspace, hints in _PROJECT_HINTS.items():
        if any(hint in haystack for hint in hints):
            return workspace
    if profile_key:
        return profile_key
    return "admin"


def classify_session_importance(
    row: dict[str, Any], *, user_texts: list[str], final_assistant_text: str
) -> dict[str, Any]:
    title = str(row.get("title") or "")
    cwd = str(row.get("cwd") or row.get("git_repo_root") or "")
    haystack = f"{title} {cwd} {' '.join(user_texts[:5])} {final_assistant_text[:500]}".lower()
    tool_calls = int(row.get("tool_call_count") or 0)
    message_count = int(row.get("message_count") or 0)
    input_tokens = int(row.get("input_tokens") or 0)
    important_hits = sum(1 for term in _IMPORTANT_TERMS if term in haystack)
    project_hits = sum(1 for hints in _PROJECT_HINTS.values() for hint in hints if hint in haystack)
    noisy_hits = sum(1 for term in _NOISY_TERMS if term in haystack)
    tool_ratio = tool_calls / max(message_count, 1)
    score = important_hits * 2 + project_hits + min(message_count // 40, 3)
    if input_tokens > 250000:
        score += 1
    if tool_ratio > 0.75:
        score -= 3
    if noisy_hits:
        score -= noisy_hits
    if str(row.get("archived") or "0") not in {"", "0", "False", "false"}:
        score -= 2

    if tool_ratio > 0.85 and score <= 2:
        tier, lifecycle, surface, activity = "cold", "archived", "archive_hint", "passive"
    elif score >= 7:
        tier, lifecycle, surface, activity = "hot", "active", "supplemental_evidence", "active"
    elif score >= 3:
        tier, lifecycle, surface, activity = "warm", "active", "supplemental_evidence", "active"
    else:
        tier, lifecycle, surface, activity = "cold", "archived", "archive_hint", "passive"
    return {
        "score": score,
        "tier": tier,
        "lifecycle": lifecycle,
        "surface": surface,
        "memory_activity": activity,
        "tool_ratio": round(tool_ratio, 4),
        "important_hits": important_hits,
        "project_hits": project_hits,
        "noisy_hits": noisy_hits,
    }


def _session_row(con: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    con.row_factory = sqlite3.Row
    row = con.execute("select * from sessions where id = ?", (session_id,)).fetchone()
    return dict(row) if row is not None else None


def _session_messages(con: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select id, role, content, tool_name, timestamp, active, compacted from messages where session_id = ? order by id",
        (session_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _brief_items(items: list[str], *, limit: int, max_chars: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = redact_sensitive_text(item, max_chars=max_chars)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def summarize_session(
    db_path: str | Path, session_id: str, *, profile: str = "default"
) -> SessionSummary | None:
    db = Path(db_path)
    con = sqlite3.connect(db)
    try:
        row = _session_row(con, session_id)
        if row is None:
            return None
        messages = _session_messages(con, session_id)
    finally:
        con.close()

    user_texts_all = [
        _content_text(msg.get("content")) for msg in messages if msg.get("role") == "user"
    ]
    assistant_texts = [
        _content_text(msg.get("content"))
        for msg in messages
        if msg.get("role") == "assistant" and _content_text(msg.get("content"))
    ]
    user_texts = _brief_items(user_texts_all, limit=4, max_chars=350)
    final_assistant = redact_sensitive_text(
        assistant_texts[-1] if assistant_texts else "", max_chars=700
    )
    title = redact_sensitive_text(str(row.get("title") or session_id), max_chars=160)
    workspace = detect_workspace(
        profile,
        str(row.get("git_repo_root") or row.get("cwd") or ""),
        title,
    )
    classified = classify_session_importance(
        row, user_texts=user_texts, final_assistant_text=final_assistant
    )
    first_id = messages[0].get("id") if messages else ""
    last_id = messages[-1].get("id") if messages else ""
    lines = [
        f"Hermes session summary: {title}",
        "",
        f"Profile/workspace: {profile} -> {workspace}",
        f"Session id: {session_id}",
        f"Started: {_iso(row.get('started_at'))}",
    ]
    cwd = redact_sensitive_text(str(row.get("cwd") or ""), max_chars=220)
    repo = redact_sensitive_text(str(row.get("git_repo_root") or ""), max_chars=220)
    branch = redact_sensitive_text(str(row.get("git_branch") or ""), max_chars=80)
    if cwd:
        lines.append(f"CWD: {cwd}")
    if repo:
        lines.append(f"Git repo: {repo}")
    if branch:
        lines.append(f"Git branch: {branch}")
    lines.extend(
        [
            f"Messages/tool calls: {row.get('message_count') or 0}/{row.get('tool_call_count') or 0}",
            f"Classification: tier={classified['tier']}, lifecycle={classified['lifecycle']}, surface={classified['surface']}, score={classified['score']}",
            "",
            "User intent excerpts:",
        ]
    )
    if user_texts:
        lines.extend(f"- {text}" for text in user_texts)
    else:
        lines.append("- No user text captured.")
    if final_assistant:
        lines.extend(["", "Final assistant outcome excerpt:", f"- {final_assistant}"])
    lines.extend(
        [
            "",
            "Migration note: raw tool outputs and full transcript were not copied; use session_search/state.db for exact evidence.",
        ]
    )
    tags = [
        "session-migration",
        f"profile:{profile}",
        f"source:{row.get('source') or 'unknown'}",
        f"tier:{classified['tier']}",
        f"workspace:{workspace}",
    ]
    if repo:
        tags.append(f"repo:{Path(repo).name}")
    metadata = {
        "source": "hermes_state_session_summary",
        "session_id": session_id,
        "profile": profile,
        "workspace_id": workspace,
        "title": title,
        "source_db": str(db),
        "session_source": row.get("source") or "",
        "started_at": _iso(row.get("started_at")),
        "ended_at": _iso(row.get("ended_at")),
        "cwd": cwd,
        "git_repo_root": repo,
        "git_branch": branch,
        "message_count": int(row.get("message_count") or 0),
        "tool_call_count": int(row.get("tool_call_count") or 0),
        "input_tokens": int(row.get("input_tokens") or 0),
        "output_tokens": int(row.get("output_tokens") or 0),
        "first_message_id": first_id,
        "last_message_id": last_id,
        "classification": classified,
        "memory_activity": classified["memory_activity"],
        "migration_policy": "summary_only_no_raw_tool_outputs",
        "privacy_class": "normal",
    }
    memory_id = f"session_{_slug(profile, 32)}_{_slug(session_id, 80)}"
    evidence = [
        EvidenceRef(
            ref_type="hermes_session",
            ref_id=f"{profile}/{session_id}#{first_id}-{last_id}",
            produced_by="hermes_lethyx_sessions_summarize",
        )
    ]
    return SessionSummary(
        memory_id=memory_id,
        workspace_id=workspace,
        title=f"Session summary: {title}",
        content="\n".join(lines),
        tags=tags,
        metadata=metadata,
        evidence_refs=evidence,
        tier=classified["tier"],
        lifecycle=classified["lifecycle"],
        surface=classified["surface"],
        memory_activity=classified["memory_activity"],
    )


def profile_state_dbs(hermes_home: str | Path) -> dict[str, Path]:
    home = Path(hermes_home)
    out: dict[str, Path] = {}
    default = home / "state.db"
    if default.exists():
        out["default"] = default
    profiles = home / "profiles"
    if profiles.exists():
        for path in sorted(profiles.iterdir()):
            db = path / "state.db"
            if db.exists():
                out[path.name] = db
    return out


def _session_ids(db_path: Path, *, limit: int = 0) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "select id from sessions order by coalesce(started_at, 0) desc"
        ).fetchall()
    finally:
        con.close()
    ids = [str(row[0]) for row in rows]
    return ids[:limit] if limit and limit > 0 else ids


def _client(repo_root: str | Path, workspace: str, backend: str) -> MemoryClient:
    return MemoryClient(repo_root=repo_root, workspace_id=workspace, backend=backend)


def _profile_home(hermes_home: str | Path, profile: str) -> Path:
    home = Path(hermes_home)
    if profile in {"", "default", "admin"}:
        return home
    return home / "profiles" / profile


def _memory_files_for_profile(home: Path) -> list[tuple[str, Path]]:
    return [("memory", home / "memories/MEMORY.md"), ("user", home / "memories/USER.md")]


def _split_memory_entries(text: str) -> list[str]:
    entries: list[str] = []
    for chunk in str(text or "").replace("\r\n", "\n").split("\n§\n"):
        cleaned = chunk.strip()
        if cleaned:
            entries.append(cleaned)
    return entries


def _builtin_memory_id(profile: str, target: str, idx: int, workspace: str) -> str:
    prefix = "user_profile" if target == "user" else "profile_memory"
    return f"{prefix}_{_slug(workspace, 40)}_{_slug(profile or 'default', 32)}_{idx:03d}"


def _builtin_memory_title(profile: str, target: str, idx: int) -> str:
    label = "User profile" if target == "user" else "Profile memory"
    return f"{label}: {profile or 'default'} #{idx}"


def import_builtin_memories(
    *,
    hermes_home: str | Path,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    profiles: list[str] | None = None,
    include_global_user: bool = True,
) -> MigrationResult:
    """Import Hermes bounded MEMORY.md/USER.md into a scoped Lethyx workspace.

    The built-in Hermes files remain the bounded curated source of truth.  This
    import mirrors each entry as a typed, evidenced Lethyx block so enabling the
    Lethyx provider does not hide the memory the user already curated.
    """

    selected_profiles = profiles or ["default"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for profile in selected_profiles:
        home = _profile_home(hermes_home, profile)
        for target, path in _memory_files_for_profile(home):
            if not path.exists():
                continue
            try:
                entries = _split_memory_entries(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                errors.append(f"{profile}/{target}: {type(exc).__name__}: {exc}")
                continue
            for idx, entry in enumerate(entries, start=1):
                target_workspaces = [workspace]
                if include_global_user and target == "user" and workspace != "global":
                    target_workspaces.append("global")
                for workspace_id in target_workspaces:
                    try:
                        row = _remember_block(
                            repo_root=repo_root,
                            backend=backend,
                            workspace=workspace_id,
                            memory_id=_builtin_memory_id(profile, target, idx, workspace_id),
                            title=_builtin_memory_title(profile, target, idx),
                            content=redact_sensitive_text(entry, max_chars=2200),
                            kind="user_profile" if target == "user" else "profile_memory",
                            tags=[
                                "current-memory",
                                "hermes-builtins",
                                f"profile:{profile or 'default'}",
                                f"target:{target}",
                                f"workspace:{workspace_id}",
                            ],
                            metadata={
                                "source": "hermes_builtin_memory_bootstrap",
                                "profile": profile or "default",
                                "target": target,
                                "source_path": str(path),
                                "memory_activity": "active",
                                "builtin_memory_target": target,
                            },
                            evidence_refs=[
                                EvidenceRef(
                                    ref_type="file",
                                    ref_id=str(path),
                                    produced_by="hermes_lethyx_import_builtins",
                                )
                            ],
                            lifecycle="active",
                            surface="supplemental_evidence",
                            tier="hot",
                        )
                        row.update(
                            {"profile": profile, "target": target, "workspace": workspace_id}
                        )
                        rows.append(row)
                    except Exception as exc:
                        errors.append(f"{profile}/{target}/{idx}: {type(exc).__name__}: {exc}")
    return MigrationResult(
        ok=not errors,
        count=len(rows),
        saved=sum(1 for row in rows if row.get("ok")),
        rows=rows,
        errors=errors,
    )


def _remember_summary(
    repo_root: str | Path, backend: str, summary: SessionSummary
) -> dict[str, Any]:
    client = _client(repo_root, summary.workspace_id, backend)
    try:
        result = client.remember(
            summary.content,
            title=summary.title,
            kind=summary.kind,
            tags=summary.tags,
            metadata={**summary.metadata, "tier": summary.tier},
            evidence_refs=summary.evidence_refs,
            lifecycle=summary.lifecycle,
            surface=summary.surface,
            tier=summary.tier,
            trust_level=summary.trust_level,
            memory_id=summary.memory_id,
        )
        return {
            "ok": result.saved,
            "memory_id": result.memory_id,
            "workspace": summary.workspace_id,
            "backend": result.backend,
            "tier": summary.tier,
            "lifecycle": summary.lifecycle,
        }
    finally:
        close = getattr(client.backend, "close", None)
        if callable(close):
            close()


def migrate_sessions(
    *,
    hermes_home: str | Path,
    repo_root: str | Path,
    backend: str = "composite",
    profiles: list[str] | None = None,
    limit_per_profile: int = 0,
    dry_run: bool = False,
) -> MigrationResult:
    dbs = profile_state_dbs(hermes_home)
    selected = set(profiles or dbs.keys())
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    saved = 0
    skipped = 0
    for profile, db in dbs.items():
        if profile not in selected:
            continue
        for session_id in _session_ids(db, limit=limit_per_profile):
            try:
                summary = summarize_session(db, session_id, profile=profile)
                if summary is None:
                    skipped += 1
                    continue
                row = summary.to_json()
                if dry_run:
                    row.update({"ok": True, "dry_run": True})
                else:
                    row.update(_remember_summary(repo_root, backend, summary))
                    if row.get("ok"):
                        saved += 1
                rows.append(row)
            except Exception as exc:
                errors.append(f"{profile}/{session_id}: {type(exc).__name__}: {exc}")
    return MigrationResult(
        ok=not errors, count=len(rows), saved=saved, skipped=skipped, rows=rows, errors=errors
    )


def _path_under(path_text: Any, root: Path) -> bool:
    if not path_text:
        return False
    try:
        path = Path(str(path_text)).expanduser().resolve()
    except Exception:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return path == root


def _session_belongs_to_project(row: dict[str, Any], *, workspace: str, project_root: Path) -> bool:
    if _path_under(row.get("git_repo_root"), project_root) or _path_under(
        row.get("cwd"), project_root
    ):
        return True
    title = str(row.get("title") or "").lower().replace("-", "_")
    root_name = project_root.name.lower().replace("-", "_")
    workspace_key = workspace.lower().replace("-", "_")
    return bool(workspace_key and workspace_key in title) or bool(root_name and root_name in title)


def _summary_for_workspace(summary: SessionSummary, workspace: str) -> SessionSummary:
    if summary.workspace_id == workspace:
        return summary
    tags = [tag for tag in summary.tags if not str(tag).startswith("workspace:")]
    tags.append(f"workspace:{workspace}")
    metadata = {
        **summary.metadata,
        "workspace_id": workspace,
        "workspace_forced_by": "project_bootstrap",
    }
    return replace(
        summary,
        workspace_id=workspace,
        memory_id=f"session_{_slug(workspace, 32)}_{_slug(metadata.get('profile', 'default'), 32)}_{_slug(metadata.get('session_id', summary.memory_id), 80)}",
        tags=tags,
        metadata=metadata,
    )


def migrate_project_sessions(
    *,
    hermes_home: str | Path,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    project_path: str | Path,
    profiles: list[str] | None = None,
    limit: int = 20,
    dry_run: bool = False,
) -> MigrationResult:
    """Summarize Hermes SQLite/FTS5 sessions relevant to one project.

    Raw transcripts and tool logs stay in state.db/session_search. Lethyx gets
    bounded session-summary evidence with source pointers, so the provider can
    recall past work without putting the entire SQLite history in prompt memory.
    """

    project_root = Path(project_path).expanduser().resolve()
    dbs = profile_state_dbs(hermes_home)
    selected = set(profiles or dbs.keys())
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    saved = 0
    skipped = 0
    for profile, db in dbs.items():
        if profile not in selected:
            continue
        con = sqlite3.connect(db)
        try:
            for session_id in _session_ids(db, limit=0):
                if limit and len(rows) >= limit:
                    break
                row = _session_row(con, session_id)
                if row is None or not _session_belongs_to_project(
                    row, workspace=workspace, project_root=project_root
                ):
                    skipped += 1
                    continue
                summary = summarize_session(db, session_id, profile=profile)
                if summary is None:
                    skipped += 1
                    continue
                summary = _summary_for_workspace(summary, workspace)
                if summary.lifecycle != "active":
                    skipped += 1
                    continue
                item = summary.to_json()
                if dry_run:
                    item.update({"ok": True, "dry_run": True})
                else:
                    try:
                        item.update(_remember_summary(repo_root, backend, summary))
                    except Exception as exc:
                        errors.append(f"{profile}/{session_id}: {type(exc).__name__}: {exc}")
                        continue
                    if item.get("ok"):
                        saved += 1
                rows.append(item)
        except Exception as exc:
            errors.append(f"{profile}: {type(exc).__name__}: {exc}")
        finally:
            con.close()
    return MigrationResult(
        ok=not errors, count=len(rows), saved=saved, skipped=skipped, rows=rows, errors=errors
    )


def _remember_block(
    *,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    memory_id: str,
    title: str,
    content: str,
    kind: str,
    tags: list[str],
    metadata: dict[str, Any],
    evidence_refs: list[EvidenceRef],
    lifecycle: str = "active",
    surface: str = "supplemental_evidence",
    tier: str = "warm",
) -> dict[str, Any]:
    client = _client(repo_root, workspace, backend)
    try:
        result = client.remember(
            content,
            title=title,
            kind=kind,
            tags=tags,
            metadata={**metadata, "tier": tier},
            evidence_refs=evidence_refs,
            lifecycle=lifecycle,
            surface=surface,
            tier=tier,
            trust_level="workspace_verified",
            memory_id=memory_id,
        )
        return {
            "ok": result.saved,
            "memory_id": result.memory_id,
            "workspace": workspace,
            "backend": result.backend,
        }
    finally:
        close = getattr(client.backend, "close", None)
        if callable(close):
            close()


def _add_edge(
    *,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    src_id: str,
    dst_id: str,
    edge_type: str,
    reason: str,
) -> dict[str, Any]:
    from lethyx.memory.backends.contracts import MemoryEdgeRecord

    client = _client(repo_root, workspace, backend)
    try:
        edge = MemoryEdgeRecord(
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            workspace_id=workspace,
            reason=reason,
            evidence_refs=(
                EvidenceRef(
                    ref_type="migration", ref_id=reason, produced_by="hermes_smart_migration"
                ),
            ),
            metadata={"source": "hermes_smart_migration"},
        )
        return dict(client.backend.add_edge(edge) or {})
    finally:
        close = getattr(client.backend, "close", None)
        if callable(close):
            close()


def ensure_source_node(
    *,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    source_name: str,
    title: str,
    content: str,
) -> dict[str, Any]:
    memory_id = f"source_{_slug(workspace, 40)}_{_slug(source_name, 80)}"
    return _remember_block(
        repo_root=repo_root,
        backend=backend,
        workspace=workspace,
        memory_id=memory_id,
        title=title,
        content=content,
        kind="source_index",
        tags=["source-node", f"workspace:{workspace}", f"source:{source_name}"],
        metadata={"source": "hermes_smart_migration", "memory_activity": "active"},
        evidence_refs=[
            EvidenceRef(ref_type="source", ref_id=source_name, produced_by="hermes_smart_migration")
        ],
        lifecycle="active",
        surface="supplemental_evidence",
        tier="hot",
    )


def _read_skill_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    front = text[3:end]
    data: dict[str, Any] = {}
    for line in front.splitlines():
        if ":" not in line or line.startswith(" "):
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def index_skills(
    *,
    repo_root: str | Path,
    backend: str,
    hermes_home: str | Path,
    workspace: str = "global",
    limit: int = 0,
) -> MigrationResult:
    homes = [Path(hermes_home) / "skills"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    count = 0
    for root in homes:
        if not root.exists():
            continue
        for skill in sorted(root.rglob("SKILL.md")):
            if limit and count >= limit:
                break
            fm = _read_skill_frontmatter(skill)
            name = str(fm.get("name") or skill.parent.name)
            desc = str(fm.get("description") or "")
            target_workspace = detect_workspace("default", str(skill), f"{name} {desc}")
            if target_workspace == "admin" and workspace != "global":
                target_workspace = workspace
            memory_id = f"skill_index_{_slug(target_workspace, 32)}_{_slug(name, 80)}"
            content = "\n".join(
                [
                    f"Skill pointer: {name}",
                    f"Description: {redact_sensitive_text(desc, max_chars=600)}",
                    f"Path: {skill}",
                    "Use the skill file/skill_view as procedural authority; this block is only an index pointer.",
                ]
            )
            try:
                row = _remember_block(
                    repo_root=repo_root,
                    backend=backend,
                    workspace=target_workspace if workspace == "auto" else workspace,
                    memory_id=memory_id
                    if workspace == "auto"
                    else f"skill_index_{_slug(workspace, 32)}_{_slug(name, 80)}",
                    title=f"Skill pointer: {name}",
                    content=content,
                    kind="skill_index",
                    tags=["skill-index", f"skill:{name}", f"workspace:{target_workspace}"],
                    metadata={
                        "source": "hermes_skill_index",
                        "skill_name": name,
                        "skill_path": str(skill),
                        "description": desc,
                        "memory_activity": "active",
                    },
                    evidence_refs=[
                        EvidenceRef(
                            ref_type="file",
                            ref_id=str(skill),
                            produced_by="hermes_lethyx_index_skills",
                        )
                    ],
                    lifecycle="active",
                    surface="supplemental_evidence",
                    tier="hot" if target_workspace != "admin" else "warm",
                )
                rows.append(row)
                count += 1
            except Exception as exc:
                errors.append(f"{skill}: {type(exc).__name__}: {exc}")
    return MigrationResult(
        ok=not errors,
        count=len(rows),
        saved=sum(1 for row in rows if row.get("ok")),
        rows=rows,
        errors=errors,
    )


def _safe_yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _profile_config_paths(hermes_home: str | Path) -> dict[str, Path]:
    home = Path(hermes_home)
    out = {"default": home / "config.yaml"}
    profiles = home / "profiles"
    if profiles.exists():
        for profile in sorted(profiles.iterdir()):
            if profile.is_dir():
                out[profile.name] = profile / "config.yaml"
    return {profile: path for profile, path in out.items() if path.exists()}


def index_mcp(*, repo_root: str | Path, backend: str, hermes_home: str | Path) -> MigrationResult:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for profile, config_path in _profile_config_paths(hermes_home).items():
        data = _safe_yaml_load(config_path)
        servers = data.get("mcp_servers") or {}
        if not isinstance(servers, dict):
            servers = {}
        workspace = PROFILE_WORKSPACES.get(profile, profile)
        safe_servers = []
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                continue
            env = spec.get("env") or {}
            env_keys = sorted(env.keys()) if isinstance(env, dict) else []
            safe_servers.append(
                {
                    "name": name,
                    "command": spec.get("command"),
                    "args": spec.get("args"),
                    "enabled": spec.get("enabled", True),
                    "env_keys": env_keys,
                }
            )
        content = "Hermes MCP inventory for profile `{}`.\n\nServers:\n{}".format(
            profile,
            json.dumps(safe_servers, ensure_ascii=False, indent=2),
        )
        try:
            row = _remember_block(
                repo_root=repo_root,
                backend=backend,
                workspace=workspace,
                memory_id=f"mcp_inventory_{_slug(profile, 32)}",
                title=f"MCP inventory: {profile}",
                content=content,
                kind="mcp_inventory",
                tags=["mcp-inventory", f"profile:{profile}", f"workspace:{workspace}"],
                metadata={
                    "source": "hermes_config_mcp_inventory",
                    "profile": profile,
                    "config_path": str(config_path),
                    "server_count": len(safe_servers),
                    "memory_activity": "active",
                },
                evidence_refs=[
                    EvidenceRef(
                        ref_type="file",
                        ref_id=str(config_path),
                        produced_by="hermes_lethyx_index_mcp",
                    )
                ],
                lifecycle="active",
                surface="supplemental_evidence",
                tier="hot",
            )
            rows.append(row)
        except Exception as exc:
            errors.append(f"{profile}: {type(exc).__name__}: {exc}")
    return MigrationResult(
        ok=not errors,
        count=len(rows),
        saved=sum(1 for row in rows if row.get("ok")),
        rows=rows,
        errors=errors,
    )


def index_project_docs(
    *,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    project_path: str | Path,
    limit: int = 40,
) -> MigrationResult:
    root = Path(project_path)
    candidates: list[Path] = []
    for pattern in _PROJECT_DOC_PATTERNS:
        candidates.extend(sorted(root.glob(pattern)))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.is_file():
            continue
        rel_parts = (
            {part.lower() for part in path.relative_to(root).parts}
            if path.is_relative_to(root)
            else set()
        )
        if rel_parts & _PROJECT_DOC_EXCLUDE_PARTS:
            continue
        seen.add(resolved)
        if len(rows) >= limit:
            break
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        title = next(
            (
                line.strip("# ").strip()
                for line in text.splitlines()
                if line.strip().startswith("#")
            ),
            str(rel),
        )
        excerpt = redact_sensitive_text(text, max_chars=900)
        content = f"Project document pointer: {rel}\nTitle: {title}\nPath: {path}\n\nExcerpt:\n{excerpt}\n\nSource file remains authoritative; this is a pointer/excerpt only."
        try:
            row = _remember_block(
                repo_root=repo_root,
                backend=backend,
                workspace=workspace,
                memory_id=f"docptr_{_slug(workspace, 32)}_{_slug(str(rel), 90)}",
                title=f"Doc pointer: {title}",
                content=content,
                kind="doc_pointer",
                tags=["doc-pointer", f"workspace:{workspace}", f"path:{rel}"],
                metadata={
                    "source": "project_docs_index",
                    "project_path": str(root),
                    "path": str(path),
                    "relative_path": str(rel),
                    "memory_activity": "active",
                },
                evidence_refs=[
                    EvidenceRef(
                        ref_type="file",
                        ref_id=str(path),
                        produced_by="hermes_lethyx_index_project_docs",
                    )
                ],
                lifecycle="active",
                surface="supplemental_evidence",
                tier="warm",
            )
            rows.append(row)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return MigrationResult(
        ok=not errors,
        count=len(rows),
        saved=sum(1 for row in rows if row.get("ok")),
        rows=rows,
        errors=errors,
    )


def link_source_edges(
    *, repo_root: str | Path, backend: str, workspaces: list[str]
) -> MigrationResult:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for workspace in workspaces:
        client = _client(repo_root, workspace, backend)
        try:
            blocks = list(client.backend.export_blocks() or [])
        finally:
            close = getattr(client.backend, "close", None)
            if callable(close):
                close()
        source_nodes: dict[str, str] = {}
        for source_name, title in [
            ("sessions", "Hermes session summaries"),
            ("skills", "Hermes skill pointers"),
            ("mcp", "Hermes MCP inventories"),
            ("docs", "Project document pointers"),
        ]:
            node = ensure_source_node(
                repo_root=repo_root,
                backend=backend,
                workspace=workspace,
                source_name=source_name,
                title=title,
                content=f"Source node for {source_name} in workspace {workspace}.",
            )
            source_nodes[source_name] = str(
                node.get("memory_id") or f"source_{workspace}_{source_name}"
            )
        for block in blocks:
            block_id = getattr(block, "block_id", "")
            kind = getattr(block, "kind", "")
            if not block_id or str(block_id).startswith("source_"):
                continue
            if kind == "session_summary" or str(block_id).startswith("session_"):
                dst = source_nodes["sessions"]
            elif kind in {"skill_index"}:
                dst = source_nodes["skills"]
            elif kind in {"mcp_inventory"}:
                dst = source_nodes["mcp"]
            elif kind in {"doc_pointer", "repo_capsule", "code_chunk"}:
                dst = source_nodes["docs"]
            else:
                continue
            try:
                row = _add_edge(
                    repo_root=repo_root,
                    backend=backend,
                    workspace=workspace,
                    src_id=block_id,
                    dst_id=dst,
                    edge_type="derived_from",
                    reason=f"{kind or block_id} derived from {dst}",
                )
                rows.append(
                    {"workspace": workspace, "src": block_id, "dst": dst, "ok": row.get("ok", True)}
                )
            except Exception as exc:
                errors.append(f"{workspace}/{block_id}: {type(exc).__name__}: {exc}")
    return MigrationResult(
        ok=not errors,
        count=len(rows),
        saved=sum(1 for row in rows if row.get("ok", True)),
        rows=rows,
        errors=errors,
    )


def _export_obsidian_workspace(
    *,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    vault_dir: str | Path,
    project_title: str = "",
) -> dict[str, Any]:
    from lethyx.memory.obsidian_export import export_backend_to_obsidian

    client = _client(repo_root, workspace, backend)
    try:
        result = export_backend_to_obsidian(
            backend=client.backend,
            vault_dir=vault_dir,
            workspace_id=workspace,
            project_title=project_title or workspace,
            include_global=False,
        )
        return {
            "ok": True,
            "workspace": workspace,
            "index_path": str(result.index_path),
            "files": len(result.written_files),
            "blocks": len(result.block_paths),
        }
    finally:
        close = getattr(client.backend, "close", None)
        if callable(close):
            close()


def bootstrap_project(
    *,
    hermes_home: str | Path,
    repo_root: str | Path,
    backend: str,
    workspace: str,
    project_path: str | Path,
    profiles: list[str] | None = None,
    session_limit: int = 20,
    doc_limit: int = 80,
    vault_dir: str | Path | None = None,
    export: bool = False,
) -> MigrationResult:
    """Bootstrap a real Hermes project memory surface into Lethyx.

    This is the high-level path users expect when they enable Lethyx: bounded
    Hermes MEMORY/USER entries, project rule/doc pointers, and relevant session
    summaries all become typed Lethyx records.  Exact raw sessions remain in
    SQLite/FTS5 and source files remain authoritative.
    """

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    def add_stage(stage: str, result: MigrationResult | dict[str, Any]) -> None:
        payload = result.to_json() if hasattr(result, "to_json") else dict(result)
        payload["stage"] = stage
        rows.append(payload)
        if not payload.get("ok", False):
            for error in payload.get("errors") or []:
                errors.append(f"{stage}: {error}")
            if payload.get("error"):
                errors.append(f"{stage}: {payload['error']}")

    selected_profiles = profiles or ["default"]
    add_stage(
        "builtins",
        import_builtin_memories(
            hermes_home=hermes_home,
            repo_root=repo_root,
            backend=backend,
            workspace=workspace,
            profiles=selected_profiles,
            include_global_user=True,
        ),
    )
    add_stage(
        "project_docs",
        index_project_docs(
            repo_root=repo_root,
            backend=backend,
            workspace=workspace,
            project_path=project_path,
            limit=doc_limit,
        ),
    )
    if session_limit:
        add_stage(
            "sessions",
            migrate_project_sessions(
                hermes_home=hermes_home,
                repo_root=repo_root,
                backend=backend,
                workspace=workspace,
                project_path=project_path,
                profiles=selected_profiles,
                limit=session_limit,
            ),
        )
    add_stage(
        "source_edges",
        link_source_edges(repo_root=repo_root, backend=backend, workspaces=[workspace]),
    )
    if export and vault_dir:
        add_stage(
            "obsidian_export",
            _export_obsidian_workspace(
                repo_root=repo_root,
                backend=backend,
                workspace=workspace,
                vault_dir=vault_dir,
                project_title=workspace,
            ),
        )

    saved = 0
    count = 0
    for row in rows:
        if isinstance(row.get("saved"), int):
            saved += int(row.get("saved") or 0)
        if isinstance(row.get("count"), int):
            count += int(row.get("count") or 0)
    return MigrationResult(ok=not errors, count=count, saved=saved, rows=rows, errors=errors)


def export_static_graph_viewer(
    *, repo_root: str | Path, backend: str, workspace: str, output_path: str | Path
) -> MigrationResult:
    client = _client(repo_root, workspace, backend)
    try:
        graph = client.export_graph()
    finally:
        close = getattr(client.backend, "close", None)
        if callable(close):
            close()
    blocks = []
    for block in graph.get("blocks") or []:
        metadata = dict(getattr(block, "metadata", {}) or {})
        blocks.append(
            {
                "id": getattr(block, "block_id", ""),
                "title": getattr(block, "title", ""),
                "kind": getattr(block, "kind", ""),
                "tier": metadata.get("tier") or getattr(block, "tier", ""),
                "lifecycle": getattr(block, "lifecycle", ""),
                "surface": getattr(block, "surface", ""),
                "trust": getattr(block, "trust_level", ""),
                "tags": list(getattr(block, "tags", ()) or ()),
                "source": metadata.get("source", ""),
                "content": getattr(block, "content", ""),
                "metadata": metadata,
            }
        )
    raw_edges = [
        {
            "src": getattr(edge, "src_id", ""),
            "dst": getattr(edge, "dst_id", ""),
            "type": getattr(edge, "edge_type", ""),
            "reason": getattr(edge, "reason", ""),
        }
        for edge in graph.get("edges") or []
    ]
    seen_edges: set[tuple[str, str, str, str]] = set()
    edges = []
    for edge in raw_edges:
        key = (str(edge["src"]), str(edge["dst"]), str(edge["type"]), str(edge["reason"]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(edge)
    payload = {"workspace": workspace, "backend": backend, "blocks": blocks, "edges": edges}
    json_blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html_text = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Lethyx Graph - {html.escape(workspace)}</title>
<style>
body{{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:#0b1020;color:#e7ecff}}header{{padding:16px 22px;background:#111936;border-bottom:1px solid #26345f}}h1{{margin:0;font-size:20px}}#controls{{padding:12px 22px;display:flex;gap:10px;flex-wrap:wrap}}input,select{{background:#111936;color:#e7ecff;border:1px solid #334777;border-radius:8px;padding:8px}}#layout{{display:grid;grid-template-columns:1fr 420px;gap:12px;padding:0 22px 22px}}#graph{{min-height:720px;background:#080d1d;border:1px solid #26345f;border-radius:14px;overflow:hidden}}#details{{background:#111936;border:1px solid #26345f;border-radius:14px;padding:14px;max-height:720px;overflow:auto}}.node{{cursor:pointer}}.badge{{display:inline-block;margin:2px;padding:2px 6px;border-radius:999px;background:#26345f;color:#b8c7ff;font-size:11px}}pre{{white-space:pre-wrap;color:#d8e0ff}}line{{stroke:#445c99;stroke-width:1.2;opacity:.55}}circle{{stroke:#dbe6ff;stroke-width:1.5}}text{{fill:#eef3ff;font-size:11px;pointer-events:none}}</style></head>
<body><header><h1>Lethyx Graph · {html.escape(workspace)} · {len(blocks)} blocks · {len(edges)} edges</h1></header>
<div id=\"controls\"><input id=\"q\" placeholder=\"filter title/content/tags\"><select id=\"tier\"><option value=\"\">all tiers</option></select><select id=\"life\"><option value=\"\">all lifecycle</option></select></div>
<div id=\"layout\"><svg id=\"graph\"></svg><aside id=\"details\">Click a node.</aside></div>
<script>const DATA={json_blob};
const svg=document.getElementById('graph'), details=document.getElementById('details'), q=document.getElementById('q'), tier=document.getElementById('tier'), life=document.getElementById('life');
function addOption(sel,v){{const o=document.createElement('option'); o.value=v; o.textContent=v; sel.appendChild(o);}}
for (const v of [...new Set(DATA.blocks.map(b=>b.tier).filter(Boolean))].sort()) addOption(tier,v);
for (const v of [...new Set(DATA.blocks.map(b=>b.lifecycle).filter(Boolean))].sort()) addOption(life,v);
function color(b){{if(b.lifecycle==='archived')return '#667085'; if(b.tier==='hot')return '#ffb454'; if(b.tier==='cold')return '#5b7cff'; if(b.kind==='source_index')return '#7ee787'; return '#9d7bff';}}
function filtered(){{const s=q.value.toLowerCase();return DATA.blocks.filter(b=>(!tier.value||b.tier===tier.value)&&(!life.value||b.lifecycle===life.value)&&(!s||JSON.stringify(b).toLowerCase().includes(s)));}}
function el(tag,txt,cls){{const n=document.createElement(tag); if(cls)n.className=cls; if(txt!==undefined)n.textContent=String(txt); return n;}}
function badge(txt){{return el('span',txt,'badge');}}
function draw(){{const nodes=filtered(); const ids=new Set(nodes.map(n=>n.id)); const edges=DATA.edges.filter(e=>ids.has(e.src)&&ids.has(e.dst)); const W=svg.clientWidth||900,H=svg.clientHeight||720,cx=W/2,cy=H/2,R=Math.max(180,Math.min(W,H)/2-70); svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`); svg.replaceChildren(); const pos={{}}; nodes.forEach((n,i)=>{{const a=2*Math.PI*i/Math.max(nodes.length,1);pos[n.id]=[cx+R*Math.cos(a),cy+R*Math.sin(a)];}}); for(const e of edges){{const a=pos[e.src],b=pos[e.dst]; if(!a||!b)continue; const line=document.createElementNS('http://www.w3.org/2000/svg','line'); line.setAttribute('x1',String(a[0])); line.setAttribute('y1',String(a[1])); line.setAttribute('x2',String(b[0])); line.setAttribute('y2',String(b[1])); const title=document.createElementNS('http://www.w3.org/2000/svg','title'); title.textContent=(e.type||'')+': '+(e.reason||''); line.appendChild(title); svg.appendChild(line);}} nodes.forEach(n=>{{const p=pos[n.id]; const label=(n.title||n.id).slice(0,34); const g=document.createElementNS('http://www.w3.org/2000/svg','g'); g.setAttribute('class','node'); const c=document.createElementNS('http://www.w3.org/2000/svg','circle'); c.setAttribute('cx',String(p[0])); c.setAttribute('cy',String(p[1])); c.setAttribute('r','12'); c.setAttribute('fill',color(n)); const t=document.createElementNS('http://www.w3.org/2000/svg','text'); t.setAttribute('x',String(p[0]+16)); t.setAttribute('y',String(p[1]+4)); t.textContent=label; g.appendChild(c); g.appendChild(t); g.onclick=()=>show(n); svg.appendChild(g);}});}}
function show(b){{details.replaceChildren(); details.appendChild(el('h2',b.title||b.id)); for(const v of [b.kind,b.tier,b.lifecycle,b.surface,b.trust].filter(Boolean)) details.appendChild(badge(v)); details.appendChild(el('h3','Tags')); const tags=el('p'); for(const t of (b.tags||[])) tags.appendChild(badge(t)); details.appendChild(tags); details.appendChild(el('h3','Content')); details.appendChild(el('pre',b.content||'')); details.appendChild(el('h3','Metadata')); details.appendChild(el('pre',JSON.stringify(b.metadata,null,2)));}}
q.oninput=tier.onchange=life.onchange=draw; draw();</script></body></html>"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return MigrationResult(
        ok=True,
        count=len(blocks),
        saved=len(edges),
        output_path=str(out),
        rows=[{"blocks": len(blocks), "edges": len(edges), "path": str(out)}],
    )
