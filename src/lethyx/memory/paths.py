"""Filesystem layout for per-workspace memory isolation.

Workspace-scoped memory lives under ``workspaces/<id>/.memory/``.
Manager (cross-workspace) memory stays under ``.Lethyx/memory/``.
Each workspace gets its own MemPalace Chroma directory under ``.../.memory/palace/``.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lethyx.memory.store import MemoryStore


def _safe_workspace_segment(workspace_id: str) -> str:
    """Return a single workspace directory name (never ``workspaces/<id>``)."""
    raw = (workspace_id or "").strip().replace("\\", "/").strip("/")
    while raw.casefold().startswith("./"):
        raw = raw[2:].strip("/")
    if raw.casefold().startswith(".workspaces/"):
        raw = raw[1:].strip("/")
    parts = [part for part in raw.split("/") if part]
    if parts and parts[0].casefold() == "workspaces":
        parts = parts[1:]
    if not parts:
        return ""
    seg = parts[0]
    if ".." in Path(seg).parts:
        raise ValueError("workspace_id must not contain path traversal")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", seg).strip("._-")


def normalize_workspace_id(workspace_id: str) -> str:
    """Public alias for :func:`_safe_workspace_segment`."""
    return _safe_workspace_segment(workspace_id)


def parse_palace_path_hint(
    palace_path: str,
    *,
    workspace_id: str = "",
    default_kind: str = "observation",
) -> tuple[str, str, str]:
    """Parse ``palace_path`` into ``(workspace_id, event_type, room)``.

    Accepts repo-relative ``workspaces/<id>/research``, workspace-relative
    ``research/plan``, or mistaken ``workspaces/<id>/.memory/...`` without
    treating those segments as extra on-disk directories under ``.memory``.
    """
    ws = _safe_workspace_segment(workspace_id)
    text = str(palace_path or "").strip().replace("\\", "/").strip("/")
    while text.startswith("./"):
        text = text[2:].strip("/")
    if text.casefold().startswith(".workspaces/"):
        text = text[1:].strip("/")
    if ws:
        for prefix in (f"workspaces/{ws}/", f"{ws}/", "workspaces/"):
            while text.casefold().startswith(prefix.casefold()):
                text = text[len(prefix) :].strip("/")
        while text.casefold().startswith(f"{ws}/".casefold()):
            text = text[len(f"{ws}/") :].strip("/")
    elif text.casefold().startswith("workspaces/"):
        parts = text.split("/")
        if len(parts) >= 2:
            ws = _safe_workspace_segment(parts[1])
            text = "/".join(parts[2:]).strip("/")
    if text.casefold().startswith(".memory/"):
        text = text[len(".memory/") :].strip("/")
    elif text.casefold() == ".memory":
        text = ""
    if not text:
        return ws, default_kind, ""
    parts = text.split("/")
    event_type = parts[0] or default_kind
    room = "/".join(parts) if parts else ""
    return ws, event_type, room


def workspace_memory_root(repo_root: Path, workspace_id: str) -> Path:
    """``repo_root/workspaces/<id>/.memory/`` (resolved)."""
    seg = _safe_workspace_segment(workspace_id)
    if not seg:
        raise ValueError("workspace_id is required for workspace_memory_root")
    return (repo_root / "workspaces" / seg / ".memory").resolve()


def manager_memory_root(repo_root: Path) -> Path:
    """``repo_root/.Lethyx/memory/`` — manager lessons, gaps, signals, ideas."""
    return (repo_root / ".Lethyx" / "memory").resolve()


def manager_core_root(repo_root: Path) -> Path:
    """``repo_root/.Lethyx/memory/core/`` — always-loaded manager core memory."""
    return (manager_memory_root(repo_root) / "core").resolve()


def workspace_core_root(repo_root: Path, workspace_id: str) -> Path:
    """``workspaces/<id>/.memory/core/`` — always-loaded workspace core memory."""
    return (workspace_memory_root(repo_root, workspace_id) / "core").resolve()


def workspace_root(repo_root: Path, workspace_id: str) -> Path:
    """``repo_root/workspaces/<id>/`` — workspace project root (not .memory)."""
    seg = _safe_workspace_segment(workspace_id)
    if not seg:
        raise ValueError("workspace_id is required for workspace_root")
    return (repo_root / "workspaces" / seg).resolve()


def manager_palace_root(repo_root: Path) -> Path:
    """Cross-workspace MemPalace (Chroma) under ``.Lethyx/palace/``."""
    return (repo_root / ".Lethyx" / "palace").resolve()


def palace_path_for(repo_root: Path, workspace_id: str) -> Path:
    """Per-workspace palace dir, or manager palace when ``workspace_id`` is empty."""
    seg = _safe_workspace_segment(workspace_id)
    if not seg:
        return manager_palace_root(repo_root)
    return (workspace_memory_root(repo_root, seg) / "palace").resolve()


def hierarchical_root_for_palace(palace_dir: Path | str) -> Path:
    """Directory that holds ``ideas.jsonl`` for the given MemPalace data dir."""
    p = Path(palace_dir).resolve()
    if p.name != "palace":
        return p
    parent = p.parent
    if parent.name == ".memory":
        return parent
    # ``.Lethyx/palace`` -> ``.Lethyx/memory``
    return parent / "memory"


def get_workspace_store(repo_root: Path, workspace_id: str) -> "MemoryStore":
    """Return a :class:`MemoryStore` for manager (``workspace_id == ""``) or a workspace."""
    from lethyx.memory.models import MemoryConfig
    from lethyx.memory.store import MemoryStore

    seg = _safe_workspace_segment(workspace_id)
    if not seg:
        root = manager_memory_root(repo_root)
    else:
        root = workspace_memory_root(repo_root, seg)
    root.mkdir(parents=True, exist_ok=True)
    return MemoryStore(
        MemoryConfig(
            memory_root=root,
            lessons_path=root / "lessons.jsonl",
            gaps_path=root / "gaps.jsonl",
            signals_path=root / "signals.jsonl",
        )
    )


__all__ = [
    "get_workspace_store",
    "hierarchical_root_for_palace",
    "manager_core_root",
    "manager_memory_root",
    "manager_palace_root",
    "normalize_workspace_id",
    "palace_path_for",
    "parse_palace_path_hint",
    "workspace_core_root",
    "workspace_memory_root",
    "workspace_root",
]
