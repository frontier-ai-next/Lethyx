"""Filesystem layout for per-workspace memory isolation.

Workspace-scoped memory lives under ``workspaces/<id>/.memory/``.
Manager (cross-workspace) memory stays under ``.Lethyx/memory/``.
Each workspace gets its own MemPalace Chroma directory under ``.../.memory/palace/``.
"""

import re
from pathlib import Path


def normalize_workspace_id(workspace_id: str) -> str:
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


def workspace_memory_root(repo_root: Path, workspace_id: str) -> Path:
    """``repo_root/workspaces/<id>/.memory/`` (resolved)."""
    seg = normalize_workspace_id(workspace_id)
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
    seg = normalize_workspace_id(workspace_id)
    if not seg:
        raise ValueError("workspace_id is required for workspace_root")
    return (repo_root / "workspaces" / seg).resolve()


def palace_path_for(repo_root: Path, workspace_id: str) -> Path:
    """Per-workspace palace dir, or manager palace when ``workspace_id`` is empty."""
    seg = normalize_workspace_id(workspace_id)
    if not seg:
        return (repo_root / ".Lethyx" / "palace").resolve()
    return (workspace_memory_root(repo_root, seg) / "palace").resolve()


__all__ = [
    "manager_core_root",
    "manager_memory_root",
    "normalize_workspace_id",
    "palace_path_for",
    "workspace_core_root",
    "workspace_memory_root",
    "workspace_root",
]
