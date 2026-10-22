"""Filesystem layout for per-workspace and manager memory.

Workspace-scoped memory lives under ``workspaces/<id>/.memory/``.
The store-level manager layer (cross-workspace user memory) lives under
``.UM-MEM/memory/`` and is addressed by ``MANAGER_WORKSPACE_ID``.
Each workspace gets its own MemPalace Chroma directory under ``.../.memory/palace/``.
"""

import re
from pathlib import Path

MANAGER_WORKSPACE_ID = "um-mem-manager"


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
    """``repo_root/workspaces/<id>/.memory/`` (resolved).

    The reserved manager identity maps to the store-level memory root so the
    manager layer never becomes one more workspace directory.
    """

    seg = normalize_workspace_id(workspace_id)
    if not seg:
        raise ValueError("workspace_id is required for workspace_memory_root")
    if seg == MANAGER_WORKSPACE_ID:
        return manager_memory_root(repo_root)
    return (repo_root / "workspaces" / seg / ".memory").resolve()


def is_manager_workspace(workspace_id: str) -> bool:
    """Return whether this identity addresses the store-level manager layer."""

    return normalize_workspace_id(workspace_id) == MANAGER_WORKSPACE_ID


def manager_memory_root(repo_root: Path) -> Path:
    """``repo_root/.UM-MEM/memory/`` — manager lessons, gaps, signals, ideas."""
    return (repo_root / ".UM-MEM" / "memory").resolve()


def manager_core_root(repo_root: Path) -> Path:
    """``repo_root/.UM-MEM/memory/core/`` — always-loaded manager core memory."""
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
        return (repo_root / ".UM-MEM" / "palace").resolve()
    return (workspace_memory_root(repo_root, seg) / "palace").resolve()


__all__ = [
    "MANAGER_WORKSPACE_ID",
    "is_manager_workspace",
    "manager_core_root",
    "manager_memory_root",
    "normalize_workspace_id",
    "palace_path_for",
    "workspace_core_root",
    "workspace_memory_root",
    "workspace_root",
]
