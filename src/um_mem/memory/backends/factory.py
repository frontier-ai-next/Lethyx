"""Factory helpers for selectable durable memory backends."""

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ALLOWED_BACKENDS = ("mempalace", "hindsight", "mnemosyne", "zvec")


def allowed_memory_backend_modes() -> tuple[str, ...]:
    return _ALLOWED_BACKENDS


def selected_memory_backend_mode(mode: str | None = None) -> str:
    if mode is not None and not isinstance(mode, str):
        raise ValueError("memory backend mode must be a string")
    if os.getenv("UM_MEM_MEMORY_BACKEND") is not None:
        raise ValueError("UM_MEM_MEMORY_BACKEND is not supported; use UM_MEM_BACKEND")
    selected = (mode or os.getenv("UM_MEM_BACKEND") or "").strip().lower()
    if not selected:
        selected = "mempalace"
    if selected not in _ALLOWED_BACKENDS:
        raise ValueError(f"unknown memory backend: {selected}")
    return selected


def _create_single_backend(
    root: Path,
    *,
    workspace_id: str,
    selected: str,
    reconcile_pending_writes: bool,
) -> Any:
    if selected == "mempalace":
        from um_mem.memory.backends.mempalace import MemPalaceBackend

        return MemPalaceBackend(
            repo_root=root,
            workspace_id=workspace_id,
            reconcile_pending_writes=reconcile_pending_writes,
        )
    if selected == "hindsight":
        from um_mem.memory.backends.hindsight import HindsightBackend

        return HindsightBackend.from_env(
            repo_root=root,
            workspace_id=workspace_id,
            reconcile_pending_writes=reconcile_pending_writes,
        )
    if selected == "mnemosyne":
        from um_mem.memory.backends.mnemosyne import MnemosyneBackend

        return MnemosyneBackend.from_env(
            repo_root=root,
            workspace_id=workspace_id,
            reconcile_pending_writes=reconcile_pending_writes,
        )
    if selected == "zvec":
        from um_mem.memory.backends.zvec import ZvecBackend

        return ZvecBackend(
            repo_root=root,
            workspace_id=workspace_id,
            reconcile_pending_writes=reconcile_pending_writes,
        )
    raise ValueError(f"unknown single memory backend: {selected}")


def create_durable_backend(
    repo_root: Path,
    *,
    workspace_id: str = "",
    mode: str | None = None,
    reconcile_pending_writes: bool = True,
) -> Any:
    """Create the selected durable memory backend."""
    selected = selected_memory_backend_mode(mode)
    return _create_single_backend(
        Path(repo_root),
        workspace_id=workspace_id,
        selected=selected,
        reconcile_pending_writes=reconcile_pending_writes,
    )
