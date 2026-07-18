"""Factory helpers for selectable durable memory backends."""

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ALLOWED_BACKENDS = ("mempalace", "hindsight", "mnemosyne", "composite")
_COMPOSITE_ALIASES = {"blocks", "memory-blocks", "memory_blocks", "block", "memory-block"}
_REMOVED_SINGLE_BACKEND_ALIASES = {"canonical"}
_REMOVED_MULTI_BACKEND_MODES = {
    "dual",
    "dual_" + "mnemosyne",
    "canonical" + "+mnemosyne",
    "canonical" + "+hindsight",
}


def allowed_memory_backend_modes() -> tuple[str, ...]:
    return _ALLOWED_BACKENDS


def selected_memory_backend_mode(mode: str | None = None) -> str:
    selected = (
        str(mode or os.getenv("LETHYX_BACKEND") or os.getenv("LETHYX_MEMORY_BACKEND") or "")
        .strip()
        .lower()
    )
    if not selected:
        selected = "mempalace"
    if selected in _COMPOSITE_ALIASES:
        selected = "composite"
    if selected in _REMOVED_SINGLE_BACKEND_ALIASES:
        raise ValueError("memory backend 'canonical' was removed; use 'mempalace'")
    if selected in _REMOVED_MULTI_BACKEND_MODES:
        raise ValueError("legacy dual memory modes were removed; use 'composite' memory blocks")
    if selected not in _ALLOWED_BACKENDS:
        raise ValueError(f"unknown memory backend: {selected}")
    return selected


def _create_single_backend(root: Path, *, workspace_id: str, selected: str) -> Any:
    if selected == "mempalace":
        from lethyx.memory.backends.mempalace import MemPalaceBackend

        return MemPalaceBackend(repo_root=root, workspace_id=workspace_id)
    if selected == "hindsight":
        from lethyx.memory.backends.hindsight import HindsightBackend

        return HindsightBackend.from_env(repo_root=root, workspace_id=workspace_id)
    if selected == "mnemosyne":
        from lethyx.memory.backends.mnemosyne import MnemosyneBackend

        return MnemosyneBackend.from_env(repo_root=root, workspace_id=workspace_id)
    raise ValueError(f"unknown single memory backend: {selected}")


def create_durable_backend(
    repo_root: Path,
    *,
    workspace_id: str = "",
    mode: str | None = None,
) -> Any:
    """Create the selected durable memory backend.

    Single modes return one concrete technology adapter.  ``composite`` returns
    a phase-routed wrapper over multiple concrete memory technology blocks.
    """
    selected = selected_memory_backend_mode(mode)
    root = Path(repo_root)
    if selected != "composite":
        return _create_single_backend(root, workspace_id=workspace_id, selected=selected)

    from lethyx.memory.backends.composite import (
        CompositeMemoryBackend,
        build_memory_blocks_from_specs,
        memory_block_specs_from_env,
    )

    def backend_factory(backend_id: str) -> Any:
        return _create_single_backend(root, workspace_id=workspace_id, selected=backend_id)

    blocks = build_memory_blocks_from_specs(
        memory_block_specs_from_env(),
        backend_factory=backend_factory,
    )
    return CompositeMemoryBackend(root, workspace_id=workspace_id, blocks=blocks)


# Backward-compatible alias for callers that import a generic creator name.
def create_memory_backend(*args: Any, **kwargs: Any) -> Any:
    return create_durable_backend(*args, **kwargs)
