"""Load built-in, manager, and workspace memory block definitions."""

from pathlib import Path
from typing import Any

import yaml

from lethyx.memory.blocks.models import MemoryBlock


def _coerce_block(raw: dict[str, Any]) -> MemoryBlock | None:
    if not isinstance(raw, dict) or not raw.get("block_id"):
        return None
    tags = raw.get("tags") or ()
    return MemoryBlock(
        block_id=str(raw.get("block_id") or ""),
        block_type=str(raw.get("block_type") or "generic"),
        store=str(raw.get("store") or "palace.idea"),
        tier=str(raw.get("tier") or "warm"),
        scope=str(raw.get("scope") or "cross_run_durable"),
        authority=str(raw.get("authority") or "passive_evidence"),
        default_surface=str(raw.get("default_surface") or "supplemental_evidence"),
        tags=tuple(str(t) for t in tags if str(t).strip()),
        owner=str(raw.get("owner") or "Lethyx"),
        applies_to=dict(raw.get("applies_to") or {}),
        retrieval_policy=dict(raw.get("retrieval_policy") or {}),
        graph_policy=dict(raw.get("graph_policy") or {}),
        promotion_policy=dict(raw.get("promotion_policy") or {}),
        token_budget=int(raw.get("token_budget") or 1000),
        lifecycle_policy=dict(raw.get("lifecycle_policy") or {}),
    )


def _load_yaml(path: Path) -> list[MemoryBlock]:
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    raw_blocks = data.get("blocks") if isinstance(data, dict) else data
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[MemoryBlock] = []
    for raw in raw_blocks:
        block = _coerce_block(raw)
        if block is not None:
            blocks.append(block)
    return blocks


def default_blocks_path() -> Path:
    return Path(__file__).with_name("default_blocks.yaml")


def load_memory_blocks(repo_root: Path, workspace_id: str = "") -> list[MemoryBlock]:
    sources = [default_blocks_path(), Path(repo_root) / ".Lethyx" / "memory" / "blocks.yaml"]
    if workspace_id:
        sources.append(Path(repo_root) / "workspaces" / workspace_id / ".memory" / "blocks.yaml")
    by_id: dict[str, MemoryBlock] = {}
    for source in sources:
        for block in _load_yaml(source):
            if block.block_id in by_id and source != default_blocks_path():
                current = by_id[block.block_id]
                # Workspace config may narrow or add blocks, but cannot upgrade passive to directive authority.
                if (
                    current.authority == "directive_candidate"
                    and block.authority != current.authority
                ):
                    continue
            by_id[block.block_id] = block
    return list(by_id.values())
