"""Normalize agent-authored memory into taxonomy/block route decisions."""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from um_mem.memory.authoring.models import MemoryRouteDecision, MemoryWriteProposal
from um_mem.memory.authoring.policy import route_by_taxonomy
from um_mem.memory.blocks.catalog import compile_semantic_catalog
from um_mem.memory.blocks.registry import load_memory_blocks


def route_memory_write(
    repo_root: Path,
    write: MemoryWriteProposal,
    *,
    profile_id: str = "default",
) -> MemoryRouteDecision:
    blocks = load_memory_blocks(
        Path(repo_root),
        write.workspace_id,
        profile_id=profile_id,
        strict=True,
    )
    decision = route_by_taxonomy(write, blocks)
    return replace(
        decision,
        catalog_revision_id=compile_semantic_catalog(blocks).revision_id,
    )


def write_authoring_event(
    drive_root: Path, write: MemoryWriteProposal, decision: MemoryRouteDecision
) -> None:
    state = Path(drive_root) / "state"
    state.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "write": asdict(write),
        "decision": asdict(decision),
    }
    (state / "memory_authoring_report_latest.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (state / "memory_authoring_events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
