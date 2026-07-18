"""Normalize agent-authored memory into taxonomy/block route decisions."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from lethyx.memory.authoring.models import MemoryRouteDecision, MemoryWriteProposal
from lethyx.memory.authoring.policy import route_by_taxonomy
from lethyx.memory.blocks.registry import load_memory_blocks


def route_memory_write(repo_root: Path, write: MemoryWriteProposal) -> MemoryRouteDecision:
    blocks = load_memory_blocks(Path(repo_root), write.workspace_id)
    return route_by_taxonomy(write, blocks)


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
