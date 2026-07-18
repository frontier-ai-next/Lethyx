"""Export memory graph JSON."""

import json
from dataclasses import asdict
from pathlib import Path

from lethyx.memory.graph.store import MemoryGraphStore


def export_graph_json(drive_root: Path | None, output_path: Path) -> Path:
    store = MemoryGraphStore(drive_root)
    payload = {
        "schema_version": "1",
        "nodes": [asdict(n) for n in store.nodes()],
        "edges": [asdict(e) for e in store.edges()],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
