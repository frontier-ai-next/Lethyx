"""Small JSONL graph store; no-op safe when drive root is absent."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lethyx.memory.graph.models import GraphEdge, GraphNode


class MemoryGraphStore:
    def __init__(self, drive_root: Path | None):
        self.drive_root = Path(drive_root) if drive_root is not None else None

    @property
    def graph_dir(self) -> Path | None:
        if self.drive_root is None:
            return None
        path = self.drive_root / "graph"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add_node(self, node: GraphNode) -> dict[str, Any]:
        graph_dir = self.graph_dir
        if graph_dir is None:
            return {"ok": False, "skipped": True, "reason": "drive_root_unavailable"}
        with (graph_dir / "nodes.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(node), ensure_ascii=False, sort_keys=True) + "\n")
        self._write_index()
        return {"ok": True, "node_id": node.node_id}

    def add_edge(self, edge: GraphEdge) -> dict[str, Any]:
        graph_dir = self.graph_dir
        if graph_dir is None:
            return {"ok": False, "skipped": True, "reason": "drive_root_unavailable"}
        with (graph_dir / "edges.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(edge), ensure_ascii=False, sort_keys=True) + "\n")
        self._write_index()
        return {"ok": True, "src_id": edge.src_id, "dst_id": edge.dst_id}

    def nodes(self) -> list[GraphNode]:
        rows: list[GraphNode] = []
        graph_dir = self.graph_dir
        if graph_dir is None:
            return rows
        path = graph_dir / "nodes.jsonl"
        if not path.is_file():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(
                    GraphNode(
                        **{k: v for k, v in data.items() if k in GraphNode.__dataclass_fields__}
                    )
                )
        return rows

    def edges(self) -> list[GraphEdge]:
        rows: list[GraphEdge] = []
        graph_dir = self.graph_dir
        if graph_dir is None:
            return rows
        path = graph_dir / "edges.jsonl"
        if not path.is_file():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(
                    GraphEdge(
                        **{k: v for k, v in data.items() if k in GraphEdge.__dataclass_fields__}
                    )
                )
        return rows

    def _write_index(self) -> None:
        graph_dir = self.graph_dir
        if graph_dir is None:
            return
        payload = {
            "schema_version": "1",
            "node_count": len(self.nodes()),
            "edge_count": len(self.edges()),
        }
        (graph_dir / "index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
