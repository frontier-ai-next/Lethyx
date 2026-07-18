"""Tree helpers for containment/path hierarchy."""

from pathlib import Path

from lethyx.memory.graph.models import GraphEdge, GraphNode
from lethyx.memory.graph.store import MemoryGraphStore


def add_tree_node(
    drive_root: Path | None,
    node_id: str,
    parent_id: str,
    path: str,
    node_type: str = "MemoryTreeNode",
) -> dict[str, object]:
    store = MemoryGraphStore(drive_root)
    result = store.add_node(
        GraphNode(node_id=node_id, node_type=node_type, parent_id=parent_id, path=path)
    )
    if parent_id:
        store.add_edge(GraphEdge(parent_id, node_id, "contains", "tree_parent"))
        store.add_edge(GraphEdge(node_id, parent_id, "child_of", "tree_parent"))
    return result


def children(drive_root: Path | None, parent_id: str) -> list[str]:
    return [
        edge.dst_id
        for edge in MemoryGraphStore(drive_root).edges()
        if edge.src_id == parent_id and edge.edge_type == "contains"
    ]
