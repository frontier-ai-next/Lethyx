"""Graph path queries."""

from pathlib import Path

from lethyx.memory.graph.models import GraphEdge
from lethyx.memory.graph.store import MemoryGraphStore


def _incoming(candidate_id: str, drive_root: Path | None, types: set[str]) -> list[GraphEdge]:
    return [
        edge
        for edge in MemoryGraphStore(drive_root).edges()
        if edge.dst_id == candidate_id and edge.edge_type in types
    ]


def find_supporting_paths(
    candidate_id: str, drive_root: Path | None, max_depth: int = 3
) -> list[list[str]]:
    return [
        [edge.src_id, edge.dst_id]
        for edge in _incoming(candidate_id, drive_root, {"supports", "verified_by"})
    ]


def find_contradiction_paths(
    candidate_id: str, drive_root: Path | None, max_depth: int = 3
) -> list[list[str]]:
    edges = [
        edge
        for edge in MemoryGraphStore(drive_root).edges()
        if candidate_id in {edge.src_id, edge.dst_id} and edge.edge_type == "contradicts"
    ]
    return [[edge.src_id, edge.dst_id] for edge in edges]


def find_superseded_rules(rule_id: str, drive_root: Path | None) -> list[str]:
    return [edge.src_id for edge in _incoming(rule_id, drive_root, {"superseded_by", "supersedes"})]
