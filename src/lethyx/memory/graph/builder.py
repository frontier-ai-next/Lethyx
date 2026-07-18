"""Convenience graph builders for memory events and BKB rules."""

from pathlib import Path
from typing import Any

from lethyx.memory.graph.models import GraphEdge, GraphNode
from lethyx.memory.graph.store import MemoryGraphStore


def add_memory_event(event: Any, canonical_id: str, drive_root: Path | None) -> dict[str, Any]:
    node = GraphNode(
        node_id=canonical_id,
        node_type="MemoryEventNode",
        store=str(getattr(event, "metadata", {}).get("palace_store", ""))
        if getattr(event, "metadata", None)
        else "",
        tier=str(getattr(event, "tier", "")),
        scope=str(getattr(event, "scope", "")),
        surface=str(getattr(event, "surface", "supplemental_evidence")),
        lifecycle=str(getattr(event, "lifecycle", "active")),
        trust_level=str(getattr(event, "trust_level", "agent_claim")),
        workspace_id=str(getattr(event, "workspace_id", "")),
        run_id=str(getattr(event, "run_id", "")),
        phase_id=str(getattr(event, "phase_id", "")),
    )
    store = MemoryGraphStore(drive_root)
    result = store.add_node(node)
    for ref in getattr(event, "evidence_refs", ()) or ():
        ref_id = str(
            getattr(ref, "ref_id", "") or (ref.get("ref_id") if isinstance(ref, dict) else "")
        )
        if ref_id:
            store.add_edge(
                GraphEdge(
                    src_id=ref_id, dst_id=canonical_id, edge_type="supports", reason="evidence_ref"
                )
            )
    return result


def add_bkb_rule(rule: Any, drive_root: Path | None) -> dict[str, Any]:
    node = GraphNode(
        node_id=str(getattr(rule, "id", "")),
        node_type="BKBRuleNode",
        surface="directive",
        lifecycle=str(getattr(rule, "status", "active")),
        trust_level=str(getattr(rule, "trust", "verified")),
    )
    return MemoryGraphStore(drive_root).add_node(node)


def add_evidence_ref(ref: dict[str, Any], drive_root: Path | None) -> dict[str, Any]:
    node_id = str(ref.get("ref_id") or "")
    node = GraphNode(
        node_id=node_id,
        node_type="EvidenceRefNode",
        trust_level=str(ref.get("trust") or ""),
        metadata=dict(ref),
    )
    return MemoryGraphStore(drive_root).add_node(node)


def link_support(src: str, dst: str, drive_root: Path | None, reason: str = "") -> dict[str, Any]:
    return MemoryGraphStore(drive_root).add_edge(GraphEdge(src, dst, "supports", reason))


def link_contradiction(a: str, b: str, drive_root: Path | None, reason: str = "") -> dict[str, Any]:
    store = MemoryGraphStore(drive_root)
    result = store.add_edge(GraphEdge(a, b, "contradicts", reason, weight=1.0))
    store.add_edge(GraphEdge(b, a, "contradicts", reason, weight=1.0))
    return result
