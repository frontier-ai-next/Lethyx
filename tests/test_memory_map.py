"""Coverage for the self-contained memory graph."""

from __future__ import annotations

from lethyx.memory import memory_map

_DATA = {
    "workspace_id": "opencode",
    "profile_id": "default",
    "backend": "mnemosyne",
    "tree": [
        {
            "block_id": "root",
            "name": "Memory",
            "synopsis": "root",
            "records": 0,
            "writable": False,
            "index_only": True,
            "children": [
                {
                    "block_id": "leaf",
                    "name": "OpenCode workspace setup",
                    "synopsis": "facts",
                    "records": 2,
                    "writable": True,
                    "index_only": False,
                    "children": [],
                }
            ],
        }
    ],
    "records": [
        {
            "record_id": "mem_fact",
            "title": "OpenCode config locations",
            "content": "Global config lives in the config folder.",
            "kind": "project_fact",
            "tags": ["opencode"],
            "status": "candidate",
            "recorded_at": "2026-10-20T00:00:00Z",
            "block": "leaf",
            "steps": [],
            "freshness": "",
        },
        {
            "record_id": "mem_procedure",
            "title": "Restart the local service",
            "content": "Restart after editing the source.",
            "kind": "procedure",
            "tags": ["lethyx"],
            "status": "candidate",
            "recorded_at": "2026-10-21T00:00:00Z",
            "block": "leaf",
            "steps": ["edit the source", "restart the service"],
            "freshness": "fresh",
        },
    ],
    "record_count": 2,
    "indexed_records": 2,
    "semantic_model": "intfloat/multilingual-e5-large",
    "kinds": {"project_fact": 1, "procedure": 1},
    "edges": [
        {"source": "mem_fact", "target": "mem_procedure", "kind": "semantic", "weight": 0.91}
    ],
    "generated_at": "2026-10-22T00:00:00+00:00",
}


def test_memory_graph_renders_one_self_contained_page() -> None:
    page = memory_map.render_memory_map(_DATA)
    assert page.startswith("<!doctype html>")
    assert "Lethyx memory graph" in page
    assert "OpenCode config locations" in page
    assert "Restart the local service" in page
    assert "edit the source" in page
    assert "intfloat/multilingual-e5-large" in page
    # Manual layout and inline block previews are part of the page contract.
    assert "drag a card to move it" in page
    assert "function resetLayout" in page
    assert "node.subtext" in page
    assert "data-reset" in page
    # The page must work offline: no external stylesheet, script or image.
    assert "src=\"http" not in page
    assert "href=\"http" not in page
    assert "@import" not in page


def test_memory_graph_links_blocks_records_and_similar_records() -> None:
    graph = memory_map._graph_data(_DATA)
    identifiers = {node["id"] for node in graph["nodes"]}
    assert {"root", "leaf", "mem_fact", "mem_procedure"} <= identifiers
    kinds = {link["kind"] for link in graph["links"]}
    assert "tree" in kinds
    assert "membership" in kinds
    assert "semantic" in kinds
    fact = next(node for node in graph["nodes"] if node["id"] == "mem_fact")
    assert fact["type"] == "record"
    assert fact["parent"] == "leaf"
    assert "config" in fact["hay"]


def test_memory_map_summary_reports_the_essentials() -> None:
    summary = memory_map.memory_map_summary(_DATA)
    assert '"records": 2' in summary
    assert '"indexed": 2' in summary
    assert "procedure" in summary
    assert '"semantic_links": 1' in summary
