"""Canonical eligibility and session isolation across shared memory clients."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest
from test_memory_semantic import FakeEmbedder
from test_record_dependency_invalidation import _correct, _record

from um_mem import MemoryClient
from um_mem.contracts import RecordBlockMembership
from um_mem.memory import local_tier, mcp_facade, mcp_server, semantic
from um_mem.memory.backends.contracts import EdgeType, MemoryEdgeRecord
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


@pytest.fixture
def memory(tmp_path):
    client = MemoryClient(
        tmp_path,
        workspace_id="shared-safety",
        backend="mnemosyne",
        session_id="test-session",
        task_id="test-task",
    )
    root = client.block_map()[0]["block_id"]
    block = client.create_leaf_block(
        name="Configuration", synopsis="Configuration facts", parent_id=root, reason="Test"
    )
    yield client, block["block_id"]
    client.backend.close()


def test_index_loader_rejects_old_heads_and_retractions(memory):
    client, leaf = memory
    first = client.remember("Old configuration", requested_block_id=leaf)
    original = client.read_exact_record(record_id=first.memory_id)
    amended = client.amend(
        first.memory_id,
        expected_revision=original.revision_id,
        content="New configuration",
        reason="Changed",
    )
    loader = mcp_facade._record_loader(client)
    assert [
        r["record_id"]
        for r in loader([first.memory_id, amended["new_id"], "missing", amended["new_id"]])
    ] == [amended["new_id"]]
    client.forget(amended["new_id"])
    assert loader([first.memory_id, amended["new_id"]]) == []
    assert client.read_exact_record(record_id=first.memory_id).content == "Old configuration"


@pytest.mark.parametrize("name", ["recall", "build_context"])
@pytest.mark.parametrize(
    "filters",
    [{"tags": ["different"]}, {"include_candidates": False}, {"source_speakers": ["Someone else"]}],
)
def test_global_index_obeys_filters_even_when_recent_window_is_empty(
    memory, monkeypatch, name, filters
):
    client, leaf = memory
    saved = client.remember(
        "The config is in the global folder",
        title="Config",
        requested_block_id=leaf,
        tags=["configuration"],
    )
    record = client.read_exact_record(record_id=saved.memory_id)
    embedder = FakeEmbedder()
    monkeypatch.setenv("UM_MEM_SEMANTIC", "1")
    monkeypatch.setattr(semantic, "get_embedder", lambda **kwargs: embedder)
    index = semantic.SemanticIndex(
        semantic.semantic_index_path(client.repo_root, client.workspace_id)
    )
    semantic.embed_record(
        index, embedder, record_id=record.block_id, title=record.title, content=record.content
    )
    arguments = {
        "query": "где конфигурация",
        "limit": 4,
        "include_candidates": True,
        "scope_chain": [f"workspace:{client.workspace_id}"],
    }

    def retrieve(options):
        return local_tier.apply_local_tier(
            {"ok": True, "hits": [], "items": []},
            name=name,
            arguments=options,
            records_provider=list,
            record_loader=mcp_facade._record_loader(client, options),
            repo_root=client.repo_root,
            workspace_id=client.workspace_id,
        )

    assert retrieve(arguments)["fallback"] == "semantic"
    denied = retrieve({**arguments, **filters})
    assert not denied.get("hits") and not denied.get("items")


def test_scoped_fallback_respects_task_scope(memory, monkeypatch):
    client, leaf = memory
    client.remember("The config is in the global folder", requested_block_id=leaf)
    monkeypatch.setenv("UM_MEM_SEMANTIC", "0")
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda value: None)
    for function in (mcp_facade.um_mem_recall, mcp_facade.um_mem_build_context):
        result = function(
            "config",
            logical_block_ids=[leaf],
            scope_chain=["task:test-task"],
            include_candidates=True,
        )
        assert result["ok"] is True
        assert not result.get("hits") and not result.get("items")
    allowed = mcp_facade.um_mem_recall(
        "config",
        logical_block_ids=[leaf],
        scope_chain=[f"workspace:{client.workspace_id}"],
        include_candidates=True,
    )
    assert allowed["hits"]


def test_mcp_processes_share_records_but_have_independent_masks(memory, monkeypatch):
    client, leaf = memory
    environment = {**os.environ, "UM_MEM_SESSION_ID": "host-mcp"}
    command = [
        sys.executable,
        "-c",
        "from um_mem.memory.mcp_server import _session_id; print(_session_id())",
    ]
    identities = [
        subprocess.check_output(command, env=environment, text=True, timeout=30).strip()
        for _ in range(2)
    ]
    assert len(set(identities)) == 2
    first = MemoryClient(
        client.repo_root,
        workspace_id=client.workspace_id,
        backend="mnemosyne",
        session_id=identities[0],
    )
    second = MemoryClient(
        client.repo_root,
        workspace_id=client.workspace_id,
        backend="mnemosyne",
        session_id=identities[1],
    )
    try:
        one = first.set_active_mask(
            selected_block_ids=(leaf,), token_budget=1000, activation_reason="First client"
        )
        assert second.get_active_mask() is None
        two = second.set_active_mask(
            selected_block_ids=(leaf,), token_budget=1000, activation_reason="Second client"
        )
        assert one.mask_id != two.mask_id
        assert first.block_map() == second.block_map()
    finally:
        first.backend.close()
        second.backend.close()
    monkeypatch.setenv("UM_MEM_SESSION_ID", "host-owned-session")
    assert mcp_server._session_id() == "host-owned-session"


def test_vector_cache_isolated_between_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic.time, "strftime", lambda *args: "same-second")
    first = semantic.SemanticIndex(tmp_path / "first.sqlite")
    second = semantic.SemanticIndex(tmp_path / "second.sqlite")
    first.store("model", "default", {"one|full": ("first", [1.0, 0.0])})
    second.store("model", "default", {"two|full": ("second", [0.0, 1.0])})
    assert set(semantic.cached_vectors(first, "model", "default")) == {"one"}
    assert set(semantic.cached_vectors(second, "model", "default")) == {"two"}


def test_vector_cache_observes_same_second_replacements(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic.time, "strftime", lambda *args: "same-second")
    index = semantic.SemanticIndex(tmp_path / "index.sqlite")
    index.store("model", "default", {"one|full": ("old", [1.0, 0.0])})
    assert semantic.cached_vectors(index, "model", "default")["one"]["full"][0] == "old"
    # A separate index handle represents another client or process.
    writer = semantic.SemanticIndex(index.path)
    writer.store("model", "default", {"one|full": ("new", [0.0, 1.0])})
    assert semantic.cached_vectors(index, "model", "default")["one"]["full"][0] == "new"
    writer.delete("model", "default", "one")
    writer.store("model", "default", {"two|full": ("other", [1.0, 1.0])})
    assert set(semantic.cached_vectors(index, "model", "default")) == {"two"}


def test_all_fallback_sources_revalidate_derived_dependencies(tmp_path, monkeypatch):
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
    try:
        root = client.block_map()[0]["block_id"]
        leaf = client.create_leaf_block(
            name="Findings", synopsis="Derived configuration", parent_id=root, reason="Test"
        )["block_id"]
        source, derived = [
            replace(
                _record(name),
                content="Config location is /old/config",
                block_membership=RecordBlockMembership(
                    primary_block_id=leaf, catalog_revision_id="catalog:fixture"
                ),
            )
            for name in ("source", "derived")
        ]
        for record in (source, derived):
            index.retain_block(record)
        index.add_edge(
            MemoryEdgeRecord(
                src_id="derived",
                dst_id="source",
                edge_type=EdgeType.DERIVED_FROM,
                workspace_id="alpha",
            )
        )
        options = {
            "query": "config",
            "include_candidates": True,
            "scope_chain": ["workspace:alpha"],
        }
        monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
        monkeypatch.setattr(mcp_facade, "_close_client", lambda value: None)
        monkeypatch.setattr(
            mcp_server,
            "_service_call",
            lambda identity, lineage, name, arguments: {
                "recent": mcp_facade.um_mem_recent,
                "recall": mcp_facade.um_mem_recall,
            }[name](**arguments),
        )
        providers = [
            lambda: mcp_facade._local_window(client, options),
            lambda: mcp_facade._scoped_window(client, [leaf], options),
            lambda: mcp_facade._record_loader(client, options)(["derived"]),
            lambda: mcp_server._fallback_records(None, None, options),
            lambda: mcp_server._fallback_records(
                None, None, {**options, "logical_block_ids": [leaf], "retrieval_mode": "hybrid"}
            ),
        ]
        for provider in providers:
            assert "derived" in {r.get("record_id") or r.get("block_id") for r in provider()}
        _correct(index, source)
        assert client.validate_dependencies("derived").status == "invalidated"
        for provider in providers:
            assert "derived" not in {r.get("record_id") or r.get("block_id") for r in provider()}
    finally:
        index.close()
