from dataclasses import replace
from pathlib import Path

import pytest

from um_mem import MemoryClient
from um_mem.memory.backends.factory import create_durable_backend
from um_mem.memory.backends.mempalace import _index_path
from um_mem.memory.paths import (
    MANAGER_WORKSPACE_ID,
    is_manager_workspace,
    manager_memory_root,
    workspace_memory_root,
)


def _manager_client(root: Path, *, user_id: str = "") -> MemoryClient:
    return MemoryClient(
        root, workspace_id=MANAGER_WORKSPACE_ID, user_id=user_id, backend="mempalace"
    )


def _writable_leaf(client: MemoryClient, name: str, *, scope: str = "workspace") -> str:
    root_block = next(
        block for block in client.block_map() if block["index_only"] and not block["parent_id"]
    )
    leaf = client.create_leaf_block(
        name=name,
        synopsis=f"{name} synopsis",
        parent_id=root_block["block_id"],
        reason="manager layer test",
        scope=scope,
    )
    return leaf["block_id"]


def _remember_rule(
    client: MemoryClient, text: str, *, block_name: str = "Store rules", scope: str = "workspace"
) -> None:
    client.remember(
        text,
        title="rule",
        requested_block_id=_writable_leaf(client, block_name, scope=scope),
    )


def test_manager_identity_uses_the_store_memory_root(tmp_path: Path) -> None:
    assert is_manager_workspace(MANAGER_WORKSPACE_ID)
    assert not is_manager_workspace("default")
    assert workspace_memory_root(tmp_path, MANAGER_WORKSPACE_ID) == manager_memory_root(tmp_path)
    assert _index_path(tmp_path, MANAGER_WORKSPACE_ID) == (
        manager_memory_root(tmp_path) / "mempalace" / "UM_MEM_index.sqlite"
    )


def test_manager_memory_is_read_only_on_explicit_opt_in(tmp_path: Path) -> None:
    manager = _manager_client(tmp_path, user_id="alice")
    _remember_rule(
        manager, "Manager rule: run focused tests before claiming a fix.", scope="user_global"
    )

    workspace = MemoryClient(tmp_path, workspace_id="alpha", user_id="alice", backend="mempalace")

    assert workspace.recall("focused tests", search_all=True, include_candidates=True) == []

    manager_hits = workspace.recall(
        "focused tests",
        search_all=True,
        include_candidates=True,
        include_manager=True,
    )
    assert len(manager_hits) == 1
    assert manager_hits[0].metadata["memory_level"] == "manager"
    assert manager_hits[0].metadata["authority"] == "advisory"
    assert "Manager rule" in manager_hits[0].text


def test_manager_opt_in_never_crosses_user_or_workspace_scope(tmp_path: Path) -> None:
    manager = _manager_client(tmp_path, user_id="alice")
    _remember_rule(manager, "Private manager-only filing note")
    _remember_rule(
        manager,
        "Alice global preference: use focused tests",
        block_name="Alice preferences",
        scope="user_global",
    )

    alice = MemoryClient(tmp_path, workspace_id="alpha", user_id="alice", backend="mempalace")
    bob = MemoryClient(tmp_path, workspace_id="beta", user_id="bob", backend="mempalace")
    anonymous = MemoryClient(tmp_path, workspace_id="gamma", backend="mempalace")
    alice_hits = alice.recall(
        "manager tests filing", search_all=True, include_candidates=True, include_manager=True
    )
    assert len(alice_hits) == 1
    assert "Alice global preference" in alice_hits[0].text
    assert (
        bob.recall("focused tests", search_all=True, include_candidates=True, include_manager=True)
        == []
    )
    assert (
        anonymous.recall(
            "focused tests", search_all=True, include_candidates=True, include_manager=True
        )
        == []
    )
    assert (
        alice.recall(
            "focused tests",
            search_all=True,
            scope_chain=("workspace:alpha",),
            include_candidates=True,
            include_manager=True,
        )
        == []
    )


def test_manager_block_must_opt_in_to_cross_workspace_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from um_mem import api

    manager = _manager_client(tmp_path, user_id="alice")
    _remember_rule(manager, "Alice global preference: use focused tests", scope="user_global")
    blocks = api.load_memory_blocks(tmp_path, MANAGER_WORKSPACE_ID)
    monkeypatch.setattr(
        api,
        "load_memory_blocks",
        lambda *args, **kwargs: [
            replace(
                block,
                isolation_policy={"dimension": "user_global", "cross_workspace": False},
            )
            if block.scope == "user_global"
            else block
            for block in blocks
        ],
    )

    workspace = MemoryClient(tmp_path, workspace_id="alpha", user_id="alice", backend="mempalace")
    assert (
        workspace.recall(
            "focused tests", search_all=True, include_candidates=True, include_manager=True
        )
        == []
    )


def test_manager_results_do_not_exceed_recall_limit(tmp_path: Path) -> None:
    manager = _manager_client(tmp_path, user_id="alice")
    _remember_rule(manager, "Use focused tests across all projects", scope="user_global")
    workspace = MemoryClient(tmp_path, workspace_id="alpha", user_id="alice", backend="mempalace")
    _remember_rule(workspace, "Use focused tests inside alpha")

    hits = workspace.recall(
        "focused tests", search_all=True, include_candidates=True, include_manager=True, limit=1
    )
    assert len(hits) == 1
    assert hits[0].metadata.get("memory_level") is None


def test_workspace_memory_never_leaks_into_the_manager_layer(tmp_path: Path) -> None:
    workspace = MemoryClient(tmp_path, workspace_id="alpha", backend="mempalace")
    _remember_rule(
        workspace, "Alpha secret: the parser lives in src/parser.py.", block_name="Alpha"
    )

    manager = _manager_client(tmp_path)
    assert manager.recall("parser lives in src", search_all=True, include_candidates=True) == []


def test_user_global_record_in_a_workspace_is_not_auto_copied_to_manager(tmp_path: Path) -> None:
    alpha = MemoryClient(tmp_path, workspace_id="alpha", user_id="alice", backend="mempalace")
    _remember_rule(alpha, "Alpha global preference: focus tests", scope="user_global")
    beta = MemoryClient(tmp_path, workspace_id="beta", user_id="alice", backend="mempalace")

    assert alpha.recall(
        "focus tests",
        search_all=True,
        scope_chain=("user_global:alice",),
        include_candidates=True,
    )
    assert (
        beta.recall("focus tests", search_all=True, include_candidates=True, include_manager=True)
        == []
    )


def test_manager_client_does_not_merge_itself(tmp_path: Path) -> None:
    manager = _manager_client(tmp_path)
    _remember_rule(manager, "Manager rule: keep the store layer advisory.")

    hits = manager.recall(
        "store layer advisory",
        search_all=True,
        include_candidates=True,
        include_manager=True,
    )
    assert hits
    assert all(hit.metadata.get("memory_level") is None for hit in hits)


def test_build_context_include_manager_marks_the_layer(tmp_path: Path) -> None:
    manager = _manager_client(tmp_path, user_id="alice")
    _remember_rule(
        manager, "Manager rule: never cross workspace boundaries implicitly.", scope="user_global"
    )

    workspace = MemoryClient(tmp_path, workspace_id="alpha", user_id="alice", backend="mempalace")
    items = workspace.build_context(
        "cross workspace boundaries",
        search_all=True,
        include_candidates=True,
        include_manager=True,
    )

    assert any(item["metadata"].get("memory_level") == "manager" for item in items)


def test_include_manager_rejects_an_untyped_backend_instance(tmp_path: Path) -> None:
    backend = create_durable_backend(tmp_path, workspace_id="alpha", mode="mempalace")
    client = MemoryClient(tmp_path, workspace_id="alpha", user_id="alice", backend_instance=backend)

    with pytest.raises(ValueError, match="explicit backend mode"):
        client.recall("anything", search_all=True, include_manager=True)
