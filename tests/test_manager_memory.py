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


def _manager_client(root: Path) -> MemoryClient:
    return MemoryClient(root, workspace_id=MANAGER_WORKSPACE_ID, backend="mempalace")


def _writable_leaf(client: MemoryClient, name: str) -> str:
    root_block = next(
        block for block in client.block_map() if block["index_only"] and not block["parent_id"]
    )
    leaf = client.create_leaf_block(
        name=name,
        synopsis=f"{name} synopsis",
        parent_id=root_block["block_id"],
        reason="manager layer test",
    )
    return leaf["block_id"]


def _remember_rule(client: MemoryClient, text: str, *, block_name: str = "Store rules") -> None:
    client.remember(
        text,
        title="rule",
        requested_block_id=_writable_leaf(client, block_name),
    )


def test_manager_identity_uses_the_store_memory_root(tmp_path: Path) -> None:
    assert is_manager_workspace(MANAGER_WORKSPACE_ID)
    assert not is_manager_workspace("default")
    assert workspace_memory_root(tmp_path, MANAGER_WORKSPACE_ID) == manager_memory_root(tmp_path)
    assert _index_path(tmp_path, MANAGER_WORKSPACE_ID) == (
        manager_memory_root(tmp_path) / "mempalace" / "UM_MEM_index.sqlite"
    )


def test_manager_memory_is_read_only_on_explicit_opt_in(tmp_path: Path) -> None:
    manager = _manager_client(tmp_path)
    _remember_rule(manager, "Manager rule: run focused tests before claiming a fix.")

    workspace = MemoryClient(tmp_path, workspace_id="alpha", backend="mempalace")

    assert workspace.recall("focused tests", search_all=True, include_candidates=True) == []

    manager_hits = workspace.recall(
        "focused tests",
        search_all=True,
        include_candidates=True,
        include_manager=True,
    )
    assert len(manager_hits) == 1
    assert manager_hits[0].metadata["memory_level"] == "manager"
    assert "Manager rule" in manager_hits[0].text


def test_workspace_memory_never_leaks_into_the_manager_layer(tmp_path: Path) -> None:
    workspace = MemoryClient(tmp_path, workspace_id="alpha", backend="mempalace")
    _remember_rule(
        workspace, "Alpha secret: the parser lives in src/parser.py.", block_name="Alpha"
    )

    manager = _manager_client(tmp_path)
    assert manager.recall("parser lives in src", search_all=True, include_candidates=True) == []


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
    manager = _manager_client(tmp_path)
    _remember_rule(manager, "Manager rule: never cross workspace boundaries implicitly.")

    workspace = MemoryClient(tmp_path, workspace_id="alpha", backend="mempalace")
    items = workspace.build_context(
        "cross workspace boundaries",
        search_all=True,
        include_candidates=True,
        include_manager=True,
    )

    assert any(item["metadata"].get("memory_level") == "manager" for item in items)


def test_include_manager_rejects_an_untyped_backend_instance(tmp_path: Path) -> None:
    backend = create_durable_backend(tmp_path, workspace_id="alpha", mode="mempalace")
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="explicit backend mode"):
        client.recall("anything", search_all=True, include_manager=True)
