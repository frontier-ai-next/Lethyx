import pytest

from um_mem import MemoryClient
from um_mem.memory.mcp_compact import compact_response


def _tree(tmp_path, count=7):
    client = MemoryClient(tmp_path, workspace_id="pages", backend_instance=object())
    root = client.block_map()[0]["block_id"]
    branch = client.create_index_block(
        name="Ideas", synopsis="Independently evolving ideas", parent_id=root, reason="test"
    )
    children = [
        client.create_index_block(
            name=f"Topic {index}",
            synopsis=f"Topic {index}",
            parent_id=branch["block_id"],
            reason="test",
        )
        for index in range(count)
    ]
    for child in children:
        client.create_leaf_block(
            name="Evidence", synopsis="Source evidence", parent_id=child["block_id"], reason="test"
        )
    return client, root, branch, children


def test_pages_cover_children_once_and_keep_branch_semantics(tmp_path, monkeypatch):
    client, root, branch, children = _tree(tmp_path)
    cursor = ""
    seen = []
    monkeypatch.setenv("UM_MEM_MCP_RESPONSE_MODE", "compact")
    while True:
        page = client.expand_block(branch["block_id"], limit=3, cursor=cursor)
        assert page["path_ids"] == [root, branch["block_id"]]
        assert page["child_count"] == 7
        assert len(page["children"]) <= 3
        assert page["root"]["child_ids"] == [child["block_id"] for child in page["children"]]
        for child in page["children"]:
            assert child["child_count"] == 1
            assert child["child_ids"] == []
            assert child["expandable"] is True
            assert child["writable"] is False
            assert child["next_cursor"]
        seen.extend(child["block_id"] for child in page["children"])
        compact = compact_response("expand_block", {"ok": True, **page})
        assert compact["next_cursor"] == page["next_cursor"]
        assert compact["catalog_revision"] == page["catalog_revision"]
        assert compact["children"][0]["child_count"] == 1
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert seen == [child["block_id"] for child in children]


def test_navigation_cursor_rejects_stale_revision_scope_and_branch(tmp_path):
    client, root, branch, children = _tree(tmp_path, count=3)
    cursor = client.expand_block(branch["block_id"], limit=1)["next_cursor"]
    with pytest.raises(ValueError, match="scope"):
        client.expand_block(root, limit=1, cursor=cursor)
    other = MemoryClient(tmp_path, workspace_id="other", backend_instance=object())
    with pytest.raises(ValueError, match="scope"):
        other.expand_block(branch["block_id"], limit=1, cursor=cursor)
    client.amend_block(
        children[0]["block_id"], expected_revision=1, name="Corrected topic", reason="correction"
    )
    with pytest.raises(ValueError, match="revision"):
        client.expand_block(branch["block_id"], limit=1, cursor=cursor)


def test_root_map_limits_children_and_has_a_bounded_deep_view(tmp_path):
    client, root, _branch, _children = _tree(tmp_path)
    overview = client.block_map(depth=3, child_limit=2)[0]
    assert overview["block_id"] == root
    shown = overview["children"][0]
    assert shown["child_count"] == 7
    assert shown["children_truncated"] is True
    assert len(shown["children"]) == 2
    assert shown["next_cursor"]
    assert all(child["child_ids"] == [] for child in shown["children"])


@pytest.mark.parametrize("limit", [0, -1, True, 201])
def test_navigation_rejects_invalid_page_limits(tmp_path, limit):
    client = MemoryClient(tmp_path, workspace_id="pages", backend_instance=object())
    root = client.block_map()[0]["block_id"]
    with pytest.raises(ValueError, match="limit"):
        client.expand_block(root, limit=limit)


def test_protected_meta_pages_and_cursor_restart(tmp_path):
    client = MemoryClient(tmp_path, workspace_id="pages", backend_instance=object())
    meta = next(root for root in client.block_map(include_meta=True) if root["meta"])
    first = client.expand_block(meta["block_id"], limit=1)
    assert len(first["children"]) == 1
    assert first["next_cursor"]
    second = client.expand_block(meta["block_id"], limit=1, cursor=first["next_cursor"])
    assert first["children"][0]["block_id"] != second["children"][0]["block_id"]
    mapped = next(
        root for root in client.block_map(include_meta=True, child_limit=1) if root["meta"]
    )
    from_map = client.expand_block(meta["block_id"], limit=1, cursor=mapped["next_cursor"])
    assert from_map["children"] == second["children"]
    with pytest.raises(ValueError, match="cursor"):
        client.expand_block(meta["block_id"], cursor="not*base64")


def test_mcp_pages_preserve_cursor_and_fail_explicitly_after_mutation(tmp_path, monkeypatch):
    from um_mem.memory import mcp_facade

    client, _root, branch, children = _tree(tmp_path, count=3)
    monkeypatch.setattr(mcp_facade, "_client", lambda **kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda client: None)
    page = mcp_facade.um_mem_expand_block(branch["block_id"], limit=1)
    assert page["ok"] is True
    assert len(page["children"]) == 1
    second = mcp_facade.um_mem_expand_block(branch["block_id"], limit=1, cursor=page["next_cursor"])
    assert second["children"][0]["block_id"] == children[1]["block_id"]
    client.amend_block(children[0]["block_id"], expected_revision=1, name="Changed", reason="test")
    stale = mcp_facade.um_mem_expand_block(branch["block_id"], limit=1, cursor=page["next_cursor"])
    assert stale["ok"] is False
    assert "revision" in stale["error"]


def test_deep_map_budget_discloses_unexpanded_branches(tmp_path):
    from dataclasses import replace
    from uuid import uuid4

    from um_mem.memory.catalog_store import WorkspaceCatalogStore

    client, _root, _branch, children = _tree(tmp_path, count=14)
    store = WorkspaceCatalogStore(tmp_path, "pages", "default")
    try:
        prototype = store.get_block(children[0]["block_id"]).block
        for parent in children:
            for _ in range(14):
                block_id = str(uuid4())
                store.create_block(
                    replace(prototype, block_id=block_id, parent_id=parent["block_id"]),
                    backend_partition_id=block_id,
                    actor="llm",
                    reason="budget fixture",
                )
    finally:
        store.close()
    nodes = client.block_map(depth=8, child_limit=200)
    seen = []
    while nodes:
        node = nodes.pop()
        seen.append(node)
        nodes.extend(node.get("children", []))
    assert len(seen) <= 201
    assert any(node["child_count"] and node["next_cursor"] for node in seen)
