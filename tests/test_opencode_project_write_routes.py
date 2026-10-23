from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from um_mem.memory import mcp_server
from um_mem.memory.blocks.registry import load_memory_blocks
from um_mem.memory.mcp_server import _OPENCODE_PROJECT_WRITE_TWINS, _route_opencode_project_write
from um_mem.memory.scopes import ScopeLineage


@pytest.fixture
def blocks() -> list:
    return load_memory_blocks(Path(__file__).resolve().parents[1])


def test_project_twins_preserve_all_policy_except_scope_identity(blocks: list) -> None:
    by_id = {block.block_id: block for block in blocks}
    for source_id, twin_id in _OPENCODE_PROJECT_WRITE_TWINS.items():
        source = by_id[source_id]
        twin = by_id[twin_id]
        assert source.scope == "workspace"
        assert twin.scope == "project"
        assert twin.isolation_policy == {"dimension": "project", "cross_workspace": False}
        assert twin == replace(
            source,
            block_id=twin.block_id,
            name=twin.name,
            scope="project",
            isolation_policy=twin.isolation_policy,
        )


def test_opencode_remaps_only_supported_workspace_leaves(blocks: list) -> None:
    for source_id, twin_id in _OPENCODE_PROJECT_WRITE_TWINS.items():
        routed = _route_opencode_project_write(
            {"requested_block_id": source_id},
            enabled=True,
            project_id="local-project-test",
            blocks=blocks,
        )
        assert routed == twin_id

    for special_id in (
        "user_global_preferences",
        "user_global_core_directives",
        "task_state",
        "execution_working_state",
        "transient_logs",
    ):
        assert (
            _route_opencode_project_write(
                {"requested_block_id": special_id},
                enabled=True,
                project_id="local-project-test",
                blocks=blocks,
            )
            == special_id
        )

    with pytest.raises(ValueError, match="create an explicit project-scoped block"):
        _route_opencode_project_write(
            {"requested_block_id": "workspace_core_directives"},
            enabled=True,
            project_id="local-project-test",
            blocks=blocks,
        )


def test_read_only_scope_flag_does_not_enable_write_routing(blocks: list) -> None:
    assert (
        _route_opencode_project_write(
            {"requested_block_id": "stable_project_facts"},
            enabled=False,
            project_id="local-project-test",
            blocks=blocks,
        )
        == ""
    )


def test_project_only_write_requires_explicit_route(blocks: list) -> None:
    with pytest.raises(ValueError, match="require an explicit requested_block_id"):
        _route_opencode_project_write(
            {}, enabled=True, project_id="local-project-test", blocks=blocks
        )


def test_mcp_remember_routes_project_only_and_replay_keeps_same_target(
    blocks: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = SimpleNamespace(
        repo_root=Path(__file__).resolve().parents[1],
        workspace_id="test-workspace",
        profile_id="default",
        backend="sqlite",
    )
    routed: list[dict] = []
    monkeypatch.setenv("UM_MEM_WRITE_SCOPE_PROJECT_ONLY", "1")
    monkeypatch.delenv("UM_MEM_ACTIVE_SCOPE_PROJECT_ONLY", raising=False)
    monkeypatch.setattr(mcp_server, "build_local_service_identity", lambda **_: identity)
    monkeypatch.setattr(
        mcp_server,
        "local_service_scope_lineage",
        lambda *args, **kwargs: ScopeLineage(
            workspace_id="test-workspace", project_id="local-project-test"
        ),
    )
    monkeypatch.setattr(mcp_server, "load_memory_blocks", lambda *args, **kwargs: blocks)
    monkeypatch.setattr(
        mcp_server,
        "_service_call",
        lambda _identity, _lineage, name, args: routed.append(dict(args)) or {"ok": True},
    )
    monkeypatch.setattr(mcp_server, "_await_projection", lambda result, *_: result)

    args = {"requested_block_id": "reflection_candidates", "operation_id": "stable-replay"}
    assert mcp_server._call_tool("remember", args)["ok"] is True
    assert mcp_server._call_tool("remember", args)["ok"] is True
    assert [call["requested_block_id"] for call in routed] == [
        "project_scope_reflection_candidates",
        "project_scope_reflection_candidates",
    ]
    assert [call["operation_id"] for call in routed] == ["stable-replay", "stable-replay"]

    routed.clear()
    refusal = mcp_server._call_tool(
        "remember",
        {
            "requested_block_id": "stable_project_facts",
            "operation": "correct",
            "memory_id": "legacy-workspace-record",
            "operation_id": "legacy-correction",
        },
    )
    assert refusal["ok"] is False
    assert "will not retarget a correction or repair" in refusal["error"]
    assert routed == []

    monkeypatch.setenv("UM_MEM_ACTIVE_SCOPE_PROJECT_ONLY", "1")
    monkeypatch.setenv("UM_MEM_WRITE_SCOPE_PROJECT_ONLY", "0")
    assert (
        mcp_server._call_tool(
            "remember",
            {"requested_block_id": "stable_project_facts", "operation_id": "read-only-flag"},
        )["ok"]
        is True
    )
    assert routed[0]["requested_block_id"] == "stable_project_facts"
