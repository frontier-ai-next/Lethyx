import inspect
from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

from um_mem import MemoryClient
from um_mem.contracts import MemoryBlockRecord, MemoryScope
from um_mem.contracts.memory import memory_scope_key
from um_mem.memory.mcp_facade import um_mem_build_context, um_mem_recall
from um_mem.memory.recall_plan import RecallPlan


class _CapturingBackend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def recall(self, payload: dict[str, object]) -> list[object]:
        self.payloads.append(payload)
        return []


def test_memory_read_guidance_reaches_python_mcp_and_hermes() -> None:
    from um_mem.integrations.hermes_provider import UmMemMemoryProvider
    from um_mem.memory.mcp_server import create_mcp_server

    guidance = MemoryClient.agent_instructions()
    assert create_mcp_server().instructions == guidance
    assert guidance in UmMemMemoryProvider().system_prompt_block()
    assert "empty active mask does not mean" in guidance
    assert "refine the query" in guidance
    assert "explicit broad search" in guidance


def test_recall_plan_uses_only_canonical_selection_fields() -> None:
    field_names = {field.name for field in fields(RecallPlan)}

    assert {"record_ids", "source_horizon", "scope_chain"} <= field_names
    assert {"selected_block_ids", "include_passive", "include_global"}.isdisjoint(field_names)

    plan = RecallPlan.compile(
        "exact Parser record",
        workspace_id="workspace-a",
        scope_chain=("workspace:workspace-a", "project:project-a"),
        record_ids=("record-a",),
        source_horizon="archive",
    )

    assert plan.record_ids == ("record-a",)
    assert plan.source_horizon == "archive"
    assert plan.query_type == "exact_lookup"
    assert plan.to_backend_payload()["record_ids"] == ["record-a"]
    assert plan.to_backend_payload()["source_horizon"] == "archive"
    assert set(plan.to_backend_payload()).isdisjoint(
        {"selected_block_ids", "include_passive", "include_global"}
    )


@pytest.mark.parametrize(
    "legacy_field",
    ("selected_block_ids", "include_passive", "include_global"),
)
def test_recall_plan_rejects_retired_selection_fields(legacy_field: str) -> None:
    with pytest.raises(ValueError, match="unknown RecallPlan fields"):
        RecallPlan.from_mapping({"query": "parser", legacy_field: True})


def test_recall_plan_source_horizon_is_closed_and_defaults_to_durable() -> None:
    assert RecallPlan.compile("parser").source_horizon == "durable"

    with pytest.raises(ValueError, match="source_horizon"):
        RecallPlan.compile("parser", source_horizon="passive")


def test_memory_client_sends_the_canonical_plan_to_the_selected_backend(tmp_path) -> None:
    backend = _CapturingBackend()
    client = MemoryClient(tmp_path, workspace_id="workspace-a", backend_instance=backend)

    assert (
        client.recall(
            "parser",
            record_ids=("record-a",),
            search_all=True,
            source_horizon="archive",
            scope_chain=("workspace:workspace-a",),
        )
        == []
    )

    assert backend.payloads
    for payload in backend.payloads:
        assert payload["record_ids"] == ["record-a"]
        assert payload["source_horizon"] == "archive"
        assert payload["scope_chain"] == ["workspace:workspace-a"]
        assert set(payload).isdisjoint({"selected_block_ids", "include_passive", "include_global"})


def test_mcp_recall_and_context_expose_the_same_canonical_selection_fields() -> None:
    for function in (um_mem_recall, um_mem_build_context):
        parameters = inspect.signature(function).parameters
        assert {"record_ids", "source_horizon", "scope_chain"} <= set(parameters)
        assert {"selected_block_ids", "include_passive", "include_global"}.isdisjoint(parameters)


def test_memory_scope_is_canonical_and_user_global_requires_an_exact_user() -> None:
    assert set(get_args(MemoryScope)) == {
        "subtask",
        "task",
        "session",
        "workspace",
        "project",
        "user_global",
    }

    with pytest.raises(ValueError, match="user_global scope requires user_id"):
        MemoryBlockRecord(block_id="global", content="preference", scope="user_global")

    record = MemoryBlockRecord(
        block_id="global",
        content="preference",
        scope="user_global",
        user_id="user-a",
    )

    assert memory_scope_key(record) == "user_global:user-a"


def test_caller_supplied_logical_ids_remain_strict(tmp_path: Path) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=object())

    plan = client.compile_recall_plan(
        "celestial cobalt orchard",
        logical_block_ids=["strict-leaf"],
    )

    assert plan is not None
    assert plan.logical_block_ids == ("strict-leaf",)
    assert plan.reason == "explicit_block_first_recall"
    assert "routing_trace" not in plan.metadata
