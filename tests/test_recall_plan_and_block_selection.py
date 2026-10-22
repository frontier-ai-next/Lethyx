from pathlib import Path

import pytest

from um_mem.memory.backends.query_policy import normalize_query_payload
from um_mem.memory.blocks.registry import load_memory_blocks
from um_mem.memory.recall_plan import RecallPlan


def test_recall_plan_payload_keeps_logical_blocks_and_record_ids_separate() -> None:
    plan = RecallPlan(
        query="python parser",
        workspace_id="alpha",
        run_id="run-1",
        logical_block_ids=("code_resources",),
        record_ids=("record-1",),
        tags=("python",),
        source_horizon="archive",
        include_candidates=True,
        scope_chain=("workspace:alpha", "user_global:user-1"),
        max_tokens=512,
        limit=4,
        reason="test",
    )

    payload = plan.to_backend_payload()

    assert set(payload) == {
        "query",
        "normalized_query",
        "query_type",
        "retrieval_mode",
        "workspace_id",
        "profile_id",
        "run_id",
        "logical_block_ids",
        "search_all",
        "record_ids",
        "paths",
        "symbols",
        "entities",
        "tags",
        "source_horizon",
        "include_candidates",
        "scope_chain",
        "temporal_mode",
        "as_of",
        "known_as_of",
        "revision_id",
        "max_tokens",
        "limit",
    }
    assert payload["query"] == "python parser"
    assert payload["profile_id"] == ""
    assert payload["logical_block_ids"] == ["code_resources"]
    assert payload["search_all"] is False
    assert payload["record_ids"] == ["record-1"]
    assert payload["source_horizon"] == "archive"
    assert payload["include_candidates"] is True
    assert payload["scope_chain"] == ["workspace:alpha", "user_global:user-1"]
    assert payload["limit"] == 4
    assert payload["max_tokens"] == 512


def test_recall_plan_mapping_requires_canonical_fields_and_types() -> None:
    plan = RecallPlan.from_mapping(
        {
            "query": "parser",
            "source_horizon": "working",
            "include_candidates": True,
            "scope_chain": ["workspace:alpha"],
            "max_tokens": 0,
            "limit": 1,
        }
    )

    assert plan.source_horizon == "working"
    assert plan.include_candidates is True
    assert plan.scope_chain == ("workspace:alpha",)
    assert (plan.max_tokens, plan.limit) == (0, 1)

    for payload, message in (
        ({"query": "parser", "source_horizon": "false"}, "source_horizon"),
        ({"query": "parser", "limit": True}, "limit"),
        ({"query": "parser", "unknown": "value"}, "unknown RecallPlan fields"),
    ):
        with pytest.raises(ValueError, match=message):
            RecallPlan.from_mapping(payload)


def test_recall_plan_rejects_legacy_question_alias() -> None:
    with pytest.raises(ValueError, match="unknown RecallPlan fields: question"):
        RecallPlan.from_mapping({"question": "parser"})


def test_recall_plan_rejects_invalid_history_timestamps_at_the_public_boundary() -> None:
    with pytest.raises(ValueError, match="as_of"):
        RecallPlan(query="previous result", temporal_mode="historical", as_of="not-a-time")


@pytest.mark.parametrize(
    ("alias", "value"),
    [
        ("n", 3),
        ("token_budget", 128),
        ("block_ids", ["record-1"]),
        ("selected_block_ids", ["record-1"]),
        ("selected_blocks", ["record-1"]),
        ("logical_blocks", ["code_resources"]),
        ("include_passive", True),
        ("passive_memory", True),
        ("include_global", True),
        ("include_cross_workspace", True),
    ],
)
def test_native_query_payload_rejects_recall_plan_aliases(alias: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"unknown recall query fields: {alias}"):
        normalize_query_payload({"query": "parser", alias: value})


def test_recall_plan_keeps_agent_selected_blocks_and_source_horizon() -> None:
    plan = RecallPlan.compile(
        "episodic workspace event memory",
        workspace_id="alpha",
        logical_block_ids=("workspace_episodes",),
        source_horizon="durable",
    )

    assert plan.logical_block_ids == ("workspace_episodes",)
    assert plan.source_horizon == "durable"


def test_recall_plan_requires_search_all_to_be_explicit_and_exclusive() -> None:
    plan = RecallPlan.compile("all memory", workspace_id="alpha", search_all=True)

    assert plan.search_all is True
    assert plan.logical_block_ids == ()
    with pytest.raises(ValueError, match="cannot be combined"):
        RecallPlan.compile(
            "all memory",
            workspace_id="alpha",
            logical_block_ids=("workspace_episodes",),
            search_all=True,
        )


@pytest.mark.parametrize(
    "routing_yaml",
    [
        "unexpected_field: true",
        "retrieval_policy: {unexpected_field: true}",
    ],
)
def test_block_registry_rejects_unknown_fields(tmp_path: Path, routing_yaml: str) -> None:
    config = tmp_path / ".UM-MEM" / "memory" / "blocks.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"""blocks:
  - block_id: local_evidence
    block_type: evidence
    temperature: warm
    scope: workspace
    authority: advisory
    delivery_policy: on_demand
    retention: durable
    {routing_yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid block"):
        load_memory_blocks(tmp_path, workspace_id="alpha", strict=True)


def test_default_memory_blocks_form_a_consistent_hierarchical_index(tmp_path: Path) -> None:
    blocks = load_memory_blocks(tmp_path, workspace_id="alpha")
    by_id = {block.block_id: block for block in blocks}
    root = by_id["memory_root_index"]

    assert root.index_only is True
    assert root.parent_id == ""
    assert set(root.child_ids) == {
        "directive_memory_index",
        "working_memory_index",
        "durable_knowledge_index",
        "semantic_memory_index",
        "archive_memory_index",
    }
    for child_id in root.child_ids:
        child = by_id[child_id]
        assert child.index_only is True
        assert child.parent_id == root.block_id
        for leaf_id in child.child_ids:
            assert by_id[leaf_id].parent_id == child.block_id

    assert set(by_id["working_memory_index"].child_ids) == {
        "task_state",
        "execution_working_state",
    }
    assert by_id["task_state"].block_type == "typed_task_state"
    assert by_id["execution_working_state"].block_type == "execution_working_state"
    assert set(by_id["semantic_memory_index"].child_ids) == {
        "user_global_preferences",
        "project_resources",
        "code_resources",
        "stable_project_facts",
        "mutable_project_state",
        "decisions_and_contracts",
        "artifact_claims",
        "workspace_episodes",
    }
    assert {
        "external_memory_index",
        "github_repo_capsules",
        "github_code_chunks",
        "workspace_facts",
        "current_run_state",
        "subtask_scratchpad",
        "hindsight_candidates",
        "mnemosyne_workspace_episodic",
    }.isdisjoint(by_id)
