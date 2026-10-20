import pytest

from lethyx.memory.recall_plan import RecallPlan
from lethyx.memory.scopes import (
    MemoryPrecedenceCandidate,
    OrderedScopeChain,
    ScopeLineage,
    order_by_memory_precedence,
)


def _lineage() -> ScopeLineage:
    return ScopeLineage(
        user_id="user-1",
        project_id="project-1",
        workspace_id="workspace-1",
        session_id="session-1",
        task_id="task-1",
        subtask_id="subtask-1",
    )


def test_ordered_scope_chain_is_narrow_to_broad_and_global_is_explicit() -> None:
    local = OrderedScopeChain.from_lineage(_lineage())
    shared = OrderedScopeChain.from_lineage(_lineage(), include_user_global=True)

    assert local.keys == (
        "subtask:subtask-1",
        "task:task-1",
        "session:session-1",
        "workspace:workspace-1",
        "project:project-1",
    )
    assert local.includes("user_global") is False
    assert shared.keys[-1] == "user_global:user-1"


def test_typed_lineage_rejects_missing_parent_identity() -> None:
    workspace_only = ScopeLineage(workspace_id="workspace-only")
    assert OrderedScopeChain.from_lineage(workspace_only).keys == ("workspace:workspace-only",)
    with pytest.raises(ValueError, match="task_id requires session_id"):
        ScopeLineage(workspace_id="workspace-1", task_id="task-orphan")
    with pytest.raises(ValueError, match="subtask_id requires task_id"):
        ScopeLineage(
            user_id="user-1",
            project_id="project-1",
            workspace_id="workspace-1",
            session_id="session-1",
            subtask_id="subtask-orphan",
        )
    with pytest.raises(ValueError, match="workspace_id is required"):
        ScopeLineage(workspace_id="", user_id="user-only")


def test_recall_plan_carries_canonical_ordered_scope_chain() -> None:
    chain = OrderedScopeChain.from_lineage(_lineage(), include_user_global=True)
    plan = RecallPlan.compile(
        query="release constraints",
        workspace_id="workspace-1",
        scope_chain=chain.keys,
    )

    assert plan.scope_chain == chain.keys
    assert plan.to_backend_payload()["scope_chain"] == list(chain.keys)


def test_workspace_evidence_precedes_explicit_and_inferred_global_memory() -> None:
    candidates = (
        MemoryPrecedenceCandidate(
            candidate_id="inferred-global",
            precedence_class="inferred_global_candidate",
            scope="user_global",
            verified=False,
            explicit=False,
            score=1.0,
        ),
        MemoryPrecedenceCandidate(
            candidate_id="explicit-global",
            precedence_class="explicit_user_global_preference",
            scope="user_global",
            verified=True,
            explicit=True,
            score=1.0,
        ),
        MemoryPrecedenceCandidate(
            candidate_id="workspace-procedure",
            precedence_class="workspace_procedure",
            scope="workspace",
            verified=True,
            explicit=True,
            score=0.1,
        ),
        MemoryPrecedenceCandidate(
            candidate_id="workspace-decision",
            precedence_class="workspace_decision",
            scope="workspace",
            verified=True,
            explicit=True,
            score=0.1,
        ),
        MemoryPrecedenceCandidate(
            candidate_id="workspace-fact",
            precedence_class="verified_workspace_fact",
            scope="workspace",
            verified=True,
            explicit=True,
            score=0.1,
        ),
    )

    assert tuple(item.candidate_id for item in order_by_memory_precedence(candidates)) == (
        "workspace-fact",
        "workspace-decision",
        "workspace-procedure",
        "explicit-global",
        "inferred-global",
    )
