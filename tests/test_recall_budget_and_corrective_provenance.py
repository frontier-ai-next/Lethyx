from lethyx.memory.backends.base import MemoryHit
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mempalace import MemPalaceBackend
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.mcp_facade import _plan_diagnostics
from lethyx.memory.recall_plan import RecallPlan
from lethyx.memory.recall_strategy import (
    allocate_recall_budget,
    provenance_reference_record,
)


def test_budget_allocation_preserves_the_exact_caller_ceiling() -> None:
    plan = RecallPlan.compile(
        "Compare source, decision, failure, test, and rationale evidence",
        query_type="synthesis",
        logical_block_ids=("workspace_episodes",),
        evidence_roles=("source", "decision", "failure", "test", "rationale"),
        max_tokens=2048,
    )

    allocation = allocate_recall_budget(plan)

    assert allocation.requested_tokens == 2048
    assert allocation.allocated_tokens == 2048
    assert allocation.evidence_role_count == 5
    assert allocate_recall_budget(plan.with_overrides(max_tokens=256)).allocated_tokens == 256


def test_mcp_diagnostics_publish_the_same_requested_and_allocated_budget() -> None:
    plan = RecallPlan.compile(
        "release evidence",
        logical_block_ids=("workspace_episodes",),
        max_tokens=1536,
    )

    diagnostics = _plan_diagnostics(
        plan,
        source_horizon="durable",
        include_candidates=False,
        max_tokens=plan.max_tokens,
    )

    assert diagnostics["requested_context_token_budget"] == 1536
    assert diagnostics["allocated_context_token_budget"] == 1536
    assert diagnostics["context_char_budget"] == 1536 * 4


def test_reference_only_provenance_hit_can_cross_the_mcp_record_transport() -> None:
    hit = MemoryHit(
        text="Canonical source reference: raw-episode",
        source="provenance:raw-episode",
        score=0.8,
        tags=["provenance_source"],
        metadata={
            "block_id": "provenance_ref:abc",
            "workspace_id": "alpha",
            "reverse_provenance_reference_only": True,
            "reverse_provenance_depth": 1,
            "reverse_provenance_derived_id": "derived:release",
            "reverse_provenance_root_id": "derived-record",
            "canonical_source_ref": "raw-episode",
        },
    )

    record = provenance_reference_record(hit)

    assert record is not None
    assert record.block_id == "provenance_ref:abc"
    assert record.authority == "advisory"
    assert record.metadata["canonical_source_ref"] == "raw-episode"


def test_selected_backends_keep_the_same_explicit_reverse_provenance_contract() -> None:
    assert callable(MemPalaceBackend.reverse_provenance_sources)
    assert callable(HindsightBackend.reverse_provenance_sources)
    assert callable(MnemosyneBackend.reverse_provenance_sources)
