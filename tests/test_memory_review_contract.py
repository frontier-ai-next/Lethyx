import pytest

from um_mem.contracts import MemoryReviewDecision, MemoryReviewReceipt


def _decision(**overrides: object) -> MemoryReviewDecision:
    values: dict[str, object] = {
        "workspace_id": "workspace:alpha",
        "session_id": "session:one",
        "task_id": "task:one",
        "outcome": "changed",
        "operations": ("remembered", "selected"),
        "candidate_record_ids": ("record:one",),
        "referenced_block_ids": ("block:one",),
        "selected_block_ids": ("block:one",),
        "reason_code": "reusable_state_recorded",
    }
    values.update(overrides)
    return MemoryReviewDecision(**values)


def test_memory_review_round_trip_is_content_free_and_exact() -> None:
    decision = _decision()
    restored_decision = MemoryReviewDecision.from_payload(decision.to_payload())
    receipt = MemoryReviewReceipt(
        receipt_id="review:one",
        decision=restored_decision,
        observed_operations=("remembered", "selected"),
        observed_candidate_record_ids=("record:one",),
        observed_selected_block_ids=("block:one",),
    )

    assert restored_decision == decision
    assert MemoryReviewReceipt.from_payload(receipt.to_payload()) == receipt
    assert set(receipt.to_payload()) == {
        "receipt_id",
        "decision",
        "observed_operations",
        "observed_candidate_record_ids",
        "observed_selected_block_ids",
        "recorded_at",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"outcome": "unknown"}, "memory review outcome"),
        ({"reason_code": "explanation with content"}, "memory review reason"),
        ({"operations": ("invented",)}, "memory review operation"),
        ({"operations": ("remembered", "remembered")}, "operations must be unique"),
        (
            {
                "outcome": "changed",
                "candidate_record_ids": (),
                "selected_block_ids": (),
                "reason_code": "reusable_state_recorded",
            },
            "changed review",
        ),
        (
            {
                "outcome": "unchanged",
                "operations": ("remembered",),
                "candidate_record_ids": ("record:one",),
                "reason_code": "no_reusable_state",
            },
            "unchanged review",
        ),
        (
            {
                "outcome": "unavailable",
                "operations": ("recalled",),
                "candidate_record_ids": (),
                "referenced_block_ids": ("block:one",),
                "selected_block_ids": (),
                "reason_code": "memory_unavailable",
            },
            "unavailable review",
        ),
    ),
)
def test_memory_review_decision_rejects_inconsistent_claims(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _decision(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"observed_candidate_record_ids": ()}, "candidate records"),
        ({"observed_selected_block_ids": ()}, "selected blocks"),
        ({"observed_operations": ("remembered",)}, "operations"),
        (
            {
                "observed_operations": ("remembered", "selected"),
                "observed_candidate_record_ids": ("record:one", "record:extra"),
            },
            "candidate records",
        ),
    ),
)
def test_memory_review_receipt_rejects_a_decision_that_does_not_match_observation(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "receipt_id": "review:one",
        "decision": _decision(),
        "observed_operations": ("remembered", "selected"),
        "observed_candidate_record_ids": ("record:one",),
        "observed_selected_block_ids": ("block:one",),
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        MemoryReviewReceipt(**values)


def test_unavailable_memory_review_carries_no_false_activity() -> None:
    decision = _decision(
        outcome="unavailable",
        operations=(),
        candidate_record_ids=(),
        referenced_block_ids=(),
        selected_block_ids=(),
        reason_code="memory_unavailable",
    )
    receipt = MemoryReviewReceipt(
        receipt_id="review:unavailable",
        decision=decision,
        observed_operations=(),
        observed_candidate_record_ids=(),
        observed_selected_block_ids=(),
    )

    assert receipt.decision.outcome == "unavailable"
    assert receipt.observed_operations == ()
