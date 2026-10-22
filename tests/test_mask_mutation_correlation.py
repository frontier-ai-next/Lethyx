"""A retried mask mutation must replay instead of executing twice."""

from um_mem.memory.mcp_server import _stable_mutation_correlation


def test_identical_mask_mutations_share_one_correlation() -> None:
    arguments = {
        "mask_id": "m1",
        "expected_revision": 1,
        "activation_reason": "keep evidence in focus",
        "token_budget": 40,
    }
    first = _stable_mutation_correlation("amend_active_mask", dict(arguments))
    second = _stable_mutation_correlation("amend_active_mask", dict(arguments))
    assert first == second
    assert first.startswith("mask:")


def test_a_new_revision_target_keeps_its_own_correlation() -> None:
    base = {
        "mask_id": "m1",
        "activation_reason": "keep evidence in focus",
        "token_budget": 40,
    }
    first = _stable_mutation_correlation("amend_active_mask", {**base, "expected_revision": 1})
    second = _stable_mutation_correlation("amend_active_mask", {**base, "expected_revision": 2})
    other_tool = _stable_mutation_correlation(
        "transition_active_mask", {**base, "expected_revision": 1}
    )
    assert len({first, second, other_tool}) == 3
