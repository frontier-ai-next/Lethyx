"""Lifecycle decision policies."""

from lethyx.memory.kernel.value import MemoryValueStats
from lethyx.memory.lifecycle.models import LifecycleProposal
from lethyx.memory.proactive.models import BeliefRule


def propose_lifecycle_action(
    rule: BeliefRule, stats: MemoryValueStats | None = None
) -> LifecycleProposal:
    if rule.rule_type == "invariant" or rule.scope in {"constitution", "identity"}:
        return LifecycleProposal(rule.id, "keep_active", "safety_invariant_protected", False)
    if rule.contradiction_count > rule.support_count:
        return LifecycleProposal(
            rule.id, "quarantine_candidate", "contradiction_count_exceeds_support", False
        )
    if stats is not None and stats.confidence >= 0.4 and stats.estimated_value < -0.5:
        return LifecycleProposal(
            rule.id,
            "downrank_under_budget",
            "stale_low_value",
            True,
            {"estimated_value": stats.estimated_value},
        )
    if rule.status in {"candidate", "quarantined"}:
        return LifecycleProposal(
            rule.id, "needs_evidence", "non_active_candidate_requires_gate", True
        )
    return LifecycleProposal(rule.id, "keep_active", "no_lifecycle_change", False)
