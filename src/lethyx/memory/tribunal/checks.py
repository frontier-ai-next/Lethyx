"""Deterministic tribunal checks."""

from lethyx.memory.edge_graph import score_memory_edges
from lethyx.memory.tribunal.models import TribunalCheckResult, TribunalInput


def evidence_check(inp: TribunalInput) -> TribunalCheckResult:
    typed = [
        ref
        for ref in inp.source_evidence
        if isinstance(ref, dict)
        and ref.get("ref_type")
        and ref.get("ref_id")
        and ref.get("produced_by")
    ]
    if not typed:
        return TribunalCheckResult("EvidenceCheck", False, "blocking", "missing_typed_evidence")
    return TribunalCheckResult(
        "EvidenceCheck", True, "info", "typed_evidence_present", {"count": len(typed)}
    )


def trust_check(inp: TribunalInput) -> TribunalCheckResult:
    if inp.source_backend in {"hindsight", "mnemosyne"} and inp.actor == inp.source_backend:
        return TribunalCheckResult(
            "TrustCheck", False, "blocking", "passive_backend_cannot_self_promote"
        )
    if inp.actor in {"supervisor", "verifier", "harness", "watcher", "Lethyx"}:
        return TribunalCheckResult("TrustCheck", True, "info", "trusted_actor")
    return TribunalCheckResult("TrustCheck", False, "blocking", "untrusted_actor")


def scope_check(inp: TribunalInput) -> TribunalCheckResult:
    if inp.target_workspace_id and inp.workspace_id and inp.target_workspace_id != inp.workspace_id:
        return TribunalCheckResult(
            "ScopeCheck", False, "blocking", "cross_workspace_candidate_blocked"
        )
    return TribunalCheckResult("ScopeCheck", True, "info", "scope_ok")


def lifecycle_check(inp: TribunalInput) -> TribunalCheckResult:
    for rule in inp.rules:
        status = str(rule.get("status") or "candidate")
        if status in {"deprecated", "retracted", "superseded"}:
            return TribunalCheckResult("LifecycleCheck", False, "blocking", f"status_{status}")
    return TribunalCheckResult("LifecycleCheck", True, "info", "lifecycle_ok")


def contradiction_check(inp: TribunalInput) -> TribunalCheckResult:
    if not inp.graph_edges:
        return TribunalCheckResult("ContradictionCheck", True, "info", "graph_unavailable")
    score = score_memory_edges(inp.candidate_id, inp.graph_edges)
    if score.hard_contradiction:
        return TribunalCheckResult(
            "ContradictionCheck",
            False,
            "blocking",
            "hard_contradiction_path",
            {"proof_score": score.score},
        )
    return TribunalCheckResult(
        "ContradictionCheck", True, "info", "no_contradiction", {"proof_score": score.score}
    )


def utility_check(inp: TribunalInput) -> TribunalCheckResult:
    if inp.utility_value is not None and inp.utility_value < -0.5:
        return TribunalCheckResult("UtilityCheck", True, "warning", "low_utility_warning_only")
    return TribunalCheckResult("UtilityCheck", True, "info", "utility_ok")
