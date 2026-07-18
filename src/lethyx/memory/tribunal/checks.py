"""Deterministic tribunal checks."""

from pathlib import Path

from lethyx.memory.graph.scoring import proof_score
from lethyx.memory.tribunal.models import TribunalCheckResult, TribunalInput

_PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "bypass verification",
    "disable tests",
    "skip evidence",
    "do not tell the user",
)


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
    if not inp.drive_root:
        return TribunalCheckResult("ContradictionCheck", True, "info", "graph_unavailable")
    score = proof_score(inp.candidate_id, Path(inp.drive_root))
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


def safety_prompt_check(inp: TribunalInput) -> TribunalCheckResult:
    text = "\n".join(
        str(rule.get("title") or "") + "\n" + str(rule.get("rule") or "") for rule in inp.rules
    ).lower()
    for marker in _PROMPT_INJECTION_MARKERS:
        if marker in text:
            return TribunalCheckResult(
                "SafetyPromptCheck",
                False,
                "quarantine",
                "prompt_injection_like_candidate",
                {"marker": marker},
            )
    return TribunalCheckResult("SafetyPromptCheck", True, "info", "prompt_safety_ok")
