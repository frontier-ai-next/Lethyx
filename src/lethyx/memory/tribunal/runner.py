"""Run deterministic memory tribunal."""

from lethyx.memory.tribunal.checks import (
    contradiction_check,
    evidence_check,
    lifecycle_check,
    safety_prompt_check,
    scope_check,
    trust_check,
    utility_check,
)
from lethyx.memory.tribunal.models import TribunalDecision, TribunalInput


def run_tribunal(inp: TribunalInput) -> TribunalDecision:
    checks = (
        evidence_check(inp),
        trust_check(inp),
        scope_check(inp),
        lifecycle_check(inp),
        contradiction_check(inp),
        utility_check(inp),
        safety_prompt_check(inp),
    )
    quarantine = [c for c in checks if not c.passed and c.severity == "quarantine"]
    blocking = [c for c in checks if not c.passed and c.severity in {"blocking", "error"}]
    if quarantine:
        return TribunalDecision(
            inp.candidate_id,
            "quarantine",
            quarantine[0].reason,
            checks,
            tuple(c.reason for c in checks if c.severity == "warning"),
        )
    if blocking:
        if blocking[0].reason == "missing_typed_evidence":
            return TribunalDecision(
                inp.candidate_id,
                "needs_more_evidence",
                blocking[0].reason,
                checks,
                tuple(c.reason for c in checks if c.severity == "warning"),
            )
        return TribunalDecision(
            inp.candidate_id,
            "reject",
            blocking[0].reason,
            checks,
            tuple(c.reason for c in checks if c.severity == "warning"),
        )
    return TribunalDecision(
        inp.candidate_id,
        "accept",
        "accepted",
        checks,
        tuple(c.reason for c in checks if c.severity == "warning"),
    )
