"""Phase transition decisions derived from typed contract issues."""

from typing import Any

from lethyx.contracts.models import ContractIssue, PhaseDecision, TaskRiskProfile


def decide_phase_transition(
    *,
    phase: str,
    issues: list[ContractIssue],
    manifest: Any = None,
    risk: TaskRiskProfile | None = None,
) -> PhaseDecision:
    del manifest, risk
    human = [issue for issue in issues if issue.severity == "human_required"]
    if human:
        return PhaseDecision(
            action="human_checkpoint",
            target_phase=phase,
            blocking_issue_codes=tuple(issue.code for issue in human),
            reason=human[0].message or "Contract validation requires human checkpoint.",
        )
    blocking = [issue for issue in issues if issue.severity == "blocking"]
    if blocking:
        if any(
            issue.code in {"policy_violation", "verifier_mutation_attempt"} for issue in blocking
        ):
            action = "abort"
            target = None
        elif all(issue.code == "proof_stale_rerun_required" for issue in blocking):
            action = "verify"
            target = phase
        else:
            action = "loop_back"
            target = phase
        return PhaseDecision(
            action=action,  # type: ignore[arg-type]
            target_phase=target,
            blocking_issue_codes=tuple(issue.code for issue in blocking),
            reason=blocking[0].message or f"Contract issue `{blocking[0].code}`.",
        )
    return PhaseDecision(action="continue", target_phase=None, reason="contracts ok")


class PhaseDecisionEngine:
    @staticmethod
    def decide(
        *,
        phase: str,
        issues: list[ContractIssue],
        manifest: Any = None,
        risk: TaskRiskProfile | None = None,
    ) -> PhaseDecision:
        return decide_phase_transition(
            phase=phase,
            issues=issues,
            manifest=manifest,
            risk=risk,
        )
