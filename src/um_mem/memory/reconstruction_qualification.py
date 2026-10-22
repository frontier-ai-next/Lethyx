"""Product qualification gate for optional reconstructive recall."""

import hashlib
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReconstructionQualificationCase:
    case_id: str
    declared_multi_hop: bool
    one_hop_coverage: float
    reconstruction_coverage: float
    one_hop_latency_ms: float
    reconstruction_latency_ms: float
    latency_budget_ms: float
    one_hop_tokens: int
    reconstruction_tokens: int
    token_budget: int
    one_hop_safety_violations: int = 0
    reconstruction_safety_violations: int = 0
    reconstruction_steps: int = 0
    exact_lookup: bool = False

    def __post_init__(self) -> None:
        _identifier(self.case_id, field_name="case_id")
        for name in ("declared_multi_hop", "exact_lookup"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name in ("one_hop_coverage", "reconstruction_coverage"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be between zero and one")
        for name in (
            "one_hop_latency_ms",
            "reconstruction_latency_ms",
            "latency_budget_ms",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        for name in (
            "one_hop_tokens",
            "reconstruction_tokens",
            "token_budget",
            "one_hop_safety_violations",
            "reconstruction_safety_violations",
            "reconstruction_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")


@dataclass(frozen=True)
class ReconstructionProductQualification:
    evidence_revision: str
    qualified: bool
    reasons: tuple[str, ...]
    case_count: int
    declared_multi_hop_cases: int
    safe_one_hop_fallback: bool
    qualification_id: str

    def __post_init__(self) -> None:
        _identifier(self.evidence_revision, field_name="evidence_revision")
        if not isinstance(self.qualified, bool) or not isinstance(self.safe_one_hop_fallback, bool):
            raise ValueError("qualification flags must be bool values")
        if any(not isinstance(value, str) or not value for value in self.reasons):
            raise ValueError("qualification reasons must be identifiers")
        for name in ("case_count", "declared_multi_hop_cases"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected = _qualification_id(self._payload_without_id())
        if self.qualification_id != expected:
            raise ValueError("qualification_id does not match qualification content")

    def _payload_without_id(self) -> dict[str, object]:
        return {
            "evidence_revision": self.evidence_revision,
            "qualified": self.qualified,
            "reasons": list(self.reasons),
            "case_count": self.case_count,
            "declared_multi_hop_cases": self.declared_multi_hop_cases,
            "safe_one_hop_fallback": self.safe_one_hop_fallback,
        }


def qualify_reconstructive_product(
    cases: tuple[ReconstructionQualificationCase, ...],
    *,
    evidence_revision: str,
) -> ReconstructionProductQualification:
    """Qualify reconstruction only when it beats a safe one-hop fallback."""

    if any(not isinstance(case, ReconstructionQualificationCase) for case in cases):
        raise TypeError("cases must contain ReconstructionQualificationCase values")
    revision = _identifier(evidence_revision, field_name="evidence_revision")
    reasons: list[str] = []
    if not cases:
        reasons.append("no_qualification_cases")
    multi_hop = tuple(case for case in cases if case.declared_multi_hop)
    if not multi_hop:
        reasons.append("no_declared_multi_hop_case")

    safe_one_hop = bool(cases) and all(
        case.one_hop_safety_violations == 0
        and case.one_hop_latency_ms <= case.latency_budget_ms
        and case.one_hop_tokens <= case.token_budget
        for case in cases
    )
    if not safe_one_hop:
        reasons.append("unsafe_one_hop_fallback")
    if any(case.reconstruction_safety_violations for case in cases):
        reasons.append("reconstruction_safety_regression")
    if any(case.reconstruction_latency_ms > case.latency_budget_ms for case in cases):
        reasons.append("reconstruction_latency_budget_regression")
    if any(case.reconstruction_tokens > case.token_budget for case in cases):
        reasons.append("reconstruction_token_budget_regression")
    if any(case.reconstruction_coverage <= case.one_hop_coverage for case in multi_hop):
        reasons.append("multi_hop_coverage_not_improved")
    if any(case.exact_lookup and case.reconstruction_steps for case in cases):
        reasons.append("exact_lookup_paid_reconstruction_cost")

    payload = {
        "evidence_revision": revision,
        "qualified": not reasons,
        "reasons": reasons or ["qualified_against_safe_one_hop_fallback"],
        "case_count": len(cases),
        "declared_multi_hop_cases": len(multi_hop),
        "safe_one_hop_fallback": safe_one_hop,
    }
    return ReconstructionProductQualification(
        evidence_revision=revision,
        qualified=not reasons,
        reasons=tuple(reasons or ["qualified_against_safe_one_hop_fallback"]),
        case_count=len(cases),
        declared_multi_hop_cases=len(multi_hop),
        safe_one_hop_fallback=safe_one_hop,
        qualification_id=_qualification_id(payload),
    )


def _qualification_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 1_024:
        raise ValueError(f"{field_name} must be a bounded non-empty identifier")
    return text
