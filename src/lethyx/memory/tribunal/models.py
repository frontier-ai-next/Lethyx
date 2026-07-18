"""Models for deterministic memory candidate tribunal."""

from dataclasses import dataclass, field
from typing import Any, Literal

Decision = Literal["accept", "reject", "quarantine", "needs_more_evidence"]


@dataclass(frozen=True)
class TribunalInput:
    candidate_id: str
    rules: tuple[dict[str, Any], ...]
    source_evidence: tuple[dict[str, Any], ...]
    actor: str = "Lethyx"
    source_backend: str = ""
    workspace_id: str = ""
    target_workspace_id: str = ""
    drive_root: str = ""
    utility_value: float | None = None


@dataclass(frozen=True)
class TribunalCheckResult:
    name: str
    passed: bool
    severity: str = "info"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TribunalDecision:
    candidate_id: str
    decision: Decision
    reason: str
    checks: tuple[TribunalCheckResult, ...]
    warnings: tuple[str, ...] = ()
    report_path: str = ""
