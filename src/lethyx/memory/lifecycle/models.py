"""Lifecycle action models."""

from dataclasses import dataclass, field
from typing import Any, Literal

LifecycleAction = Literal[
    "keep_active",
    "downrank_under_budget",
    "refresh_needed",
    "archive_to_warm",
    "mark_deprecated",
    "supersede_candidate",
    "quarantine_candidate",
    "needs_evidence",
]


@dataclass(frozen=True)
class LifecycleProposal:
    rule_id: str
    action: LifecycleAction
    reason: str
    safe_to_apply: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
