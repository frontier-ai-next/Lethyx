"""Dataclasses for proactive memory overlay."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OverlaySection:
    name: str
    content: str
    source_refs: list[str] = field(default_factory=list)
    source_hashes: list[str] = field(default_factory=list)
    trust: str = "curated"
    token_count: int = 0
    injection_id: str = ""
    estimated_value: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "source_refs": list(self.source_refs),
            "source_hashes": list(self.source_hashes),
            "trust": self.trust,
            "token_count": self.token_count,
            "injection_id": self.injection_id,
            "estimated_value": self.estimated_value,
            "reason": self.reason,
        }


@dataclass
class BeliefRule:
    id: str
    title: str
    scope: str
    rule_type: str
    status: str
    trust: str
    strength: float
    rule: dict[str, Any]
    applies_to: dict[str, Any] = field(default_factory=dict)
    source_evidence: list[dict[str, Any]] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    confidence: float = 0.0
    support_count: int = 0
    contradiction_count: int = 0
    created_at: str = ""
    last_verified_at: str = ""
    expires_at: str = ""
    source_backend: str = ""
    lifecycle_reason: str = ""
    trigger: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BkbConflictError(RuntimeError):
    """Unresolved hard invariant conflict in BKB overlay."""


@dataclass
class ProactiveMemoryOverlay:
    sections: list[OverlaySection] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    archive_hints: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def render_markdown(self) -> str:
        lines = ["## [ALWAYS-LOADED MEMORY]"]
        for section in self.sections:
            lines.append(f"### {section.name}")
            if section.source_refs:
                lines.append(f"_Sources: {', '.join(section.source_refs[:5])}_")
            lines.append(section.content.strip())
            lines.append("")
        if self.archive_hints:
            lines.append("### Archive hints (non-directive)")
            for hint in self.archive_hints[:5]:
                lines.append(f"- {hint}")
            lines.append("")
        lines.append("## [/ALWAYS-LOADED MEMORY]")
        return "\n".join(lines).strip()

    def to_payload(self) -> dict[str, Any]:
        telemetry = dict(self.telemetry)
        injection_audit = telemetry.get("injection_audit")
        payload: dict[str, Any] = {
            "sections": [s.to_dict() for s in self.sections],
            "conflicts": list(self.conflicts),
            "archive_hints": list(self.archive_hints),
            "telemetry": telemetry,
        }
        if isinstance(injection_audit, dict):
            payload["injection_audit"] = injection_audit
        return payload
