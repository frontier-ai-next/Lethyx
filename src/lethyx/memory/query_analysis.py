"""Exact structural query terms and explicit evidence requirements."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, cast


EvidenceRole = Literal[
    "policy",
    "source",
    "decision",
    "failure",
    "test",
    "temporal_state",
    "rationale",
    "negative_evidence",
]
EVIDENCE_ROLES = (
    "policy",
    "source",
    "decision",
    "failure",
    "test",
    "temporal_state",
    "rationale",
    "negative_evidence",
)
_EVIDENCE_ROLE_SET = frozenset(EVIDENCE_ROLES)


@dataclass(frozen=True)
class EvidenceRequirement:
    """One caller-declared fact role that selected evidence should satisfy."""

    subject: str
    role: EvidenceRole
    accepted_literals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        subject = str(self.subject or "").strip()
        if not subject:
            raise ValueError("evidence requirement subject must not be blank")
        role = validate_evidence_roles((str(self.role),))[0]
        if not isinstance(self.accepted_literals, (list, tuple)):
            raise ValueError("evidence requirement accepted_literals must be a sequence")
        literals = tuple(
            dict.fromkeys(
                str(value).strip() for value in self.accepted_literals if str(value).strip()
            )
        )
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "accepted_literals", literals)

    @property
    def requirement_id(self) -> str:
        payload = json.dumps(
            [self.subject, self.role, list(self.accepted_literals)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "req_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def to_mapping(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "role": self.role,
            "accepted_literals": list(self.accepted_literals),
            "requirement_id": self.requirement_id,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "EvidenceRequirement":
        if not isinstance(value, dict):
            raise ValueError("evidence requirement must be a mapping")
        unknown = set(value).difference({"subject", "role", "accepted_literals", "requirement_id"})
        if unknown:
            raise ValueError("unknown evidence requirement fields: " + ", ".join(sorted(unknown)))
        requirement = cls(
            subject=str(value.get("subject") or ""),
            role=cast(EvidenceRole, value.get("role")),
            accepted_literals=tuple(value.get("accepted_literals") or ()),
        )
        supplied_id = str(value.get("requirement_id") or "")
        if supplied_id and supplied_id != requirement.requirement_id:
            raise ValueError("evidence requirement id does not match its content")
        return requirement


def matches_evidence_literal(text: str, literal: str) -> bool:
    """Match one caller-declared literal on exact alphanumeric boundaries."""

    haystack = str(text or "").replace("\\", "/").casefold()
    needle = str(literal or "").strip().replace("\\", "/").casefold()
    if not haystack or not needle:
        return False
    start = haystack.find(needle)
    while start >= 0:
        end = start + len(needle)
        before = haystack[start - 1] if start else ""
        after = haystack[end] if end < len(haystack) else ""
        if (not before or not before.isalnum()) and (not after or not after.isalnum()):
            return True
        start = haystack.find(needle, start + 1)
    return False


def validate_evidence_roles(value: tuple[str, ...]) -> tuple[EvidenceRole, ...]:
    roles = tuple(dict.fromkeys(str(role).strip() for role in value if str(role).strip()))
    unknown = sorted(set(roles) - _EVIDENCE_ROLE_SET)
    if unknown:
        raise ValueError("unknown evidence roles: " + ", ".join(unknown))
    return cast(tuple[EvidenceRole, ...], roles)


def validate_evidence_requirements(
    value: tuple[EvidenceRequirement, ...],
) -> tuple[EvidenceRequirement, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(requirement, EvidenceRequirement) for requirement in value
    ):
        raise ValueError("evidence requirements must contain EvidenceRequirement values")
    by_id: dict[str, EvidenceRequirement] = {}
    for requirement in value:
        by_id.setdefault(requirement.requirement_id, requirement)
    return tuple(by_id.values())
