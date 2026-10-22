"""Canonical typed memory records, provenance, and transition payloads."""

import hashlib
import json
import os
import time
import unicodedata
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

from um_mem.contracts.models import APPROVED_TRUST_LEVELS, EvidenceRef, TrustLevel

CURRENT_SCHEMA_VERSION = 9


MemoryKind = str
BackendId = Literal["mempalace", "hindsight", "mnemosyne", "zvec"]
MemoryStatus = Literal[
    "candidate",
    "accepted",
    "quarantined",
    "superseded",
    "deprecated",
    "retracted",
]
MemoryScope = Literal[
    "subtask",
    "task",
    "session",
    "workspace",
    "project",
    "user_global",
]
MemoryRetention = Literal["ephemeral", "session", "durable", "archive"]
DeliveryPolicy = Literal[
    "always_on",
    "working_set",
    "on_demand",
    "reflection_only",
    "hidden",
]
MemoryTemperature = Literal["hot", "warm", "cold"]
MemoryAuthority = Literal["advisory", "directive"]
Sensitivity = Literal["public", "internal", "confidential", "secret"]
ContentMode = Literal["full", "pointer_only", "non_injectable"]
OperationalRisk = Literal["low", "medium", "high", "critical"]
VerificationStatus = Literal[
    "unverified",
    "observed",
    "evidence_backed",
    "verified",
    "disputed",
]
VerificationMethod = Literal[
    "test_run",
    "verification_report",
    "human_review",
    "artifact_inspection",
    "diff_review",
    "tool_replay",
]
PromotionDecision = Literal["not_requested", "accepted", "rejected"]
OriginChannel = Literal[
    "host_event",
    "user_message",
    "assistant_message",
    "tool_result",
    "file_snapshot",
    "test_run",
    "artifact",
    "external_source",
    "derived",
]
OriginIntegrity = Literal["untrusted", "reported", "observed", "verified", "endorsed"]
DerivationTransformation = Literal["extract", "summarize", "rephrase", "tool_echo"]
TurnChildClaimSourceRole = Literal["user", "assistant", "conversation"]
ProspectiveStatus = Literal["pending", "armed", "fired", "satisfied", "cancelled", "expired"]
ProspectivePredicateType = Literal[
    "time",
    "deadline",
    "host_event",
    "tool_state",
    "test_state",
    "file_state",
    "registered_condition",
]
ProspectiveTriggerOperator = Literal["eq", "neq", "contains", "prefix", "gte", "lte", "exists"]
ProspectiveRecurrencePolicy = Literal["once", "repeat"]
ProspectiveScope = Literal["workspace", "run", "task", "subtask"]
ProspectiveLifecycleEventType = Literal["arm", "fire", "satisfy", "cancel", "expire"]
MemoryDecisionAction = Literal["recall", "defer", "skip"]
MemoryReviewOutcome = Literal["changed", "unchanged", "unavailable", "forbidden"]
MemoryReviewOperation = Literal["remembered", "amended", "recalled", "selected"]
MemoryReviewReason = Literal[
    "reusable_state_recorded",
    "existing_memory_sufficient",
    "no_reusable_state",
    "memory_unavailable",
    "memory_forbidden",
]
MemoryUsageType = Literal[
    "candidate", "selected", "rendered", "cited", "applied", "not_applied", "blocked"
]
OutcomeAttributionBasis = Literal[
    "deterministic_verifier",
    "trusted_approval",
    "proven_stale_application",
    "proven_incorrect_application",
    "proven_contamination",
    "proven_regression",
    "insufficient_linkage",
]
ConsolidationStatus = Literal["proposed", "assessed", "accepted", "rejected", "committed"]
ConsolidationAction = Literal["retain", "defer", "split", "consolidate", "quarantine"]
ConsolidationTrigger = Literal[
    "recurrence",
    "novelty_surprise",
    "correction_conflict",
    "explicit_decision_preference",
    "high_risk_authority",
    "rare_irreversible_fact",
    "subgoal_boundary",
    "task_boundary",
    "session_boundary",
    "verified_reuse_harm",
    "capacity_pressure",
    "source_freshness",
    "prospective_deadline",
    "session_end",
    "task_end",
    "token_pressure",
    "duplicate_density",
    "source_change",
    "unresolved_conflict",
    "repeated_verified_reuse",
]
TransferStatus = Literal["pending", "accepted", "rejected"]
AssociationStatus = Literal["candidate", "strengthened", "weakened", "quarantined"]
FailureStage = Literal[
    "storage",
    "retrieval",
    "selection",
    "rendering",
    "comprehension",
    "action",
    "verifier",
]
MemoryOutcomeDisposition = Literal["helped", "hurt", "unnecessary", "neutral", "unknown"]
ContinuationDimension = Literal[
    "goals_constraints",
    "paths_symbols",
    "diff",
    "error_literals",
    "tests",
    "open_hypotheses",
    "rejected_reasons",
    "safety_state",
    "raw_page_back_coverage",
]
ContinuationSufficiency = Literal["sufficient", "insufficient"]
RecallTemporalMode = Literal["current", "historical", "change_history", "why_changed"]
RecallWideningPolicy = Literal[
    "backend_only",
    "broad",
]


_PROTECTED_SENSITIVITIES = frozenset({"confidential", "secret"})
_DECISIVE_EVIDENCE_TYPES = frozenset({"verification_report", "test_run", "human_approval"})


_RESERVED_METADATA_KEYS = frozenset(
    {
        "authority",
        "authority_class",
        "backend",
        "backend_id",
        "backend_reference",
        "backend_revision",
        "block_id",
        "canonical_id",
        "canonical_key",
        "child_claims",
        "commit_revision",
        "content_mode",
        "delivery_policy",
        "duplicate_cluster_id",
        "evidence_refs",
        "idempotency_key",
        "lifecycle",
        "logical_block_id",
        "memory_id",
        "mutation_correlation_id",
        "native_id",
        "native_reference",
        "native_revision",
        "operational_risk",
        "promotion_decision",
        "promotion_gate",
        "promotion_reason",
        "promotion_result",
        "promotion_status",
        "recorded_at",
        "record_id",
        "revision_id",
        "revision_of",
        "run_id",
        "privacy_class",
        "retention",
        "risk",
        "risk_level",
        "safety_class",
        "safety_policy",
        "selected_backend",
        "service_receipt_sha256",
        "sensitivity",
        "source_backend",
        "scope",
        "scope_chain",
        "session_id",
        "stage_id",
        "status",
        "subtask_id",
        "surface",
        "temperature",
        "tier",
        "trust_level",
        "task_id",
        "supersedes_ids",
        "user_id",
        "verification_methods",
        "verification_status",
        "verified",
        "workspace_id",
        "project_id",
        "parent_session_id",
        "mutation_of",
        "supersedes",
        "superseded_by",
        "token_budget",
    }
)


def reserved_memory_metadata_keys(metadata: object) -> tuple[str, ...]:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    reserved: list[str] = []
    for key in metadata:
        text = str(key)
        if (
            text in _RESERVED_METADATA_KEYS
            or text == "store"
            or text.endswith("_store")
            or text.split("_")[-2:] == ["write", "policy"]
        ):
            reserved.append(text)
    return tuple(sorted(reserved))


def _literal_values(shape: object) -> frozenset[str]:
    return frozenset(cast(tuple[str, ...], get_args(shape)))


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not (text := value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _closed_value(value: object, *, field_name: str, shape: object) -> str:
    text = _required_text(value, field_name=field_name)
    if text not in _literal_values(shape):
        raise ValueError(f"{field_name} must be one of {sorted(_literal_values(shape))}")
    return text


def validate_memory_safety_axes(
    sensitivity: object,
    content_mode: object,
    operational_risk: object,
) -> tuple[Sensitivity, ContentMode, OperationalRisk]:
    return (
        cast(
            Sensitivity,
            _closed_value(sensitivity, field_name="sensitivity", shape=Sensitivity),
        ),
        cast(
            ContentMode,
            _closed_value(content_mode, field_name="content_mode", shape=ContentMode),
        ),
        cast(
            OperationalRisk,
            _closed_value(operational_risk, field_name="operational_risk", shape=OperationalRisk),
        ),
    )


def is_protected_memory(
    sensitivity: object,
    content_mode: object,
    operational_risk: object,
) -> bool:
    try:
        sensitivity, content_mode, _ = validate_memory_safety_axes(
            sensitivity, content_mode, operational_risk
        )
    except ValueError:
        return True
    return sensitivity in _PROTECTED_SENSITIVITIES or content_mode == "non_injectable"


def has_decisive_evidence(refs: tuple[EvidenceRef, ...]) -> bool:
    return any(ref.ref_type in _DECISIVE_EVIDENCE_TYPES and ref.producer_id for ref in refs)


def _evidence_refs(value: object) -> tuple[EvidenceRef, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("evidence_refs must be a list or tuple")
    refs: list[EvidenceRef] = []
    for item in value:
        if isinstance(item, EvidenceRef):
            refs.append(item)
        elif isinstance(item, dict):
            refs.append(EvidenceRef.from_mapping(item))
        else:
            raise ValueError("evidence_refs must contain typed EvidenceRef values")
    return tuple(refs)


def _text_tuple(value: object, *, field_name: str, maximum: int = 0) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    texts: list[str] = []
    for raw in value:
        text = _required_text(raw, field_name=field_name)
        if maximum and len(text) > maximum:
            raise ValueError(f"{field_name} exceeds {maximum} characters")
        if text not in texts:
            texts.append(text)
    return tuple(texts)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _optional_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return unicodedata.normalize("NFC", value.strip())


def normalize_identity_path(value: object, *, field_name: str) -> str:
    """Resolve an identity path to its Unicode- and platform-stable form."""
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field_name} must be a path string")
    raw = unicodedata.normalize("NFC", str(value).strip())
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFC", str(Path(raw).expanduser().resolve()))
    return os.path.normcase(normalized) if os.name == "nt" else normalized


def _valid_time(value: object, *, field_name: str) -> tuple[str, datetime | None]:
    text = _optional_text(value, field_name=field_name)
    if not text:
        return text, None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return text, parsed.astimezone(UTC)


def _positive_version(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _normalize_record_identity(
    record: "MemoryEvent | MemoryBlockRecord | WorkingStateCheckpoint",
) -> None:
    for field_name in (
        "user_id",
        "workspace_id",
        "session_id",
        "task_id",
        "parent_session_id",
        "artifact_id",
        "artifact_revision",
        "project_id",
        "repository_id",
        "branch",
        "revision",
        "revision_of",
        "dirty_state",
        "semantic_catalog_version",
    ):
        if not hasattr(record, field_name):
            continue
        object.__setattr__(
            record,
            field_name,
            _optional_text(getattr(record, field_name), field_name=field_name),
        )
    for field_name in ("checkout", "worktree"):
        if not hasattr(record, field_name):
            continue
        object.__setattr__(
            record,
            field_name,
            normalize_identity_path(getattr(record, field_name), field_name=field_name),
        )
    if getattr(record, "parent_session_id", "") and not getattr(record, "session_id", ""):
        raise ValueError("parent_session_id requires session_id")
    if hasattr(record, "scope"):
        requirements = {
            "workspace": ("workspace_id",),
            "session": ("workspace_id", "session_id"),
            "task": ("workspace_id", "session_id", "task_id"),
            "subtask": ("workspace_id", "session_id", "task_id", "subtask_id"),
            "project": ("project_id",),
            "user_global": ("user_id",),
        }
        missing = tuple(
            field_name
            for field_name in requirements[record.scope]
            if not getattr(record, field_name, "")
        )
        if missing:
            raise ValueError(f"{record.scope} scope requires {', '.join(missing)}")
    if bool(getattr(record, "artifact_id", "")) != bool(getattr(record, "artifact_revision", "")):
        raise ValueError("artifact_id and artifact_revision must be provided together")
    valid_from, starts_at = _valid_time(record.valid_from, field_name="valid_from")
    valid_to, ends_at = _valid_time(record.valid_to, field_name="valid_to")
    if starts_at is None and ends_at is not None:
        raise ValueError("valid_to requires valid_from")
    if starts_at is not None and ends_at is not None and ends_at < starts_at:
        raise ValueError("valid_to must not precede valid_from")
    object.__setattr__(record, "valid_from", valid_from)
    object.__setattr__(record, "valid_to", valid_to)
    for field_name in ("canonical_key", "revision_id", "duplicate_cluster_id"):
        if hasattr(record, field_name):
            object.__setattr__(
                record,
                field_name,
                _optional_text(getattr(record, field_name), field_name=field_name),
            )
    revision_of = getattr(record, "revision_of", "")
    supersedes_ids = getattr(record, "supersedes_ids", ())
    if revision_of:
        if supersedes_ids != (revision_of,):
            raise ValueError("revision successors must supersede exactly revision_of")
        if revision_of == getattr(record, "block_id", ""):
            raise ValueError("a memory revision cannot supersede itself")
    elif supersedes_ids:
        raise ValueError("root memory records cannot carry supersedes_ids")
    if hasattr(record, "recorded_at"):
        recorded_at, recorded_at_value = _valid_time(record.recorded_at, field_name="recorded_at")
        if recorded_at_value is not None:
            recorded_at = recorded_at_value.isoformat().replace("+00:00", "Z")
        object.__setattr__(record, "recorded_at", recorded_at)
    object.__setattr__(
        record,
        "schema_version",
        _positive_version(record.schema_version, field_name="schema_version"),
    )
    object.__setattr__(
        record,
        "hard_policy_version",
        _positive_version(record.hard_policy_version, field_name="hard_policy_version"),
    )


@dataclass(frozen=True)
class RecordBlockMembership:
    primary_block_id: str
    catalog_revision_id: str
    backend_partition_id: str = ""
    secondary_block_ids: tuple[str, ...] = ()
    record_id: str = ""
    record_revision_id: str = ""
    workspace_id: str = ""
    valid_from: str = ""
    valid_to: str = ""
    membership_revision_id: str = ""

    def __post_init__(self) -> None:
        primary_block_id = _required_text(
            self.primary_block_id,
            field_name="primary_block_id",
        )
        catalog_revision_id = _required_text(
            self.catalog_revision_id,
            field_name="catalog_revision_id",
        )
        secondary_block_ids = _text_tuple(
            self.secondary_block_ids,
            field_name="secondary_block_id",
        )
        if len(secondary_block_ids) > 16:
            raise ValueError("record block membership accepts at most 16 secondary blocks")
        if primary_block_id in secondary_block_ids:
            raise ValueError("primary_block_id cannot also be a secondary block")
        object.__setattr__(self, "primary_block_id", primary_block_id)
        object.__setattr__(self, "catalog_revision_id", catalog_revision_id)
        partition_id = _optional_text(self.backend_partition_id, field_name="backend_partition_id")
        if not partition_id:
            partition_id = _static_backend_partition_id("default", primary_block_id)
        object.__setattr__(self, "backend_partition_id", partition_id)
        object.__setattr__(self, "secondary_block_ids", secondary_block_ids)
        for field_name in ("record_id", "record_revision_id", "workspace_id"):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name=field_name),
            )
        valid_from, starts_at = _valid_time(
            self.valid_from,
            field_name="membership.valid_from",
        )
        valid_to, ends_at = _valid_time(
            self.valid_to,
            field_name="membership.valid_to",
        )
        if starts_at is None and ends_at is not None:
            raise ValueError("membership.valid_to requires membership.valid_from")
        if starts_at is not None and ends_at is not None and ends_at < starts_at:
            raise ValueError("membership.valid_to must not precede membership.valid_from")
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_to", valid_to)
        bound_fields = (self.record_id, self.record_revision_id, self.workspace_id)
        if any(bound_fields) and not all(bound_fields):
            raise ValueError(
                "record block membership identity requires record_id, "
                "record_revision_id, and workspace_id"
            )
        revision_id = _optional_text(
            self.membership_revision_id,
            field_name="membership_revision_id",
        )
        if not all(bound_fields):
            if revision_id:
                raise ValueError("unbound record block membership cannot have a revision id")
            return
        expected = record_block_membership_revision_id(self)
        if revision_id and revision_id != expected:
            raise ValueError("membership_revision_id does not match membership payload")
        object.__setattr__(self, "membership_revision_id", expected)

    @classmethod
    def from_mapping(cls, value: object) -> "RecordBlockMembership":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("block_membership must be a RecordBlockMembership mapping")
        expected = frozenset(cls.__dataclass_fields__)
        missing = sorted(expected.difference(value))
        unknown = sorted(set(value).difference(expected))
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("block_membership fields must be exact: " + "; ".join(details))
        return cls(**dict(value))


def record_block_membership_revision_id(membership: RecordBlockMembership) -> str:
    payload = {
        "record_id": membership.record_id,
        "record_revision_id": membership.record_revision_id,
        "workspace_id": membership.workspace_id,
        "primary_block_id": membership.primary_block_id,
        "secondary_block_ids": list(membership.secondary_block_ids),
        "catalog_revision_id": membership.catalog_revision_id,
        "backend_partition_id": membership.backend_partition_id,
        "valid_from": membership.valid_from,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "membership_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _static_backend_partition_id(profile_id: str, block_id: str) -> str:
    material = json.dumps([profile_id, block_id], separators=(",", ":")).encode("utf-8")
    return "static:" + hashlib.sha256(material).hexdigest()


def _bind_record_block_membership(record: object) -> None:
    membership = getattr(record, "block_membership", None)
    if membership is None:
        return
    if not isinstance(membership, RecordBlockMembership):
        membership = RecordBlockMembership.from_mapping(membership)
    profile_id = str(getattr(record, "profile_id", "default") or "default").strip()
    if membership.backend_partition_id == _static_backend_partition_id(
        "default", membership.primary_block_id
    ):
        membership = replace(
            membership,
            backend_partition_id=_static_backend_partition_id(
                profile_id, membership.primary_block_id
            ),
            membership_revision_id="",
        )
    record_id = str(getattr(record, "block_id", "") or "").strip()
    record_revision_id = str(getattr(record, "revision_id", "") or "").strip()
    workspace_id = str(getattr(record, "workspace_id", "") or "").strip()
    valid_from = str(getattr(record, "valid_from", "") or getattr(record, "recorded_at", "") or "")
    if not record_id or not record_revision_id or not workspace_id:
        bound = replace(
            membership,
            record_id="",
            record_revision_id="",
            workspace_id="",
            valid_from=valid_from,
            valid_to=str(getattr(record, "valid_to", "") or ""),
            membership_revision_id="",
        )
    else:
        bound = replace(
            membership,
            record_id=record_id,
            record_revision_id=record_revision_id,
            workspace_id=workspace_id,
            valid_from=valid_from,
            valid_to=str(getattr(record, "valid_to", "") or ""),
            membership_revision_id="",
        )
    object.__setattr__(record, "block_membership", bound)


@dataclass(frozen=True)
class OriginRef:
    channel: OriginChannel
    principal_id: str
    integrity_class: OriginIntegrity
    origin_id: str = ""

    def __post_init__(self) -> None:
        channel = _closed_value(self.channel, field_name="origin channel", shape=OriginChannel)
        integrity = _closed_value(
            self.integrity_class,
            field_name="origin integrity_class",
            shape=OriginIntegrity,
        )
        principal_id = _required_text(self.principal_id, field_name="origin principal_id")
        object.__setattr__(self, "channel", cast(OriginChannel, channel))
        object.__setattr__(self, "integrity_class", cast(OriginIntegrity, integrity))
        object.__setattr__(self, "principal_id", principal_id)
        object.__setattr__(
            self,
            "origin_id",
            _required_text(self.origin_id, field_name="origin_id")
            if self.origin_id
            else f"{channel}:{principal_id}",
        )


_INTEGRITY_RANK = {"untrusted": 0, "reported": 1, "observed": 2, "verified": 3, "endorsed": 4}


@dataclass(frozen=True)
class OriginSegment:
    claim_id: str
    origin: OriginRef
    integrity_ceiling: OriginIntegrity
    independence_domain: str
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.origin, OriginRef):
            raise ValueError("origin must be an OriginRef")
        ceiling = _closed_value(
            self.integrity_ceiling,
            field_name="integrity_ceiling",
            shape=OriginIntegrity,
        )
        if _INTEGRITY_RANK[ceiling] > _INTEGRITY_RANK[self.origin.integrity_class]:
            raise ValueError("integrity_ceiling cannot exceed origin integrity_class")
        object.__setattr__(self, "claim_id", _required_text(self.claim_id, field_name="claim_id"))
        object.__setattr__(self, "integrity_ceiling", cast(OriginIntegrity, ceiling))
        object.__setattr__(
            self,
            "independence_domain",
            _required_text(self.independence_domain, field_name="independence_domain"),
        )
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs))


@dataclass(frozen=True)
class DerivationReceipt:
    derived_id: str
    source_segments: tuple[OriginSegment, ...]
    transformation: DerivationTransformation
    integrity_ceiling: OriginIntegrity

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_segments, tuple)
            or not self.source_segments
            or any(not isinstance(segment, OriginSegment) for segment in self.source_segments)
        ):
            raise ValueError("source_segments must contain canonical OriginSegment values")
        transformation = _closed_value(
            self.transformation,
            field_name="transformation",
            shape=DerivationTransformation,
        )
        ceiling = _closed_value(
            self.integrity_ceiling,
            field_name="integrity_ceiling",
            shape=OriginIntegrity,
        )
        inherited_ceiling = min(
            _INTEGRITY_RANK[segment.integrity_ceiling] for segment in self.source_segments
        )
        if _INTEGRITY_RANK[ceiling] > inherited_ceiling:
            raise ValueError("integrity_ceiling cannot exceed the minimum source ceiling")
        object.__setattr__(
            self, "derived_id", _required_text(self.derived_id, field_name="derived_id")
        )
        object.__setattr__(
            self,
            "transformation",
            cast(DerivationTransformation, transformation),
        )
        object.__setattr__(self, "integrity_ceiling", cast(OriginIntegrity, ceiling))


TurnChildClaimTemperature = MemoryTemperature
_TURN_CHILD_FIELDS = frozenset(
    {
        "claim_type",
        "source_role",
        "content",
        "parent_session_id",
        "origin_segments",
        "derivation_receipt",
        "temperature",
        "tags",
        "trust_level",
        "status",
        "authority",
    }
)
_TURN_CHILD_CONTROLLED = {
    "trust_level": "agent_reported",
    "status": "candidate",
    "authority": "advisory",
}


def _origin_segment_from_mapping(value: object) -> OriginSegment:
    if isinstance(value, OriginSegment):
        return value
    if not isinstance(value, dict):
        raise ValueError("origin segment must be typed")
    raw_origin = value.get("origin")
    if isinstance(raw_origin, OriginRef):
        origin = raw_origin
    elif isinstance(raw_origin, dict):
        origin = OriginRef(
            channel=cast(OriginChannel, raw_origin.get("channel")),
            principal_id=cast(str, raw_origin.get("principal_id")),
            integrity_class=cast(OriginIntegrity, raw_origin.get("integrity_class")),
            origin_id=cast(str, raw_origin.get("origin_id") or ""),
        )
    else:
        raise ValueError("origin segment origin must be typed")
    return OriginSegment(
        claim_id=cast(str, value.get("claim_id")),
        origin=origin,
        integrity_ceiling=cast(OriginIntegrity, value.get("integrity_ceiling")),
        independence_domain=cast(str, value.get("independence_domain")),
        evidence_refs=_evidence_refs(value.get("evidence_refs", ())),
    )


def _derivation_receipt_from_mapping(value: object) -> DerivationReceipt | None:
    if value is None or isinstance(value, DerivationReceipt):
        return value
    if not isinstance(value, dict):
        raise ValueError("derivation_receipt must be typed")
    raw_segments = value.get("source_segments", ())
    if not isinstance(raw_segments, (list, tuple)):
        raise ValueError("derivation_receipt source_segments must be a list or tuple")
    return DerivationReceipt(
        derived_id=cast(str, value.get("derived_id")),
        source_segments=tuple(_origin_segment_from_mapping(item) for item in raw_segments),
        transformation=cast(DerivationTransformation, value.get("transformation")),
        integrity_ceiling=cast(OriginIntegrity, value.get("integrity_ceiling")),
    )


@dataclass(frozen=True)
class TurnChildClaim:
    claim_type: str
    source_role: TurnChildClaimSourceRole
    content: str
    parent_session_id: str
    origin_segments: tuple[OriginSegment, ...]
    derivation_receipt: DerivationReceipt | None = None
    temperature: TurnChildClaimTemperature = "warm"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        claim_type = _required_text(self.claim_type, field_name="turn child claim claim_type")
        if any(character.isspace() for character in claim_type):
            raise ValueError("turn child claim claim_type cannot contain whitespace")
        if (
            not isinstance(self.origin_segments, tuple)
            or not self.origin_segments
            or any(not isinstance(segment, OriginSegment) for segment in self.origin_segments)
        ):
            raise ValueError("turn child claim origin_segments are required")
        if len(self.origin_segments) > 8:
            raise ValueError("turn child claim has too many origin_segments")
        source_role = _closed_value(
            self.source_role,
            field_name="turn child claim source_role",
            shape=TurnChildClaimSourceRole,
        )
        temperature = _closed_value(
            self.temperature,
            field_name="turn child claim temperature",
            shape=MemoryTemperature,
        )
        receipt = self.derivation_receipt
        if receipt is not None:
            if not isinstance(receipt, DerivationReceipt):
                raise ValueError("derivation_receipt must be typed")
            if receipt.source_segments != self.origin_segments:
                raise ValueError(
                    "derivation_receipt source_segments must preserve claim origin_segments"
                )
        object.__setattr__(self, "claim_type", claim_type)
        object.__setattr__(self, "source_role", cast(TurnChildClaimSourceRole, source_role))
        object.__setattr__(
            self,
            "content",
            _required_text(self.content, field_name="turn child claim content"),
        )
        object.__setattr__(
            self,
            "parent_session_id",
            _required_text(self.parent_session_id, field_name="turn child claim parent_session_id"),
        )
        object.__setattr__(
            self,
            "temperature",
            cast(TurnChildClaimTemperature, temperature),
        )
        object.__setattr__(
            self, "tags", _text_tuple(self.tags, field_name="turn child claim tag", maximum=96)
        )

    @property
    def trust_level(self) -> Literal["agent_reported"]:
        return "agent_reported"

    @property
    def status(self) -> Literal["candidate"]:
        return "candidate"

    @property
    def authority(self) -> Literal["advisory"]:
        return "advisory"

    def to_transport_payload(self) -> dict[str, object]:
        return {
            "claim_type": self.claim_type,
            "source_role": self.source_role,
            "content": self.content,
            "parent_session_id": self.parent_session_id,
            "origin_segments": [asdict(segment) for segment in self.origin_segments],
            "derivation_receipt": asdict(self.derivation_receipt)
            if self.derivation_receipt is not None
            else None,
            "temperature": self.temperature,
            "tags": list(self.tags),
            "trust_level": self.trust_level,
            "status": self.status,
            "authority": self.authority,
        }


def normalize_turn_child_claims(value: object) -> tuple[TurnChildClaim, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("turn child claims must be a list or tuple")
    if len(value) > 8:
        raise ValueError("turn event has too many child claims")
    claims: list[TurnChildClaim] = []
    for item in value:
        if isinstance(item, TurnChildClaim):
            claims.append(item)
            continue
        if not isinstance(item, dict) or any(not isinstance(key, str) for key in item):
            raise ValueError("turn child claim must be an object")
        unsupported = sorted(set(item).difference(_TURN_CHILD_FIELDS))
        if unsupported:
            raise ValueError(f"turn child claim has unsupported fields: {', '.join(unsupported)}")
        for field_name, expected in _TURN_CHILD_CONTROLLED.items():
            if field_name in item and item[field_name] != expected:
                raise ValueError(f"turn child claim {field_name} must be {expected}")
        raw_segments = item.get("origin_segments", ())
        if not isinstance(raw_segments, (list, tuple)):
            raise ValueError("turn child claim origin_segments must be a list or tuple")
        claims.append(
            TurnChildClaim(
                claim_type=cast(str, item.get("claim_type")),
                source_role=cast(TurnChildClaimSourceRole, item.get("source_role")),
                content=cast(str, item.get("content")),
                parent_session_id=cast(str, item.get("parent_session_id")),
                origin_segments=tuple(
                    _origin_segment_from_mapping(segment) for segment in raw_segments
                ),
                derivation_receipt=_derivation_receipt_from_mapping(item.get("derivation_receipt")),
                temperature=cast(TurnChildClaimTemperature, item.get("temperature", "warm")),
                tags=cast(tuple[str, ...], item.get("tags", ())),
            )
        )
    return tuple(claims)


def compact_turn_child_claims(claims: tuple[TurnChildClaim, ...]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for claim in sorted(
        claims,
        key=lambda item: (item.claim_type, item.parent_session_id, item.source_role, item.content),
    )[:8]:
        segments = sorted(
            claim.origin_segments,
            key=lambda item: (item.claim_id, item.origin.origin_id, item.independence_domain),
        )[:8]
        ceiling = min(
            [item.integrity_ceiling for item in segments]
            + ([claim.derivation_receipt.integrity_ceiling] if claim.derivation_receipt else []),
            key=_INTEGRITY_RANK.__getitem__,
        )
        compact.append(
            {
                "claim_type": claim.claim_type,
                "source_role": claim.source_role,
                "content": claim.content[:384],
                "parent_session_id": claim.parent_session_id[:160],
                "authority": claim.authority,
                "trust_level": claim.trust_level,
                "status": claim.status,
                "integrity_ceiling": ceiling,
                "origin_segments": [
                    {
                        "claim_id": item.claim_id[:160],
                        "source": {
                            "channel": item.origin.channel,
                            "principal_id": item.origin.principal_id[:160],
                            "origin_id": item.origin.origin_id[:160],
                        },
                        "integrity_ceiling": item.integrity_ceiling,
                        "independence_domain": item.independence_domain[:160],
                    }
                    for item in segments
                ],
                "derivation": (
                    {
                        "transformation": claim.derivation_receipt.transformation,
                        "integrity_ceiling": claim.derivation_receipt.integrity_ceiling,
                    }
                    if claim.derivation_receipt
                    else None
                ),
            }
        )
    return compact


def encode_turn_child_claims_json(claims: tuple[TurnChildClaim, ...]) -> str:
    if not isinstance(claims, tuple) or any(
        not isinstance(claim, TurnChildClaim) for claim in claims
    ):
        raise ValueError("turn child claims must use canonical TurnChildClaim values")
    return json.dumps(
        [claim.to_transport_payload() for claim in claims],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_turn_child_claims_json(value: object) -> tuple[TurnChildClaim, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ValueError("turn child claims JSON must be a string")
    raw = value.strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("turn child claims JSON is invalid") from exc
    if not isinstance(parsed, list):
        raise ValueError("turn child claims JSON must contain a list")
    return normalize_turn_child_claims(parsed)


def _normalize_memory_axes(record: "MemoryEvent | MemoryBlockRecord", *, kind_field: str) -> None:
    object.__setattr__(
        record,
        kind_field,
        _required_text(getattr(record, kind_field), field_name=kind_field),
    )
    for field_name, shape in (
        ("status", MemoryStatus),
        ("retention", MemoryRetention),
        ("delivery_policy", DeliveryPolicy),
        ("temperature", MemoryTemperature),
        ("authority", MemoryAuthority),
        ("trust_level", TrustLevel),
        ("scope", MemoryScope),
        ("verification_status", VerificationStatus),
        ("promotion_decision", PromotionDecision),
    ):
        object.__setattr__(
            record,
            field_name,
            _closed_value(getattr(record, field_name), field_name=field_name, shape=shape),
        )
    sensitivity, content_mode, operational_risk = validate_memory_safety_axes(
        record.sensitivity,
        record.content_mode,
        record.operational_risk,
    )
    object.__setattr__(record, "sensitivity", sensitivity)
    object.__setattr__(record, "content_mode", content_mode)
    object.__setattr__(record, "operational_risk", operational_risk)
    object.__setattr__(record, "tags", _text_tuple(record.tags, field_name="tag", maximum=96))
    object.__setattr__(
        record,
        "profile_id",
        _required_text(record.profile_id, field_name="profile_id"),
    )
    object.__setattr__(
        record,
        "supersedes_ids",
        _text_tuple(record.supersedes_ids, field_name="supersedes_id"),
    )
    object.__setattr__(record, "evidence_refs", _evidence_refs(record.evidence_refs))
    verification_methods = tuple(
        cast(
            VerificationMethod,
            _closed_value(method, field_name="verification_method", shape=VerificationMethod),
        )
        for method in _text_tuple(
            record.verification_methods,
            field_name="verification_methods",
        )
    )
    if record.verification_status in {"evidence_backed", "verified"}:
        if not verification_methods:
            raise ValueError("evidence-backed records require verification_methods")
        if not record.evidence_refs:
            raise ValueError("evidence-backed records require typed evidence_refs")
    object.__setattr__(record, "verification_methods", verification_methods)
    object.__setattr__(
        record,
        "child_claims",
        normalize_turn_child_claims(record.child_claims),
    )
    if not isinstance(record.external_refs, dict):
        raise ValueError("external_refs must be a dictionary")
    reserved = reserved_memory_metadata_keys(record.metadata)
    if reserved:
        raise ValueError(f"metadata contains reserved governance fields: {', '.join(reserved)}")
    if record.status == "retracted":
        if record.retention != "archive":
            raise ValueError("retracted memory requires archive retention")
        if record.delivery_policy != "hidden":
            raise ValueError("retracted memory requires hidden delivery_policy")
    if record.authority == "directive" and record.status != "accepted":
        raise ValueError("directive memory requires accepted status")
    if record.status == "accepted" or record.authority == "directive":
        semantic = (
            record.verification_status == "verified" and record.trust_level in APPROVED_TRUST_LEVELS
        )
        integrity = (
            record.authority == "advisory"
            and record.verification_status == "evidence_backed"
            and record.trust_level == "agent_reported"
        )
        if not (semantic or integrity):
            raise ValueError("accepted memory requires integrity-backed or semantic assurance")
        if not record.verification_methods:
            raise ValueError("accepted memory requires typed verification methods")
        if not has_decisive_evidence(record.evidence_refs):
            raise ValueError("accepted memory requires decisive typed evidence_refs")
        if record.promotion_decision != "accepted":
            raise ValueError("accepted memory requires an accepted promotion decision")


@dataclass(frozen=True)
class MemoryEvent:
    content: str
    block_membership: RecordBlockMembership | None = None
    block_id: str = ""
    canonical_key: str = ""
    revision_id: str = ""
    duplicate_cluster_id: str = ""
    recorded_at: str = ""
    title: str = ""
    memory_kind: MemoryKind = "observation"
    status: MemoryStatus = "candidate"
    retention: MemoryRetention = "durable"
    delivery_policy: DeliveryPolicy = "on_demand"
    temperature: MemoryTemperature = "warm"
    authority: MemoryAuthority = "advisory"
    trust_level: TrustLevel = "agent_reported"
    scope: MemoryScope = "workspace"
    verification_status: VerificationStatus = "unverified"
    verification_methods: tuple[VerificationMethod, ...] = ()
    promotion_decision: PromotionDecision = "not_requested"
    user_id: str = ""
    workspace_id: str = ""
    profile_id: str = "default"
    run_id: str = ""
    stage_id: str = ""
    subtask_id: str = ""
    agent_kind: str = ""
    agent_instance_id: str = ""
    producer: str = "agent"
    source_path: str = ""
    source_backend: str = "um_mem"
    tags: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    external_refs: dict[str, str] = field(default_factory=dict)
    sensitivity: Sensitivity = "internal"
    content_mode: ContentMode = "full"
    operational_risk: OperationalRisk = "low"
    revision_of: str = ""
    supersedes_ids: tuple[str, ...] = ()
    expires_at: str = ""
    child_claims: tuple[TurnChildClaim, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    task_id: str = ""
    parent_session_id: str = ""
    artifact_id: str = ""
    artifact_revision: str = ""
    valid_from: str = ""
    valid_to: str = ""
    project_id: str = ""
    repository_id: str = ""
    checkout: str = ""
    worktree: str = ""
    branch: str = ""
    revision: str = ""
    dirty_state: str = ""
    schema_version: int = CURRENT_SCHEMA_VERSION
    hard_policy_version: int = 1
    semantic_catalog_version: str = ""

    def __post_init__(self) -> None:
        _normalize_memory_axes(self, kind_field="memory_kind")
        _normalize_record_identity(self)
        _bind_record_block_membership(self)


@dataclass(frozen=True)
class MemoryBlockRecord:
    block_id: str
    content: str
    block_membership: RecordBlockMembership | None = None
    canonical_key: str = ""
    revision_id: str = ""
    duplicate_cluster_id: str = ""
    recorded_at: str = ""
    title: str = ""
    kind: MemoryKind = "observation"
    status: MemoryStatus = "candidate"
    retention: MemoryRetention = "durable"
    delivery_policy: DeliveryPolicy = "on_demand"
    temperature: MemoryTemperature = "warm"
    authority: MemoryAuthority = "advisory"
    trust_level: TrustLevel = "agent_reported"
    scope: MemoryScope = "workspace"
    verification_status: VerificationStatus = "unverified"
    verification_methods: tuple[VerificationMethod, ...] = ()
    promotion_decision: PromotionDecision = "not_requested"
    user_id: str = ""
    workspace_id: str = ""
    profile_id: str = "default"
    run_id: str = ""
    stage_id: str = ""
    subtask_id: str = ""
    producer: str = "agent"
    source_backend: str = ""
    tags: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    external_refs: dict[str, str] = field(default_factory=dict)
    sensitivity: Sensitivity = "internal"
    content_mode: ContentMode = "full"
    operational_risk: OperationalRisk = "low"
    revision_of: str = ""
    supersedes_ids: tuple[str, ...] = ()
    expires_at: str = ""
    child_claims: tuple[TurnChildClaim, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    session_id: str = ""
    task_id: str = ""
    parent_session_id: str = ""
    artifact_id: str = ""
    artifact_revision: str = ""
    valid_from: str = ""
    valid_to: str = ""
    project_id: str = ""
    repository_id: str = ""
    checkout: str = ""
    worktree: str = ""
    branch: str = ""
    revision: str = ""
    dirty_state: str = ""
    schema_version: int = CURRENT_SCHEMA_VERSION
    hard_policy_version: int = 1
    semantic_catalog_version: str = ""

    def __post_init__(self) -> None:
        if not self.block_id:
            object.__setattr__(self, "block_id", f"mem_{uuid.uuid4().hex[:16]}")
        _normalize_memory_axes(self, kind_field="kind")
        _normalize_record_identity(self)
        _bind_record_block_membership(self)
        now = _utc_now()
        if not self.created_at:
            object.__setattr__(self, "created_at", now)
        if not self.updated_at:
            object.__setattr__(self, "updated_at", now)


def memory_scope_key(record: MemoryEvent | MemoryBlockRecord) -> str:
    identity_field = {
        "subtask": "subtask_id",
        "task": "task_id",
        "session": "session_id",
        "workspace": "workspace_id",
        "project": "project_id",
        "user_global": "user_id",
    }[record.scope]
    scope_id = str(getattr(record, identity_field) or "").strip()
    if not scope_id:
        raise ValueError(f"{record.scope} scope requires {identity_field}")
    return f"{record.scope}:{scope_id}"


@dataclass(frozen=True)
class RejectedApproach:
    approach: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approach",
            _required_text(self.approach, field_name="rejected approach"),
        )
        object.__setattr__(
            self,
            "reason",
            _required_text(self.reason, field_name="rejected reason"),
        )


@dataclass(frozen=True)
class ContinuationLiteralSet:
    dimension: ContinuationDimension
    literals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        dimension = _closed_value(
            self.dimension,
            field_name="continuation dimension",
            shape=ContinuationDimension,
        )
        object.__setattr__(self, "dimension", cast(ContinuationDimension, dimension))
        if not isinstance(self.literals, (tuple, list)) or any(
            not isinstance(value, str) or not value.strip() for value in self.literals
        ):
            raise ValueError(f"{dimension} literals require exact non-empty strings")
        object.__setattr__(
            self,
            "literals",
            tuple(dict.fromkeys(self.literals)),
        )


def _continuation_literal_manifest(value: object) -> tuple[ContinuationLiteralSet, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("exact_literal_manifest must be a list or tuple")
    manifest: list[ContinuationLiteralSet] = []
    for item in value:
        if not isinstance(item, ContinuationLiteralSet):
            raise ValueError("exact_literal_manifest must contain ContinuationLiteralSet values")
        if any(existing.dimension == item.dimension for existing in manifest):
            raise ValueError(f"exact_literal_manifest repeats {item.dimension}")
        manifest.append(item)
    return tuple(manifest)


@dataclass(frozen=True)
class ContinuationSufficiencyAssessment:
    workspace_id: str
    session_id: str
    task_id: str
    branch_id: str
    branch_completed: bool
    branch_isolated: bool
    raw_page_back_verified: bool
    projection_sufficiency: ContinuationSufficiency
    source_event_sha256: str
    exact_literal_manifest: tuple[ContinuationLiteralSet, ...]
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        for field_name in ("workspace_id", "session_id", "task_id", "branch_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("branch_completed", "branch_isolated", "raw_page_back_verified"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a bool")
        object.__setattr__(
            self,
            "source_event_sha256",
            _require_sha256_ref(self.source_event_sha256, field_name="source_event_sha256"),
        )
        projection_sufficiency = _closed_value(
            self.projection_sufficiency,
            field_name="projection_sufficiency",
            shape=ContinuationSufficiency,
        )
        object.__setattr__(
            self,
            "projection_sufficiency",
            cast(ContinuationSufficiency, projection_sufficiency),
        )
        manifest = _continuation_literal_manifest(self.exact_literal_manifest)
        expected = _literal_values(ContinuationDimension)
        if {item.dimension for item in manifest} != expected:
            raise ValueError("exact_literal_manifest must cover every continuation dimension")
        object.__setattr__(self, "exact_literal_manifest", manifest)
        refs = _evidence_refs(self.evidence_refs)
        if not refs:
            raise ValueError("continuation sufficiency assessment requires evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)

    @property
    def covered_dimensions(self) -> tuple[ContinuationDimension, ...]:
        return tuple(item.dimension for item in self.exact_literal_manifest)

    @property
    def sufficient(self) -> bool:
        return (
            self.branch_completed
            and self.branch_isolated
            and self.raw_page_back_verified
            and self.projection_sufficiency == "sufficient"
        )


@dataclass(frozen=True)
class WorkingStateCheckpoint:
    workspace_id: str
    profile_id: str
    owner_generation: str
    requested_block_id: str
    working_view: str
    source_ref_sha256: tuple[tuple[str, str], ...]
    host_event_id: str
    session_id: str
    task_id: str
    source_event_refs: tuple[str, ...]
    source_event_sha256: str
    task_anchor_refs: tuple[str, ...]
    system_anchor_refs: tuple[str, ...]
    exact_recent_tail_refs: tuple[str, ...]
    evidence_page_refs: tuple[tuple[str, ...], ...]
    rollback_or_page_back_ref: str
    valid_from: str
    valid_to: str
    schema_version: int
    hard_policy_version: int
    semantic_catalog_version: str
    host_version: str
    model_version: str
    prompt_version: str
    continuation_sufficiency: ContinuationSufficiency
    evidence_refs: tuple[EvidenceRef, ...]
    exact_literal_manifest: tuple[ContinuationLiteralSet, ...] = ()
    summary_source_refs: tuple[str, ...] = ()
    transitive_raw_refs: tuple[str, ...] = ()
    preservation_verified: bool = False
    repository_id: str = ""
    checkout: str = ""
    worktree: str = ""
    branch: str = ""
    revision: str = ""
    dirty_state: str = ""
    touched_files: tuple[str, ...] = ()
    touched_symbols: tuple[str, ...] = ()
    current_diff: str = ""
    tests: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    parent_session_id: str = ""
    artifact_id: str = ""
    artifact_revision: str = ""
    project_id: str = ""
    current_goal: str = ""
    plan: tuple[str, ...] = ()
    current_subgoal: str = ""
    unresolved_constraints: tuple[str, ...] = ()
    open_hypotheses: tuple[str, ...] = ()
    verified_hypotheses: tuple[str, ...] = ()
    rejected_approaches: tuple[RejectedApproach, ...] = ()
    safety_state: tuple[str, ...] = ()
    open_branch_ids: tuple[str, ...] = ()
    completed_branch_ids: tuple[str, ...] = ()
    isolated_branch_ids: tuple[str, ...] = ()
    pending_prospective_obligations: tuple[str, ...] = ()
    remaining_work: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "WorkingStateCheckpoint":
        unknown = set(payload) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError("unknown working checkpoint field: " + min(unknown))
        values: dict[str, Any] = dict(payload)
        for name in ("source_ref_sha256", "evidence_page_refs"):
            raw = values.get(name, ())
            if not isinstance(raw, (tuple, list)) or any(
                not isinstance(item, (tuple, list)) for item in raw
            ):
                raise ValueError(f"checkpoint {name} requires an array of arrays")
            values[name] = tuple(tuple(item) for item in raw)
        for name, constructor in (
            ("evidence_refs", EvidenceRef.from_mapping),
            ("exact_literal_manifest", lambda value: ContinuationLiteralSet(**value)),
            ("rejected_approaches", lambda value: RejectedApproach(**value)),
        ):
            raw = values.get(name, ())
            if not isinstance(raw, (tuple, list)) or any(
                not isinstance(item, Mapping) for item in raw
            ):
                raise ValueError(f"checkpoint {name} requires typed objects")
            values[name] = tuple(constructor(item) for item in raw)
        return cls(**values)

    def __post_init__(self) -> None:
        _normalize_record_identity(self)
        for field_name in [
            "workspace_id",
            "profile_id",
            "owner_generation",
            "requested_block_id",
            "host_event_id",
            "session_id",
            "task_id",
            "valid_from",
            "valid_to",
            "semantic_catalog_version",
            "host_version",
            "model_version",
            "prompt_version",
            "rollback_or_page_back_ref",
        ]:
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.working_view, str) or not self.working_view.strip():
            raise ValueError("working_view must contain model-authored continuation text")
        object.__setattr__(
            self,
            "source_event_sha256",
            _require_sha256_ref(self.source_event_sha256, field_name="source_event_sha256"),
        )
        for field_name in (
            "source_event_refs",
            "task_anchor_refs",
            "system_anchor_refs",
            "exact_recent_tail_refs",
        ):
            refs = _text_tuple(getattr(self, field_name), field_name=field_name)
            if not refs:
                raise ValueError(f"{field_name} must contain at least one raw reference")
            object.__setattr__(self, field_name, refs)
        if not isinstance(self.evidence_page_refs, (list, tuple)) or not self.evidence_page_refs:
            raise ValueError("evidence_page_refs must contain at least one page")
        pages: list[tuple[str, ...]] = []
        for page in self.evidence_page_refs:
            refs = _text_tuple(page, field_name="evidence_page_refs")
            if not refs:
                raise ValueError("evidence_page_refs pages must not be empty")
            pages.append(refs)
        object.__setattr__(self, "evidence_page_refs", tuple(pages))
        if not isinstance(self.source_ref_sha256, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.source_ref_sha256
        ):
            raise ValueError("source_ref_sha256 requires exact reference/hash pairs")
        hashes = tuple(
            (
                _required_text(ref, field_name="source_ref"),
                _require_sha256_ref(digest, field_name="source_ref_sha256"),
            )
            for ref, digest in self.source_ref_sha256
        )
        referenced = {
            *self.source_event_refs,
            *self.task_anchor_refs,
            *self.system_anchor_refs,
            *self.exact_recent_tail_refs,
            *(ref for page in self.evidence_page_refs for ref in page),
            *self.transitive_raw_refs,
        }
        if len(dict(hashes)) != len(hashes) or set(dict(hashes)) != referenced:
            raise ValueError("source_ref_sha256 must pin every exact reference once")
        object.__setattr__(self, "source_ref_sha256", hashes)
        for field_name in [
            "plan",
            "unresolved_constraints",
            "touched_files",
            "touched_symbols",
            "tests",
            "errors",
            "open_hypotheses",
            "verified_hypotheses",
            "safety_state",
            "open_branch_ids",
            "completed_branch_ids",
            "isolated_branch_ids",
            "pending_prospective_obligations",
            "remaining_work",
            "summary_source_refs",
            "transitive_raw_refs",
        ]:
            object.__setattr__(
                self,
                field_name,
                _text_tuple(getattr(self, field_name), field_name=field_name),
            )
        approaches = tuple(self.rejected_approaches)
        if any(not isinstance(item, RejectedApproach) for item in approaches):
            raise ValueError("rejected_approaches must contain RejectedApproach values")
        object.__setattr__(self, "rejected_approaches", approaches)
        if set(self.open_branch_ids) & set(self.completed_branch_ids):
            raise ValueError("open and completed branch ids must be disjoint")
        if not isinstance(self.preservation_verified, bool):
            raise ValueError("preservation_verified must be a bool")
        if self.summary_source_refs and not self.transitive_raw_refs:
            raise ValueError("recursive summary requires transitive_raw_refs")
        if self.summary_source_refs and not self.preservation_verified:
            raise ValueError("recursive summary requires preservation_verified")
        referenced_raw_events = {
            *self.source_event_refs,
            *self.task_anchor_refs,
            *self.system_anchor_refs,
            *self.exact_recent_tail_refs,
            *(ref for page in self.evidence_page_refs for ref in page),
        }
        if any(ref not in referenced_raw_events for ref in self.transitive_raw_refs):
            raise ValueError("transitive_raw_refs must name referenced immutable raw events")
        sufficiency = _closed_value(
            self.continuation_sufficiency,
            field_name="continuation_sufficiency",
            shape=ContinuationSufficiency,
        )
        object.__setattr__(
            self,
            "continuation_sufficiency",
            cast(ContinuationSufficiency, sufficiency),
        )
        expected_manifest = _working_state_exact_literal_manifest(self)
        manifest = (
            _continuation_literal_manifest(self.exact_literal_manifest)
            if self.exact_literal_manifest
            else expected_manifest
        )
        if manifest != expected_manifest:
            raise ValueError("exact_literal_manifest does not match working state")
        object.__setattr__(self, "exact_literal_manifest", manifest)
        evidence = _evidence_refs(self.evidence_refs)
        if not evidence:
            raise ValueError("working-state checkpoint requires evidence_refs")
        object.__setattr__(self, "evidence_refs", evidence)


def _working_state_exact_literal_manifest(
    checkpoint: WorkingStateCheckpoint,
) -> tuple[ContinuationLiteralSet, ...]:
    return (
        ContinuationLiteralSet(
            dimension="goals_constraints",
            literals=(
                checkpoint.working_view,
                *((checkpoint.current_goal,) if checkpoint.current_goal else ()),
                *((checkpoint.current_subgoal,) if checkpoint.current_subgoal else ()),
                *checkpoint.plan,
                *checkpoint.unresolved_constraints,
            ),
        ),
        ContinuationLiteralSet(
            dimension="paths_symbols",
            literals=(*checkpoint.touched_files, *checkpoint.touched_symbols),
        ),
        ContinuationLiteralSet(
            dimension="diff", literals=(checkpoint.current_diff,) if checkpoint.current_diff else ()
        ),
        ContinuationLiteralSet(dimension="error_literals", literals=checkpoint.errors),
        ContinuationLiteralSet(dimension="tests", literals=checkpoint.tests),
        ContinuationLiteralSet(
            dimension="open_hypotheses",
            literals=checkpoint.open_hypotheses,
        ),
        ContinuationLiteralSet(
            dimension="rejected_reasons",
            literals=tuple(item.reason for item in checkpoint.rejected_approaches),
        ),
        ContinuationLiteralSet(dimension="safety_state", literals=checkpoint.safety_state),
        ContinuationLiteralSet(
            dimension="raw_page_back_coverage",
            literals=(
                *checkpoint.source_event_refs,
                *checkpoint.task_anchor_refs,
                *checkpoint.system_anchor_refs,
                *checkpoint.exact_recent_tail_refs,
                *(ref for page in checkpoint.evidence_page_refs for ref in page),
                *checkpoint.transitive_raw_refs,
                checkpoint.rollback_or_page_back_ref,
            ),
        ),
    )


def _require_sha256_ref(value: object, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    digest = text.removeprefix("sha256:")
    if not text.startswith("sha256:") or len(digest) != 64:
        raise ValueError(f"{field_name} must be a sha256 reference")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a sha256 reference") from exc
    return text


@dataclass(frozen=True)
class FoldedBranch:
    workspace_id: str
    session_id: str
    task_id: str
    branch_id: str
    source_event_refs: tuple[str, ...]
    source_event_sha256: str
    schema_version: int
    hard_policy_version: int
    semantic_catalog_version: str
    host_version: str
    model_version: str
    prompt_version: str
    exact_literal_manifest: tuple[ContinuationLiteralSet, ...]
    sufficiency_assessment: ContinuationSufficiencyAssessment
    continuation_sufficiency: ContinuationSufficiency
    expires_at: str
    rollback_or_page_back_ref: str
    result: str = ""
    remaining_risks: tuple[str, ...] = ()
    raw_span_refs: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        for field_name in [
            "workspace_id",
            "session_id",
            "task_id",
            "branch_id",
            "semantic_catalog_version",
            "host_version",
            "model_version",
            "prompt_version",
            "expires_at",
            "rollback_or_page_back_ref",
        ]:
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        refs = _text_tuple(self.source_event_refs, field_name="source_event_refs")
        if not refs:
            raise ValueError("source_event_refs must contain at least one raw reference")
        object.__setattr__(self, "source_event_refs", refs)
        object.__setattr__(
            self,
            "source_event_sha256",
            _require_sha256_ref(self.source_event_sha256, field_name="source_event_sha256"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _positive_version(self.schema_version, field_name="schema_version"),
        )
        object.__setattr__(
            self,
            "hard_policy_version",
            _positive_version(self.hard_policy_version, field_name="hard_policy_version"),
        )
        expires_at, parsed_expiry = _valid_time(self.expires_at, field_name="expires_at")
        if parsed_expiry is None:
            raise ValueError("expires_at must be an ISO-8601 timestamp")
        object.__setattr__(self, "expires_at", expires_at)
        manifest = _continuation_literal_manifest(self.exact_literal_manifest)
        object.__setattr__(self, "exact_literal_manifest", manifest)
        if not isinstance(self.sufficiency_assessment, ContinuationSufficiencyAssessment):
            raise TypeError("sufficiency_assessment must be a ContinuationSufficiencyAssessment")
        assessment = self.sufficiency_assessment
        if (
            assessment.workspace_id,
            assessment.session_id,
            assessment.task_id,
            assessment.branch_id,
        ) != (self.workspace_id, self.session_id, self.task_id, self.branch_id):
            raise ValueError("folded branch sufficiency assessment scope does not match")
        if assessment.source_event_sha256 != self.source_event_sha256:
            raise ValueError("folded branch sufficiency source hash does not match")
        if assessment.exact_literal_manifest != manifest or not assessment.sufficient:
            raise ValueError("folded branch requires a sufficient exact-literal assessment")
        sufficiency = _closed_value(
            self.continuation_sufficiency,
            field_name="continuation_sufficiency",
            shape=ContinuationSufficiency,
        )
        if sufficiency != "sufficient":
            raise ValueError("folded branch continuation_sufficiency must be sufficient")
        object.__setattr__(
            self,
            "continuation_sufficiency",
            cast(ContinuationSufficiency, sufficiency),
        )
        object.__setattr__(self, "result", _required_text(self.result, field_name="result"))
        object.__setattr__(
            self,
            "remaining_risks",
            _text_tuple(self.remaining_risks, field_name="remaining_risks"),
        )
        raw_span_refs = _text_tuple(self.raw_span_refs, field_name="raw_span_refs")
        if not raw_span_refs:
            raise ValueError("raw_span_refs must contain at least one raw reference")
        object.__setattr__(
            self,
            "raw_span_refs",
            raw_span_refs,
        )
        evidence_refs = _evidence_refs(self.evidence_refs)
        if not evidence_refs:
            raise ValueError("folded branch requires evidence_refs")
        object.__setattr__(self, "evidence_refs", evidence_refs)


@dataclass(frozen=True)
class ContinuationPacket:
    workspace_id: str
    profile_id: str
    owner_generation: str
    session_id: str
    task_id: str
    checkpoint: WorkingStateCheckpoint
    source_event_range: tuple[str, ...]
    source_event_sha256: str
    task_anchor_refs: tuple[str, ...]
    system_anchor_refs: tuple[str, ...]
    exact_recent_tail_refs: tuple[str, ...]
    evidence_page_refs: tuple[tuple[str, ...], ...]
    schema_version: int
    hard_policy_version: int
    semantic_catalog_version: str
    host_version: str
    model_version: str
    prompt_version: str
    exact_literal_manifest: tuple[ContinuationLiteralSet, ...]
    continuation_sufficiency: ContinuationSufficiency
    expires_at: str
    rollback_or_page_back_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, WorkingStateCheckpoint):
            raise TypeError("checkpoint must be a WorkingStateCheckpoint")
        for field_name in (
            "workspace_id",
            "profile_id",
            "owner_generation",
            "session_id",
            "task_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if (
            self.workspace_id,
            self.profile_id,
            self.owner_generation,
            self.session_id,
            self.task_id,
        ) != (
            self.checkpoint.workspace_id,
            self.checkpoint.profile_id,
            self.checkpoint.owner_generation,
            self.checkpoint.session_id,
            self.checkpoint.task_id,
        ):
            raise ValueError("continuation packet checkpoint scope does not match")
        expected_refs = {
            "source_event_range": self.checkpoint.source_event_refs,
            "task_anchor_refs": self.checkpoint.task_anchor_refs,
            "system_anchor_refs": self.checkpoint.system_anchor_refs,
            "exact_recent_tail_refs": self.checkpoint.exact_recent_tail_refs,
        }
        for field_name, expected in expected_refs.items():
            refs = _text_tuple(getattr(self, field_name), field_name=field_name)
            if refs != expected:
                raise ValueError(f"continuation packet {field_name} does not match checkpoint")
            object.__setattr__(self, field_name, refs)
        pages: list[tuple[str, ...]] = []
        if not isinstance(self.evidence_page_refs, (list, tuple)):
            raise ValueError("evidence_page_refs must be a list or tuple")
        for page in self.evidence_page_refs:
            pages.append(_text_tuple(page, field_name="evidence_page_refs"))
        if tuple(pages) != self.checkpoint.evidence_page_refs:
            raise ValueError("continuation packet evidence_page_refs does not match checkpoint")
        object.__setattr__(self, "evidence_page_refs", tuple(pages))
        object.__setattr__(
            self,
            "source_event_sha256",
            _require_sha256_ref(self.source_event_sha256, field_name="source_event_sha256"),
        )
        if self.source_event_sha256 != self.checkpoint.source_event_sha256:
            raise ValueError("continuation packet source_event_sha256 does not match checkpoint")
        schema_version = _positive_version(self.schema_version, field_name="schema_version")
        if schema_version != self.checkpoint.schema_version:
            raise ValueError("continuation packet schema_version does not match checkpoint")
        object.__setattr__(self, "schema_version", schema_version)
        hard_policy_version = _positive_version(
            self.hard_policy_version,
            field_name="hard_policy_version",
        )
        if hard_policy_version != self.checkpoint.hard_policy_version:
            raise ValueError("continuation packet hard_policy_version does not match checkpoint")
        object.__setattr__(self, "hard_policy_version", hard_policy_version)
        expected_text = {
            "semantic_catalog_version": self.checkpoint.semantic_catalog_version,
            "host_version": self.checkpoint.host_version,
            "model_version": self.checkpoint.model_version,
            "prompt_version": self.checkpoint.prompt_version,
            "expires_at": self.checkpoint.valid_to,
            "rollback_or_page_back_ref": self.checkpoint.rollback_or_page_back_ref,
        }
        for field_name, expected_value in expected_text.items():
            value = _required_text(getattr(self, field_name), field_name=field_name)
            if value != expected_value:
                raise ValueError(f"continuation packet {field_name} does not match checkpoint")
            object.__setattr__(self, field_name, value)
        manifest = _continuation_literal_manifest(self.exact_literal_manifest)
        if manifest != self.checkpoint.exact_literal_manifest:
            raise ValueError("continuation packet exact_literal_manifest does not match checkpoint")
        object.__setattr__(self, "exact_literal_manifest", manifest)
        sufficiency = _closed_value(
            self.continuation_sufficiency,
            field_name="continuation_sufficiency",
            shape=ContinuationSufficiency,
        )
        if sufficiency != self.checkpoint.continuation_sufficiency:
            raise ValueError("continuation packet sufficiency does not match checkpoint")
        object.__setattr__(
            self,
            "continuation_sufficiency",
            cast(ContinuationSufficiency, sufficiency),
        )


@dataclass(frozen=True)
class RecallFrame:
    query: str
    workspace_id: str = ""
    run_id: str = ""
    task_id: str = ""
    subgoal_id: str = ""
    subtask_id: str = ""
    episode_id: str = ""
    scope_chain: tuple[str, ...] = ()
    transition_reason: str = ""
    repository_id: str = ""
    worktree: str = ""
    revision_id: str = ""
    dirty_state: str = "unknown"
    actor_id: str = ""
    speaker_id: str = ""
    belief_holder_id: str = ""
    relevant_entities: tuple[str, ...] = ()
    relevant_paths: tuple[str, ...] = ()
    relevant_symbols: tuple[str, ...] = ()
    relevant_configs: tuple[str, ...] = ()
    event_time: str = ""
    mention_time: str = ""
    observation_time: str = ""
    query_time: str = ""
    tool_name: str = ""
    tool_state: str = ""
    environment_state: tuple[str, ...] = ()
    desired_temporal_mode: RecallTemporalMode = "current"
    required_evidence_roles: tuple[str, ...] = ()
    expected_source_types: tuple[str, ...] = ()
    uncertainty: float = 1.0
    coverage: float = 0.0
    widening_policy: RecallWideningPolicy = "backend_only"
    widening_allowed: bool = True
    auto_derived_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _required_text(self.query, field_name="query"))
        for field_name in (
            "workspace_id",
            "run_id",
            "task_id",
            "subgoal_id",
            "subtask_id",
            "episode_id",
            "transition_reason",
            "repository_id",
            "worktree",
            "revision_id",
            "actor_id",
            "speaker_id",
            "belief_holder_id",
            "tool_name",
            "tool_state",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name=field_name),
            )
        if self.dirty_state not in {"clean", "dirty", "unknown"}:
            raise ValueError("dirty_state must be clean, dirty, or unknown")
        for field_name in (
            "scope_chain",
            "relevant_entities",
            "relevant_paths",
            "relevant_symbols",
            "relevant_configs",
            "environment_state",
            "required_evidence_roles",
            "expected_source_types",
            "auto_derived_fields",
        ):
            object.__setattr__(
                self,
                field_name,
                _text_tuple(getattr(self, field_name), field_name=field_name, maximum=1024),
            )
        for field_name in ("event_time", "mention_time", "observation_time", "query_time"):
            normalized, _ = _valid_time(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, normalized)
        temporal_mode = _closed_value(
            self.desired_temporal_mode,
            field_name="desired_temporal_mode",
            shape=RecallTemporalMode,
        )
        object.__setattr__(self, "desired_temporal_mode", cast(RecallTemporalMode, temporal_mode))
        widening_policy = _closed_value(
            self.widening_policy,
            field_name="widening_policy",
            shape=RecallWideningPolicy,
        )
        object.__setattr__(self, "widening_policy", cast(RecallWideningPolicy, widening_policy))
        if type(self.widening_allowed) is not bool:
            raise ValueError("widening_allowed must be a bool")
        for field_name in ("uncertainty", "coverage"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number between 0 and 1")
            normalized_value = float(value)
            if not 0.0 <= normalized_value <= 1.0:
                raise ValueError(f"{field_name} must be a number between 0 and 1")
            object.__setattr__(self, field_name, normalized_value)
        allowed_derived_fields = set(self.__dataclass_fields__) - {
            "query",
            "workspace_id",
            "run_id",
            "auto_derived_fields",
        }
        unknown_derived_fields = sorted(set(self.auto_derived_fields) - allowed_derived_fields)
        if unknown_derived_fields:
            raise ValueError(
                "unknown RecallFrame auto_derived_fields: " + ", ".join(unknown_derived_fields)
            )


_PROSPECTIVE_TRANSITIONS = {
    "pending": frozenset({"armed", "cancelled", "expired"}),
    "armed": frozenset({"fired", "cancelled", "expired"}),
    "fired": frozenset({"satisfied", "cancelled", "expired"}),
    "satisfied": frozenset(),
    "cancelled": frozenset(),
    "expired": frozenset(),
}


@dataclass(frozen=True)
class AuthenticatedHostEvent:
    event_id: str
    event_type: str
    principal_id: str
    channel_id: str
    authentication_evidence_ref: EvidenceRef
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.authentication_evidence_ref, EvidenceRef) or (
            self.authentication_evidence_ref.ref_type not in {"ledger_event", "session_event"}
            or self.authentication_evidence_ref.produced_by not in {"supervisor", "harness"}
            or not self.authentication_evidence_ref.hash
        ):
            raise ValueError(
                "authenticated host event requires authentication evidence from the host gate"
            )
        if not isinstance(self.payload, dict) or any(
            not isinstance(key, str) or not key.strip() for key in self.payload
        ):
            raise ValueError("authenticated host event payload must use non-empty string keys")
        reserved = {"event_id", "event_type", "principal_id", "channel_id"}
        if reserved.intersection(self.payload):
            raise ValueError("authenticated host event payload cannot replace envelope fields")
        object.__setattr__(self, "event_id", _required_text(self.event_id, field_name="event_id"))
        object.__setattr__(
            self, "event_type", _required_text(self.event_type, field_name="event_type")
        )
        object.__setattr__(
            self, "principal_id", _required_text(self.principal_id, field_name="principal_id")
        )
        object.__setattr__(
            self, "channel_id", _required_text(self.channel_id, field_name="channel_id")
        )
        object.__setattr__(self, "payload", dict(self.payload))

    def trigger_values(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "principal_id": self.principal_id,
            "channel_id": self.channel_id,
            **self.payload,
        }


@dataclass(frozen=True)
class ProspectiveTriggerPredicate:
    predicate_type: ProspectivePredicateType
    field: str
    operator: ProspectiveTriggerOperator
    value: str | int | bool = ""

    def __post_init__(self) -> None:
        predicate_type = _closed_value(
            self.predicate_type,
            field_name="prospective predicate_type",
            shape=ProspectivePredicateType,
        )
        operator = _closed_value(
            self.operator,
            field_name="prospective trigger operator",
            shape=ProspectiveTriggerOperator,
        )
        if not isinstance(self.value, (str, int, bool)):
            raise ValueError("prospective trigger value must be a string, integer, or boolean")
        value = self.value
        if isinstance(value, str):
            value = _optional_text(value, field_name="prospective trigger value")
        if operator != "exists" and value == "":
            raise ValueError("prospective trigger value is required")
        object.__setattr__(self, "predicate_type", cast(ProspectivePredicateType, predicate_type))
        object.__setattr__(self, "field", _required_text(self.field, field_name="field"))
        object.__setattr__(self, "operator", cast(ProspectiveTriggerOperator, operator))
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class ProspectiveTelemetry:
    false_positive_count: int = 0
    missed_trigger_count: int = 0
    overloaded_trigger_count: int = 0
    attention_interruption_count: int = 0
    duplicate_fire_count: int = 0
    late_trigger_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "false_positive_count",
            "missed_trigger_count",
            "overloaded_trigger_count",
            "attention_interruption_count",
            "duplicate_fire_count",
            "late_trigger_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


def _prospective_predicates(
    value: object,
    *,
    field_name: str,
) -> tuple[ProspectiveTriggerPredicate, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    predicates: list[ProspectiveTriggerPredicate] = []
    for item in value:
        if not isinstance(item, ProspectiveTriggerPredicate):
            raise ValueError(f"{field_name} must contain ProspectiveTriggerPredicate values")
        if item not in predicates:
            predicates.append(item)
    return tuple(predicates)


_PROSPECTIVE_EVENT_TRANSITIONS: dict[
    ProspectiveLifecycleEventType,
    tuple[frozenset[ProspectiveStatus], ProspectiveStatus],
] = {
    "arm": (frozenset({"pending"}), "armed"),
    "fire": (frozenset({"armed"}), "fired"),
    "satisfy": (frozenset({"fired"}), "satisfied"),
    "cancel": (frozenset({"pending", "armed", "fired"}), "cancelled"),
    "expire": (frozenset({"pending", "armed", "fired"}), "expired"),
}


@dataclass(frozen=True)
class ProspectiveLifecycleEvent:
    event_id: str
    event_type: ProspectiveLifecycleEventType
    expected_version: int
    expected_status: ProspectiveStatus
    evidence_refs: tuple[EvidenceRef, ...]
    idempotency_key: str = ""
    satisfaction_evidence_refs: tuple[EvidenceRef, ...] = ()
    satisfaction_verifier: str = ""
    satisfaction_verifier_result: Literal["", "passed", "failed"] = ""

    def __post_init__(self) -> None:
        event_type = cast(
            ProspectiveLifecycleEventType,
            _closed_value(
                self.event_type,
                field_name="prospective lifecycle event_type",
                shape=ProspectiveLifecycleEventType,
            ),
        )
        expected_status = cast(
            ProspectiveStatus,
            _closed_value(
                self.expected_status,
                field_name="prospective expected_status",
                shape=ProspectiveStatus,
            ),
        )
        if (
            isinstance(self.expected_version, bool)
            or not isinstance(self.expected_version, int)
            or self.expected_version < 0
        ):
            raise ValueError("prospective expected_version must be a non-negative integer")
        event_id = _required_text(self.event_id, field_name="prospective event_id")
        idempotency_key = _optional_text(
            self.idempotency_key, field_name="prospective idempotency_key"
        )
        evidence_refs = _evidence_refs(self.evidence_refs)
        if not evidence_refs:
            raise ValueError("prospective lifecycle event requires evidence_refs")
        allowed_from, _ = _PROSPECTIVE_EVENT_TRANSITIONS[event_type]
        if expected_status not in allowed_from:
            allowed = ", ".join(sorted(allowed_from))
            raise ValueError(f"{event_type} event requires {allowed} expected status")
        satisfaction_refs = _evidence_refs(self.satisfaction_evidence_refs)
        verifier = _optional_text(self.satisfaction_verifier, field_name="satisfaction_verifier")
        verifier_result = _optional_text(
            self.satisfaction_verifier_result, field_name="satisfaction_verifier_result"
        )
        if verifier_result not in {"", "passed", "failed"}:
            raise ValueError("satisfaction_verifier_result is invalid")
        if event_type == "satisfy":
            if not satisfaction_refs or not verifier or verifier_result != "passed":
                raise ValueError(
                    "satisfy event requires proof-of-satisfaction and passed verifier result"
                )
        elif satisfaction_refs or verifier_result:
            raise ValueError("proof-of-satisfaction is valid only for satisfy events")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "expected_status", expected_status)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "idempotency_key", idempotency_key)
        object.__setattr__(self, "satisfaction_evidence_refs", satisfaction_refs)
        object.__setattr__(self, "satisfaction_verifier", verifier)
        object.__setattr__(self, "satisfaction_verifier_result", verifier_result)

    @property
    def status(self) -> ProspectiveStatus:
        return _PROSPECTIVE_EVENT_TRANSITIONS[self.event_type][1]


@dataclass(frozen=True)
class ProspectiveObligation:
    obligation_id: str
    scope: ProspectiveScope
    evidence_refs: tuple[EvidenceRef, ...]
    scope_id: str = ""
    required_intervention: str = ""
    trigger_predicates: tuple[ProspectiveTriggerPredicate, ...] = ()
    semantic_trigger_cues: tuple[str, ...] = ()
    earliest_at: str = ""
    deadline_at: str = ""
    expires_at: str = ""
    recurrence_policy: ProspectiveRecurrencePolicy = "once"
    recurrence_interval_seconds: int = 0
    suppress_predicates: tuple[ProspectiveTriggerPredicate, ...] = ()
    cancel_predicates: tuple[ProspectiveTriggerPredicate, ...] = ()
    satisfaction_verifier: str = ""
    satisfaction_verifier_result: Literal["", "passed", "failed"] = ""
    satisfaction_evidence_refs: tuple[EvidenceRef, ...] = ()
    telemetry: ProspectiveTelemetry = field(default_factory=ProspectiveTelemetry)
    status: ProspectiveStatus = "pending"
    transition_from: ProspectiveStatus | None = None
    transition_evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        status = _closed_value(
            self.status, field_name="prospective status", shape=ProspectiveStatus
        )
        refs = _evidence_refs(self.evidence_refs)
        if not refs:
            raise ValueError("prospective obligation requires evidence_refs")
        object.__setattr__(
            self, "obligation_id", _required_text(self.obligation_id, field_name="obligation_id")
        )
        scope = cast(
            ProspectiveScope,
            _closed_value(self.scope, field_name="prospective scope", shape=ProspectiveScope),
        )
        scope_id = _optional_text(self.scope_id, field_name="scope_id")
        if scope != "workspace" and not scope_id:
            raise ValueError("non-workspace prospective scope requires scope_id")
        if scope == "workspace" and scope_id:
            raise ValueError("workspace prospective scope cannot carry scope_id")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "status", cast(ProspectiveStatus, status))
        object.__setattr__(self, "evidence_refs", refs)
        trigger_predicates = _prospective_predicates(
            self.trigger_predicates, field_name="trigger_predicates"
        )
        suppress_predicates = _prospective_predicates(
            self.suppress_predicates, field_name="suppress_predicates"
        )
        cancel_predicates = _prospective_predicates(
            self.cancel_predicates, field_name="cancel_predicates"
        )
        semantic_cues = _text_tuple(
            self.semantic_trigger_cues,
            field_name="semantic_trigger_cues",
            maximum=256,
        )
        intervention = _optional_text(
            self.required_intervention, field_name="required_intervention"
        )
        if (trigger_predicates or semantic_cues) and not intervention:
            raise ValueError("prospective trigger requires required_intervention")
        recurrence = _closed_value(
            self.recurrence_policy,
            field_name="prospective recurrence_policy",
            shape=ProspectiveRecurrencePolicy,
        )
        if (
            isinstance(self.recurrence_interval_seconds, bool)
            or not isinstance(self.recurrence_interval_seconds, int)
            or self.recurrence_interval_seconds < 0
        ):
            raise ValueError("recurrence_interval_seconds must be a non-negative integer")
        if recurrence == "repeat" and self.recurrence_interval_seconds <= 0:
            raise ValueError("repeat recurrence requires recurrence_interval_seconds")
        if recurrence == "once" and self.recurrence_interval_seconds != 0:
            raise ValueError("once recurrence cannot carry recurrence_interval_seconds")
        earliest_at, earliest = _valid_time(self.earliest_at, field_name="earliest_at")
        deadline_at, deadline = _valid_time(self.deadline_at, field_name="deadline_at")
        expires_at, expires = _valid_time(self.expires_at, field_name="expires_at")
        if intervention and not (trigger_predicates or semantic_cues or deadline_at):
            raise ValueError(
                "prospective intervention cannot use an unconditioned always-remind strategy"
            )
        if earliest is not None and deadline is not None and deadline < earliest:
            raise ValueError("deadline_at must not precede earliest_at")
        if earliest is not None and expires is not None and expires < earliest:
            raise ValueError("expires_at must not precede earliest_at")
        if deadline is not None and expires is not None and expires < deadline:
            raise ValueError("expires_at must not precede deadline_at")
        satisfaction_refs = _evidence_refs(self.satisfaction_evidence_refs)
        verifier = _optional_text(self.satisfaction_verifier, field_name="satisfaction_verifier")
        verifier_result = _optional_text(
            self.satisfaction_verifier_result, field_name="satisfaction_verifier_result"
        )
        if verifier_result not in {"", "passed", "failed"}:
            raise ValueError("satisfaction_verifier_result is invalid")
        if not isinstance(self.telemetry, ProspectiveTelemetry):
            raise ValueError("telemetry must use ProspectiveTelemetry")
        if status != "satisfied" and satisfaction_refs:
            raise ValueError("proof-of-satisfaction is valid only for satisfied obligations")
        object.__setattr__(self, "required_intervention", intervention)
        object.__setattr__(self, "trigger_predicates", trigger_predicates)
        object.__setattr__(self, "semantic_trigger_cues", semantic_cues)
        object.__setattr__(self, "earliest_at", earliest_at)
        object.__setattr__(self, "deadline_at", deadline_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "recurrence_policy", cast(ProspectiveRecurrencePolicy, recurrence))
        object.__setattr__(self, "suppress_predicates", suppress_predicates)
        object.__setattr__(self, "cancel_predicates", cancel_predicates)
        object.__setattr__(self, "satisfaction_verifier", verifier)
        object.__setattr__(self, "satisfaction_verifier_result", verifier_result)
        object.__setattr__(self, "satisfaction_evidence_refs", satisfaction_refs)
        if status == "pending":
            if self.transition_from is not None or self.transition_evidence_refs:
                raise ValueError("pending prospective obligation cannot carry a transition")
            return
        if self.transition_from is None:
            raise ValueError("prospective state requires transition_from")
        previous = _closed_value(
            self.transition_from,
            field_name="prospective transition_from",
            shape=ProspectiveStatus,
        )
        if status not in _PROSPECTIVE_TRANSITIONS[previous]:
            raise ValueError(f"invalid prospective transition: {previous} -> {status}")
        transition_refs = _evidence_refs(self.transition_evidence_refs)
        if not transition_refs:
            raise ValueError("prospective transition requires evidence_refs")
        if status == "satisfied" and (
            not satisfaction_refs or not verifier or verifier_result != "passed"
        ):
            raise ValueError(
                "satisfied prospective obligation requires proof-of-satisfaction and passed verifier result"
            )
        object.__setattr__(self, "transition_from", cast(ProspectiveStatus, previous))
        object.__setattr__(self, "transition_evidence_refs", transition_refs)

    def transition_to(
        self,
        status: ProspectiveStatus,
        *,
        evidence_refs: tuple[EvidenceRef, ...],
        satisfaction_evidence_refs: tuple[EvidenceRef, ...] = (),
        satisfaction_verifier: str = "",
        satisfaction_verifier_result: Literal["", "passed", "failed"] = "",
    ) -> "ProspectiveObligation":
        return replace(
            self,
            status=status,
            transition_from=self.status,
            transition_evidence_refs=_evidence_refs(evidence_refs),
            satisfaction_evidence_refs=_evidence_refs(satisfaction_evidence_refs),
            satisfaction_verifier=satisfaction_verifier or self.satisfaction_verifier,
            satisfaction_verifier_result=satisfaction_verifier_result,
        )

    def with_telemetry(
        self,
        *,
        false_positive_delta: int = 0,
        missed_trigger_delta: int = 0,
        overloaded_trigger_delta: int = 0,
        attention_interruption_delta: int = 0,
        duplicate_fire_delta: int = 0,
        late_trigger_delta: int = 0,
    ) -> "ProspectiveObligation":
        deltas = (
            false_positive_delta,
            missed_trigger_delta,
            overloaded_trigger_delta,
            attention_interruption_delta,
            duplicate_fire_delta,
            late_trigger_delta,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in deltas
        ):
            raise ValueError("prospective telemetry deltas must be non-negative integers")
        return replace(
            self,
            telemetry=ProspectiveTelemetry(
                false_positive_count=(self.telemetry.false_positive_count + false_positive_delta),
                missed_trigger_count=self.telemetry.missed_trigger_count + missed_trigger_delta,
                overloaded_trigger_count=(
                    self.telemetry.overloaded_trigger_count + overloaded_trigger_delta
                ),
                attention_interruption_count=(
                    self.telemetry.attention_interruption_count + attention_interruption_delta
                ),
                duplicate_fire_count=self.telemetry.duplicate_fire_count + duplicate_fire_delta,
                late_trigger_count=self.telemetry.late_trigger_count + late_trigger_delta,
            ),
        )


@dataclass(frozen=True)
class MemoryDecisionEvent:
    decision_id: str
    workspace_id: str
    run_id: str
    task_signature: str
    action_trace_id: str
    action: MemoryDecisionAction
    canonical_snapshot_sha256: str
    context_snapshot_sha256: str
    live_state_snapshot_sha256: str
    live_trigger: str
    candidate_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    policy_revision: str
    catalog_revision: str
    cost_tokens: int
    cost_latency_ms: int
    downstream_outcome: str
    evidence_refs: tuple[EvidenceRef, ...]
    created_at: str = ""

    def __post_init__(self) -> None:
        candidates = _text_tuple(self.candidate_ids, field_name="candidate_ids", maximum=256)
        selected = _text_tuple(
            self.selected_candidate_ids,
            field_name="selected_candidate_ids",
            maximum=256,
        )
        if len(candidates) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if len(selected) != len(self.selected_candidate_ids):
            raise ValueError("selected_candidate_ids must be unique")
        if not set(selected).issubset(candidates):
            raise ValueError("selected_candidate_ids must be available candidates")
        for field_name in ("cost_tokens", "cost_latency_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        evidence_refs = _evidence_refs(self.evidence_refs)
        if not evidence_refs:
            raise ValueError("memory decision event requires evidence_refs")
        action = _closed_value(
            self.action,
            field_name="memory decision action",
            shape=MemoryDecisionAction,
        )
        object.__setattr__(
            self, "decision_id", _required_text(self.decision_id, field_name="decision_id")
        )
        object.__setattr__(
            self, "workspace_id", _required_text(self.workspace_id, field_name="workspace_id")
        )
        object.__setattr__(self, "run_id", _required_text(self.run_id, field_name="run_id"))
        object.__setattr__(
            self,
            "task_signature",
            _required_text(self.task_signature, field_name="task_signature"),
        )
        object.__setattr__(
            self,
            "action_trace_id",
            _required_text(self.action_trace_id, field_name="action_trace_id"),
        )
        object.__setattr__(self, "action", cast(MemoryDecisionAction, action))
        for field_name in (
            "canonical_snapshot_sha256",
            "context_snapshot_sha256",
            "live_state_snapshot_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_sha256_ref(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "live_trigger",
            "policy_revision",
            "catalog_revision",
            "downstream_outcome",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "selected_candidate_ids", selected)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        created_at = _optional_text(self.created_at, field_name="created_at") or _utc_now()
        object.__setattr__(self, "created_at", created_at)


_MEMORY_REVIEW_REASONS: dict[str, frozenset[str]] = {
    "changed": frozenset({"reusable_state_recorded"}),
    "unchanged": frozenset({"existing_memory_sufficient", "no_reusable_state"}),
    "unavailable": frozenset({"memory_unavailable"}),
    "forbidden": frozenset({"memory_forbidden"}),
}


@dataclass(frozen=True)
class MemoryReviewDecision:
    """One agent-authored, content-free decision at a managed work boundary."""

    workspace_id: str
    session_id: str
    task_id: str
    outcome: MemoryReviewOutcome
    operations: tuple[MemoryReviewOperation, ...]
    candidate_record_ids: tuple[str, ...]
    referenced_block_ids: tuple[str, ...]
    selected_block_ids: tuple[str, ...]
    reason_code: MemoryReviewReason

    def __post_init__(self) -> None:
        for field_name in ("workspace_id", "session_id", "task_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        outcome = _closed_value(
            self.outcome,
            field_name="memory review outcome",
            shape=MemoryReviewOutcome,
        )
        reason = _closed_value(
            self.reason_code,
            field_name="memory review reason",
            shape=MemoryReviewReason,
        )
        if reason not in _MEMORY_REVIEW_REASONS[outcome]:
            raise ValueError("memory review reason does not match the outcome")
        operations = _text_tuple(self.operations, field_name="memory review operation")
        if len(operations) != len(self.operations):
            raise ValueError("memory review operations must be unique")
        for operation in operations:
            _closed_value(
                operation,
                field_name="memory review operation",
                shape=MemoryReviewOperation,
            )
        candidates = _text_tuple(
            self.candidate_record_ids,
            field_name="candidate_record_ids",
            maximum=512,
        )
        blocks = _text_tuple(
            self.referenced_block_ids,
            field_name="referenced_block_ids",
            maximum=512,
        )
        selected = _text_tuple(
            self.selected_block_ids,
            field_name="selected_block_ids",
            maximum=512,
        )
        if len(candidates) != len(self.candidate_record_ids):
            raise ValueError("memory review candidate record IDs must be unique")
        if len(blocks) != len(self.referenced_block_ids):
            raise ValueError("memory review referenced block IDs must be unique")
        if len(selected) != len(self.selected_block_ids):
            raise ValueError("memory review selected block IDs must be unique")
        if not set(selected).issubset(blocks):
            raise ValueError("memory review selected blocks must be referenced")
        changed_operations = {"remembered", "amended", "selected"}
        if outcome == "changed" and (
            not (candidates or selected) or not changed_operations.intersection(operations)
        ):
            raise ValueError("changed review requires observed memory changes")
        if outcome == "unchanged" and (
            candidates or selected or changed_operations.intersection(operations)
        ):
            raise ValueError("unchanged review cannot claim memory changes")
        if outcome in {"unavailable", "forbidden"} and (
            operations or candidates or blocks or selected
        ):
            raise ValueError(f"{outcome} review cannot claim memory activity")
        object.__setattr__(self, "outcome", cast(MemoryReviewOutcome, outcome))
        object.__setattr__(self, "reason_code", cast(MemoryReviewReason, reason))
        object.__setattr__(self, "operations", cast(tuple[MemoryReviewOperation, ...], operations))
        object.__setattr__(self, "candidate_record_ids", candidates)
        object.__setattr__(self, "referenced_block_ids", blocks)
        object.__setattr__(self, "selected_block_ids", selected)

    def to_payload(self) -> dict[str, object]:
        return {
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "outcome": self.outcome,
            "operations": list(self.operations),
            "candidate_record_ids": list(self.candidate_record_ids),
            "referenced_block_ids": list(self.referenced_block_ids),
            "selected_block_ids": list(self.selected_block_ids),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_payload(cls, value: object) -> "MemoryReviewDecision":
        if not isinstance(value, Mapping) or set(value) != {
            "workspace_id",
            "session_id",
            "task_id",
            "outcome",
            "operations",
            "candidate_record_ids",
            "referenced_block_ids",
            "selected_block_ids",
            "reason_code",
        }:
            raise ValueError("memory review decision fields are invalid")
        return cls(
            workspace_id=value["workspace_id"],
            session_id=value["session_id"],
            task_id=value["task_id"],
            outcome=value["outcome"],
            operations=value["operations"],
            candidate_record_ids=value["candidate_record_ids"],
            referenced_block_ids=value["referenced_block_ids"],
            selected_block_ids=value["selected_block_ids"],
            reason_code=value["reason_code"],
        )


@dataclass(frozen=True)
class MemoryReviewReceipt:
    """Host observation that binds an agent decision to exact journal activity."""

    receipt_id: str
    decision: MemoryReviewDecision
    observed_operations: tuple[MemoryReviewOperation, ...]
    observed_candidate_record_ids: tuple[str, ...]
    observed_selected_block_ids: tuple[str, ...]
    recorded_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_id",
            _required_text(self.receipt_id, field_name="receipt_id"),
        )
        if not isinstance(self.decision, MemoryReviewDecision):
            raise ValueError("memory review receipt requires a typed decision")
        operations = _text_tuple(
            self.observed_operations,
            field_name="observed memory review operation",
        )
        for operation in operations:
            _closed_value(
                operation,
                field_name="observed memory review operation",
                shape=MemoryReviewOperation,
            )
        candidates = _text_tuple(
            self.observed_candidate_record_ids,
            field_name="observed_candidate_record_ids",
            maximum=512,
        )
        selected = _text_tuple(
            self.observed_selected_block_ids,
            field_name="observed_selected_block_ids",
            maximum=512,
        )
        if operations != self.decision.operations:
            raise ValueError("memory review receipt operations do not match the decision")
        if candidates != self.decision.candidate_record_ids:
            raise ValueError("memory review receipt candidate records do not match the decision")
        if selected != self.decision.selected_block_ids:
            raise ValueError("memory review receipt selected blocks do not match the decision")
        _recorded_at, parsed_recorded_at = _valid_time(
            self.recorded_at or _utc_now(),
            field_name="recorded_at",
        )
        if parsed_recorded_at is None:
            raise ValueError("memory review receipt requires recorded_at")
        object.__setattr__(self, "observed_operations", operations)
        object.__setattr__(self, "observed_candidate_record_ids", candidates)
        object.__setattr__(self, "observed_selected_block_ids", selected)
        object.__setattr__(
            self,
            "recorded_at",
            parsed_recorded_at.isoformat().replace("+00:00", "Z"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "decision": self.decision.to_payload(),
            "observed_operations": list(self.observed_operations),
            "observed_candidate_record_ids": list(self.observed_candidate_record_ids),
            "observed_selected_block_ids": list(self.observed_selected_block_ids),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_payload(cls, value: object) -> "MemoryReviewReceipt":
        if not isinstance(value, Mapping) or set(value) != {
            "receipt_id",
            "decision",
            "observed_operations",
            "observed_candidate_record_ids",
            "observed_selected_block_ids",
            "recorded_at",
        }:
            raise ValueError("memory review receipt fields are invalid")
        return cls(
            receipt_id=value["receipt_id"],
            decision=MemoryReviewDecision.from_payload(value["decision"]),
            observed_operations=value["observed_operations"],
            observed_candidate_record_ids=value["observed_candidate_record_ids"],
            observed_selected_block_ids=value["observed_selected_block_ids"],
            recorded_at=value["recorded_at"],
        )


@dataclass(frozen=True)
class MemoryUsageEvent:
    event_id: str
    decision_id: str
    memory_id: str
    workspace_id: str
    task_signature: str
    action_trace_id: str
    usage_type: MemoryUsageType
    sequence: int
    evidence_refs: tuple[EvidenceRef, ...]
    created_at: str = ""

    def __post_init__(self) -> None:
        usage_type = _closed_value(
            self.usage_type,
            field_name="memory usage_type",
            shape=MemoryUsageType,
        )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("memory usage sequence must be a non-negative integer")
        evidence_refs = _evidence_refs(self.evidence_refs)
        if not evidence_refs:
            raise ValueError("memory usage event requires evidence_refs")
        for field_name in (
            "event_id",
            "decision_id",
            "memory_id",
            "workspace_id",
            "task_signature",
            "action_trace_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "usage_type", cast(MemoryUsageType, usage_type))
        object.__setattr__(self, "evidence_refs", evidence_refs)
        created_at = _optional_text(self.created_at, field_name="created_at") or _utc_now()
        object.__setattr__(self, "created_at", created_at)


@dataclass(frozen=True)
class MemoryDecisionTrace:
    decision: MemoryDecisionEvent
    usage_events: tuple[MemoryUsageEvent, ...]
    channel: Literal["memory_decision_usage_trace"] = field(
        default="memory_decision_usage_trace", init=False
    )
    ordinary_recall: bool = field(default=False, init=False)
    host_action_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, MemoryDecisionEvent):
            raise ValueError("decision trace requires MemoryDecisionEvent")
        if not isinstance(self.usage_events, tuple) or any(
            not isinstance(event, MemoryUsageEvent) for event in self.usage_events
        ):
            raise ValueError("decision trace requires typed usage events")
        sequences = tuple(event.sequence for event in self.usage_events)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
            raise ValueError("decision trace usage sequence is invalid")
        for event in self.usage_events:
            if (
                event.decision_id != self.decision.decision_id
                or event.workspace_id != self.decision.workspace_id
                or event.task_signature != self.decision.task_signature
                or event.action_trace_id != self.decision.action_trace_id
            ):
                raise ValueError("decision trace usage linkage is invalid")
        for memory_id in {event.memory_id for event in self.usage_events}:
            observed = {
                event.usage_type for event in self.usage_events if event.memory_id == memory_id
            }
            if {"applied", "not_applied"}.issubset(observed):
                raise ValueError("decision trace contains contradictory action-use observations")


@dataclass(frozen=True)
class OutcomeCreditEvent:
    credit_id: str
    decision_id: str
    workspace_id: str
    task_signature: str
    action_trace_id: str
    memory_ids: tuple[str, ...]
    applied_usage_event_ids: tuple[str, ...]
    outcome: Literal["helped", "hurt", "neutral"]
    attribution_basis: OutcomeAttributionBasis
    verifier_result: Literal["", "passed", "failed", "inconclusive"]
    approval_principal_id: str
    task_family: str
    stage: str
    project_id: str
    host_id: str
    profile_id: str
    independence_domain: str
    evidence_refs: tuple[EvidenceRef, ...]
    reason: str
    created_at: str = ""

    def __post_init__(self) -> None:
        outcome = _closed_value(
            self.outcome,
            field_name="outcome credit outcome",
            shape=Literal["helped", "hurt", "neutral"],
        )
        basis = _closed_value(
            self.attribution_basis,
            field_name="outcome attribution_basis",
            shape=OutcomeAttributionBasis,
        )
        memory_ids = _text_tuple(self.memory_ids, field_name="memory_ids", maximum=256)
        usage_event_ids = _text_tuple(
            self.applied_usage_event_ids,
            field_name="applied_usage_event_ids",
            maximum=256,
        )
        if len(memory_ids) != len(self.memory_ids):
            raise ValueError("outcome credit memory_ids must be unique")
        if len(usage_event_ids) != len(self.applied_usage_event_ids):
            raise ValueError("outcome credit applied_usage_event_ids must be unique")
        evidence_refs = _evidence_refs(self.evidence_refs)
        if not evidence_refs:
            raise ValueError("outcome credit requires evidence_refs")
        verifier_result = _optional_text(
            self.verifier_result,
            field_name="verifier_result",
        )
        if verifier_result not in {"", "passed", "failed", "inconclusive"}:
            raise ValueError("outcome credit verifier_result is invalid")
        approval_principal_id = _optional_text(
            self.approval_principal_id,
            field_name="approval_principal_id",
        )
        if outcome in {"helped", "hurt"} and (not memory_ids or not usage_event_ids):
            raise ValueError("helped or hurt outcome credit requires applied memory linkage")
        if outcome == "helped":
            if basis == "deterministic_verifier":
                if verifier_result != "passed" or not any(
                    ref.ref_type in {"test_run", "verification_report"} and ref.producer_id
                    for ref in evidence_refs
                ):
                    raise ValueError(
                        "helped credit requires passed deterministic verifier evidence"
                    )
            elif basis == "trusted_approval":
                if not approval_principal_id or not any(
                    ref.ref_type == "human_approval" and ref.producer_id for ref in evidence_refs
                ):
                    raise ValueError("helped credit requires trusted approval evidence")
            else:
                raise ValueError("helped credit requires verifier or trusted approval basis")
        elif outcome == "hurt":
            if basis not in {
                "proven_stale_application",
                "proven_incorrect_application",
                "proven_contamination",
                "proven_regression",
            }:
                raise ValueError("hurt credit requires a proven harm basis")
        elif basis != "insufficient_linkage":
            raise ValueError("neutral outcome credit requires insufficient_linkage basis")
        for field_name in (
            "credit_id",
            "decision_id",
            "workspace_id",
            "task_signature",
            "action_trace_id",
            "task_family",
            "stage",
            "project_id",
            "host_id",
            "profile_id",
            "independence_domain",
            "reason",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "outcome", cast(Literal["helped", "hurt", "neutral"], outcome))
        object.__setattr__(self, "attribution_basis", cast(OutcomeAttributionBasis, basis))
        object.__setattr__(self, "memory_ids", memory_ids)
        object.__setattr__(self, "applied_usage_event_ids", usage_event_ids)
        object.__setattr__(self, "verifier_result", verifier_result)
        object.__setattr__(self, "approval_principal_id", approval_principal_id)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        created_at = _optional_text(self.created_at, field_name="created_at") or _utc_now()
        _valid_time(created_at, field_name="outcome credit created_at")
        object.__setattr__(self, "created_at", created_at)


@dataclass(frozen=True)
class ContextualUtilitySummary:
    summary_id: str
    workspace_id: str
    memory_id: str
    task_family: str
    stage: str
    project_id: str
    host_id: str
    profile_id: str
    minimum_support: int
    minimum_independent_outcomes: int
    helped_count: int
    hurt_count: int
    neutral_count: int
    outcome_diversity: int
    oldest_credit_at: str = ""
    newest_credit_at: str = ""
    support_count: int = field(default=0, init=False)
    shadow_utility: float = field(default=0.0, init=False)
    shadow_eligible: bool = field(default=False, init=False)
    ranking_mode: Literal["shadow_only"] = field(default="shadow_only", init=False)
    affects_trust: bool = field(default=False, init=False)
    affects_truth: bool = field(default=False, init=False)
    affects_deletion: bool = field(default=False, init=False)
    affects_authority: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "summary_id",
            "workspace_id",
            "memory_id",
            "task_family",
            "stage",
            "project_id",
            "host_id",
            "profile_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "minimum_support",
            "minimum_independent_outcomes",
            "helped_count",
            "hurt_count",
            "neutral_count",
            "outcome_diversity",
        ):
            value = getattr(self, field_name)
            minimum = 1 if field_name.startswith("minimum_") else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer of at least {minimum}")
        support_count = self.helped_count + self.hurt_count
        if self.outcome_diversity > support_count:
            raise ValueError("outcome_diversity cannot exceed verified support")
        oldest = _optional_text(self.oldest_credit_at, field_name="oldest_credit_at")
        newest = _optional_text(self.newest_credit_at, field_name="newest_credit_at")
        if oldest:
            _valid_time(oldest, field_name="oldest_credit_at")
        if newest:
            _valid_time(newest, field_name="newest_credit_at")
        object.__setattr__(self, "oldest_credit_at", oldest)
        object.__setattr__(self, "newest_credit_at", newest)
        object.__setattr__(self, "support_count", support_count)
        utility = (self.helped_count - self.hurt_count) / support_count if support_count else 0.0
        object.__setattr__(self, "shadow_utility", round(utility, 6))
        object.__setattr__(
            self,
            "shadow_eligible",
            support_count >= self.minimum_support
            and self.outcome_diversity >= self.minimum_independent_outcomes,
        )


@dataclass(frozen=True)
class ProcedurePromotionGate:
    verified_outcome_ids: tuple[str, ...]
    independence_domains: tuple[str, ...]
    minimum_verified_outcomes: int = 2
    allowed: bool = field(default=False, init=False)
    reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        outcome_ids = _text_tuple(
            self.verified_outcome_ids,
            field_name="verified_outcome_ids",
            maximum=256,
        )
        domains = _text_tuple(
            self.independence_domains,
            field_name="independence_domains",
            maximum=256,
        )
        if (
            isinstance(self.minimum_verified_outcomes, bool)
            or not isinstance(self.minimum_verified_outcomes, int)
            or self.minimum_verified_outcomes < 2
        ):
            raise ValueError("minimum_verified_outcomes must be an integer of at least 2")
        allowed = (
            len(outcome_ids) >= self.minimum_verified_outcomes
            and len(domains) >= self.minimum_verified_outcomes
        )
        object.__setattr__(self, "verified_outcome_ids", outcome_ids)
        object.__setattr__(self, "independence_domains", domains)
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(
            self,
            "reason",
            "independent_verified_outcomes_satisfied"
            if allowed
            else "insufficient_independent_verified_outcomes",
        )


@dataclass(frozen=True)
class ProcedureReuseDecision:
    memory_id: str
    source_task_signature: str
    requested_task_signature: str
    allowed: bool
    reason: str
    transfer_envelope_id: str = ""
    promotion_authorized: bool = field(default=False, init=False)
    host_action_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "memory_id",
            "source_task_signature",
            "requested_task_signature",
            "reason",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "transfer_envelope_id",
            _optional_text(self.transfer_envelope_id, field_name="transfer_envelope_id"),
        )
        if not isinstance(self.allowed, bool):
            raise ValueError("procedure reuse allowed must be a boolean")


@dataclass(frozen=True)
class AdvisoryAssociation:
    association_id: str
    memory_ids: tuple[str, str]
    status: AssociationStatus
    helped_count: int
    hurt_count: int
    outcome_diversity: int
    advisory_weight: float
    evidence_refs: tuple[EvidenceRef, ...]
    evidence_retained: bool = field(default=True, init=False)
    affects_authority: bool = field(default=False, init=False)
    durable_procedure: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        memory_ids = _text_tuple(self.memory_ids, field_name="association memory_ids")
        if len(memory_ids) != 2:
            raise ValueError("advisory association requires exactly two memory_ids")
        status = _closed_value(
            self.status,
            field_name="association status",
            shape=AssociationStatus,
        )
        for field_name in ("helped_count", "hurt_count", "outcome_diversity"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.advisory_weight, (int, float)) or isinstance(
            self.advisory_weight, bool
        ):
            raise ValueError("advisory_weight must be numeric")
        weight = float(self.advisory_weight)
        if weight < -1.0 or weight > 1.0:
            raise ValueError("advisory_weight must be between -1 and 1")
        refs = _evidence_refs(self.evidence_refs)
        if not refs:
            raise ValueError("advisory association requires evidence_refs")
        object.__setattr__(
            self, "association_id", _required_text(self.association_id, field_name="association_id")
        )
        object.__setattr__(self, "memory_ids", cast(tuple[str, str], memory_ids))
        object.__setattr__(self, "status", cast(AssociationStatus, status))
        object.__setattr__(self, "advisory_weight", round(weight, 6))
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class UtilitySelectionCandidate:
    memory_id: str
    task_family: str
    shadow_utility: float
    support_count: int
    newest_credit_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_id",
            _required_text(self.memory_id, field_name="utility candidate memory_id"),
        )
        object.__setattr__(
            self,
            "task_family",
            _required_text(self.task_family, field_name="utility candidate task_family"),
        )
        if not isinstance(self.shadow_utility, (int, float)) or isinstance(
            self.shadow_utility, bool
        ):
            raise ValueError("shadow_utility must be numeric")
        utility = float(self.shadow_utility)
        if utility < -1.0 or utility > 1.0:
            raise ValueError("shadow_utility must be between -1 and 1")
        if (
            isinstance(self.support_count, bool)
            or not isinstance(self.support_count, int)
            or self.support_count < 0
        ):
            raise ValueError("support_count must be a non-negative integer")
        timestamp = _required_text(
            self.newest_credit_at,
            field_name="utility candidate newest_credit_at",
        )
        _valid_time(timestamp, field_name="utility candidate newest_credit_at")
        object.__setattr__(self, "shadow_utility", round(utility, 6))
        object.__setattr__(self, "newest_credit_at", timestamp)


@dataclass(frozen=True)
class MemoryOutcomeReportEntry:
    memory_id: str
    disposition: MemoryOutcomeDisposition
    usage_event_ids: tuple[str, ...]
    credit_ids: tuple[str, ...]
    reason: str
    evidence_refs: tuple[EvidenceRef, ...]

    def __post_init__(self) -> None:
        disposition = _closed_value(
            self.disposition,
            field_name="memory outcome disposition",
            shape=MemoryOutcomeDisposition,
        )
        refs = _evidence_refs(self.evidence_refs)
        if not refs:
            raise ValueError("memory outcome report entry requires evidence_refs")
        object.__setattr__(
            self, "memory_id", _required_text(self.memory_id, field_name="memory_id")
        )
        object.__setattr__(
            self,
            "disposition",
            cast(MemoryOutcomeDisposition, disposition),
        )
        object.__setattr__(
            self,
            "usage_event_ids",
            _text_tuple(self.usage_event_ids, field_name="usage_event_ids"),
        )
        object.__setattr__(
            self,
            "credit_ids",
            _text_tuple(self.credit_ids, field_name="credit_ids"),
        )
        object.__setattr__(self, "reason", _required_text(self.reason, field_name="reason"))
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class MemoryOutcomeReport:
    report_id: str
    workspace_id: str
    decision_id: str
    task_signature: str
    action_trace_id: str
    entries: tuple[MemoryOutcomeReportEntry, ...]
    channel: Literal["memory_outcome_operator_report"] = field(
        default="memory_outcome_operator_report",
        init=False,
    )
    ordinary_recall: bool = field(default=False, init=False)
    host_action_authorized: bool = field(default=False, init=False)
    affects_authority: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "report_id",
            "workspace_id",
            "decision_id",
            "task_signature",
            "action_trace_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, MemoryOutcomeReportEntry) for entry in self.entries
        ):
            raise ValueError("memory outcome report requires typed entries")
        if not self.entries:
            raise ValueError("memory outcome report requires entries")
        if len({entry.memory_id for entry in self.entries}) != len(self.entries):
            raise ValueError("memory outcome report entries must be unique by memory_id")


@dataclass(frozen=True)
class ConsolidationDecision:
    decision_id: str
    workspace_id: str
    source_episode_ids: tuple[str, ...]
    grouping_key: str
    task_family: str
    trigger: ConsolidationTrigger
    expected_applicability: str
    risk: OperationalRisk
    cost_budget_tokens: int
    rationale: str
    evidence_refs: tuple[EvidenceRef, ...]
    action: ConsolidationAction = "defer"
    status: ConsolidationStatus = "proposed"
    forced_rewrite: bool = field(default=False, init=False)
    host_action_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id",
            "workspace_id",
            "grouping_key",
            "task_family",
            "expected_applicability",
            "rationale",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        action = _closed_value(
            self.action,
            field_name="consolidation action",
            shape=ConsolidationAction,
        )
        trigger = _closed_value(
            self.trigger,
            field_name="consolidation trigger",
            shape=ConsolidationTrigger,
        )
        risk = _closed_value(
            self.risk,
            field_name="consolidation risk",
            shape=OperationalRisk,
        )
        if self.status != "proposed":
            raise ValueError("consolidation decisions must start proposed")
        source_episode_ids = _text_tuple(
            self.source_episode_ids,
            field_name="source_episode_ids",
        )
        if not source_episode_ids:
            raise ValueError("source_episode_ids are required")
        if len(set(source_episode_ids)) != len(source_episode_ids):
            raise ValueError("source_episode_ids must be unique")
        if (
            isinstance(self.cost_budget_tokens, bool)
            or not isinstance(self.cost_budget_tokens, int)
            or self.cost_budget_tokens < 1
        ):
            raise ValueError("cost_budget_tokens must be a positive integer")
        refs = _evidence_refs(self.evidence_refs)
        if not refs:
            raise ValueError("consolidation decision requires evidence_refs")
        object.__setattr__(self, "action", cast(ConsolidationAction, action))
        object.__setattr__(self, "trigger", cast(ConsolidationTrigger, trigger))
        object.__setattr__(self, "risk", cast(OperationalRisk, risk))
        object.__setattr__(self, "source_episode_ids", tuple(sorted(source_episode_ids)))
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class SourceCoverageDiff:
    before_source_ids: tuple[str, ...]
    after_source_ids: tuple[str, ...]
    added_source_ids: tuple[str, ...] = field(init=False)
    lost_source_ids: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        before = _text_tuple(self.before_source_ids, field_name="before_source_ids")
        after = _text_tuple(self.after_source_ids, field_name="after_source_ids")
        if not before:
            raise ValueError("before_source_ids are required")
        if len(set(before)) != len(before) or len(set(after)) != len(after):
            raise ValueError("source coverage IDs must be unique")
        before_set = set(before)
        after_set = set(after)
        object.__setattr__(self, "before_source_ids", tuple(sorted(before_set)))
        object.__setattr__(self, "after_source_ids", tuple(sorted(after_set)))
        object.__setattr__(self, "added_source_ids", tuple(sorted(after_set - before_set)))
        object.__setattr__(self, "lost_source_ids", tuple(sorted(before_set - after_set)))


_CONSOLIDATION_TRANSITIONS = {
    "proposed": frozenset({"assessed", "rejected"}),
    "assessed": frozenset({"accepted", "rejected"}),
    "accepted": frozenset({"committed"}),
    "rejected": frozenset(),
    "committed": frozenset(),
}


@dataclass(frozen=True)
class ConsolidationTransition:
    transition_id: str
    from_status: ConsolidationStatus
    to_status: ConsolidationStatus
    decision_id: str = ""
    pre_state_snapshot_sha256: str = ""
    new_evidence_refs: tuple[EvidenceRef, ...] = ()
    proposed_operations: tuple[str, ...] = ()
    touched_record_ids: tuple[str, ...] = ()
    touched_view_ids: tuple[str, ...] = ()
    post_state_candidate: str = ""
    post_state_candidate_sha256: str = ""
    source_coverage_diff: SourceCoverageDiff | None = None
    policy_version: str = ""
    model_version: str = ""
    prompt_version: str = ""
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        previous = _closed_value(
            self.from_status,
            field_name="consolidation from_status",
            shape=ConsolidationStatus,
        )
        target = _closed_value(
            self.to_status,
            field_name="consolidation to_status",
            shape=ConsolidationStatus,
        )
        if target not in _CONSOLIDATION_TRANSITIONS[previous]:
            raise ValueError(f"invalid consolidation transition: {previous} -> {target}")
        for field_name in (
            "transition_id",
            "decision_id",
            "post_state_candidate",
            "policy_version",
            "model_version",
            "prompt_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "pre_state_snapshot_sha256",
            _require_sha256_ref(
                self.pre_state_snapshot_sha256,
                field_name="pre_state_snapshot_sha256",
            ),
        )
        object.__setattr__(
            self,
            "post_state_candidate_sha256",
            _require_sha256_ref(
                self.post_state_candidate_sha256,
                field_name="post_state_candidate_sha256",
            ),
        )
        try:
            post_state_value = json.loads(self.post_state_candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("post_state_candidate must be valid JSON") from exc
        canonical_post_state = json.dumps(
            post_state_value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        expected_post_state_sha256 = (
            f"sha256:{hashlib.sha256(canonical_post_state.encode('utf-8')).hexdigest()}"
        )
        if self.post_state_candidate_sha256 != expected_post_state_sha256:
            raise ValueError("post_state_candidate_sha256 does not match the candidate")
        object.__setattr__(self, "post_state_candidate", canonical_post_state)
        new_refs = _evidence_refs(self.new_evidence_refs)
        if not new_refs:
            raise ValueError("consolidation transition requires new_evidence_refs")
        operations = _text_tuple(self.proposed_operations, field_name="proposed_operations")
        if not operations:
            raise ValueError("consolidation transition requires proposed_operations")
        touched_records = _text_tuple(self.touched_record_ids, field_name="touched_record_ids")
        if not touched_records:
            raise ValueError("consolidation transition requires touched_record_ids")
        touched_views = _text_tuple(self.touched_view_ids, field_name="touched_view_ids")
        if not isinstance(self.source_coverage_diff, SourceCoverageDiff):
            raise ValueError("consolidation transition requires SourceCoverageDiff")
        refs = _evidence_refs(self.evidence_refs)
        if target in {"accepted", "committed"} and not refs:
            raise ValueError("consolidation transition requires evidence_refs")
        object.__setattr__(self, "from_status", cast(ConsolidationStatus, previous))
        object.__setattr__(self, "to_status", cast(ConsolidationStatus, target))
        object.__setattr__(self, "new_evidence_refs", new_refs)
        object.__setattr__(self, "proposed_operations", operations)
        object.__setattr__(self, "touched_record_ids", tuple(sorted(touched_records)))
        object.__setattr__(self, "touched_view_ids", tuple(sorted(touched_views)))
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class MemoryTransitionAssessment:
    assessment_id: str
    transition_id: str
    decision_id: str
    pre_state_snapshot_sha256: str
    post_state_candidate_sha256: str
    source_coverage_diff: SourceCoverageDiff
    policy_version: str
    model_version: str
    prompt_version: str
    verdict: Literal["accept", "reject", "defer"]
    reasons: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    canonical_mutation_applied: bool = field(default=False, init=False)
    host_action_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "assessment_id",
            "transition_id",
            "decision_id",
            "policy_version",
            "model_version",
            "prompt_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "pre_state_snapshot_sha256",
            _require_sha256_ref(
                self.pre_state_snapshot_sha256,
                field_name="pre_state_snapshot_sha256",
            ),
        )
        object.__setattr__(
            self,
            "post_state_candidate_sha256",
            _require_sha256_ref(
                self.post_state_candidate_sha256,
                field_name="post_state_candidate_sha256",
            ),
        )
        if not isinstance(self.source_coverage_diff, SourceCoverageDiff):
            raise ValueError("memory transition assessment requires SourceCoverageDiff")
        verdict = _closed_value(
            self.verdict,
            field_name="memory transition verdict",
            shape=Literal["accept", "reject", "defer"],
        )
        reasons = _text_tuple(self.reasons, field_name="assessment reasons")
        if not reasons:
            raise ValueError("memory transition assessment requires reasons")
        refs = _evidence_refs(self.evidence_refs)
        if not refs:
            raise ValueError("memory transition assessment requires evidence_refs")
        object.__setattr__(self, "verdict", cast(Literal["accept", "reject", "defer"], verdict))
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class ConsolidationAssessment:
    assessment_id: str
    decision_id: str = ""
    verdict: Literal["accept", "reject", "defer"] = "defer"


@dataclass(frozen=True)
class ConsolidationCommitReceipt:
    receipt_id: str
    evidence_refs: tuple[EvidenceRef, ...]
    committed_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        refs = _evidence_refs(self.evidence_refs)
        committed_ids = _text_tuple(self.committed_ids, field_name="committed_ids")
        if not refs:
            raise ValueError("consolidation commit requires evidence_refs")
        if not committed_ids:
            raise ValueError("consolidation commit requires committed_ids")
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "committed_ids", committed_ids)


@dataclass(frozen=True)
class MetamemoryPolicyScorecard:
    revision_id: str
    action_correctness: float
    state_correctness: float
    coverage: float
    tokens_max: int
    latency_ms_max: int
    storage_bytes_max: int
    safety_violations: int
    churn_rate: float
    stability_runs: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "revision_id",
            _required_text(self.revision_id, field_name="scorecard revision_id"),
        )
        for field_name in ("action_correctness", "state_correctness", "coverage", "churn_rate"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field_name} must be numeric")
            normalized = float(value)
            if not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{field_name} must be between zero and one")
            object.__setattr__(self, field_name, normalized)
        for field_name in (
            "tokens_max",
            "latency_ms_max",
            "storage_bytes_max",
            "safety_violations",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if (
            isinstance(self.stability_runs, bool)
            or not isinstance(self.stability_runs, int)
            or not 2 <= self.stability_runs <= 5
        ):
            raise ValueError("stability_runs must be between two and five")


@dataclass(frozen=True)
class TransferEnvelope:
    envelope_id: str
    evidence_refs: tuple[EvidenceRef, ...] = ()
    status: TransferStatus = "pending"
    source_scope: str = ""
    target_scope: str = ""
    source_task_signature: str = ""
    target_task_signature: str = ""
    qualified_memory_ids: tuple[str, ...] = ()
    verified_outcome_ids: tuple[str, ...] = ()
    independence_domains: tuple[str, ...] = ()
    source_workflow: str = ""
    target_workflow: str = ""
    source_project_id: str = ""
    target_project_id: str = ""
    source_host: str = ""
    target_host: str = ""
    source_profile: str = ""
    target_profile: str = ""
    source_stages: tuple[str, ...] = ()
    target_stages: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    contraindications: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    support_count: int = 0
    source_revision_id: str = ""
    valid_from: str = ""
    expires_at: str = ""
    drift_fingerprint: str = ""

    def __post_init__(self) -> None:
        status = _closed_value(self.status, field_name="transfer status", shape=TransferStatus)
        refs = _evidence_refs(self.evidence_refs)
        qualified_memory_ids = _text_tuple(
            self.qualified_memory_ids,
            field_name="qualified_memory_ids",
            maximum=256,
        )
        verified_outcome_ids = _text_tuple(
            self.verified_outcome_ids,
            field_name="verified_outcome_ids",
            maximum=256,
        )
        independence_domains = _text_tuple(
            self.independence_domains,
            field_name="independence_domains",
            maximum=256,
        )
        tuple_fields: dict[str, tuple[str, ...]] = {}
        for field_name in (
            "source_stages",
            "target_stages",
            "prerequisites",
            "contraindications",
            "exclusions",
        ):
            tuple_fields[field_name] = _text_tuple(
                getattr(self, field_name),
                field_name=field_name,
                maximum=256,
            )
            object.__setattr__(self, field_name, tuple_fields[field_name])
        if status == "accepted":
            source = _required_text(self.source_scope, field_name="accepted transfer source_scope")
            target = _required_text(self.target_scope, field_name="accepted transfer target_scope")
            if source == target:
                raise ValueError(
                    "accepted transfer requires distinct source_scope and target_scope"
                )
            if not refs:
                raise ValueError("accepted transfer requires evidence_refs")
            source_task_signature = _required_text(
                self.source_task_signature,
                field_name="accepted transfer source_task_signature",
            )
            target_task_signature = _required_text(
                self.target_task_signature,
                field_name="accepted transfer target_task_signature",
            )
            if source_task_signature == target_task_signature:
                raise ValueError(
                    "accepted transfer requires distinct source and target task signatures"
                )
            if not qualified_memory_ids:
                raise ValueError("accepted transfer requires qualified_memory_ids")
            text_fields: dict[str, str] = {}
            for field_name in (
                "source_workflow",
                "target_workflow",
                "source_project_id",
                "target_project_id",
                "source_host",
                "target_host",
                "source_profile",
                "target_profile",
                "source_revision_id",
                "drift_fingerprint",
            ):
                text_fields[field_name] = _required_text(
                    getattr(self, field_name),
                    field_name=f"accepted transfer {field_name}",
                )
                object.__setattr__(self, field_name, text_fields[field_name])
            if text_fields["source_workflow"] == text_fields["target_workflow"]:
                raise ValueError("accepted transfer requires distinct source and target workflows")
            if text_fields["source_project_id"] == text_fields["target_project_id"]:
                raise ValueError("accepted transfer requires distinct source and target projects")
            if not tuple_fields["source_stages"] or not tuple_fields["target_stages"]:
                raise ValueError("accepted transfer requires source and target stages")
            if (
                isinstance(self.support_count, bool)
                or not isinstance(self.support_count, int)
                or self.support_count < 2
                or self.support_count < len(verified_outcome_ids)
            ):
                raise ValueError("accepted transfer requires declared independent support count")
            valid_from, valid_from_time = _valid_time(
                self.valid_from,
                field_name="accepted transfer valid_from",
            )
            expires_at, expires_at_time = _valid_time(
                self.expires_at,
                field_name="accepted transfer expires_at",
            )
            if valid_from_time is None or expires_at_time is None:
                raise ValueError("accepted transfer requires validity and expiry")
            if valid_from_time >= expires_at_time:
                raise ValueError("accepted transfer expiry must follow valid_from")
            object.__setattr__(self, "valid_from", valid_from)
            object.__setattr__(self, "expires_at", expires_at)
            gate = ProcedurePromotionGate(
                verified_outcome_ids=verified_outcome_ids,
                independence_domains=independence_domains,
            )
            if not gate.allowed:
                raise ValueError(
                    "accepted transfer requires multiple independently verified outcomes"
                )
            if any(not domain.startswith("project:") for domain in independence_domains):
                raise ValueError("accepted transfer requires independent project evidence domains")
            object.__setattr__(self, "source_scope", source)
            object.__setattr__(self, "target_scope", target)
            object.__setattr__(self, "source_task_signature", source_task_signature)
            object.__setattr__(self, "target_task_signature", target_task_signature)
        object.__setattr__(self, "status", cast(TransferStatus, status))
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "qualified_memory_ids", qualified_memory_ids)
        object.__setattr__(self, "verified_outcome_ids", verified_outcome_ids)
        object.__setattr__(self, "independence_domains", independence_domains)
