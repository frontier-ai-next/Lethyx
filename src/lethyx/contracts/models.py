"""Typed evidence and trust primitives used by the memory boundary."""

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, cast, get_args

Actor = Literal["agent", "supervisor", "verifier", "watcher", "harness", "human"]
IssueSeverity = Literal["info", "warning", "error", "blocking", "human_required"]
EvidenceRefType = Literal[
    "ledger_event",
    "session_event",
    "message",
    "tool_call",
    "tool_result",
    "file_snapshot",
    "source_span",
    "diff",
    "test_run",
    "artifact",
    "verification_report",
    "human_approval",
]
HostObservationKind = Literal[
    "conversation_turn",
    "file_created",
    "file_changed",
    "symbol_observed",
    "test_passed",
    "test_failed",
    "run_receipt",
    "trace_assertion",
    "source_opened",
    "source_snapshot",
    "source_claim",
    "session_episode",
    "handoff",
    "checkpoint",
    "report_created",
    "analysis_report",
    "decision",
    "contract",
    "task_checkpoint",
    "subtask_checkpoint",
    "lesson_candidate",
    "skill_artifact",
    "ambiguous",
]
TrustLevel = Literal[
    "untrusted",
    "agent_reported",
    "source_observed",
    "workspace_verified",
    "external_verified",
    "supervisor_verified",
    "human_approved",
    "system_curated",
]
APPROVED_TRUST_LEVELS = frozenset(
    {
        "workspace_verified",
        "external_verified",
        "supervisor_verified",
        "human_approved",
        "system_curated",
    }
)
_EVIDENCE_REF_FIELDS = frozenset(
    {
        "ref_type",
        "ref_id",
        "hash",
        "produced_by",
        "producer_id",
        "stage_id",
        "subtask_id",
        "created_after_event",
    }
)


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        if isinstance(value, type):
            raise TypeError("json_ready requires a dataclass instance")
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(item) for item in value]
    return value


@dataclass(frozen=True)
class EvidenceRef:
    ref_type: EvidenceRefType
    ref_id: str
    hash: str | None = None
    produced_by: Actor = "agent"
    producer_id: str = ""
    stage_id: str | None = None
    subtask_id: str | None = None
    created_after_event: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref_type, str) or not (ref_type := self.ref_type.strip()):
            raise ValueError("evidence ref_type must be a non-empty string")
        if ref_type not in get_args(EvidenceRefType):
            raise ValueError("evidence ref_type is not a supported EvidenceRefType")
        if not isinstance(self.ref_id, str) or not (ref_id := self.ref_id.strip()):
            raise ValueError("evidence ref_id must be a non-empty string")
        if not isinstance(self.produced_by, str) or not (produced_by := self.produced_by.strip()):
            raise ValueError("evidence produced_by must be a non-empty Actor")
        if produced_by not in get_args(Actor):
            raise ValueError("evidence produced_by is not a supported Actor")
        if not isinstance(self.producer_id, str) or not (producer_id := self.producer_id.strip()):
            raise ValueError("evidence producer_id must be a non-empty identity")
        object.__setattr__(self, "ref_type", cast(EvidenceRefType, ref_type))
        object.__setattr__(self, "ref_id", ref_id)
        object.__setattr__(self, "produced_by", cast(Actor, produced_by))
        object.__setattr__(self, "producer_id", producer_id)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EvidenceRef":
        unknown_fields = sorted(str(key) for key in set(value).difference(_EVIDENCE_REF_FIELDS))
        if unknown_fields:
            raise ValueError(f"unknown evidence fields: {', '.join(unknown_fields)}")

        def optional_text(key: str) -> str | None:
            raw = value.get(key)
            if raw is None:
                return None
            if not isinstance(raw, str):
                raise ValueError(f"evidence {key} must be a string")
            return raw.strip() or None

        return cls(
            ref_type=cast(EvidenceRefType, value.get("ref_type")),
            ref_id=cast(str, value.get("ref_id")),
            hash=optional_text("hash"),
            produced_by=cast(Actor, value.get("produced_by", "agent")),
            producer_id=cast(str, value.get("producer_id")),
            stage_id=optional_text("stage_id"),
            subtask_id=optional_text("subtask_id"),
            created_after_event=optional_text("created_after_event"),
        )


@dataclass(frozen=True)
class HostObservation:
    """One authenticated, typed observation crossing a host retention boundary."""

    event_id: str
    event_kind: HostObservationKind
    raw_evidence: str
    observed_at: str
    producer_id: str
    atomic_claim: str = ""
    source_path: str = ""
    source_uri: str = ""
    artifact_hash: str = ""
    tool_name: str = ""
    workspace_root: str = ""
    changed_files: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    speaker: str = ""
    source_occurred_at: str = ""

    def __post_init__(self) -> None:
        for field_name in ("event_id", "raw_evidence", "observed_at", "producer_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"host observation {field_name} is required")
            object.__setattr__(
                self, field_name, value.strip() if field_name != "raw_evidence" else value
            )
        if self.event_kind not in get_args(HostObservationKind):
            raise ValueError("host observation event_kind is unsupported")
        for field_name in (
            "atomic_claim",
            "source_path",
            "source_uri",
            "artifact_hash",
            "tool_name",
            "workspace_root",
            "speaker",
            "source_occurred_at",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"host observation {field_name} must be a string")
            object.__setattr__(
                self,
                field_name,
                value if field_name in {"speaker", "source_occurred_at"} else value.strip(),
            )
        normalized_tags = tuple(
            dict.fromkeys(str(tag).strip() for tag in self.tags if str(tag).strip())
        )
        object.__setattr__(self, "tags", normalized_tags)
        normalized_changed_files = tuple(
            dict.fromkeys(
                str(path).replace("\\", "/").strip()
                for path in self.changed_files
                if str(path).strip()
            )
        )
        object.__setattr__(self, "changed_files", normalized_changed_files)
        if self.event_kind not in {"conversation_turn", "ambiguous"} and not self.atomic_claim:
            raise ValueError("typed host observation requires a compact atomic_claim")
        if len(self.atomic_claim) > 512:
            raise ValueError("host observation atomic_claim exceeds 512 characters")


class AmbiguousEvidenceRefError(ValueError):
    """An exact evidence reference identifies more than one canonical record."""


@dataclass(frozen=True)
class ContractIssue:
    code: str
    severity: IssueSeverity
    stage_id: str = ""
    subtask_id: str = ""
    message: str = ""
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @property
    def is_blocking(self) -> bool:
        return self.severity in {"blocking", "human_required"}


@dataclass(frozen=True)
class MemorySemanticReview:
    """Host-owned independent review of one exact immutable memory subject."""

    record_id: str
    revision_id: str
    content_hash: str
    source_authority_id: str
    source_authority_hash: str
    review_authority_id: str
    review_authority_hash: str
    manifest_hash: str

    def __post_init__(self) -> None:
        for value in (
            self.record_id,
            self.revision_id,
            self.source_authority_id,
            self.review_authority_id,
        ):
            if not value.strip() or len(value) > 512 or "\x00" in value:
                raise ValueError("semantic review requires bounded exact identities")
        for value in (
            self.content_hash,
            self.source_authority_hash,
            self.review_authority_hash,
            self.manifest_hash,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("semantic review requires exact SHA-256 subjects")
        if (
            self.source_authority_id == self.review_authority_id
            or self.source_authority_hash == self.review_authority_hash
        ):
            raise ValueError("semantic review requires independent source and review authorities")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MemorySemanticReview":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected or any(not isinstance(item, str) for item in value.values()):
            raise ValueError("semantic review omitted its exact typed subject")
        return cls(**value)


@dataclass(frozen=True)
class CandidateVerificationSubject:
    workspace_id: str
    profile_id: str
    memory_id: str
    revision_id: str
    payload_hash: str
    source_evidence_hash: str
    target_block_id: str
    authority: str
    scope: str
    scope_id: str

    def __post_init__(self) -> None:
        for name in (
            "workspace_id",
            "profile_id",
            "memory_id",
            "revision_id",
            "payload_hash",
            "source_evidence_hash",
            "target_block_id",
            "authority",
            "scope",
            "scope_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"verification subject {name} must be a non-empty identity")
        if self.authority not in {"advisory", "directive"}:
            raise ValueError("verification subject authority is unsupported")


@dataclass(frozen=True)
class VerificationReportRef:
    report_id: str
    report_hash: str
    workspace_hash: str
    diff_hash: str
    produced_after_event_id: str
    verifier_id: str
    passed: bool
    ledger_hash: str | None = None

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("verification report passed must be a bool")
        for name in (
            "report_id",
            "report_hash",
            "workspace_hash",
            "diff_hash",
            "produced_after_event_id",
            "verifier_id",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"verification report {name} must be a string")
        if self.ledger_hash is not None and not isinstance(self.ledger_hash, str):
            raise ValueError("verification report ledger_hash must be a string or null")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "VerificationReportRef":
        allowed = {
            "report_id",
            "report_hash",
            "workspace_hash",
            "diff_hash",
            "produced_after_event_id",
            "verifier_id",
            "passed",
            "ledger_hash",
        }
        if unknown := set(value) - allowed:
            raise ValueError("unknown verification fields: " + ", ".join(sorted(unknown)))
        passed = value.get("passed")
        if not isinstance(passed, bool):
            raise ValueError("verification report passed must be a bool")
        return cls(
            report_id=value.get("report_id", ""),
            report_hash=value.get("report_hash", ""),
            workspace_hash=value.get("workspace_hash", ""),
            diff_hash=value.get("diff_hash", ""),
            produced_after_event_id=value.get("produced_after_event_id", ""),
            verifier_id=value.get("verifier_id", ""),
            passed=passed,
            ledger_hash=(
                value.get("ledger_hash") if value.get("ledger_hash") is not None else None
            ),
        )

    def evidence_ref(self, *, stage_id: str = "", subtask_id: str = "") -> EvidenceRef:
        return EvidenceRef(
            ref_type="verification_report",
            ref_id=self.report_id,
            hash=self.ledger_hash,
            produced_by="verifier",
            producer_id=self.verifier_id,
            stage_id=stage_id or None,
            subtask_id=subtask_id or None,
            created_after_event=self.produced_after_event_id or None,
        )


@dataclass(frozen=True)
class WorkspaceContext:
    repo_root: str
    workspace_root: str
    workspace_id: str
    current_workspace_hash: str = ""
    current_diff_hash: str = ""
    last_patch_event_id: str = ""
