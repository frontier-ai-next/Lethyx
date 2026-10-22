"""Canonical memory mutation policy for block updates and supersession."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

from um_mem.contracts import ConsolidationCommitReceipt, EvidenceRef, MemoryBlockRecord
from um_mem.contracts.memory import (
    DeliveryPolicy,
    MemoryAuthority,
    MemoryRetention,
    MemoryStatus,
    _closed_value,
)
from um_mem.memory.backends.contracts import MemoryEdgeRecord, validate_status_transition

_ORDER15_MUTATION_ACTIONS = frozenset(
    {
        "insert",
        "supersede",
        "correction",
        "deprecate",
        "retract",
        "rollback",
        "promote",
        "quarantine",
    }
)


@dataclass(frozen=True, slots=True)
class MemoryRetractionReceipt:
    source_record_id: str
    record_id: str
    revision_id: str
    outcome: Literal["retracted", "already_retracted", "superseded"]


def _event_identity(value: object) -> str:
    """Keep opaque host identifiers distinct without turning them into block ids."""

    return value.strip() if isinstance(value, str) else ""


def derive_canonical_key(
    *,
    workspace_id: str,
    profile_id: str = "default",
    kind: str,
    record_id: str,
) -> str:
    if not record_id:
        raise ValueError("canonical creation requires an explicit record identity")
    raw = json.dumps([workspace_id, profile_id, kind, record_id], ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    prefix = (
        f"{workspace_id}:{kind}"
        if profile_id == "default"
        else f"{workspace_id}:{profile_id}:{kind}"
    )
    return f"{prefix}:{digest}"


def block_id_for_canonical_key(canonical_key: str) -> str:
    digest = hashlib.sha256(str(canonical_key).encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


def block_id_for_revision_id(revision_id: str) -> str:
    digest = hashlib.sha256(str(revision_id).encode("utf-8")).hexdigest()[:16]
    return f"mem_rev_{digest}"


def _duplicate_cluster_id_for_record(record: MemoryBlockRecord) -> str:
    identity_axes = [record.workspace_id.strip()]
    if record.profile_id != "default":
        identity_axes.append(f"profile:{record.profile_id.strip()}")
    material = json.dumps(
        [
            *identity_axes,
            record.kind,
            record.content,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "dup_" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _revision_payload(canonical_key: str, record: MemoryBlockRecord) -> dict[str, Any]:
    payload = asdict(record)
    for field_name in (
        "block_id",
        "canonical_key",
        "revision_id",
        "duplicate_cluster_id",
        "recorded_at",
        "created_at",
        "updated_at",
        "status",
        "retention",
        "delivery_policy",
        "temperature",
        "authority",
        "trust_level",
        "verification_status",
        "verification_methods",
        "promotion_decision",
        "revision_of",
        "supersedes_ids",
        "semantic_catalog_version",
    ):
        payload.pop(field_name, None)
    metadata = dict(payload.get("metadata") or {})
    for key in (
        "block_id",
        "canonical_id",
        "canonical_key",
        "backend_reference",
        "backend_revision",
        "commit_revision",
        "idempotency_key",
        "memory_id",
        "mutation_correlation_id",
        "native_id",
        "native_reference",
        "native_revision",
        "recorded_at",
        "record_id",
        "revision_id",
        "service_receipt_sha256",
    ):
        metadata.pop(key, None)
    payload["canonical_key"] = canonical_key
    payload["metadata"] = metadata
    membership = payload.get("block_membership")
    if isinstance(membership, dict):
        payload["block_membership"] = {
            "primary_block_id": membership.get("primary_block_id"),
            "secondary_block_ids": membership.get("secondary_block_ids") or (),
        }
    return payload


def revision_id_for_record(
    canonical_key: str,
    record: MemoryBlockRecord,
    *,
    mutation_action: str,
    predecessor_revision_id: str = "",
) -> str:
    payload = _revision_payload(canonical_key, record)
    if predecessor_revision_id:
        payload.pop("valid_from", None)
        payload.pop("valid_to", None)
    payload["mutation_action"] = mutation_action
    payload["predecessor_revision_id"] = predecessor_revision_id
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "rev_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _same_semantic_revision(
    existing: MemoryBlockRecord,
    record: MemoryBlockRecord,
    *,
    canonical_key: str,
) -> bool:
    existing_payload = _revision_payload(existing.canonical_key, existing)
    record_payload = _revision_payload(canonical_key, record)
    if existing.revision_of:
        existing_payload.pop("valid_from", None)
        existing_payload.pop("valid_to", None)
        record_payload.pop("valid_from", None)
        record_payload.pop("valid_to", None)
    return existing_payload == record_payload


def _recorded_at() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CanonicalMutationIdentity:
    """Committed identity required to recognize an exact canonical retry."""

    action: str
    predecessor_id: str
    successor_id: str
    rollback_target_id: str = ""

    def __post_init__(self) -> None:
        action = self.action.strip()
        predecessor_id = self.predecessor_id.strip()
        successor_id = self.successor_id.strip()
        rollback_target_id = self.rollback_target_id.strip()
        if action not in _ORDER15_MUTATION_ACTIONS:
            raise ValueError(f"unsupported Order15 mutation action: {action}")
        if not successor_id:
            raise ValueError("canonical mutation identity requires a successor")
        if action == "rollback" and not rollback_target_id:
            raise ValueError("rollback mutation identity requires a rollback target")
        if rollback_target_id and action != "rollback":
            raise ValueError("only rollback mutation identities may carry a rollback target")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "predecessor_id", predecessor_id)
        object.__setattr__(self, "successor_id", successor_id)
        object.__setattr__(self, "rollback_target_id", rollback_target_id)


@dataclass(frozen=True)
class AmendmentCommitReceipt:
    receipt_id: str
    request_hash: str
    workspace_id: str
    profile_id: str
    predecessor_id: str
    predecessor_revision_id: str
    record_id: str
    revision_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"amendment receipt {name} requires an exact identity")

    def response(self, backend: str) -> dict[str, object]:
        return {
            "ok": True,
            "saved": True,
            "backend": backend,
            "workspace_id": self.workspace_id,
            "old_id": self.predecessor_id,
            "new_id": self.record_id,
            "memory_id": self.record_id,
            "old_revision": self.predecessor_revision_id,
            "new_revision": self.revision_id,
        }


@dataclass(frozen=True)
class PublicationCommitReceipt:
    receipt_id: str
    request_hash: str
    workspace_id: str
    profile_id: str
    source_id: str
    source_revision_id: str
    record_id: str
    revision_id: str
    target_block_id: str
    target_scope: str
    target_scope_id: str
    authority: str
    materialized_copy: bool = False

    def __post_init__(self) -> None:
        for name in (
            "receipt_id",
            "request_hash",
            "workspace_id",
            "profile_id",
            "source_id",
            "source_revision_id",
            "record_id",
            "revision_id",
            "target_block_id",
            "target_scope",
            "target_scope_id",
            "authority",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"publication receipt {name} requires an exact identity")
        if type(self.materialized_copy) is not bool or self.authority not in {
            "advisory",
            "directive",
        }:
            raise ValueError("publication receipt governance is invalid")

    def response(self, backend: str) -> dict[str, object]:
        return {
            "ok": True,
            "saved": True,
            "block_id": self.record_id,
            "source_memory_id": self.source_id,
            "result_memory_id": self.record_id,
            "promotion_receipt_id": self.receipt_id,
            "revision_id": self.revision_id,
            "requested_block_id": self.target_block_id,
            "actual_block_id": self.target_block_id,
            "target_scope": self.target_scope,
            "target_scope_id": self.target_scope_id,
            "authority": self.authority,
            "status": "accepted",
            "verification_status": "verified",
            "promotion_decision": "accepted",
            "tribunal_decision": "accept",
            "backend": backend,
        }


@dataclass(frozen=True)
class MutationDecision:
    record: MemoryBlockRecord
    superseded_block_id: str = ""
    action: str = "insert"
    edge: MemoryEdgeRecord | None = None
    rollback_target_id: str = ""
    affected_view_ids: tuple[str, ...] = ()
    consolidation_receipt: ConsolidationCommitReceipt | None = None
    publication_receipt: PublicationCommitReceipt | None = None
    amendment_receipt: AmendmentCommitReceipt | None = None

    def __post_init__(self) -> None:
        action = self.action.strip()
        if action not in _ORDER15_MUTATION_ACTIONS | {"retry"}:
            raise ValueError(f"unsupported Order15 mutation action: {action}")
        if action == "rollback" and not self.rollback_target_id:
            raise ValueError("rollback mutation requires a rollback target")
        if self.rollback_target_id and action != "rollback":
            raise ValueError("only rollback mutations may carry a rollback target")
        receipt = self.consolidation_receipt
        if receipt is not None:
            if not isinstance(receipt, ConsolidationCommitReceipt):
                raise TypeError("consolidation receipt must be a typed commit receipt")
            if self.record.block_id not in receipt.committed_ids:
                raise ValueError("consolidation receipt must commit the mutation successor")
        object.__setattr__(self, "action", action)
        publication = self.publication_receipt
        if publication is not None and (
            not isinstance(publication, PublicationCommitReceipt)
            or publication.record_id != self.record.block_id
            or publication.revision_id != self.record.revision_id
            or publication.workspace_id != self.record.workspace_id
            or publication.profile_id != self.record.profile_id
        ):
            raise ValueError("publication receipt must bind the exact mutation successor")
        amendment = self.amendment_receipt
        if amendment is not None and (
            not isinstance(amendment, AmendmentCommitReceipt)
            or self.action != "correction"
            or amendment.record_id != self.record.block_id
            or amendment.revision_id != self.record.revision_id
            or amendment.workspace_id != self.record.workspace_id
            or amendment.profile_id != self.record.profile_id
            or amendment.predecessor_id != self.record.revision_of
        ):
            raise ValueError("amendment receipt must bind the exact correction successor")


@dataclass(frozen=True)
class MemoryEraseRequest:
    request_id: str
    memory_id: str
    workspace_id: str
    reason: str
    authorization: EvidenceRef
    evidence_refs: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class MemoryEraseCertificate:
    certificate_sha256: str
    request_id_sha256: str
    target_id_sha256: str
    backend: str
    issued_at: str
    native_exact_id_absent: bool
    native_edges_absent: bool
    raw_evidence_retained: bool
    sidecar_status: str

    def as_dict(self) -> dict[str, str | bool | dict[str, str | bool]]:
        return {
            "certificate_sha256": self.certificate_sha256,
            "request_id_sha256": self.request_id_sha256,
            "target_id_sha256": self.target_id_sha256,
            "backend": self.backend,
            "issued_at": self.issued_at,
            "closure": {
                "native_exact_id_absent": self.native_exact_id_absent,
                "native_edges_absent": self.native_edges_absent,
                "raw_evidence_retained": self.raw_evidence_retained,
                "sidecar_status": self.sidecar_status,
            },
        }


def resolve_canonical_revision_head(
    root: MemoryBlockRecord,
    *,
    canonical_key: str,
    workspace_id: str,
    get_block: Callable[[str], MemoryBlockRecord | None],
    edges_for: Callable[[str], list[MemoryEdgeRecord]] | None = None,
) -> MemoryBlockRecord:
    """Resolve the sole current head of one canonical revision chain."""

    current = root
    visited: set[str] = set()
    while True:
        if current.block_id in visited:
            raise ValueError("canonical memory revision chain contains a cycle")
        visited.add(current.block_id)
        successors: set[str] = set()
        if edges_for is not None:
            successors.update(
                str(edge.src_id or "").strip()
                for edge in edges_for(current.block_id)
                if edge.edge_type == "supersedes" and edge.dst_id == current.block_id
            )
            successors.discard("")
        if not successors:
            return current
        if len(successors) != 1:
            raise ValueError("canonical memory revision chain has multiple current heads")
        successor_id = successors.pop()
        successor = get_block(successor_id)
        if not isinstance(successor, MemoryBlockRecord):
            raise ValueError("canonical memory revision successor is unavailable")
        if successor.workspace_id != workspace_id:
            raise ValueError("canonical memory revision successor is outside the workspace")
        if successor.canonical_key != canonical_key:
            raise ValueError("canonical memory revision successor has a different canonical key")
        current = successor


def plan_memory_mutation(
    record: MemoryBlockRecord,
    *,
    existing: MemoryBlockRecord | None,
    action: str = "",
    rollback_target_id: str = "",
    affected_view_ids: tuple[str, ...] = (),
    consolidation_receipt: ConsolidationCommitReceipt | None = None,
    publication_receipt: PublicationCommitReceipt | None = None,
    current_identity: CanonicalMutationIdentity | None = None,
) -> MutationDecision:
    """Plan one immutable canonical successor or an exact semantic retry."""

    if not record.duplicate_cluster_id:
        record = replace(
            record,
            duplicate_cluster_id=_duplicate_cluster_id_for_record(record),
        )
    canonical_key = (
        record.canonical_key
        or (existing.canonical_key if existing is not None else "")
        or derive_canonical_key(
            workspace_id=record.workspace_id,
            profile_id=record.profile_id,
            kind=record.kind,
            record_id=(
                "host:"
                + _event_identity(record.metadata.get("host_event_id"))
                + ":"
                + str(record.metadata.get("record_kind") or record.kind)
                if record.producer == "host_observer" and record.metadata.get("host_event_id")
                else record.block_id
            ),
        )
    )
    requested_action = action.strip()
    rollback_target_id = rollback_target_id.strip()
    if requested_action and requested_action not in _ORDER15_MUTATION_ACTIONS:
        raise ValueError(f"unsupported Order15 mutation action: {requested_action}")
    if current_identity is not None and not isinstance(current_identity, CanonicalMutationIdentity):
        raise TypeError("canonical mutation retry requires a typed mutation identity")
    if existing is not None:
        existing_membership = existing.block_membership
        requested_membership = record.block_membership
        if (
            existing.workspace_id != record.workspace_id
            or existing.profile_id != record.profile_id
            or existing_membership is None
            or requested_membership is None
        ):
            raise ValueError("canonical revisions cannot supersede across workspaces or profiles")
        if (
            existing_membership.primary_block_id != requested_membership.primary_block_id
            and requested_action != "correction"
        ):
            raise ValueError(
                "only an explicit correction may move a canonical revision between primary blocks"
            )
    mutation_action = requested_action or ("supersede" if existing is not None else "insert")
    if (
        existing is not None
        and not affected_view_ids
        and consolidation_receipt is None
        and publication_receipt is None
        and _same_semantic_revision(
            existing,
            record,
            canonical_key=canonical_key,
        )
        and (
            (not requested_action and not rollback_target_id)
            or (
                bool(requested_action)
                and current_identity is not None
                and current_identity.action == requested_action
                and current_identity.predecessor_id == existing.revision_of
                and current_identity.successor_id == existing.block_id
                and current_identity.rollback_target_id == rollback_target_id
            )
        )
    ):
        return MutationDecision(
            record=existing,
            action="retry",
        )
    mutation_timestamp = (
        _recorded_at() if existing is not None else (record.recorded_at or _recorded_at())
    )
    source = (
        replace(record, recorded_at=mutation_timestamp, valid_from=mutation_timestamp, valid_to="")
        if existing is not None
        else replace(record, recorded_at=mutation_timestamp)
    )
    predecessor_revision_id = existing.revision_id if existing is not None else ""
    revision_id = revision_id_for_record(
        canonical_key,
        source,
        mutation_action=mutation_action,
        predecessor_revision_id=predecessor_revision_id,
    )
    planned = replace(
        source,
        block_id=block_id_for_canonical_key(canonical_key),
        canonical_key=canonical_key,
        revision_id=revision_id,
        revision_of="",
        supersedes_ids=(),
    )
    if existing is None:
        return MutationDecision(
            record=planned,
            action=mutation_action,
            rollback_target_id=rollback_target_id,
            affected_view_ids=tuple(dict.fromkeys(affected_view_ids)),
            consolidation_receipt=consolidation_receipt,
            publication_receipt=publication_receipt,
        )

    superseding = replace(
        planned,
        block_id=block_id_for_revision_id(revision_id),
        revision_of=existing.block_id,
        supersedes_ids=(existing.block_id,),
    )
    edge = MemoryEdgeRecord(
        src_id=superseding.block_id,
        dst_id=existing.block_id,
        edge_type="supersedes",
        workspace_id=superseding.workspace_id,
        run_id=superseding.run_id,
        subtask_id=superseding.subtask_id,
        reason=mutation_action,
        evidence_refs=superseding.evidence_refs,
        metadata={
            "canonical_key_sha256": hashlib.sha256(canonical_key.encode("utf-8")).hexdigest(),
            "mutation_action": mutation_action,
            "rollback_target_id": rollback_target_id,
        },
    )
    return MutationDecision(
        record=superseding,
        superseded_block_id=existing.block_id,
        action=mutation_action,
        edge=edge,
        rollback_target_id=rollback_target_id,
        affected_view_ids=tuple(dict.fromkeys(affected_view_ids)),
        consolidation_receipt=consolidation_receipt,
        publication_receipt=publication_receipt,
    )


def plan_lifecycle_mutation(
    current: MemoryBlockRecord,
    *,
    action: str,
    status: str,
    reason: str = "",
    evidence_refs: tuple[EvidenceRef, ...] = (),
    authority: str | None = None,
    retention: str | None = None,
    delivery_policy: str | None = None,
    rollback_target: MemoryBlockRecord | None = None,
    current_identity: CanonicalMutationIdentity | None = None,
) -> MutationDecision:
    """Plan a lifecycle successor without changing the historical record."""

    target = rollback_target or current
    if rollback_target is not None:
        if rollback_target.workspace_id != current.workspace_id:
            raise ValueError("rollback target is outside the current workspace")
        if rollback_target.canonical_key != current.canonical_key:
            raise ValueError("rollback target has a different canonical key")
        status = rollback_target.status
        authority = rollback_target.authority
        retention = rollback_target.retention
        delivery_policy = rollback_target.delivery_policy
    else:
        validate_status_transition(
            current,
            status,
            reason=reason,
            evidence_refs=list(evidence_refs),
            authority=authority,
            retention=retention,
            delivery_policy=delivery_policy,
        )
    merged_refs = tuple(dict.fromkeys([*target.evidence_refs, *evidence_refs]))
    updated = replace(
        target,
        status=cast(MemoryStatus, _closed_value(status, field_name="status", shape=MemoryStatus)),
        authority=cast(
            MemoryAuthority,
            _closed_value(
                authority or target.authority, field_name="authority", shape=MemoryAuthority
            ),
        ),
        retention=cast(
            MemoryRetention,
            _closed_value(
                retention or target.retention, field_name="retention", shape=MemoryRetention
            ),
        ),
        delivery_policy=cast(
            DeliveryPolicy,
            _closed_value(
                delivery_policy or target.delivery_policy,
                field_name="delivery_policy",
                shape=DeliveryPolicy,
            ),
        ),
        evidence_refs=merged_refs,
        valid_from=current.valid_from,
        valid_to=current.valid_to,
        metadata={
            **dict(target.metadata),
            **({"status_reason": reason} if reason else {}),
        },
    )
    return plan_memory_mutation(
        updated,
        existing=current,
        action=action,
        rollback_target_id=rollback_target.block_id if rollback_target is not None else "",
        current_identity=current_identity,
    )
