"""Versioned semantic catalog and derived logical-block index nodes."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from um_mem.contracts.models import EvidenceRef
from um_mem.memory.backends.contracts import MemoryBlockRecord, MemoryIndexNode
from um_mem.memory.blocks.models import MemoryBlock

RefinementCaseType = Literal[
    "capture_miss",
    "selection_miss",
    "overretrieval",
    "reader_misuse",
    "novel_concept",
    "label_disagreement",
    "stale_guidance",
    "weak_model_failure",
]
FailureAttributionStage = Literal[
    "storage",
    "retrieval",
    "selection",
    "rendering",
    "comprehension",
    "action",
    "verifier",
]
RefinementTarget = Literal[
    "capture",
    "retrieval",
    "taxonomy",
    "renderer",
    "reader",
    "host_action",
    "verifier",
]
_REFINEMENT_CASE_TYPES = frozenset(
    {
        "capture_miss",
        "selection_miss",
        "overretrieval",
        "reader_misuse",
        "novel_concept",
        "label_disagreement",
        "stale_guidance",
        "weak_model_failure",
    }
)
_FAILURE_STAGES = frozenset(
    {
        "storage",
        "retrieval",
        "selection",
        "rendering",
        "comprehension",
        "action",
        "verifier",
    }
)
_REFINEMENT_TARGETS = {
    "storage": "capture",
    "retrieval": "retrieval",
    "selection": "taxonomy",
    "rendering": "renderer",
    "comprehension": "reader",
    "action": "host_action",
    "verifier": "verifier",
}
_DYNAMIC_LABEL_SCOPES = frozenset({"task", "session", "workspace", "shared", "global"})


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _text_tuple(value: Iterable[object] | object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[object] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        raise ValueError(f"{field_name} must be an iterable of strings")
    return tuple(dict.fromkeys(_required_text(item, field_name=field_name) for item in values))


def _parse_time(value: str, *, field_name: str) -> datetime:
    normalized = _required_text(value, field_name=field_name).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _revision(prefix: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class BlockGuidanceCard:
    block_id: str
    synopsis: str
    use_when: tuple[str, ...]
    avoid_when: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _required_text(self.block_id, field_name="block_id"))
        object.__setattr__(self, "synopsis", _required_text(self.synopsis, field_name="synopsis"))
        object.__setattr__(self, "use_when", _text_tuple(self.use_when, field_name="use_when"))
        object.__setattr__(
            self,
            "avoid_when",
            _text_tuple(self.avoid_when, field_name="avoid_when") if self.avoid_when else (),
        )


@dataclass(frozen=True)
class BlockDescriptor:
    block_id: str
    block_type: str
    parent_id: str
    scope: str
    authority: str
    delivery_policy: str
    retention: str
    producer: str
    consumers: tuple[str, ...]
    recall_path: str
    lifecycle_policy: dict[str, Any]
    isolation_policy: dict[str, Any]
    token_budget: int
    semantic_memberships: tuple[str, ...]
    guidance: BlockGuidanceCard
    descriptor_revision_id: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "block_id",
            "block_type",
            "scope",
            "authority",
            "delivery_policy",
            "retention",
            "producer",
            "recall_path",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "parent_id", str(self.parent_id or "").strip())
        object.__setattr__(
            self,
            "consumers",
            _text_tuple(self.consumers, field_name="consumers"),
        )
        object.__setattr__(
            self,
            "semantic_memberships",
            _text_tuple(self.semantic_memberships, field_name="semantic_memberships"),
        )
        if not isinstance(self.guidance, BlockGuidanceCard):
            raise TypeError("guidance must be a BlockGuidanceCard")
        if self.guidance.block_id != self.block_id:
            raise ValueError("guidance block_id must match descriptor block_id")
        object.__setattr__(self, "lifecycle_policy", dict(self.lifecycle_policy))
        object.__setattr__(self, "isolation_policy", dict(self.isolation_policy))
        if type(self.token_budget) is not int or self.token_budget < 0:
            raise ValueError("descriptor token_budget must be a non-negative int")
        payload = {
            key: value for key, value in asdict(self).items() if key != "descriptor_revision_id"
        }
        expected_revision = _revision("descriptor", payload)
        if self.descriptor_revision_id and self.descriptor_revision_id != expected_revision:
            raise ValueError("descriptor_revision_id does not match descriptor payload")
        object.__setattr__(self, "descriptor_revision_id", expected_revision)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicLabel:
    label_id: str
    label: str
    semantic_memberships: tuple[str, ...]
    scope: str
    workspace_id: str
    task_id: str = ""
    session_id: str = ""
    status: str = "active"
    authority: str = "advisory"
    created_by: str = "llm"
    created_at: str = ""
    expires_at: str = ""

    def __post_init__(self) -> None:
        for field_name in ("label_id", "label", "workspace_id", "created_at"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "semantic_memberships",
            _text_tuple(self.semantic_memberships, field_name="semantic_memberships"),
        )
        if self.scope not in _DYNAMIC_LABEL_SCOPES:
            raise ValueError("unsupported dynamic-label scope")
        if self.authority != "advisory":
            raise ValueError("dynamic labels must remain advisory")
        _parse_time(self.created_at, field_name="created_at")
        if self.scope == "task" and not str(self.task_id).strip():
            raise ValueError("task dynamic labels require task_id")
        if self.scope == "session" and not str(self.session_id).strip():
            raise ValueError("session dynamic labels require session_id")
        if self.scope == "workspace":
            if self.status != "experimental":
                raise ValueError("workspace dynamic labels must be experimental")
            if not self.expires_at:
                raise ValueError("workspace dynamic labels require TTL expiry")
        if self.scope in {"shared", "global"} and self.status not in {"shadow", "canary"}:
            raise ValueError("shared/global dynamic labels require shadow or canary status")
        if self.scope in {"task", "session"} and self.status != "active":
            raise ValueError("task/session dynamic labels must be immediately active")
        if self.expires_at:
            created = _parse_time(self.created_at, field_name="created_at")
            expires = _parse_time(self.expires_at, field_name="expires_at")
            if expires <= created:
                raise ValueError("dynamic-label expiry must follow creation")


@dataclass(frozen=True)
class RefinementCase:
    case_id: str
    case_type: RefinementCaseType
    failure_stage: FailureAttributionStage
    query: str
    decision_id: str
    task_signature: str
    action_trace_id: str
    evidence_refs: tuple[EvidenceRef, ...]
    reason: str
    selected_descriptor_ids: tuple[str, ...] = ()
    expected_descriptor_ids: tuple[str, ...] = ()
    status: str = "candidate"
    refinement_target: RefinementTarget = field(default="capture", init=False)
    taxonomy_change_recommended: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text(self.case_id, field_name="case_id"))
        object.__setattr__(self, "query", _required_text(self.query, field_name="query"))
        if self.case_type not in _REFINEMENT_CASE_TYPES:
            raise ValueError("unsupported refinement case type")
        if self.failure_stage not in _FAILURE_STAGES:
            raise ValueError("unsupported refinement failure stage")
        if self.status != "candidate":
            raise ValueError("refinement cases begin as candidates")
        for field_name in ("decision_id", "task_signature", "action_trace_id", "reason"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        refs = tuple(self.evidence_refs)
        if not refs or any(not isinstance(ref, EvidenceRef) for ref in refs):
            raise ValueError("refinement cases require typed evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)
        for field_name in ("selected_descriptor_ids", "expected_descriptor_ids"):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                _text_tuple(value, field_name=field_name) if value else (),
            )
        target = _REFINEMENT_TARGETS[self.failure_stage]
        object.__setattr__(self, "refinement_target", target)
        object.__setattr__(self, "taxonomy_change_recommended", target == "taxonomy")


@dataclass(frozen=True)
class SemanticCatalog:
    descriptors: tuple[BlockDescriptor, ...]
    dynamic_labels: tuple[DynamicLabel, ...] = ()
    refinement_cases: tuple[RefinementCase, ...] = ()
    revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        descriptors = tuple(sorted(self.descriptors, key=lambda item: item.block_id))
        if not descriptors:
            raise ValueError("semantic catalog requires descriptors")
        if len({item.block_id for item in descriptors}) != len(descriptors):
            raise ValueError("semantic catalog descriptor ids must be unique")
        labels = tuple(sorted(self.dynamic_labels, key=lambda item: item.label_id))
        cases = tuple(sorted(self.refinement_cases, key=lambda item: item.case_id))
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "dynamic_labels", labels)
        object.__setattr__(self, "refinement_cases", cases)
        payload = {
            "descriptors": [item.to_mapping() for item in descriptors],
            "dynamic_labels": [asdict(item) for item in labels],
            "refinement_cases": [asdict(item) for item in cases],
        }
        object.__setattr__(self, "revision_id", _revision("catalog", payload))

    def descriptor(self, block_id: str) -> BlockDescriptor:
        target = str(block_id or "").strip()
        for descriptor in self.descriptors:
            if descriptor.block_id == target:
                return descriptor
        raise KeyError(f"unknown semantic catalog block: {target}")

    def with_dynamic_label(
        self,
        label: DynamicLabel,
        *,
        workspace_quota: int = 32,
    ) -> "SemanticCatalog":
        if any(item.label_id == label.label_id for item in self.dynamic_labels):
            raise ValueError("dynamic label id already exists")
        admitted = admit_dynamic_label(
            label,
            existing=self.dynamic_labels,
            workspace_quota=workspace_quota,
        )
        return SemanticCatalog(
            descriptors=self.descriptors,
            dynamic_labels=(*self.dynamic_labels, admitted),
            refinement_cases=self.refinement_cases,
        )

    def with_refinement_case(self, case: RefinementCase) -> "SemanticCatalog":
        if not isinstance(case, RefinementCase):
            raise TypeError("case must be a RefinementCase")
        if any(item.case_id == case.case_id for item in self.refinement_cases):
            raise ValueError("refinement case id already exists")
        return SemanticCatalog(
            descriptors=self.descriptors,
            dynamic_labels=self.dynamic_labels,
            refinement_cases=(*self.refinement_cases, case),
        )


def _guidance_for(block: MemoryBlock) -> BlockGuidanceCard:
    guidance = dict(block.guidance)
    synopsis = str(guidance.get("synopsis") or "").strip()
    if not synopsis:
        synopsis = f"{block.block_id.replace('_', ' ')} ({block.block_type})"
    use_when = guidance.get("use_when") or (*block.semantic_memberships, *block.tags)
    avoid_when = guidance.get("avoid_when") or ()
    return BlockGuidanceCard(
        block_id=block.block_id,
        synopsis=synopsis,
        use_when=_text_tuple(use_when, field_name="use_when"),
        avoid_when=_text_tuple(avoid_when, field_name="avoid_when") if avoid_when else (),
    )


def _memberships_for(block: MemoryBlock) -> tuple[str, ...]:
    if block.semantic_memberships:
        return block.semantic_memberships
    candidates = [block.block_type, *block.tags]
    return tuple(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))


def _descriptor_for(block: MemoryBlock) -> BlockDescriptor:
    memberships = _memberships_for(block)
    guidance = _guidance_for(block)
    return BlockDescriptor(
        block_id=block.block_id,
        block_type=block.block_type,
        parent_id=block.parent_id,
        scope=block.scope,
        authority=block.authority,
        delivery_policy=block.delivery_policy,
        retention=block.retention,
        producer=block.producer or block.owner,
        consumers=block.consumers or ("memory_agent",),
        recall_path=block.recall_path or "hybrid_recall",
        lifecycle_policy=block.lifecycle_policy,
        isolation_policy=block.isolation_policy,
        token_budget=block.token_budget,
        semantic_memberships=memberships,
        guidance=guidance,
    )


def compile_semantic_catalog(blocks: Iterable[MemoryBlock]) -> SemanticCatalog:
    return SemanticCatalog(descriptors=tuple(_descriptor_for(block) for block in blocks))


def admit_dynamic_label(
    label: DynamicLabel,
    *,
    existing: Iterable[DynamicLabel],
    workspace_quota: int = 32,
) -> DynamicLabel:
    if not isinstance(label, DynamicLabel):
        raise TypeError("label must be a DynamicLabel")
    labels = tuple(existing)
    if any(not isinstance(item, DynamicLabel) for item in labels):
        raise TypeError("existing labels must be DynamicLabel values")
    if label.scope == "workspace":
        if workspace_quota < 1:
            raise ValueError("workspace dynamic-label quota must be positive")
        used = sum(
            item.scope == "workspace" and item.workspace_id == label.workspace_id for item in labels
        )
        if used >= workspace_quota:
            raise ValueError("workspace dynamic-label quota exceeded")
    return label


def block_descriptor_node_id(workspace_id: str, block_id: str) -> str:
    workspace = _required_text(workspace_id, field_name="workspace_id")
    block = _required_text(block_id, field_name="block_id")
    return f"block_descriptor:{workspace}:{block}"


def block_descriptor_from_mapping(value: object) -> BlockDescriptor:
    if not isinstance(value, dict):
        raise ValueError("persisted block descriptor must be a mapping")
    unknown = sorted(set(value) - set(BlockDescriptor.__dataclass_fields__))
    if unknown:
        raise ValueError("unknown persisted block descriptor fields: " + ", ".join(unknown))
    payload = dict(value)
    guidance = payload.get("guidance")
    if not isinstance(guidance, dict):
        raise ValueError("persisted block descriptor guidance must be a mapping")
    payload["guidance"] = BlockGuidanceCard(
        block_id=guidance.get("block_id", ""),
        synopsis=guidance.get("synopsis", ""),
        use_when=tuple(guidance.get("use_when") or ()),
        avoid_when=tuple(guidance.get("avoid_when") or ()),
    )
    payload["semantic_memberships"] = tuple(payload.get("semantic_memberships") or ())
    payload["consumers"] = tuple(payload.get("consumers") or ())
    return BlockDescriptor(**payload)


def descriptor_from_index_node(node: MemoryIndexNode) -> BlockDescriptor:
    if not isinstance(node, MemoryIndexNode) or node.node_type != "block_descriptor":
        raise ValueError("index node is not a block descriptor")
    return block_descriptor_from_mapping(node.metadata.get("descriptor"))


def sync_block_descriptor_index(
    backend: object,
    *,
    catalog: SemanticCatalog,
    record: MemoryBlockRecord,
    workspace_id: str,
    update_reason: str = "write",
) -> dict[str, Any]:
    membership = record.block_membership
    if membership is None:
        return {"ok": True, "status": "not_applicable"}
    logical_block_id = membership.primary_block_id
    if record.status == "retracted" and not any(
        item.block_id == logical_block_id for item in catalog.descriptors
    ):
        return {
            "ok": True,
            "status": "not_applicable",
            "reason": "hidden retraction has no active catalog descriptor",
        }
    retain = getattr(backend, "retain_index_node", None)
    if not callable(retain):
        return {"ok": False, "status": "unsupported"}
    descriptor = catalog.descriptor(logical_block_id)
    node_id = block_descriptor_node_id(workspace_id, logical_block_id)
    existing: MemoryIndexNode | None = None
    get_node = getattr(backend, "get_index_node", None)
    if callable(get_node):
        candidate = get_node(node_id, workspace_id=workspace_id)
        if isinstance(candidate, MemoryIndexNode):
            existing = candidate
    related_ids: set[str] = set()
    source_revision_ids: set[str] = set()
    current_records: list[MemoryBlockRecord] | None = None
    snapshot: dict[str, Any] | None = None
    previous_index_revision = 0
    previous_catalog_revision = ""
    if existing is not None:
        related_ids.update(str(item) for item in existing.metadata.get("related_block_ids", ()))
        source_revision_ids.update(
            str(item) for item in existing.metadata.get("source_revision_ids", ()) if str(item)
        )
        previous_index_revision = int(existing.metadata.get("index_revision") or 0)
        previous_catalog_revision = str(existing.metadata.get("catalog_revision_id") or "")
    snapshot_reader = getattr(backend, "logical_block_descriptor_snapshot", None)
    if callable(snapshot_reader):
        snapshot = snapshot_reader(
            logical_block_id,
            workspace_id=workspace_id,
            profile_id=record.profile_id,
        )
        related_ids = set(snapshot["related_block_ids"])
        source_revision_ids = set(snapshot["source_revision_ids"])
    record_ids_for_blocks = getattr(backend, "record_ids_for_logical_blocks", None)
    get_block = getattr(backend, "get_block", None)
    if snapshot is None and callable(record_ids_for_blocks) and callable(get_block):
        current_records = []
        for record_id in record_ids_for_blocks(
            (logical_block_id,),
            workspace_id=workspace_id,
        ):
            candidate = get_block(record_id, workspace_id=workspace_id)
            if not isinstance(candidate, MemoryBlockRecord):
                continue
            if candidate.status in {"superseded", "deprecated", "retracted", "quarantined"}:
                continue
            current_records.append(candidate)
        related_ids = {item.block_id for item in current_records}
        source_revision_ids = {
            str(item.revision_id or item.block_id).strip() for item in current_records
        }
    elif snapshot is None:
        related_ids.add(record.block_id)
        source_revision_ids.add(str(record.revision_id or record.block_id).strip())
    previous_source_revision_ids = {
        str(item)
        for item in (existing.metadata.get("source_revision_ids", ()) if existing else ())
        if str(item)
    }
    record_count = int(snapshot["record_count"]) if snapshot is not None else len(related_ids)
    previous_count = (
        int((existing.metadata.get("coverage") or {}).get("canonical_record_count") or 0)
        if existing
        else 0
    )
    source_changed = (
        source_revision_ids != previous_source_revision_ids
        or record_count != previous_count
        or (
            snapshot is not None
            and existing is not None
            and snapshot["latest_updated_at"] != existing.updated_at
        )
    )
    catalog_changed = previous_catalog_revision not in {"", catalog.revision_id}
    index_revision = previous_index_revision + int(
        existing is None or source_changed or catalog_changed
    )
    previous_metadata = dict(existing.metadata) if existing is not None else {}
    latest_record = (
        max(
            current_records,
            key=lambda item: (
                item.updated_at or item.recorded_at or item.created_at,
                item.block_id,
            ),
        )
        if current_records
        else None
    )
    if snapshot is not None:
        latest_record_title = str(snapshot["latest_record_title"])
        latest_updated_at = str(
            snapshot["latest_updated_at"] or record.updated_at or record.recorded_at
        )
        exact_dimensions = list(snapshot["exact_dimensions"])
    else:
        latest_record_title = (
            latest_record.title
            if latest_record is not None
            else (
                ""
                if current_records == []
                else (
                    record.title
                    if existing is None or source_changed
                    else str(previous_metadata.get("latest_record_title") or record.title)
                )
            )
        )
        latest_updated_at = (
            latest_record.updated_at or latest_record.recorded_at or latest_record.created_at
            if latest_record is not None
            else record.updated_at or record.recorded_at
        )
        exact_dimensions = sorted(
            key
            for key in (
                "paths",
                "symbols",
                "entities",
                "config_keys",
                "endpoints",
                "artifact_hashes",
            )
            if any(
                item.metadata.get(key)
                for item in (current_records if current_records is not None else [record])
            )
        )
    synopsis = (
        f"{descriptor.guidance.synopsis} "
        f"Coverage: {record_count} canonical revision(s); "
        f"latest: {latest_record_title or record.kind}."
    )
    coverage = {
        "canonical_revision_count": record_count,
        "canonical_record_count": record_count,
        "semantic_memberships": list(descriptor.semantic_memberships),
        "exact_dimensions": exact_dimensions,
        "record_ids_truncated": bool(snapshot["truncated"]) if snapshot is not None else False,
    }
    index_revision_id = _revision(
        "block-index",
        {
            "block_id": logical_block_id,
            "catalog_revision_id": catalog.revision_id,
            "index_revision": index_revision,
            "source_revision_ids": sorted(source_revision_ids),
            "record_count": record_count,
        },
    )
    node = MemoryIndexNode(
        node_id=node_id,
        title=f"Logical block: {logical_block_id}",
        node_type="block_descriptor",
        workspace_id=workspace_id,
        tags=descriptor.semantic_memberships[:32],
        metadata={
            "block_id": logical_block_id,
            "catalog_revision_id": catalog.revision_id,
            "descriptor": descriptor.to_mapping(),
            "related_block_ids": sorted(related_ids),
            "source_revision_ids": sorted(source_revision_ids),
            "index_revision": index_revision,
            "index_revision_id": index_revision_id,
            "synopsis": synopsis,
            "coverage": coverage,
            "latest_record_title": latest_record_title,
            "last_update_reason": (
                "consolidation" if update_reason == "consolidate" else str(update_reason or "write")
            )
            if existing is None or source_changed or catalog_changed
            else str(previous_metadata.get("last_update_reason") or "write"),
        },
        created_at=(
            existing.created_at if existing is not None else record.created_at or record.recorded_at
        ),
        updated_at=latest_updated_at,
    )
    result = dict(retain(node) or {})
    return {
        "ok": bool(result.get("ok")),
        "status": "updated" if existing is not None else "created",
        "node_id": node_id,
        "catalog_revision_id": catalog.revision_id,
        "index_revision": index_revision,
        "index_revision_id": index_revision_id,
    }
