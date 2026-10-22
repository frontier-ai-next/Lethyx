"""Compiled exact, structural, temporal, and provenance index inputs."""

import hashlib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from um_mem.contracts import MemoryBlockRecord, TransferEnvelope, memory_scope_key
from um_mem.memory.backends.contracts import MemoryIndexNode

_EXACT_METADATA_FIELDS = {
    "package": ("package", "packages", "package_id", "package_ids"),
    "component": ("component", "components", "component_id", "component_ids"),
    "path": ("path", "paths", "source_path", "source_paths", "file_path", "touched_files"),
    "symbol": ("symbol", "symbols", "target_symbols", "touched_symbols"),
    "entity": ("entity", "entities"),
    "dependency": ("dependency", "dependencies"),
    "configuration": ("configuration", "configurations"),
    "config_key": ("config_key", "config_keys"),
    "command": ("command", "commands"),
    "version": ("version", "versions"),
    "temporal_state": ("temporal_state", "temporal_states"),
    "endpoint": ("endpoint", "endpoints"),
    "artifact_hash": ("artifact_hash", "artifact_hashes"),
    "checkpoint_source_range": (
        "checkpoint_source_range_ref",
        "checkpoint_source_range_refs",
    ),
    "fold_source_range": ("fold_source_range_ref", "fold_source_range_refs"),
    "prospective_trigger": ("prospective_obligation_id", "prospective_obligation_ids"),
}
_HIERARCHY_TERM_TYPES = frozenset({"repository", "package", "component", "path", "symbol"})
_RETRIEVAL_FACET_TERM_TYPES = frozenset(
    {
        "entity",
        "dependency",
        "configuration",
        "config_key",
        "command",
        "version",
        "temporal_state",
        "endpoint",
    }
)


@dataclass(frozen=True, order=True)
class ExactIndexTerm:
    term_type: str
    term_value: str


@dataclass(frozen=True)
class MemoryIndexDocument:
    exact_terms: tuple[ExactIndexTerm, ...]
    structure_nodes: tuple[MemoryIndexNode, ...]
    reverse_provenance_links: tuple[tuple[str, str], ...]


def _text(value: object) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def _values(metadata: dict[str, Any], *keys: str) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw = metadata.get(key)
        candidates: Iterable[object]
        if isinstance(raw, str):
            candidates = (raw,)
        elif isinstance(raw, (list, tuple, set, frozenset)):
            candidates = raw
        else:
            candidates = ()
        for candidate in candidates:
            text = _text(candidate)
            if text and text not in values:
                values.append(text)
    return tuple(values)


def _structure_node_id(
    workspace_id: str,
    structure_kind: str,
    structure_value: str,
    parent_id: str,
) -> str:
    material = f"{workspace_id}\x00{structure_kind}\x00{structure_value}\x00{parent_id}"
    return f"project_structure:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def _structure_nodes(
    record: MemoryBlockRecord, exact: dict[str, tuple[str, ...]]
) -> tuple[MemoryIndexNode, ...]:
    metadata = dict(record.metadata)
    levels: list[tuple[str, str]] = []
    repository_id = _text(record.repository_id)
    if repository_id:
        levels.append(("repository", repository_id))
    package_id = next(iter(_values(metadata, "package_id", "package")), "")
    if package_id:
        levels.append(("package", package_id))
    component_id = next(iter(_values(metadata, "component_id", "component")), "")
    if component_id:
        levels.append(("component", component_id))
    path = next(iter(exact.get("path", ())), "")
    if path:
        levels.append(("file", path))
    symbol = next(iter(exact.get("symbol", ())), "")
    if symbol:
        levels.append(("symbol", symbol))

    nodes: list[MemoryIndexNode] = []
    parent_id = ""
    for structure_kind, structure_value in levels:
        node_id = _structure_node_id(
            record.workspace_id,
            structure_kind,
            structure_value,
            parent_id,
        )
        nodes.append(
            MemoryIndexNode(
                node_id=node_id,
                title=f"{structure_kind}: {structure_value}",
                node_type="project_structure",
                workspace_id=record.workspace_id,
                tags=("project_structure", structure_kind),
                metadata={
                    "structure_kind": structure_kind,
                    "structure_value": structure_value,
                    "parent_id": parent_id,
                    "related_record_ids": [record.block_id],
                },
                created_at=record.created_at or record.recorded_at,
                updated_at=record.updated_at or record.recorded_at,
            )
        )
        parent_id = node_id
    return tuple(nodes)


def _facet_nodes(
    record: MemoryBlockRecord,
    exact: dict[str, tuple[str, ...]],
) -> tuple[MemoryIndexNode, ...]:
    nodes: list[MemoryIndexNode] = []
    for term_type in sorted(exact):
        if term_type not in _RETRIEVAL_FACET_TERM_TYPES:
            continue
        for term_value in exact[term_type]:
            nodes.append(
                MemoryIndexNode(
                    node_id=_structure_node_id(
                        record.workspace_id,
                        f"facet:{term_type}",
                        term_value,
                        "",
                    ),
                    title=f"{term_type}: {term_value}",
                    node_type="exact_facet",
                    workspace_id=record.workspace_id,
                    tags=("exact_facet", term_type),
                    metadata={
                        "term_type": term_type,
                        "term_value": term_value,
                        "related_record_ids": [record.block_id],
                    },
                    created_at=record.created_at or record.recorded_at,
                    updated_at=record.updated_at or record.recorded_at,
                )
            )
    return tuple(nodes)


def compile_memory_index_document(
    record: MemoryBlockRecord,
    *,
    transfer_envelope: TransferEnvelope | None = None,
) -> MemoryIndexDocument:
    if not isinstance(record, MemoryBlockRecord):
        raise ValueError("record must use MemoryBlockRecord")
    if transfer_envelope is not None and not isinstance(transfer_envelope, TransferEnvelope):
        raise ValueError("transfer_envelope must use TransferEnvelope")
    metadata = dict(record.metadata)
    values_by_type = {
        term_type: _values(metadata, *keys) for term_type, keys in _EXACT_METADATA_FIELDS.items()
    }
    if record.artifact_revision:
        values_by_type["artifact_hash"] = tuple(
            dict.fromkeys((*values_by_type["artifact_hash"], record.artifact_revision))
        )
    if record.repository_id:
        values_by_type["repository"] = (record.repository_id,)
    values_by_type["profile_id"] = (record.profile_id,)
    values_by_type["scope_key"] = (memory_scope_key(record),)
    if record.block_membership is not None:
        values_by_type["primary_block_id"] = (record.block_membership.primary_block_id,)
    evidence_values = [value for ref in record.evidence_refs for value in (ref.ref_id, ref.hash)]
    for segment in record.metadata.get("source_segments", ()):
        if isinstance(segment, dict):
            evidence_values.extend(
                (str(segment.get("source_id") or ""), str(segment.get("sha256") or ""))
            )
    values_by_type["evidence_ref"] = tuple(
        dict.fromkeys(value for value in evidence_values if value)
    )

    reverse_links: set[tuple[str, str]] = set()
    for claim in record.child_claims:
        for segment in claim.origin_segments:
            values_by_type.setdefault("origin_channel", ())
            values_by_type["origin_channel"] = tuple(
                dict.fromkeys((*values_by_type["origin_channel"], segment.origin.channel))
            )
            values_by_type.setdefault("independence_domain", ())
            values_by_type["independence_domain"] = tuple(
                dict.fromkeys((*values_by_type["independence_domain"], segment.independence_domain))
            )
            raw_refs = (segment.origin.origin_id, *(ref.ref_id for ref in segment.evidence_refs))
            values_by_type.setdefault("raw_source_ref", ())
            values_by_type["raw_source_ref"] = tuple(
                dict.fromkeys((*values_by_type["raw_source_ref"], *raw_refs))
            )
            if claim.derivation_receipt is not None:
                reverse_links.update(
                    (claim.derivation_receipt.derived_id, raw_ref) for raw_ref in raw_refs
                )
    if transfer_envelope is not None:
        values_by_type["transfer_envelope_id"] = (transfer_envelope.envelope_id,)
        values_by_type["transfer_status"] = (transfer_envelope.status,)
        if transfer_envelope.source_scope:
            values_by_type["transfer_source_scope"] = (transfer_envelope.source_scope,)
        if transfer_envelope.target_scope:
            values_by_type["transfer_target_scope"] = (transfer_envelope.target_scope,)

    terms = tuple(
        sorted(
            {
                ExactIndexTerm(term_type=term_type, term_value=_text(value))
                for term_type, values in values_by_type.items()
                for value in values
                if _text(value)
            }
        )
    )
    return MemoryIndexDocument(
        exact_terms=terms,
        structure_nodes=(
            *_structure_nodes(record, values_by_type),
            *_facet_nodes(record, values_by_type),
        ),
        reverse_provenance_links=tuple(sorted(reverse_links)),
    )
