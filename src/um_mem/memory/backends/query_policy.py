"""Shared query normalization and safety filtering for native memory backends."""

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass, replace
from math import isfinite
from typing import Any, Literal, cast

from um_mem.contracts import MemoryBlockRecord, is_protected_memory
from um_mem.contracts.memory import CURRENT_SCHEMA_VERSION, memory_scope_key
from um_mem.memory.backends.base import MemoryQuery
from um_mem.memory.recall_plan import RecallPlan
from um_mem.memory.selection import MemorySelectionPolicy, SelectionContext

_QUERY_PAYLOAD_FIELDS = frozenset(
    {
        "include_candidates",
        "scope_chain",
        "source_horizon",
        "limit",
        "logical_block_ids",
        "search_all",
        "max_tokens",
        "query",
        "normalized_query",
        "query_type",
        "retrieval_mode",
        "revision_id",
        "run_id",
        "record_ids",
        "tags",
        "temporal_mode",
        "as_of",
        "known_as_of",
        "workspace_id",
        "profile_id",
        "paths",
        "symbols",
        "entities",
        "config_keys",
        "endpoints",
        "artifact_hashes",
        "facets",
    }
)

NATIVE_BLOCK_PROJECTION_MARKER = "block_partition_v1"
NATIVE_BLOCK_SCOPING_UNSUPPORTED = "native_block_scoping_unsupported"
NATIVE_PROJECTION_REBUILD_REQUIRED = "native_projection_rebuild_required"
RESERVED_NATIVE_BLOCK_TAG_PREFIX = "um_mem:block:"


def query_values(payload: dict[str, Any], *keys: str) -> set[str]:
    values: set[str] = set()
    for key in keys:
        raw = payload.get(key)
        candidates: Iterable[object]
        if isinstance(raw, str):
            candidates = (raw,)
        elif isinstance(raw, (list, tuple, set, frozenset)):
            candidates = raw
        else:
            candidates = ()
        values.update(str(item).strip() for item in candidates if str(item).strip())
    return values


def query_sequence(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = payload.get(key)
    candidates: Iterable[object]
    if isinstance(raw, str):
        candidates = raw.replace(";", ",").split(",")
    elif isinstance(raw, (set, frozenset)):
        candidates = sorted(raw, key=str)
    elif isinstance(raw, (list, tuple)):
        candidates = raw
    else:
        candidates = ()
    return tuple(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))


def normalize_query_payload(
    query: MemoryQuery | dict[str, Any] | Any,
    *,
    default_workspace_id: str = "",
    default_limit: int = 10,
) -> dict[str, Any]:
    if isinstance(query, dict):
        payload = dict(query)
    elif is_dataclass(query):
        if isinstance(query, type):
            raise ValueError("recall query requires a dataclass instance")
        payload = asdict(query)
    else:
        payload = dict(getattr(query, "__dict__", {}) or {})
    unknown = sorted(set(payload).difference(_QUERY_PAYLOAD_FIELDS))
    if unknown:
        raise ValueError("unknown recall query fields: " + ", ".join(unknown))
    if default_workspace_id and not payload.get("workspace_id"):
        payload["workspace_id"] = default_workspace_id
    if "limit" not in payload:
        payload["limit"] = default_limit
    temporal_mode = str(payload.get("temporal_mode") or "current")
    if temporal_mode not in {"current", "historical"}:
        raise ValueError("temporal_mode must be current or historical")
    if temporal_mode == "current" and payload.get("revision_id"):
        raise ValueError("revision_id requires historical temporal_mode")
    payload["temporal_mode"] = temporal_mode
    canonical_plan = RecallPlan.compile(
        str(payload.get("query") or ""),
        normalized_query=str(payload.get("normalized_query") or ""),
        query_type=payload.get("query_type"),
        retrieval_mode=payload.get("retrieval_mode", "semantic"),
        workspace_id=str(payload.get("workspace_id") or ""),
        profile_id=str(payload.get("profile_id") or "default"),
        logical_block_ids=query_sequence(payload, "logical_block_ids"),
        record_ids=query_sequence(payload, "record_ids"),
        search_all=payload.get("search_all", False),
        tags=query_sequence(payload, "tags"),
        paths=query_sequence(payload, "paths"),
        symbols=query_sequence(payload, "symbols"),
        entities=query_sequence(payload, "entities"),
        source_horizon=payload.get("source_horizon", "durable"),
        include_candidates=payload.get("include_candidates", False),
        scope_chain=query_sequence(payload, "scope_chain"),
        temporal_mode=temporal_mode,
        as_of=payload.get("as_of", ""),
        known_as_of=payload.get("known_as_of", ""),
        revision_id=payload.get("revision_id", ""),
    )
    payload["normalized_query"] = canonical_plan.normalized_query
    payload["query_type"] = canonical_plan.query_type
    payload["profile_id"] = canonical_plan.profile_id
    payload["retrieval_mode"] = canonical_plan.retrieval_mode
    payload["known_as_of"] = canonical_plan.known_as_of
    payload["record_ids"] = list(canonical_plan.record_ids)
    payload["logical_block_ids"] = list(canonical_plan.logical_block_ids)
    payload["search_all"] = canonical_plan.search_all
    payload["source_horizon"] = canonical_plan.source_horizon
    payload["scope_chain"] = list(canonical_plan.scope_chain)
    tags = query_values(payload, "tags")
    if tags:
        payload["tags"] = sorted(tags)
    for field_name in (
        "paths",
        "symbols",
        "entities",
        "config_keys",
        "endpoints",
        "artifact_hashes",
    ):
        values = query_values(payload, field_name)
        if values:
            payload[field_name] = sorted(values)
        else:
            payload.pop(field_name, None)
    facets = payload.get("facets")
    if facets is not None:
        if not isinstance(facets, dict):
            raise ValueError("facets must be a mapping of exact facet names to values")
        normalized_facets: dict[str, list[str]] = {}
        for facet_name, raw_values in facets.items():
            if not isinstance(facet_name, str) or not facet_name.strip():
                raise ValueError("facet names must be non-empty strings")
            values = query_values({"values": raw_values}, "values")
            if not values:
                raise ValueError("facet values must be non-empty strings")
            normalized_facets[facet_name.strip()] = sorted(values)
        payload["facets"] = normalized_facets
    return payload


def record_ids(payload: dict[str, Any]) -> set[str]:
    return query_values(payload, "record_ids")


def record_id_sequence(payload: dict[str, Any]) -> tuple[str, ...]:
    return query_sequence(payload, "record_ids")


def logical_block_ids(payload: dict[str, Any]) -> set[str]:
    return query_values(payload, "logical_block_ids")


def native_primary_block_id(record: MemoryBlockRecord) -> str:
    membership = record.block_membership
    if membership is None:
        raise ValueError("native projection requires a primary block membership")
    return membership.primary_block_id


def native_block_tag(logical_block_id: str) -> str:
    block_id = str(logical_block_id).strip()
    if not block_id:
        raise ValueError("native block tag requires a logical block id")
    return f"{RESERVED_NATIVE_BLOCK_TAG_PREFIX}{block_id}"


def native_block_source(logical_block_id: str) -> str:
    block_id = str(logical_block_id).strip()
    if not block_id:
        raise ValueError("native block source requires a logical block id")
    return f"um_mem:block:{block_id}"


def merge_native_block_results(
    pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]],
    *,
    limit: int,
    score_mode: Literal["cosine_similarity", "partition_rank"],
) -> list[MemoryBlockRecord]:
    """Compare scores only inside a backend-qualified shared embedding space."""
    ranked = [
        (logical_block_id, rank, score, record)
        for logical_block_id, rows in pools
        for rank, (record, score) in enumerate(rows, start=1)
    ]
    if score_mode == "cosine_similarity":
        comparable: list[tuple[str, int, float, MemoryBlockRecord]] = []
        for block_id, rank, score, record in ranked:
            if score is None or not isfinite(score):
                raise ValueError("native cosine recall requires finite similarity scores")
            comparable.append((block_id, rank, score, record))
        comparable.sort(
            key=lambda item: (-item[2], item[1], item[3].workspace_id, item[3].block_id)
        )
        ranked = list(comparable)
    elif score_mode == "partition_rank":
        # Native rank is meaningful within each pool; its score need not be
        # comparable to a separately ranked bank or runtime.
        ranked.sort(key=lambda item: item[1])
    else:
        raise ValueError("unknown native recall score mode")

    unique: list[tuple[str, int, float | None, MemoryBlockRecord]] = []
    seen: set[tuple[str, str]] = set()
    for item in ranked:
        record = item[3]
        identity = (record.workspace_id, record.block_id)
        if identity not in seen:
            seen.add(identity)
            unique.append(item)
    candidate_pool_ids = [record.block_id for *_, record in unique]
    selected = unique[: max(0, limit)]
    selected_ids = [record.block_id for *_, record in selected]
    return [
        replace(
            record,
            metadata={
                **dict(record.metadata or {}),
                "native_logical_block_id": logical_block_id,
                "native_rank": native_rank,
                "native_score": native_score,
                "candidate_pool_ids": candidate_pool_ids,
                "selected_record_ids": selected_ids,
                "selection_mode": (
                    "native_similarity" if score_mode == "cosine_similarity" else "block_rank_round"
                ),
                "native_score_mode": score_mode,
                **({"retrieval_score": native_score} if score_mode == "cosine_similarity" else {}),
            },
        )
        for logical_block_id, native_rank, native_score, record in selected
    ]


def merge_hybrid_block_results(
    semantic_pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]],
    lexical_pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]],
    *,
    limit: int,
) -> list[MemoryBlockRecord]:
    dense = [
        (block_id, score, record) for block_id, rows in semantic_pools for record, score in rows
    ]
    if any(score is None or not isfinite(score) for _, score, _ in dense):
        raise ValueError("hybrid semantic recall requires finite cosine similarity scores")
    dense.sort(
        key=lambda item: (-cast(float, item[1]), item[0], item[2].block_id, item[2].revision_id)
    )
    lexical = [
        (local_rank, block_id, record)
        for block_id, rows in lexical_pools
        for local_rank, (record, _) in enumerate(rows, start=1)
    ]
    lexical.sort(key=lambda item: (item[0], item[1], item[2].block_id, item[2].revision_id))
    identities: dict[tuple[str, str, str, str], MemoryBlockRecord] = {}
    ranks: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for channel, records in (
        ("semantic", [record for _, _, record in dense]),
        ("lexical", [record for _, _, record in lexical]),
    ):
        channel_rank = 0
        seen: set[tuple[str, str, str, str]] = set()
        for record in records:
            identity = (record.workspace_id, record.profile_id, record.block_id, record.revision_id)
            if identity in seen:
                continue
            seen.add(identity)
            channel_rank += 1
            identities[identity] = record
            ranks.setdefault(identity, {})[channel] = channel_rank
    scores = {
        identity: sum(1 / (60 + rank) for rank in values.values())
        for identity, values in ranks.items()
    }
    ordered = sorted(identities, key=lambda identity: (-scores[identity], identity))
    selected = ordered[: max(0, limit)]
    return [
        replace(
            identities[identity],
            metadata={
                **dict(identities[identity].metadata),
                "retrieval_mode": "hybrid",
                "native_logical_block_id": native_primary_block_id(identities[identity]),
                "selection_mode": "rrf",
                "native_score_mode": "rrf",
                "rrf_constant": 60,
                "channel_ranks": ranks[identity],
                "retrieval_score": scores[identity],
                "candidate_pool_ids": [item[2] for item in ordered],
                "selected_record_ids": [item[2] for item in selected],
            },
        )
        for identity in selected
    ]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def record_is_index_only(record: MemoryBlockRecord) -> bool:
    metadata = dict(record.metadata or {})
    return bool(
        record.kind == "skill_index"
        or _truthy(metadata.get("index_only"))
        or _truthy(metadata.get("routing_only"))
    )


def ensure_record_is_storable(record: MemoryBlockRecord) -> None:
    if type(record.schema_version) is not int or record.schema_version != CURRENT_SCHEMA_VERSION:
        raise ValueError(f"canonical memory record schema_version must be {CURRENT_SCHEMA_VERSION}")
    if record_is_index_only(record):
        raise ValueError("index-only memory blocks cannot store evidence records")
    native_primary_block_id(record)
    memory_scope_key(record)
    if is_protected_memory(record.sensitivity, record.content_mode, record.operational_risk):
        raise ValueError("protected or non-injectable memory cannot enter selected backend storage")


def record_allowed_for_query(record: MemoryBlockRecord, payload: dict[str, Any]) -> bool:
    if record_is_index_only(record):
        return False
    if record.profile_id != str(payload.get("profile_id") or "default"):
        return False
    policy = MemorySelectionPolicy()
    context = SelectionContext(
        query=str(payload.get("query") or ""),
        scope_chain=query_sequence(payload, "scope_chain"),
        run_id=str(payload.get("run_id") or ""),
        source_horizon=payload.get("source_horizon", "durable"),
        include_candidates=bool(payload.get("include_candidates", False)),
        historical=(
            str(payload.get("temporal_mode") or "current") == "historical"
            or bool(payload.get("known_as_of"))
        ),
        as_of="" if payload.get("known_as_of") else str(payload.get("as_of") or ""),
        tags=tuple(sorted(query_values(payload, "tags"))),
    )
    if not policy.eligible_for_recall(record, context):
        return False

    selected = record_ids(payload)
    if selected and record.block_id not in selected:
        return False

    logical = logical_block_ids(payload)
    if logical:
        membership = record.block_membership
        if membership is None or membership.primary_block_id not in logical:
            return False

    return True


def filter_records(
    records: list[MemoryBlockRecord], payload: dict[str, Any]
) -> list[MemoryBlockRecord]:
    return [record for record in records if record_allowed_for_query(record, payload)]
