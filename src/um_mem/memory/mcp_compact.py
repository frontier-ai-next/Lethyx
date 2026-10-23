"""Compact projections for the UM-MEM MCP surface.

The local service returns rich, audit-oriented envelopes. That is right for
operators and wrong for a model context window: a single 40-word memory hit
carries several hundred tokens of provenance. This module keeps the fields an
agent needs to act (identity, revision, score, text) and drops the rest.

Set ``UM_MEM_MCP_RESPONSE_MODE=compact`` to opt into the token-efficient
projection; full service payloads are the default.
"""

import os
from typing import Any

_BLOCK_FIELDS = (
    "block_id",
    "name",
    "synopsis",
    "tags",
    "parent_id",
    "child_ids",
    "child_count",
    "children_truncated",
    "next_cursor",
    "record_count",
    "canonical_record_count",
    "writable",
    "status",
    "revision",
    "latest_title",
    "expandable",
    "meta",
    "node_kind",
    "index_only",
    "diagnostics_status",
    "metrics_scope",
)

_RECORD_FIELDS = (
    "record_id",
    "revision_id",
    "logical_block_id",
    "title",
    "content",
    "tags",
    "kind",
    "recorded_at",
    "status",
    "trust_level",
    "authority",
    "truncated",
)

_ERROR_FIELDS = (
    "ok",
    "error",
    "error_type",
    "reason",
    "reason_code",
    "retryable",
    "workspace_id",
    "backend",
    "hits",
    "hint",
    "parent_record_count",
    "auto_archived",
    "auto_archive_error",
    "auto_index_parent_error",
    "supported_retrieval_modes",
    "saved",
    "memory_id",
    "record_id",
    "new_id",
    "old_id",
    "new_revision",
    "operation",
    "requested_block_id",
    "route_rejected",
    "policy_issues",
    "records_remaining",
    "archived",
    "deleted",
    "cascade_failures",
    "degraded",
    "backend_error",
    "relaxed_tried",
    "fallback",
)


def response_mode() -> str:
    # Full payloads are the library default and preserve the documented
    # Python/MCP/Hermes contract. Deployments that want the token-efficient
    # projection opt in explicitly with UM_MEM_MCP_RESPONSE_MODE=compact.
    return os.getenv("UM_MEM_MCP_RESPONSE_MODE", "full").strip().lower() or "full"


def _pick(source: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {key: source[key] for key in fields if key in source}


def _compact_block(block: Any) -> dict[str, Any]:
    compact = _pick(block, _BLOCK_FIELDS)
    children = block.get("children") if isinstance(block, dict) else None
    if isinstance(children, (list, tuple)):
        compact["children"] = [_compact_block(child) for child in children]
    return compact


def _compact_blocks(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_compact_block(value) for value in values]


def _compact_record(record: Any) -> dict[str, Any]:
    compact = _pick(record, _RECORD_FIELDS)
    if not isinstance(record, dict):
        return compact
    if not compact.get("record_id"):
        compact["record_id"] = str(record.get("block_id") or "")
    membership = record.get("block_membership")
    if not compact.get("logical_block_id") and isinstance(membership, dict):
        compact["logical_block_id"] = str(membership.get("primary_block_id") or "")
    return compact


def _compact_records(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_compact_record(value) for value in values]


def _compact_route(route: Any) -> dict[str, Any]:
    if not isinstance(route, dict):
        return {}
    return _pick(
        route,
        (
            "block_id",
            "accepted",
            "status",
            "reason",
            "action",
            "requested_block_id",
            "suggested_block_id",
            "suggested_action",
            "scope",
            "tags",
            "policy_issues",
        ),
    )


def _hit_record_id(hit: dict[str, Any]) -> str:
    if hit.get("record_id"):
        return str(hit["record_id"])
    metadata = hit.get("metadata")
    if isinstance(metadata, dict):
        for key in ("record_id", "block_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    source = str(hit.get("source") or "").strip()
    if source.startswith("um-mem:"):
        return source.split(":", 1)[1]
    return source


def _compact_hit(hit: Any) -> dict[str, Any]:
    if not isinstance(hit, dict):
        return {"text": str(hit)}
    metadata: dict[str, Any] = hit["metadata"] if isinstance(hit.get("metadata"), dict) else {}
    membership: dict[str, Any] = (
        metadata["block_membership"] if isinstance(metadata.get("block_membership"), dict) else {}
    )
    logical_block_id = (
        metadata.get("logical_block_id")
        or membership.get("primary_block_id")
        or hit.get("logical_block_id")
        or ""
    )
    compact = {
        "record_id": _hit_record_id(hit),
        "title": hit.get("title") or "",
        "text": hit.get("text") or "",
        "score": hit.get("score"),
        "score_mode": hit.get("score_mode") or metadata.get("native_score_mode") or "",
        "tags": list(hit.get("tags") or []),
        "logical_block_id": logical_block_id,
        "kind": hit.get("kind") or metadata.get("kind") or "",
        "recorded_at": metadata.get("recorded_at") or hit.get("recorded_at") or "",
        "revision_id": metadata.get("revision_id") or hit.get("revision_id") or "",
        "authority": hit.get("authority") or "",
        "status": metadata.get("status") or hit.get("status") or "",
    }
    for key in (
        "source_speakers",
        "source_dates",
        "speaker_id",
        "subject_id",
        "verification_status",
        "valid_from",
        "valid_to",
        "expires_at",
        "historical",
        "as_of",
        "known_as_of",
        "event_time_status",
    ):
        value = metadata.get(key, hit.get(key))
        if value not in (None, "", [], ()):
            compact[key] = value
    if hit.get("truncated"):
        compact["truncated"] = True
    return compact


def _compact_hits(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_compact_hit(value) for value in values]


def _compact_lease(lease: Any) -> Any:
    if not isinstance(lease, dict):
        return lease
    compact = {
        "run_id": lease.get("run_id") or "",
        "logical_block_ids": list(lease.get("logical_block_ids") or []),
        "items": _compact_hits(lease.get("items")),
    }
    if "rendered_context" in lease:
        compact["rendered_context"] = lease.get("rendered_context")
    if "token_count" in lease:
        compact["token_count"] = lease.get("token_count")
    compact.update(_pick(lease, ("token_budget", "estimated_tokens", "metadata_tokens")))
    compact["trace"] = _pick(
        lease.get("trace"),
        (
            "mask_id",
            "mask_revision",
            "catalog_revision",
            "resolution_status",
            "scope_validation_result",
            "refresh_reason",
            "truncated_item_count",
        ),
    )
    return compact


def _compact_health(result: dict[str, Any]) -> dict[str, Any]:
    service: dict[str, Any] = result["service"] if isinstance(result.get("service"), dict) else {}
    projection_value = result.get("projection") or service.get("projection")
    projection: dict[str, Any] = projection_value if isinstance(projection_value, dict) else {}
    queue_value = result.get("background_queue") or service.get("background_queue")
    queue: dict[str, Any] = queue_value if isinstance(queue_value, dict) else {}
    lanes = queue.get("lanes") if isinstance(queue, dict) else None
    compact = {
        "ok": result.get("ok"),
        "liveness": result.get("liveness"),
        "readiness": result.get("readiness"),
        "state": result.get("state"),
        "backend": result.get("backend") or service.get("backend"),
        "workspace_id": result.get("workspace_id") or service.get("workspace_id"),
        "profile_id": result.get("profile_id") or service.get("profile_id"),
        "native_ok": result.get("native_ok"),
        "effective_mode": result.get("effective_mode") or service.get("effective_mode"),
        "scheduler_health": result.get("scheduler_health") or service.get("scheduler_health"),
        "integration_mode": result.get("integration_mode"),
        "degraded_reasons": list(result.get("degraded_reasons") or []),
        "service": {
            key: service.get(key)
            for key in (
                "state",
                "liveness",
                "readiness",
                "backend",
                "workspace_id",
                "profile_id",
                "effective_mode",
            )
            if service.get(key) is not None
        },
        "projection": {
            "pending": projection.get("pending", 0),
            "failed": projection.get("failed", 0),
            "attention": projection.get("attention", 0),
        },
        "queue": {
            "depth": queue.get("queue_depth", service.get("queue_depth", 0)),
            "lanes": [
                {
                    "lane": lane.get("lane"),
                    "pending": lane.get("pending", 0),
                    "running": lane.get("running", 0),
                    "poisoned": lane.get("poisoned", 0),
                }
                for lane in (lanes or [])
                if isinstance(lane, dict)
            ],
        },
    }
    if result.get("error"):
        compact["error"] = result["error"]
    return compact


def _compact_remember(result: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = result["raw"] if isinstance(result.get("raw"), dict) else {}
    raw_route_value = raw.get("route_decision")
    raw_route: dict[str, Any] = raw_route_value if isinstance(raw_route_value, dict) else {}
    route_value = result.get("route_decision")
    route: dict[str, Any] = route_value if isinstance(route_value, dict) else {}
    compact = {
        "ok": result.get("ok"),
        "saved": result.get("saved"),
        "memory_id": result.get("memory_id") or raw.get("block_id") or "",
        "revision_id": result.get("revision_id") or raw.get("revision_id") or "",
        "block_id": result.get("actual_block_id") or route.get("block_id") or "",
        "commit_revision": result.get("commit_revision") or raw.get("commit_revision"),
        "projection_status": result.get("projection_status"),
        "durable_status": result.get("durable_status") or raw.get("durable_status"),
        "operation_id": result.get("operation_id") or raw.get("operation_id"),
        "status": route.get("status") or raw_route.get("status"),
        "replayed": bool(result.get("replayed")),
    }
    if result.get("similar_records"):
        compact["similar_records"] = [
            {
                "record_id": item.get("record_id"),
                "title": item.get("title") or "",
                "score": item.get("score"),
                "lexical": item.get("lexical"),
                "exact": bool(item.get("exact")),
                "title_match": bool(item.get("title_match")),
                "text": str(item.get("text") or "")[:200],
            }
            for item in result["similar_records"]
            if isinstance(item, dict)
        ]
        compact["duplicate_warning"] = True
    return compact


def _compact_block_mutation(result: dict[str, Any]) -> dict[str, Any]:
    block = result.get("block") if isinstance(result.get("block"), dict) else {}
    compact = _pick(
        result,
        (
            "ok",
            "block_id",
            "name",
            "parent_id",
            "revision",
            "status",
            "writable",
            "backend",
            "workspace_id",
            "auto_index_parent",
            "hint",
        ),
    )
    if "block_id" not in compact and block:
        compact["block_id"] = block.get("block_id")
    if result.get("error"):
        compact["error"] = result["error"]
    return compact


def _compact_mask(result: dict[str, Any]) -> dict[str, Any]:
    mask = result.get("mask") if isinstance(result.get("mask"), dict) else {}
    compact = _pick(result, ("ok", "mask_id", "revision", "status", "backend", "workspace_id"))
    if mask:
        compact.setdefault("mask_id", mask.get("mask_id") or mask.get("id") or "")
        compact.setdefault("revision", mask.get("revision"))
        compact.setdefault("status", mask.get("status"))
        compact["selected_block_ids"] = list(mask.get("selected_block_ids") or [])
        compact["token_budget"] = mask.get("token_budget")
        compact["activation_reason"] = mask.get("activation_reason") or ""
        compact["pinned_record_ids"] = list(mask.get("pinned_record_ids") or [])
    elif result.get("ok") is True and "mask" in result and result["mask"] is None:
        compact["status"] = "absent"
    if result.get("error"):
        compact["error"] = result["error"]
    if "replayed" in result:
        compact["replayed"] = result["replayed"]
    return compact


def compact_response(name: str, result: Any) -> Any:
    """Project one service payload for the MCP transport."""

    if not isinstance(result, dict) or response_mode() == "full":
        return result
    if name == "health":
        # Health is diagnostic even when ok is false; keep the mode and
        # listener state instead of collapsing it into a generic error.
        return _compact_health(result)
    if result.get("ok") is False:
        payload = _pick(result, _ERROR_FIELDS)
        for key in ("route_decision", "suggested_route"):
            if isinstance(result.get(key), dict):
                payload[key] = _compact_route(result[key])
        if "hits" in result:
            payload["hits"] = _compact_hits(result.get("hits"))
        if result.get("recent_fallback"):
            payload["recent_fallback"] = [
                {
                    "record_id": str(item.get("record_id") or ""),
                    "title": str(item.get("title") or ""),
                    "text": str(item.get("text") or "")[:200],
                }
                for item in result["recent_fallback"]
                if isinstance(item, dict)
            ]
        return payload

    if name == "block_map":
        return {
            "ok": result.get("ok"),
            "workspace_id": result.get("workspace_id"),
            "backend": result.get("backend"),
            "roots": _compact_blocks(result.get("roots")),
        }
    if name == "expand_block":
        return {
            "ok": result.get("ok"),
            "workspace_id": result.get("workspace_id"),
            "backend": result.get("backend"),
            "root": _compact_block(result.get("root")),
            "children": _compact_blocks(result.get("children")),
            **_pick(result, ("path_ids", "catalog_revision", "child_count", "next_cursor")),
        }
    if name == "create_block":
        return _compact_block_mutation(result)
    if name in {"archive_block", "revive_block", "amend_block", "move_block"}:
        block = result.get("block") if isinstance(result.get("block"), dict) else {}
        compact = _pick(
            result,
            (
                "ok",
                "saved",
                "block_id",
                "name",
                "status",
                "revision",
                "previous_revision",
                "parent_id",
                "backend",
                "workspace_id",
            ),
        )
        if "block_id" not in compact and block:
            compact["block_id"] = block.get("block_id")
        if result.get("error"):
            compact["error"] = result["error"]
        return compact
    if name == "delete_block":
        tombstone = result.get("tombstone") if isinstance(result.get("tombstone"), dict) else {}
        compact = {
            "ok": result.get("ok"),
            "deleted": result.get("deleted"),
            "archived": result.get("archived"),
            "records_remaining": result.get("records_remaining"),
            "tombstone": _pick(tombstone, ("block_id", "revision", "reason", "deleted_at")),
        }
        if result.get("auto_archived"):
            compact["auto_archived"] = True
        if result.get("cascade_failures"):
            compact["cascade_failures"] = result["cascade_failures"]
        if result.get("hint"):
            compact["hint"] = result["hint"]
        if result.get("error"):
            compact["error"] = result["error"]
        return compact
    if name == "remember":
        return _compact_remember(result)
    if name == "amend":
        compact = _pick(
            result,
            (
                "ok",
                "saved",
                "memory_id",
                "record_id",
                "new_id",
                "revision_id",
                "new_revision",
                "old_id",
                "old_revision",
                "requested_block_id",
                "commit_revision",
                "projection_status",
                "durable_status",
                "replayed",
            ),
        )
        if not compact.get("record_id"):
            compact["record_id"] = compact.get("new_id") or compact.get("memory_id") or ""
        return compact
    if name == "forget":
        compact = _pick(
            result,
            (
                "ok",
                "saved",
                "memory_id",
                "commit_revision",
                "projection_status",
                "durable_status",
                "replayed",
            ),
        )
        # ``forget`` returns the id of the retraction record it wrote; calling
        # that field ``block_id`` invited confusion with logical blocks.
        if result.get("block_id"):
            compact["revision_record_id"] = result["block_id"]
        return compact
    if name == "memory_map":
        return {
            "ok": result.get("ok"),
            "workspace_id": str(result.get("workspace_id") or ""),
            "path": str(result.get("path") or ""),
            "summary": str(result.get("summary") or ""),
        }
    if name == "procedure":
        procedures: list[dict[str, Any]] = []
        for item in result.get("procedures") or []:
            if not isinstance(item, dict):
                continue
            procedures.append(
                {
                    "record_id": str(item.get("record_id") or ""),
                    "title": str(item.get("title") or ""),
                    "purpose": str(item.get("purpose") or "")[:400],
                    "steps": [str(step) for step in item.get("steps") or []],
                    "verify": str(item.get("verify") or ""),
                    "applies_to": str(item.get("applies_to") or ""),
                    "verified_at": str(item.get("verified_at") or ""),
                    "age_days": int(item.get("age_days") or 0),
                    "freshness": str(item.get("freshness") or "unverified"),
                    "reason": str(item.get("reason") or ""),
                    "tags": [str(tag) for tag in item.get("tags") or []],
                }
            )
        return {
            "ok": result.get("ok"),
            "query": str(result.get("query") or ""),
            "count": int(result.get("count") or 0),
            "procedures": procedures,
        }
    if name in {"recall", "build_context"}:
        compact = {
            "ok": result.get("ok"),
            "items": _compact_hits(result.get("hits") or result.get("items")),
        }
        if result.get("rendered_context"):
            compact["rendered_context"] = result["rendered_context"]
        if result.get("next_cursor"):
            compact["next_cursor"] = result["next_cursor"]
        if result.get("relaxed"):
            compact["relaxed"] = True
            compact["relaxed_terms"] = list(result.get("relaxed_terms") or [])
        if result.get("relaxed_tried"):
            compact["relaxed_tried"] = True
        if result.get("fallback"):
            compact["fallback"] = result["fallback"]
        if result.get("semantic"):
            compact["semantic"] = True
            compact["semantic_model"] = str(result.get("semantic_model") or "")
        if result.get("suggested_blocks"):
            compact["suggested_blocks"] = [
                {
                    "block_id": str(item.get("block_id") or ""),
                    "score": float(item.get("score") or 0.0),
                    "hits": int(item.get("hits") or 0),
                }
                for item in result["suggested_blocks"]
                if isinstance(item, dict)
            ]
        if result.get("degraded"):
            compact["degraded"] = True
        if result.get("backend_error"):
            compact["backend_error"] = result["backend_error"]
        if result.get("supported_retrieval_modes"):
            compact["supported_retrieval_modes"] = list(result["supported_retrieval_modes"])
        if result.get("recent_fallback"):
            compact["recent_fallback"] = [
                {
                    "record_id": str(item.get("record_id") or ""),
                    "title": str(item.get("title") or ""),
                    "text": str(item.get("text") or "")[:200],
                }
                for item in result["recent_fallback"]
                if isinstance(item, dict)
            ]
        if result.get("hint"):
            compact["hint"] = result["hint"]
        return compact
    if name == "recent":
        items: list[dict[str, Any]] = []
        for record in result.get("records") or []:
            if not isinstance(record, dict):
                continue
            membership = record.get("block_membership")
            logical_block_id = ""
            if isinstance(membership, dict):
                logical_block_id = str(membership.get("primary_block_id") or "")
            items.append(
                {
                    "record_id": record.get("record_id") or record.get("block_id") or "",
                    "title": record.get("title") or "",
                    "text": record.get("content") or "",
                    "tags": list(record.get("tags") or []),
                    "logical_block_id": logical_block_id,
                    "kind": record.get("kind") or "",
                    "recorded_at": record.get("recorded_at") or "",
                    "status": record.get("status") or "",
                }
            )
        return {
            "ok": result.get("ok"),
            "items": items,
            "count": result.get("count"),
            "since": result.get("since") or "",
        }
    if name == "navigation_envelope":
        block = result.get("block") if isinstance(result.get("block"), dict) else {}
        related = result.get("related_blocks")
        related_ids = [
            str(item.get("block_id") or "")
            for item in related or []
            if isinstance(item, dict) and item.get("block_id")
        ]
        health: dict[str, Any] = (
            result["health_hints"] if isinstance(result.get("health_hints"), dict) else {}
        )
        return {
            "ok": result.get("ok", True),
            "workspace_id": result.get("workspace_id"),
            "profile_id": result.get("profile_id"),
            "catalog_revision": result.get("catalog_revision"),
            "tree_path": list(result.get("tree_path") or []),
            **_pick(result, ("child_count", "next_cursor")),
            "block": _compact_block(block),
            "children": _compact_blocks(result.get("children")),
            "related_block_ids": related_ids,
            "freshness": health.get("freshness") or "",
            "maintenance_observations": list(result.get("maintenance_observations") or []),
            "proposal_options": list(result.get("proposal_options") or []),
            "mutation_performed": bool(result.get("mutation_performed")),
        }
    if name == "read_exact_record":
        record = result.get("record") if isinstance(result.get("record"), dict) else {}
        return {
            "ok": result.get("ok"),
            "record": _compact_record(record),
        }
    if name == "read_exact_records":
        return {
            "ok": result.get("ok"),
            "records": _compact_records(result.get("records")),
        }
    if name == "list_records":
        return {
            "ok": result.get("ok"),
            "records": _compact_records(result.get("records")),
            "next_cursor": result.get("next_cursor", ""),
            "snapshot_count": result.get("snapshot_count"),
        }
    if name == "resolve_active_mask":
        return {
            "ok": result.get("ok"),
            "lease": _compact_lease(result.get("lease")),
        }
    if name in {
        "get_active_mask",
        "set_active_mask",
        "amend_active_mask",
        "transition_active_mask",
    }:
        return _compact_mask(result)
    return result
