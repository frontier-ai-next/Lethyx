"""Compact projections for the Lethyx MCP surface.

The local service returns rich, audit-oriented envelopes. That is right for
operators and wrong for a model context window: a single 40-word memory hit
carries several hundred tokens of provenance. This module keeps the fields an
agent needs to act (identity, revision, score, text) and drops the rest.

Set ``LETHYX_MCP_RESPONSE_MODE=full`` to get the unmodified service payloads.
"""

from __future__ import annotations

import os
from typing import Any

_BLOCK_FIELDS = (
    "block_id",
    "name",
    "synopsis",
    "tags",
    "parent_id",
    "child_ids",
    "record_count",
    "canonical_record_count",
    "writable",
    "status",
    "revision",
    "latest_title",
    "expandable",
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
    "reason_code",
    "retryable",
    "workspace_id",
    "backend",
    "hits",
)


def response_mode() -> str:
    return os.getenv("LETHYX_MCP_RESPONSE_MODE", "compact").strip().lower() or "compact"


def _pick(source: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {key: source[key] for key in fields if key in source}


def _compact_block(block: Any) -> dict[str, Any]:
    return _pick(block, _BLOCK_FIELDS)


def _compact_blocks(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_compact_block(value) for value in values]


def _compact_records(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_pick(value, _RECORD_FIELDS) for value in values]


def _hit_record_id(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata")
    if isinstance(metadata, dict):
        for key in ("record_id", "block_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    source = str(hit.get("source") or "").strip()
    if source.startswith("lethyx:"):
        return source.split(":", 1)[1]
    return source


def _compact_hit(hit: Any) -> dict[str, Any]:
    if not isinstance(hit, dict):
        return {"text": str(hit)}
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    compact = {
        "record_id": _hit_record_id(hit),
        "title": hit.get("title") or "",
        "text": hit.get("text") or "",
        "score": hit.get("score"),
        "tags": list(hit.get("tags") or []),
        "recorded_at": metadata.get("recorded_at") or hit.get("recorded_at") or "",
        "revision_id": metadata.get("revision_id") or hit.get("revision_id") or "",
        "authority": hit.get("authority") or "",
        "status": metadata.get("status") or hit.get("status") or "",
    }
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
    return compact


def _compact_health(result: dict[str, Any]) -> dict[str, Any]:
    service = result.get("service") if isinstance(result.get("service"), dict) else {}
    projection = result.get("projection") or service.get("projection") or {}
    queue = result.get("background_queue") or service.get("background_queue") or {}
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
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    raw_route = raw.get("route_decision") if isinstance(raw.get("route_decision"), dict) else {}
    route = result.get("route_decision") if isinstance(result.get("route_decision"), dict) else {}
    return {
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


def _compact_block_mutation(result: dict[str, Any]) -> dict[str, Any]:
    block = result.get("block") if isinstance(result.get("block"), dict) else {}
    compact = _pick(result, ("ok", "block_id", "name", "parent_id", "revision", "status", "writable", "backend", "workspace_id"))
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
    if result.get("error"):
        compact["error"] = result["error"]
    return compact


def compact_response(name: str, result: Any) -> Any:
    """Project one service payload for the MCP transport."""

    if not isinstance(result, dict) or response_mode() == "full":
        return result
    if result.get("ok") is False:
        payload = _pick(result, _ERROR_FIELDS)
        if "hits" in result:
            payload["hits"] = _compact_hits(result.get("hits"))
        return payload

    if name == "health":
        return _compact_health(result)
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
        return {
            "ok": result.get("ok"),
            "deleted": result.get("deleted"),
            "tombstone": _pick(tombstone, ("block_id", "revision", "reason", "deleted_at")),
        }
    if name == "remember":
        return _compact_remember(result)
    if name == "amend":
        return _pick(
            result,
            (
                "ok",
                "memory_id",
                "revision_id",
                "new_revision",
                "old_id",
                "commit_revision",
                "projection_status",
                "durable_status",
                "replayed",
            ),
        )
    if name == "forget":
        return _pick(
            result,
            (
                "ok",
                "saved",
                "memory_id",
                "block_id",
                "commit_revision",
                "projection_status",
                "durable_status",
                "replayed",
            ),
        )
    if name in {"recall", "build_context"}:
        compact = {
            "ok": result.get("ok"),
            "items": _compact_hits(result.get("hits") or result.get("items")),
        }
        if result.get("rendered_context"):
            compact["rendered_context"] = result["rendered_context"]
        if result.get("next_cursor"):
            compact["next_cursor"] = result["next_cursor"]
        return compact
    if name == "read_exact_record":
        record = result.get("record") if isinstance(result.get("record"), dict) else {}
        return {
            "ok": result.get("ok"),
            "record": _pick(record, _RECORD_FIELDS),
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
    if name in {"get_active_mask", "set_active_mask", "amend_active_mask", "transition_active_mask"}:
        return _compact_mask(result)
    return result
