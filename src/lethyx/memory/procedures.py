"""Stored procedures: freshness that is caused, not guessed.

A procedure is a record of kind ``procedure``. It carries steps, a way to
verify the result, and the date it was last confirmed. A date alone is a weak
signal: the environment may have changed the day after the confirmation.

The fingerprint closes that gap. Every procedure is stamped with a descriptor
of the environment it was written against - the Lethyx version, the semantic
model, the store it lives in. When the descriptor changes, the procedure is
reported as stale *because* the environment moved, not because a number of
days passed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

#: How old a confirmed procedure may be before it is called aging.
DEFAULT_STALE_AFTER_DAYS = 90


def current_descriptor(
    repo_root: Path | str = "", model: str = "", workspace_id: str = ""
) -> dict[str, str]:
    """Return the environment descriptor a procedure is stamped against."""

    from lethyx import __version__ as lethyx_version

    return {
        "lethyx_version": str(lethyx_version),
        "semantic_model": str(model or ""),
        "store": str(Path(repo_root).resolve()) if repo_root else "",
        "workspace": str(workspace_id or ""),
    }


def descriptor_hash(descriptor: Mapping[str, Any]) -> str:
    """Return the stable hash of one environment descriptor."""

    payload = json.dumps(dict(descriptor), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stamp_metadata(
    metadata: Mapping[str, Any] | None,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the metadata of a procedure with its environment stamp."""

    stamped = dict(metadata or {})
    stamped.setdefault("fingerprint", dict(descriptor))
    return stamped


def procedure_freshness(
    metadata: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the freshness of one procedure.

    The label is ``stale`` when the environment moved, ``unverified`` when
    nobody confirmed it, ``aging`` when it is close to its limit and ``fresh``
    when it was confirmed and nothing moved.
    """

    verified_at = str(metadata.get("verified_at") or "").strip()
    try:
        stale_after = int(metadata.get("stale_after_days") or DEFAULT_STALE_AFTER_DAYS)
    except (TypeError, ValueError):
        stale_after = DEFAULT_STALE_AFTER_DAYS
    age_days = 0
    parsed = None
    if verified_at:
        try:
            parsed = datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = max(0, (datetime.now(timezone.utc) - parsed).days)
    fingerprint = metadata.get("fingerprint")
    reason = ""
    if descriptor is not None and isinstance(fingerprint, Mapping):
        changed = [
            f"{key}: {fingerprint.get(key) or '-'} -> {descriptor.get(key) or '-'}"
            for key in sorted(descriptor)
            if str(fingerprint.get(key) or "") != str(descriptor.get(key) or "")
        ]
        if changed:
            return {
                "freshness": "stale",
                "verified_at": verified_at,
                "age_days": age_days,
                "reason": "the environment changed (" + "; ".join(changed) + ")",
            }
    if not verified_at or parsed is None:
        return {
            "freshness": "unverified",
            "verified_at": verified_at,
            "age_days": age_days,
            "reason": "nobody confirmed this procedure yet",
        }
    if age_days > stale_after:
        return {
            "freshness": "stale",
            "verified_at": verified_at,
            "age_days": age_days,
            "reason": f"not confirmed for {age_days} days",
        }
    if age_days > max(1, stale_after // 3):
        return {
            "freshness": "aging",
            "verified_at": verified_at,
            "age_days": age_days,
            "reason": f"last confirmed {age_days} days ago",
        }
    return {
        "freshness": "fresh",
        "verified_at": verified_at,
        "age_days": age_days,
        "reason": f"confirmed {age_days} days ago",
    }


def procedure_entry(
    record: Mapping[str, Any], descriptor: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return one stored procedure as a checklist with its freshness."""

    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    )
    steps = metadata.get("steps") or []
    if isinstance(steps, str):
        steps = [steps]
    if not isinstance(steps, (list, tuple)):
        steps = []
    freshness = procedure_freshness(metadata, descriptor)
    return {
        "record_id": str(record.get("record_id") or ""),
        "title": str(record.get("title") or ""),
        "purpose": str(record.get("content") or ""),
        "steps": [str(step) for step in steps],
        "verify": str(metadata.get("verify") or ""),
        "applies_to": str(metadata.get("applies_to") or ""),
        "verified_at": freshness["verified_at"],
        "age_days": freshness["age_days"],
        "freshness": freshness["freshness"],
        "reason": freshness["reason"],
        "tags": list(record.get("tags") or ()),
        "recorded_at": str(record.get("recorded_at") or ""),
    }


def procedure_entries(
    records: Sequence[Mapping[str, Any]], descriptor: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return every record as a procedure checklist."""

    return [procedure_entry(record, descriptor) for record in records]
