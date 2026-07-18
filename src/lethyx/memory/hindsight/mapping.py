"""Mapping helpers between Lethyx durable memory and Hindsight payloads."""

import hashlib
import json
import re
from typing import Any

from lethyx.contracts import json_ready


METADATA_ALLOWLIST = {
    "LETHYX_id",
    "workspace_id",
    "run_id",
    "phase_id",
    "subtask_id",
    "agent",
    "kind",
    "trust_level",
    "source_hash",
    "evidence_refs_json",
    "bkb_rule_id",
    "palace_node_id",
    "accepted_by",
    "accepted_at",
    "patch_id",
    "target",
    "candidate_fingerprint",
    "source_kind",
    "evidence_id",
    "record_id",
}

SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?([^\s,'\"]+)"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
]


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, ensure_ascii=False, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def redact_sensitive(text: str) -> str:
    redacted = str(text or "")
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(
            lambda m: m.group(1) + "=[REDACTED]" if m.groups() else "[REDACTED]", redacted
        )
    return redacted


def normalize_tags(tags: list[str]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = str(tag or "").strip().replace(" ", "_")
        if not value or value in seen:
            continue
        seen.add(value)
        clean.append(value)
    return sorted(clean)


def derived_tags(
    *,
    kind: str,
    workspace_id: str = "",
    run_id: str = "",
    phase_id: str = "",
    subtask_id: str = "",
    agent: str = "",
    trust_level: str = "",
    scope: str = "",
    store: str = "",
) -> list[str]:
    tags = ["source:Lethyx", "tier:durable"]
    if workspace_id:
        tags.append(f"workspace:{workspace_id}")
    if run_id:
        tags.append(f"run:{run_id}")
    if phase_id:
        tags.append(f"phase:{phase_id}")
    if subtask_id:
        tags.append(f"subtask:{subtask_id}")
    if agent:
        tags.append(f"agent:{agent}")
    if kind:
        tags.append(f"kind:{kind}")
    if trust_level:
        tags.append(f"trust:{trust_level}")
    if scope:
        tags.append(f"scope:{scope}")
    if store:
        tags.append(f"store:{store}")
    return normalize_tags(tags)


def normalize_metadata(
    metadata: dict[str, Any] | None,
    evidence_refs: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    raw = dict(metadata or {})
    if evidence_refs is not None:
        raw["evidence_refs_json"] = json.dumps(
            json_ready(evidence_refs),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        key_s = str(key)
        if key_s not in METADATA_ALLOWLIST:
            continue
        if isinstance(value, (dict, list, tuple, set, frozenset)):
            normalized[key_s] = json.dumps(
                json_ready(value),
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        else:
            normalized[key_s] = str(value)
    return normalized


def evidence_to_lines(evidence_refs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for ref in evidence_refs:
        if not isinstance(ref, dict):
            continue
        ref_type = str(ref.get("ref_type") or "evidence")
        ref_id = str(ref.get("ref_id") or "")
        produced_by = str(ref.get("produced_by") or "")
        digest = str(ref.get("hash") or "")
        tail = f" by {produced_by}" if produced_by else ""
        hash_tail = f" hash={digest}" if digest else ""
        lines.append(f"- {ref_type}:{ref_id}{tail}{hash_tail}")
    return lines or ["- no typed evidence refs provided"]
