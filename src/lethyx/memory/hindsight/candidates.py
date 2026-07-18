"""Hindsight reflection candidate parsing, fingerprints, and proposal queue."""

import json
import time
from pathlib import Path
from typing import Any

from lethyx.memory.backends.base import ReflectionCandidate
from lethyx.memory.hindsight.mapping import normalize_metadata, stable_hash


BKB_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": [
                    "kind",
                    "title",
                    "content",
                    "scope",
                    "confidence",
                    "evidence_refs",
                    "why_durable",
                    "risk_if_wrong",
                ],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "behavior",
                            "anti_pattern",
                            "risk",
                            "invariant",
                            "lesson",
                        ],
                    },
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": ["workspace", "manager", "agent"],
                    },
                    "confidence": {"type": "number"},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "why_durable": {"type": "string"},
                    "risk_if_wrong": {"type": "string"},
                },
            },
        }
    },
    "required": ["candidates"],
}


def build_reflection_question(*, max_candidates: int, existing_rules: str = "") -> str:
    existing = (
        "\nExisting accepted BKB rules to avoid duplicating:\n" + existing_rules.strip()
        if existing_rules.strip()
        else ""
    )
    return (
        "You are an archive analysis backend for Lethyx, not the source of truth.\n\n"
        f"From verified Lethyx memories in this bank, propose at most {max_candidates} "
        "BKB candidates that should be considered by Lethyx.\n\n"
        "Return only candidates that:\n"
        "- are supported by explicit evidence refs,\n"
        "- are durable across future runs,\n"
        "- would change agent behavior before retrieval,\n"
        "- are not merely task-local facts,\n"
        "- do not duplicate existing accepted BKB rules.\n\n"
        "Do not claim acceptance. These are proposals only." + existing
    )


def _coerce_response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    for attr in ("data", "parsed", "json", "content", "answer", "text"):
        value = getattr(response, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    if isinstance(response, str) and response.strip():
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return {"candidates": []}
        return parsed if isinstance(parsed, dict) else {"candidates": []}
    return {"candidates": []}


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def parse_reflection_candidates(
    response: Any,
    *,
    bank_id: str,
    max_candidates: int,
    budget: str,
) -> list[ReflectionCandidate]:
    payload = _coerce_response_payload(response)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[ReflectionCandidate] = []
    for raw in raw_candidates[: max(1, max_candidates)]:
        if not isinstance(raw, dict):
            continue
        evidence = raw.get("evidence_refs") if isinstance(raw.get("evidence_refs"), list) else []
        content = str(raw.get("content") or "").strip()
        title = str(raw.get("title") or "").strip()
        kind = str(raw.get("kind") or "lesson").strip()
        scope = str(raw.get("scope") or "workspace").strip()
        seed = {
            "kind": kind,
            "scope": scope,
            "title": title,
            "content": content,
            "evidence_refs": evidence,
        }
        candidates.append(
            ReflectionCandidate(
                candidate_id=stable_hash(seed)[:24].replace(":", "_"),
                kind=kind,
                title=title,
                content=content,
                confidence=_confidence(raw.get("confidence")),
                scope=scope,
                evidence_refs=[ref for ref in evidence if isinstance(ref, dict)],
                source_backend="hindsight",
                metadata={
                    "bank_id": bank_id,
                    "reflect_budget": budget,
                    "source_hash": stable_hash(seed),
                    "why_durable": str(raw.get("why_durable") or ""),
                    "risk_if_wrong": str(raw.get("risk_if_wrong") or ""),
                },
            )
        )
    return candidates


def candidate_fingerprint(candidate: ReflectionCandidate | dict[str, Any]) -> str:
    if isinstance(candidate, ReflectionCandidate):
        payload = {
            "kind": candidate.kind,
            "scope": candidate.scope,
            "title": candidate.title,
            "content": candidate.content,
            "evidence_refs": candidate.evidence_refs,
        }
    else:
        payload = {
            "kind": candidate.get("kind") or candidate.get("type"),
            "scope": candidate.get("scope"),
            "title": candidate.get("title"),
            "content": candidate.get("content") or candidate.get("rule"),
            "evidence_refs": candidate.get("evidence_refs")
            or candidate.get("source_evidence")
            or [],
        }
    return stable_hash(payload)


def bkb_fingerprint_path(repo_root: Path, workspace_id: str = "") -> Path:
    if workspace_id:
        return (
            repo_root / "workspaces" / workspace_id / ".memory" / "core" / "bkb_fingerprints.json"
        )
    return repo_root / ".Lethyx" / "memory" / "core" / "bkb_fingerprints.json"


def read_fingerprints(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(data, list):
        return {str(item) for item in data if str(item).strip()}
    if isinstance(data, dict):
        values = data.get("accepted") if isinstance(data.get("accepted"), list) else []
        return {str(item) for item in values if str(item).strip()}
    return set()


def add_accepted_fingerprints(
    *,
    repo_root: Path,
    workspace_id: str,
    fingerprints: list[str],
) -> None:
    path = bkb_fingerprint_path(repo_root, workspace_id)
    existing = read_fingerprints(path)
    existing.update(fp for fp in fingerprints if fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"accepted": sorted(existing)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def proposal_queue_dir(drive_root: Path) -> Path:
    return drive_root / "state" / "bkb_proposals"


def proposal_index_path(drive_root: Path) -> Path:
    return proposal_queue_dir(drive_root) / "index.jsonl"


def _index_fingerprints(drive_root: Path) -> set[str]:
    path = proposal_index_path(drive_root)
    if not path.is_file():
        return set()
    values: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("candidate_fingerprint"):
                values.add(str(row["candidate_fingerprint"]))
    except (OSError, json.JSONDecodeError):
        return values
    return values


def _typed_evidence(evidence_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = []
    for ref in evidence_refs:
        if not isinstance(ref, dict):
            continue
        if not ref.get("ref_type") or not ref.get("ref_id") or not ref.get("produced_by"):
            continue
        typed.append(ref)
    return typed


def candidate_to_bkb_rule(candidate: ReflectionCandidate) -> dict[str, Any]:
    rule_type = "anti_pattern" if candidate.kind == "anti_pattern" else candidate.kind
    return {
        "id": "bkb_hs_" + candidate.candidate_id[-12:].replace("_", ""),
        "title": candidate.title,
        "scope": candidate.scope,
        "type": rule_type,
        "status": "candidate",
        "trust": "candidate",
        "strength": max(0.1, min(0.95, candidate.confidence or 0.5)),
        "rule": {"behavior": candidate.content},
        "applies_to": {"workspaces": ["*"], "phases": ["*"], "agents": ["generic_agent"]},
        "confidence": candidate.confidence,
        "source_backend": "hindsight",
    }


def write_hindsight_candidates_as_pending_proposals(
    *,
    drive_root: Path,
    repo_root: Path,
    workspace_id: str,
    run_id: str,
    phase_id: str,
    candidates: list[ReflectionCandidate],
) -> dict[str, Any]:
    queue = proposal_queue_dir(drive_root)
    queue.mkdir(parents=True, exist_ok=True)
    accepted = read_fingerprints(bkb_fingerprint_path(repo_root, workspace_id))
    queued_or_seen = _index_fingerprints(drive_root)
    queued = 0
    skipped = 0
    needs_evidence = 0
    records: list[dict[str, Any]] = []

    for candidate in candidates:
        fingerprint = candidate_fingerprint(candidate)
        typed_evidence = _typed_evidence(candidate.evidence_refs)
        status = "candidate" if typed_evidence else "needs_evidence"
        if fingerprint in accepted or fingerprint in queued_or_seen:
            skipped += 1
            status = "duplicate"
        patch_id = "bkb_patch_hs_" + fingerprint.split(":", 1)[-1][:16]
        record = {
            "patch_id": patch_id,
            "status": status,
            "source": "hindsight",
            "proposed_by": "hindsight",
            "acceptor": "supervisor",
            "workspace_id": workspace_id,
            "run_id": run_id,
            "phase_id": phase_id or "reflexion",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rules": [candidate_to_bkb_rule(candidate)],
            "source_evidence": typed_evidence,
            "source_backend": "hindsight",
            "source_backend_metadata": normalize_metadata(candidate.metadata),
            "candidate_fingerprint": fingerprint,
        }
        records.append(record)
        if status == "candidate":
            (queue / f"{patch_id}.candidate.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            queued += 1
            queued_or_seen.add(fingerprint)
        elif status == "needs_evidence":
            needs_evidence += 1
            (queue / f"{patch_id}.needs_evidence.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    index_path = proposal_index_path(drive_root)
    summary_path = drive_root / "state" / "hindsight_candidate_queue.jsonl"
    with index_path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with summary_path.open("a", encoding="utf-8") as fh:
        for record in records:
            row = {
                "candidate_id": str(
                    (record.get("rules") or [{}])[0].get("id") or record.get("patch_id") or ""
                ),
                "source_backend": "hindsight",
                "lifecycle": "candidate",
                "tribunal_status": record.get("status"),
                "proof_score": 0.0,
                "accepted_bkb_rule_id": "",
                "rejected_reason": ""
                if record.get("status") != "needs_evidence"
                else "missing_canonical_typed_evidence",
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "generated": len(candidates),
        "queued": queued,
        "needs_evidence": needs_evidence,
        "duplicates_skipped": skipped,
    }
