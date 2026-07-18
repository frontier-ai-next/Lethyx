"""Evidence-bound memory write policy (shared by tools and kernel writer)."""

from collections.abc import Iterable
from typing import Any

_DURABLE_MEMORY_MARKERS = {
    "architecture_decision",
    "completion_memory",
    "durable",
    "durable_finding",
    "verified_finding",
    "manager_lesson",
    "competency_gap",
    "self_improvement_trigger",
    "verification_report",
    "bkb_rule",
    "lesson",
}

_EVIDENCE_REF_KEYS = {
    "evidence_ref",
    "evidence_refs",
    "source_id",
    "source_ids",
    "tool_call_id",
    "tool_result_id",
    "artifact_id",
    "artifact_path",
    "command_evidence_id",
    "verify_run_id",
    "verification_report_id",
    "ledger_event_id",
}

_MEMORY_TRUST_LEVELS = {
    "agent_claim",
    "observed_artifact",
    "public_verified",
    "mutation_verified",
    "hidden_verified",
    "adversarial_verified",
    "contradicted",
    "retracted",
}

_VERIFIED_MEMORY_TRUST_LEVELS = {
    "public_verified",
    "mutation_verified",
    "hidden_verified",
    "adversarial_verified",
}

_SUPERVISOR_EVIDENCE_PRODUCERS = {"supervisor", "verifier", "watcher", "harness"}


def memory_write_policy_issues(
    *,
    kind: str = "",
    tags: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Return evidence-bound memory policy issues for durable writes."""
    metadata = dict(metadata or {})
    kind_norm = str(kind or "").strip().lower()
    tag_set = {str(tag or "").strip().lower() for tag in tags if str(tag or "").strip()}
    scope = str(metadata.get("scope") or "").strip().lower()
    durable = bool(
        {kind_norm, *tag_set} & _DURABLE_MEMORY_MARKERS
        or scope in {"manager", "competency", "cross_run_durable"}
        or "durable" in tag_set
        or "verification_report" in tag_set
    )
    if not durable:
        return []
    trust_level = str(metadata.get("trust_level") or "").strip().lower()
    if trust_level not in _MEMORY_TRUST_LEVELS:
        return [
            (
                "durable memory writes require trust_level "
                "(public_verified/mutation_verified/hidden_verified/"
                "adversarial_verified, contradicted, or retracted)"
            )
        ]
    if trust_level not in _VERIFIED_MEMORY_TRUST_LEVELS:
        return [
            (
                "durable memory writes require verified-or-higher trust; "
                f"`{trust_level}` cannot be promoted as durable lesson evidence"
            )
        ]
    has_typed_ref = False
    has_untyped_ref = False
    for key in _EVIDENCE_REF_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            has_untyped_ref = True
            continue
        if isinstance(value, dict) and value.get("ref_id"):
            producer = str(value.get("produced_by") or "").strip().lower()
            if producer in _SUPERVISOR_EVIDENCE_PRODUCERS:
                has_typed_ref = True
            else:
                has_untyped_ref = True
            continue
        if isinstance(value, (list, tuple, set, frozenset)) and any(
            str(item).strip() for item in value
        ):
            for item in value:
                if isinstance(item, dict) and item.get("ref_id"):
                    producer = str(item.get("produced_by") or "").strip().lower()
                    if producer in _SUPERVISOR_EVIDENCE_PRODUCERS:
                        has_typed_ref = True
                    else:
                        has_untyped_ref = True
                elif str(item).strip():
                    has_untyped_ref = True
    if has_typed_ref:
        return []
    if has_untyped_ref:
        return [
            (
                "durable memory writes require typed EvidenceRef values "
                "produced by supervisor/verifier/watcher/harness; string refs "
                "are not sufficient"
            )
        ]
    return [
        (
            "durable memory writes require typed EvidenceRef values with "
            "ledger-backed supervisor/verifier/watcher/harness evidence"
        )
    ]
