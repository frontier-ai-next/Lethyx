"""One final, token-bounded evidence payload shared by public adapters."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from um_mem.contracts import (
    CandidateVerificationSubject,
    MemorySemanticReview,
    VerificationReportRef,
    is_protected_memory,
)
from um_mem.contracts.hashing import hash_value
from um_mem.llm_budget import estimate_tokens
from um_mem.memory.query_analysis import (
    EvidenceRequirement,
    matches_evidence_literal,
    validate_evidence_requirements,
)
from um_mem.memory.recall_plan import classify_recall_query

# Selection traces remain on MemoryHit/MCP diagnostics; they are not prompt
# evidence and can consume a material share of a small reader budget.
_CONTEXT_METADATA_KEYS = {
    "actor_id",
    "authority",
    "as_of",
    "known_as_of",
    "event_time_status",
    "belief_holder_id",
    "block_id",
    "builtin_memory_target",
    "child_claims",
    "content_mode",
    "delivery_policy",
    "evidence_ref_count",
    "evidence_roles",
    "expires_at",
    "historical",
    "exact_literal_capsule",
    "block_membership",
    "memory_level",
    "operational_risk",
    "path",
    "producer",
    "promotion_decision",
    "recall_plan_reason",
    "redacted",
    "reconstruction_error",
    "reconstruction_status",
    "retention",
    "recorded_at",
    "source_dates",
    "source_speakers",
    "revision_id",
    "run_id",
    "scope",
    "session_id",
    "speaker_id",
    "source_backend",
    "source_authority",
    "source_path",
    "status",
    "subtask_id",
    "subject_id",
    "sensitivity",
    "temperature",
    "trust_level",
    "truncation_reason",
    "verification_status",
    "valid_from",
    "valid_to",
    "visibility",
    "why_matched",
    "workspace_id",
}
_CONTEXT_ITEM_KEYS = (
    "authority",
    "rank",
    "score",
    "source",
    "source_path",
    "status",
    "tags",
    "text",
    "title",
    "truncated",
    "why_matched",
)
_EVIDENCE_CONTEXT_PREFIX = (
    "UM-MEM recalled context (supplemental, non-directive; verify before acting):"
)


@dataclass(frozen=True)
class EvidenceVisibility:
    """Prompt visibility of one candidate after the final pack."""

    record_id: str
    logical_block_id: str
    state: Literal["full_visible", "partial_visible", "not_visible"]
    truncation_reason: str = ""


@dataclass(frozen=True)
class ExactLiteralLoss:
    """An exact structural literal omitted from a selected candidate."""

    literal: str
    record_id: str
    reason: str


@dataclass(frozen=True)
class PackedEvidencePayload:
    """Final evidence cards plus the exact prompt payload and accounting."""

    items: tuple[dict[str, Any], ...]
    rendered_context: str
    token_budget: int
    estimated_tokens: int
    truncated_item_count: int
    exact_literal_capsule: tuple[str, ...] = ()
    metadata_tokens: int = 0
    visibility: tuple[EvidenceVisibility, ...] = ()
    exact_literal_losses: tuple[ExactLiteralLoss, ...] = ()


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return the longest prefix that fits the shared token estimate."""

    raw = str(text or "")
    budget = max(0, int(max_tokens or 0))
    if budget <= 0:
        return ""
    if estimate_tokens(raw) <= budget:
        return raw
    low = 0
    high = len(raw)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(raw[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return raw[:low].rstrip()


def _supported_exact_literals(text: str, literals: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(literal for literal in literals if matches_evidence_literal(text, literal))


def _window_around_exact_range(text: str, *, start: int, end: int, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    exact = text[start:end]
    if estimate_tokens(exact) > max_tokens:
        return ""
    target_chars = max(len(exact), max_tokens * 4 - 4)
    surrounding = max(0, target_chars - len(exact))
    before = min(start, surrounding // 2)
    after = min(len(text) - end, surrounding - before)
    before = min(start, surrounding - after)
    window_start = start - before
    window_end = end + after

    def render() -> str:
        excerpt = text[window_start:window_end].strip()
        if window_start:
            excerpt = f"...{excerpt}"
        if window_end < len(text):
            excerpt = f"{excerpt}..."
        return excerpt

    excerpt = render()
    while excerpt and estimate_tokens(excerpt) > max_tokens:
        if window_start < start:
            window_start += 1
        elif window_end > end:
            window_end -= 1
        else:
            return ""
        excerpt = render()
    return excerpt


def _exact_literal_excerpt(
    text: str,
    *,
    max_tokens: int,
    literals: tuple[str, ...],
) -> str:
    """Retain raw spans for exact structural literals without semantic scoring."""

    if max_tokens <= 0:
        return ""
    folded = text.casefold()
    ranges = sorted(
        (start, start + len(literal))
        for literal in literals
        if (start := folded.find(literal.casefold())) >= 0
    )
    if not ranges:
        return ""
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 64:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    separator = "\n...\n"
    minimum_tokens = [estimate_tokens(text[start:end]) for start, end in merged]
    required = sum(minimum_tokens) + max(0, len(merged) - 1) * estimate_tokens(separator)
    if required > max_tokens:
        return ""
    remaining = max_tokens - required
    allowances = list(minimum_tokens)
    for index in range(len(allowances)):
        ranges_left = len(allowances) - index
        extra = remaining // ranges_left
        allowances[index] += extra
        remaining -= extra
    excerpts = [
        _window_around_exact_range(
            text,
            start=start,
            end=end,
            max_tokens=allowances[index],
        )
        for index, (start, end) in enumerate(merged)
    ]
    rendered = separator.join(excerpt for excerpt in excerpts if excerpt)
    return rendered if estimate_tokens(rendered) <= max_tokens else ""


def excerpt_to_tokens(
    text: str,
    max_tokens: int,
    *,
    query: str = "",
    exact_literals: tuple[str, ...] = (),
) -> str:
    """Return a prefix, except when exact structural literals need a capsule.

    ``query`` remains accepted for API compatibility but never contributes
    lexical terms, aliases, relevance scores, or line selection.
    """

    del query
    raw = str(text or "")
    budget = max(0, int(max_tokens or 0))
    if budget <= 0 or estimate_tokens(raw) <= budget:
        return truncate_to_tokens(raw, budget)
    supported = _supported_exact_literals(raw, exact_literals)
    if supported:
        capsule = _exact_literal_excerpt(raw, max_tokens=budget, literals=supported)
        if capsule and all(matches_evidence_literal(capsule, literal) for literal in supported):
            return capsule
    return truncate_to_tokens(raw, budget)


def _bound_candidate_verification(source: Mapping[str, Any]) -> bool:
    receipt = source.get("candidate_semantic_verification")
    if not isinstance(receipt, Mapping) or set(receipt) != {"result", "report"}:
        return False
    result = receipt["result"]
    if not isinstance(result, dict) or not isinstance(receipt["report"], dict):
        return False
    try:
        subject = CandidateVerificationSubject(**result["subject"])
        report = VerificationReportRef.from_mapping(receipt["report"])
    except (KeyError, TypeError, ValueError):
        return False
    membership = source.get("block_membership")
    refs = source.get("evidence_refs")
    scope_field = {
        "workspace": "workspace_id",
        "project": "project_id",
        "user_global": "user_id",
        "session": "session_id",
        "task": "task_id",
        "subtask": "subtask_id",
    }.get(subject.scope)
    identities = [
        result.get(key)
        for key in (
            "verifier_id",
            "verifier_model_version",
            "proposer_model_version",
            "verifier_prompt_version",
        )
    ]
    reasons = result.get("reasons")
    return bool(
        report.passed
        and report.ledger_hash
        and report.report_hash == hash_value(result)
        and result.get("semantic_subject_hash") == hash_value(result["subject"])
        and all(isinstance(value, str) and value.strip() for value in identities)
        and result["verifier_id"] == report.verifier_id
        and result["verifier_model_version"] != result["proposer_model_version"]
        and all(
            result.get(key) is True
            for key in ("coverage", "preservation", "faithfulness", "instruction_safety")
        )
        and isinstance(reasons, (list, tuple))
        and reasons
        and all(isinstance(value, str) and value.strip() for value in reasons)
        and subject.memory_id == source.get("promotion_source_id")
        and subject.revision_id == source.get("promotion_source_revision_id")
        and subject.workspace_id == source.get("workspace_id")
        and subject.profile_id == source.get("profile_id")
        and subject.authority == source.get("authority")
        and subject.scope == source.get("scope")
        and scope_field
        and subject.scope_id == source.get(scope_field)
        and isinstance(membership, Mapping)
        and subject.target_block_id == membership.get("primary_block_id")
        and isinstance(refs, (tuple, list))
        and any(
            isinstance(ref, Mapping)
            and ref.get("ref_type") == "verification_report"
            and ref.get("ref_id") == report.report_id
            and ref.get("producer_id") == report.verifier_id
            and ref.get("hash") == report.ledger_hash
            for ref in refs
        )
    )


def project_memory_assurance(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project reviewed assurance without changing canonical historical labels."""

    source = dict(metadata or {})
    if source.get("verification_status") == "verified":
        raw_review = source.get("semantic_review")
        try:
            review = (
                MemorySemanticReview.from_mapping(dict(raw_review))
                if isinstance(raw_review, Mapping)
                else None
            )
        except ValueError:
            review = None
        refs = source.get("evidence_refs")
        bound_review = (
            review is not None
            and source.get("verification_scope") == "semantic"
            and source.get("promotion_source_id") == review.record_id
            and isinstance(refs, (tuple, list))
            and any(
                isinstance(ref, Mapping)
                and ref.get("ref_type") == "verification_report"
                and ref.get("producer_id") == review.review_authority_id
                and ref.get("hash")
                for ref in refs
            )
        )
        if "candidate_semantic_verification" in source:
            bound_review = _bound_candidate_verification(source)
        if not bound_review:
            # This is a read projection; immutable recorded lifecycle labels stay intact.
            source["verification_status"] = "unverified"
            source["trust_level"] = "agent_reported"
            if source.get("authority") == "directive":
                source["source_authority"] = "directive"
                source["authority"] = "advisory"
    return source


def compact_context_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep bounded provenance needed to interpret a context item safely."""

    source = project_memory_assurance(metadata)
    compact: dict[str, Any] = {}
    for key in _CONTEXT_METADATA_KEYS:
        value = source.get(key)
        if value in (None, "", [], (), {}):
            continue
        if key == "child_claims":
            if isinstance(value, (list, tuple)):
                compact[key] = [
                    dict(claim)
                    for claim in list(value)[:8]
                    if isinstance(claim, Mapping)
                    and isinstance(claim.get("origin_segments"), list)
                    and len(claim["origin_segments"]) <= 8
                ]
            continue
        if isinstance(value, str):
            value = value[:512]
        elif isinstance(value, (list, tuple, set, frozenset)):
            value = [str(item)[:160] for item in list(value)[:16] if str(item)]
        compact[key] = value
    for key, default in {
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "promotion_decision": "accepted",
        "sensitivity": "internal",
        "content_mode": "full",
        "operational_risk": "low",
    }.items():
        if compact.get(key) == default:
            compact.pop(key, None)
    return compact


def _context_item_is_renderable(item: Mapping[str, Any]) -> bool:
    raw_metadata = item.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    return not is_protected_memory(
        metadata.get("sensitivity", item.get("sensitivity")),
        metadata.get("content_mode", item.get("content_mode")),
        metadata.get("operational_risk", item.get("operational_risk")),
    )


def compact_context_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Remove redundant/debug fields while preserving card provenance."""

    compact = {
        key: item[key]
        for key in _CONTEXT_ITEM_KEYS
        if key in item and item[key] not in (None, "", [], (), {})
    }
    compact["metadata"] = compact_context_metadata(
        item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    )
    if (
        compact.get("authority") == "directive"
        and compact["metadata"].get("authority") == "advisory"
    ):
        compact["authority"] = "advisory"
    for key, limit in {
        "rank": 80,
        "source": 240,
        "source_path": 320,
        "status": 100,
        "title": 180,
        "why_matched": 180,
    }.items():
        if key in compact:
            compact[key] = str(compact[key])[:limit]
    if "tags" in compact:
        compact["tags"] = [str(tag)[:80] for tag in list(compact["tags"])[:16]]
    return compact


def _is_negative_evidence(item: Mapping[str, Any], *, query_type: str) -> bool:
    raw_metadata = item.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    roles = metadata.get("evidence_roles")
    return query_type == "negative_evidence" or (
        isinstance(roles, (list, tuple, set, frozenset)) and "negative_evidence" in roles
    )


def _evidence_subject_attribution(item: Mapping[str, Any]) -> tuple[str, str]:
    raw_metadata = item.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    for field_name in ("subject_id", "belief_holder_id", "speaker_id", "actor_id"):
        value = " ".join(str(metadata.get(field_name) or "").split())
        if value:
            return value[:160], field_name
    return "unattributed", "missing"


def _evidence_card_header(
    item: Mapping[str, Any],
    *,
    index: int,
    truncated: bool,
    compact: bool,
    query_type: str,
) -> str:
    raw_metadata = item.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    raw_membership = metadata.get("block_membership")
    membership = raw_membership if isinstance(raw_membership, Mapping) else {}
    primary_block_id = str(membership.get("primary_block_id") or "unknown")
    record_id = str(metadata.get("block_id") or item.get("source") or "unknown")
    title = str(item.get("title") or metadata.get("title") or "memory")
    canonical_status = str(metadata.get("status") or item.get("status") or "unknown")
    lifecycle = (
        "superseded"
        if canonical_status in {"superseded", "deprecated", "retracted", "expired"}
        else "current"
    )
    temporal_parts = []
    for key in ("source_dates", "source_speakers"):
        if metadata.get(key):
            temporal_parts.append(f"{key}={metadata[key]}")
    if metadata.get("historical") is True:
        lifecycle = "historical"
        temporal_parts.append("temporal=historical")
    for key in (
        "as_of",
        "known_as_of",
        "event_time_status",
        "valid_from",
        "valid_to",
        "expires_at",
    ):
        if compact and lifecycle == "current" and key == "valid_from":
            # Current selection already established applicability. The full
            # date remains in item metadata; reserve compact space for the fact.
            continue
        value = metadata.get(key)
        if isinstance(value, str) and value:
            temporal_parts.append(f"{key}={value[:80]}")
    temporal_suffix = f" {' '.join(temporal_parts)}" if temporal_parts else ""
    revision_id = str(metadata.get("revision_id") or "unknown")
    source = str(
        metadata.get("producer")
        or metadata.get("session_id")
        or metadata.get("source_backend")
        or item.get("source")
        or "memory"
    )
    source_time = str(metadata.get("recorded_at") or "unknown")
    trust = str(metadata.get("trust_level") or "agent_reported")
    verification = str(metadata.get("verification_status") or "unverified")
    path = str(
        metadata.get("logical_block_path")
        or (primary_block_id if primary_block_id != "unknown" else "")
        or item.get("source_path")
        or metadata.get("source_path")
        or "-"
    )
    tags = ",".join(str(tag) for tag in list(item.get("tags") or ())[:8]) or "-"
    subject_suffix = ""
    if _is_negative_evidence(item, query_type=query_type):
        subject, attribution = _evidence_subject_attribution(item)
        subject_suffix = f" subject={subject} attribution={attribution}"
    if compact:
        status_code = {
            "accepted": "a",
            "candidate": "c",
            "superseded": "s",
            "deprecated": "d",
            "retracted": "r",
            "expired": "e",
        }.get(canonical_status, canonical_status)
        trust_code = {
            "agent_reported": "ar",
            "workspace_verified": "wv",
        }.get(trust, trust)
        verification_code = {
            "unverified": "u",
            "observed": "o",
            "evidence_backed": "eb",
            "verified": "v",
            "disputed": "d",
        }.get(verification, verification)
        return (
            f"E{index}: record={record_id} block={primary_block_id} "
            f"title={title} status={lifecycle}/{status_code} "
            f"trust={trust_code}/{verification_code} "
            f"truncated={'true' if truncated else 'false'}{subject_suffix}{temporal_suffix}"
        )
    return (
        f"Evidence card {index}: block={primary_block_id} path={path} record={record_id} "
        f"title={title} status={lifecycle}/{canonical_status} revision={revision_id} "
        f"source={source}@{source_time} trust={trust}/{verification} tags={tags} "
        f"truncated={'true' if truncated else 'false'}{subject_suffix}{temporal_suffix}"
    )


def _normalized_evidence_card_item(
    item: Mapping[str, Any],
    *,
    source_fallback: str = "",
) -> dict[str, Any]:
    raw_metadata = item.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    normalized = dict(item)
    normalized["title"] = " ".join(
        str(item.get("title") or metadata.get("title") or "memory").split()
    )[:120]
    normalized["source"] = " ".join(
        str(
            item.get("source")
            or metadata.get("block_id")
            or metadata.get("source_backend")
            or source_fallback
            or "memory"
        ).split()
    )[:160]
    normalized["source_path"] = " ".join(
        str(
            item.get("source_path") or metadata.get("source_path") or metadata.get("path") or ""
        ).split()
    )[:220]
    status = " ".join(str(item.get("status") or "supplemental non-directive").split())
    authority = str(item.get("authority") or metadata.get("authority") or "non_directive")
    if authority == "non_directive" and "non-directive" not in status:
        status = f"{status} non-directive".strip()
    normalized["status"] = status[:80]
    normalized["why_matched"] = " ".join(
        str(
            item.get("why_matched")
            or metadata.get("why_matched")
            or metadata.get("recall_plan_reason")
            or "memory_recall"
        ).split()
    )[:160]
    return normalized


def _record_identity(item: Mapping[str, Any]) -> tuple[str, str]:
    raw_metadata = item.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    record_id = str(metadata.get("block_id") or item.get("source") or item.get("id") or "")
    raw_membership = metadata.get("block_membership")
    membership = raw_membership if isinstance(raw_membership, Mapping) else {}
    return record_id, str(membership.get("primary_block_id") or "")


def _minimum_exact_excerpt(
    text: str,
    *,
    literals: tuple[str, ...],
) -> str:
    if not literals:
        return excerpt_to_tokens(text, 6)
    minimum = max(6, sum(estimate_tokens(literal) for literal in literals) + 4)
    maximum = max(minimum, estimate_tokens(text))
    budget = minimum
    while budget <= maximum:
        excerpt = excerpt_to_tokens(text, budget, exact_literals=literals)
        if all(matches_evidence_literal(excerpt, literal) for literal in literals):
            return excerpt
        if budget == maximum:
            break
        budget = min(maximum, max(budget + 8, budget * 2))
    return text


def _minimum_evidence_context_chars(
    cards: list[dict[str, Any]],
    *,
    compact_headers: bool,
    query_type: str,
) -> int:
    lines = [_EVIDENCE_CONTEXT_PREFIX]
    for index, card in enumerate(cards, start=1):
        lines.append(
            _evidence_card_header(
                card["item"],
                index=index,
                truncated=False,
                compact=compact_headers,
                query_type=query_type,
            )
        )
        lines.append(f"fact={card['minimum_fact']}")
    return len("\n".join(lines))


def _metadata_token_count(
    items: Iterable[Mapping[str, Any]],
    *,
    compact_headers: bool,
    query_type: str,
) -> int:
    lines = [_EVIDENCE_CONTEXT_PREFIX]
    for index, item in enumerate(items, start=1):
        lines.append(
            _evidence_card_header(
                item,
                index=index,
                truncated=bool(item.get("truncated")),
                compact=compact_headers,
                query_type=query_type,
            )
        )
        lines.append("fact=")
    return estimate_tokens("\n".join(lines))


def _select_prefix(
    cards: list[dict[str, Any]],
    *,
    char_budget: int,
    compact_headers: bool,
    query_type: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for card in cards:
        candidate = [*selected, card]
        if (
            _minimum_evidence_context_chars(
                candidate,
                compact_headers=compact_headers,
                query_type=query_type,
            )
            > char_budget
        ):
            break
        selected = candidate
    return selected


def _fact_within_chars(card: Mapping[str, Any], allowance: int) -> str:
    original = str(card["original"])
    minimum_fact = str(card["minimum_fact"])
    if allowance >= len(original):
        return original
    literals = tuple(card["exact_literals"])
    if not literals:
        return original[:allowance].rstrip()
    candidate = excerpt_to_tokens(
        original,
        max(1, allowance // 4),
        exact_literals=literals,
    )
    if len(candidate) <= allowance and all(
        matches_evidence_literal(candidate, literal) for literal in literals
    ):
        return candidate
    return minimum_fact if len(minimum_fact) <= allowance else minimum_fact[:allowance].rstrip()


def pack_evidence_payload(
    items: Iterable[Mapping[str, Any]],
    *,
    max_tokens: int,
    query: str = "",
    source_fallback: str = "",
    requirements: tuple[EvidenceRequirement, ...] = (),
) -> PackedEvidencePayload:
    """Build one ordered, token-bounded evidence payload.

    Candidate order is the backend order. Context packing selects only a prefix
    and allocates remaining space from the first card forward. Query text never
    changes candidate or line priority; only caller-declared requirement
    literals may receive a bounded capsule.
    """

    token_budget = max(0, int(max_tokens or 0))
    char_budget = token_budget * 4
    if char_budget <= 0:
        return PackedEvidencePayload((), "", token_budget, 0, 0)
    if char_budget <= len(_EVIDENCE_CONTEXT_PREFIX):
        rendered = _EVIDENCE_CONTEXT_PREFIX[:char_budget]
        return PackedEvidencePayload(
            (),
            rendered,
            token_budget,
            estimate_tokens(rendered),
            0,
            metadata_tokens=estimate_tokens(rendered),
        )

    compiled_requirements = validate_evidence_requirements(requirements)
    exact_literals = tuple(
        dict.fromkeys(
            literal
            for requirement in compiled_requirements
            for literal in requirement.accepted_literals
        )
    )
    query_type = classify_recall_query(query)
    cards: list[dict[str, Any]] = []
    for raw_item in items:
        if not _context_item_is_renderable(raw_item):
            continue
        original = str(raw_item.get("text") or raw_item.get("content") or "").strip()
        if not original:
            continue
        item = compact_context_item(
            _normalized_evidence_card_item(raw_item, source_fallback=source_fallback)
        )
        item["text"] = original
        supported = _supported_exact_literals(original, exact_literals)
        cards.append(
            {
                "original": original,
                "item": item,
                "exact_literals": supported,
                "minimum_fact": (
                    _minimum_exact_excerpt(original, literals=supported) if supported else original
                ),
                "upstream_truncated": bool(
                    item.get("truncated") or (item.get("metadata") or {}).get("truncated")
                ),
            }
        )
    if not cards:
        return PackedEvidencePayload((), "", token_budget, 0, 0)

    selected = _select_prefix(
        cards,
        char_budget=char_budget,
        compact_headers=False,
        query_type=query_type,
    )
    compact_headers = False
    if len(selected) < len(cards):
        compact_selected = _select_prefix(
            cards,
            char_budget=char_budget,
            compact_headers=True,
            query_type=query_type,
        )
        if len(compact_selected) >= len(selected):
            selected = compact_selected
            compact_headers = True
    if not selected:
        # A single oversized first record gets the remaining budget. Do not
        # spend it on later headers or change the backend's candidate order.
        first = {
            **cards[0],
            "minimum_fact": _minimum_exact_excerpt(
                cards[0]["original"], literals=cards[0]["exact_literals"]
            ),
        }
        selected = _select_prefix(
            [first],
            char_budget=char_budget,
            compact_headers=True,
            query_type=query_type,
        )
        compact_headers = True
    if not selected:
        rendered = _EVIDENCE_CONTEXT_PREFIX[:char_budget]
        omitted_visibility = tuple(
            EvidenceVisibility(
                record_id=_record_identity(card["item"])[0],
                logical_block_id=_record_identity(card["item"])[1],
                state="not_visible",
                truncation_reason="context_budget_item_omitted",
            )
            for card in cards
        )
        return PackedEvidencePayload(
            (),
            rendered,
            token_budget,
            estimate_tokens(rendered),
            len(cards),
            metadata_tokens=estimate_tokens(rendered),
            visibility=omitted_visibility,
        )

    base_lines = [_EVIDENCE_CONTEXT_PREFIX]
    for index, card in enumerate(selected, start=1):
        base_lines.append(
            _evidence_card_header(
                card["item"],
                index=index,
                truncated=False,
                compact=compact_headers,
                query_type=query_type,
            )
        )
        base_lines.append("fact=")
    remaining_chars = max(0, char_budget - len("\n".join(base_lines)))
    fact_allowances = [len(str(card["minimum_fact"])) for card in selected]
    distributable = max(0, remaining_chars - sum(fact_allowances))
    for index, card in enumerate(selected):
        wanted = max(0, len(str(card["original"])) - fact_allowances[index])
        granted = min(distributable, wanted)
        fact_allowances[index] += granted
        distributable -= granted
        if distributable <= 0:
            break

    packed: list[dict[str, Any]] = []
    visibility: list[EvidenceVisibility] = []
    literal_losses: list[ExactLiteralLoss] = []
    for card, allowance in zip(selected, fact_allowances, strict=True):
        fact = _fact_within_chars(card, allowance)
        item = dict(card["item"])
        item["text"] = fact
        context_truncated = fact != card["original"]
        truncated = bool(card["upstream_truncated"] or context_truncated)
        item["truncated"] = truncated
        metadata = dict(item.get("metadata") or {})
        rendered_literals = [
            literal for literal in card["exact_literals"] if matches_evidence_literal(fact, literal)
        ]
        if rendered_literals:
            metadata["exact_literal_capsule"] = rendered_literals
        else:
            metadata.pop("exact_literal_capsule", None)
        if truncated:
            if card["upstream_truncated"] and context_truncated:
                reason = "upstream_and_context_budget"
            elif card["upstream_truncated"]:
                reason = "upstream_truncation"
            else:
                reason = "context_budget"
            metadata["visibility"] = "partial_visible"
            metadata["truncation_reason"] = reason
        else:
            reason = ""
            metadata["visibility"] = "full_visible"
            metadata.pop("truncation_reason", None)
        item["metadata"] = metadata
        packed.append(item)
        record_id, logical_block_id = _record_identity(item)
        visibility.append(
            EvidenceVisibility(
                record_id=record_id,
                logical_block_id=logical_block_id,
                state="partial_visible" if truncated else "full_visible",
                truncation_reason=reason,
            )
        )
        literal_losses.extend(
            ExactLiteralLoss(
                literal=literal,
                record_id=record_id,
                reason="exact_literal_exceeds_context_budget",
            )
            for literal in card["exact_literals"]
            if not matches_evidence_literal(fact, literal)
        )

    for card in cards[len(selected) :]:
        record_id, logical_block_id = _record_identity(card["item"])
        visibility.append(
            EvidenceVisibility(
                record_id=record_id,
                logical_block_id=logical_block_id,
                state="not_visible",
                truncation_reason="context_budget_item_omitted",
            )
        )
        literal_losses.extend(
            ExactLiteralLoss(
                literal=literal,
                record_id=record_id,
                reason="candidate_omitted_by_context_budget",
            )
            for literal in card["exact_literals"]
        )

    lines = [_EVIDENCE_CONTEXT_PREFIX]
    for index, item in enumerate(packed, start=1):
        lines.append(
            _evidence_card_header(
                item,
                index=index,
                truncated=bool(item.get("truncated")),
                compact=compact_headers,
                query_type=query_type,
            )
        )
        lines.append(f"fact={item['text']}")
    rendered = "\n".join(lines)
    capsule = tuple(
        dict.fromkeys(
            literal
            for item in packed
            for literal in (item.get("metadata") or {}).get("exact_literal_capsule", ())
        )
    )
    return PackedEvidencePayload(
        items=tuple(packed),
        rendered_context=rendered,
        token_budget=token_budget,
        estimated_tokens=estimate_tokens(rendered),
        truncated_item_count=sum(bool(item.get("truncated")) for item in packed),
        exact_literal_capsule=capsule,
        metadata_tokens=_metadata_token_count(
            packed,
            compact_headers=compact_headers,
            query_type=query_type,
        ),
        visibility=tuple(visibility),
        exact_literal_losses=tuple(literal_losses),
    )
