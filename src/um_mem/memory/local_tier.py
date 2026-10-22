"""The local retrieval tier: script-neutral lexical scoring plus semantics.

The native backend answers most queries. When it cannot - because it failed,
because it answered with nothing useful, or because the question is written in
a script the stored text does not share - this tier scores a bounded window of
the newest records locally.

Two scorers live here:

* a lexical one (:mod:`um_mem.memory.text`) that matches character n-grams
  and therefore works in any script, and
* a semantic one (:mod:`um_mem.memory.semantic`) that embeds the question and
  the records with a multilingual model, so a question in one script can reach
  a record written in another.

Both tiers only reorder candidates. The stored record text stays the evidence,
and a question with no real match returns nothing rather than a
plausible-looking record.

The tier lives in the service layer on purpose. The embedding model is loaded
once per service process and stays warm, so the MCP tools and the pre-tool
helper share one model instead of loading it per call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from um_mem.memory import semantic as semantic_tier
from um_mem.memory import text as universal_text

#: How many newest records the local window scores.
FALLBACK_SCAN_LIMIT = 128

#: How many newest records are surfaced when even the local tier finds nothing.
FALLBACK_RECENT_LIMIT = 3

#: The native answer below this score is not worth trusting on its own.
NATIVE_SCORE_FLOOR = 0.35


def hit_record_id(hit: Any) -> str:
    """Return the record identity of one hit, whatever shape it has."""

    if not isinstance(hit, Mapping):
        return ""
    metadata = hit.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("record_id", "block_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    for key in ("record_id", "id"):
        value = str(hit.get(key) or "").strip()
        if value:
            return value
    source = str(hit.get("source") or "").strip()
    if source.startswith("um-mem:"):
        return source.split(":", 1)[1]
    return source


def merge_hits(groups: list[Any], limit: int) -> list[dict[str, Any]]:
    """Merge hit groups by record identity, best score first."""

    merged: dict[str, tuple[float, dict[str, Any]]] = {}
    for hits in groups:
        for index, hit in enumerate(hits or []):
            if not isinstance(hit, dict):
                continue
            record_id = hit_record_id(hit) or f"anonymous:{index}"
            try:
                score = float(hit.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            current = merged.get(record_id)
            if current is None or score > current[0]:
                merged[record_id] = (score, hit)
    ordered = sorted(merged.values(), key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ordered[: max(1, limit)]]


def merge_tiered(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Order hits by tier first and keep the first copy of every record."""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in list(primary) + list(secondary):
        if not isinstance(hit, dict):
            continue
        record_id = hit_record_id(hit)
        if record_id and record_id in seen:
            continue
        if record_id:
            seen.add(record_id)
        merged.append(hit)
        if len(merged) >= limit:
            break
    return merged[:limit]


def merge_context_items(
    existing: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Keep packed context order and append non-duplicate fallback evidence."""

    merged = list(existing)
    seen = {hit_record_id(item) for item in merged if hit_record_id(item)}
    for item in fallback:
        record_id = hit_record_id(item)
        if record_id and record_id in seen:
            continue
        merged.append(item)
        if record_id:
            seen.add(record_id)
        if len(merged) >= limit:
            break
    return merged[:limit]


def scope_allows(arguments: Mapping[str, Any]) -> bool:
    """Use the local window only when it can honestly answer the same query.

    A block-scoped query is answered from the caller's scoped records, which
    keeps the tier inside the selected leaves. An exact record-id query stays
    declined because the local window cannot enumerate those identities.
    """

    if arguments.get("record_ids"):
        return False
    if str(arguments.get("temporal_mode") or "current") != "current":
        return False
    if arguments.get("as_of") or arguments.get("known_as_of") or arguments.get("revision_id"):
        return False
    return str(arguments.get("source_horizon") or "durable") == "durable"


def record_source_speakers(record: Mapping[str, Any]) -> set[str]:
    """Return every speaker name attached to one record."""

    metadata: Mapping[str, Any] = (
        record["metadata"] if isinstance(record.get("metadata"), Mapping) else {}
    )
    speakers: set[str] = set()
    for source in (record, metadata):
        for key in ("speaker", "source_speaker"):
            value = str(source.get(key) or "").strip()
            if value:
                speakers.add(value.casefold())
        values = source.get("source_speakers")
        if isinstance(values, (list, tuple, set)):
            speakers.update(str(value).strip().casefold() for value in values if str(value).strip())
    for source in (record, metadata):
        segments = source.get("source_segments")
        if not isinstance(segments, (list, tuple)):
            continue
        for segment in segments:
            if isinstance(segment, dict):
                value = str(segment.get("speaker") or "").strip()
                if value:
                    speakers.add(value.casefold())
    return speakers


def record_allowed(record: Mapping[str, Any], arguments: Mapping[str, Any]) -> bool:
    """Apply the caller's explicit filters to the bounded local window."""

    if not str(record.get("content") or "").strip():
        return False
    if record.get("delivery_policy") == "hidden":
        return False
    scopes = set(arguments.get("scope_chain") or ())
    if scopes:
        scope = str(record.get("scope") or "")
        identity_field = {
            "workspace": "workspace_id",
            "project": "project_id",
            "session": "session_id",
            "task": "task_id",
            "subtask": "subtask_id",
            "user_global": "user_id",
        }.get(scope, "")
        scope_key = record.get("scope_key") or f"{scope}:{record.get(identity_field, '')}"
        if scope_key not in scopes:
            return False
    if str(record.get("status") or "").casefold() in {
        "retracted",
        "superseded",
        "deprecated",
        "expired",
        "quarantined",
    }:
        return False
    if not arguments.get("include_candidates") and str(record.get("status") or "") == "candidate":
        return False
    wanted_tags = {str(tag).strip() for tag in (arguments.get("tags") or ()) if str(tag).strip()}
    if wanted_tags:
        record_tags = {str(tag) for tag in (record.get("tags") or ())}
        if not wanted_tags.issubset(record_tags):
            return False
    block_ids = {
        str(block_id).strip()
        for block_id in (arguments.get("logical_block_ids") or ())
        if str(block_id).strip()
    }
    if block_ids:
        membership: Mapping[str, Any] = (
            record["block_membership"]
            if isinstance(record.get("block_membership"), Mapping)
            else {}
        )
        if str(membership.get("primary_block_id") or "") not in block_ids:
            return False
    speakers = {
        str(speaker).strip().casefold()
        for speaker in (arguments.get("source_speakers") or ())
        if str(speaker).strip()
    }
    return not (speakers and not speakers.intersection(record_source_speakers(record)))


def fallback_hit(record: Mapping[str, Any], score: float, score_mode: str) -> dict[str, Any]:
    """Return one local-window hit in the shape the MCP layer expects.

    ``score_mode`` travels with the score because a cosine similarity and a
    relative lexical rank live in the same ``score`` field and are not
    comparable.
    """

    membership: Mapping[str, Any] = (
        record["block_membership"] if isinstance(record.get("block_membership"), Mapping) else {}
    )
    record_id = str(record.get("record_id") or record.get("block_id") or "")
    return {
        "source": record_id,
        "text": str(record.get("content") or ""),
        "title": str(record.get("title") or ""),
        "score": score,
        "score_mode": score_mode,
        "kind": str(record.get("kind") or "memory"),
        "tags": list(record.get("tags") or ()),
        "metadata": {
            "block_id": record_id,
            "record_id": record_id,
            "revision_id": str(record.get("revision_id") or ""),
            "logical_block_id": str(membership.get("primary_block_id") or ""),
            "block_membership": {"primary_block_id": str(membership.get("primary_block_id") or "")},
            "recorded_at": str(record.get("recorded_at") or ""),
            "status": str(record.get("status") or ""),
            "authority": str(record.get("authority") or ""),
            "kind": str(record.get("kind") or ""),
            "sensitivity": str(record.get("sensitivity") or ""),
            "content_mode": str(record.get("content_mode") or ""),
            "operational_risk": str(record.get("operational_risk") or ""),
        },
    }


def semantic_relaxed(
    records: Sequence[Mapping[str, Any]],
    query: str,
    limit: int,
    repo_root: Path | str,
    workspace_id: str,
    loader: Callable[[Sequence[str]], Sequence[Mapping[str, Any]]] | None = None,
    profile: str = "default",
) -> dict[str, Any]:
    """Score the local window with local multilingual sentence embeddings.

    When a loader is available the whole indexed workspace is scored, so a
    question reaches records that were written long ago. The window is the
    fallback for an installation without an index.
    """

    embedder = semantic_tier.get_embedder(cache_dir=semantic_tier.default_cache_dir(repo_root))
    if embedder is None or (not records and loader is None):
        return {"hits": [], "model": "", "blocks": []}
    index = semantic_tier.SemanticIndex(semantic_tier.semantic_index_path(repo_root, workspace_id))
    cross_script = semantic_tier.query_uses_non_ascii_script(query)
    threshold = semantic_tier.effective_floor(
        embedder.model,
        cross_script=cross_script,
        calibrated=semantic_tier.calibrated_floor(repo_root, workspace_id, embedder.model),
    )
    hits: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    if loader is not None:
        try:
            ranked = semantic_tier.rank_global(
                query,
                embedder=embedder,
                index=index,
                profile=profile,
                floor=threshold,
                limit=max(1, limit),
                loader=loader,
            )
        except Exception:  # pragma: no cover - advisory tier
            ranked = []
        for record_id, score, record in ranked:
            hits.append(fallback_hit(record, float(score), "semantic_cosine"))
            scores[record_id] = float(score)
    if not hits:
        try:
            ranked_semantic = semantic_tier.rank_semantic(
                query,
                records,
                embedder=embedder,
                index=index,
                profile=profile,
                floor=threshold,
                limit=max(1, limit),
            )
        except Exception:  # pragma: no cover - advisory tier
            return {"hits": [], "model": embedder.model, "blocks": []}
        by_identity = {semantic_tier.record_identity(record): record for record in records}
        for record_id, score in ranked_semantic:
            matched = by_identity.get(record_id)
            if matched is None:
                continue
            hits.append(fallback_hit(matched, float(score), "semantic_cosine"))
            scores[record_id] = float(score)
    return {
        "hits": hits,
        "model": embedder.model,
        "blocks": semantic_tier.suggest_blocks(records, scores),
    }


def relaxed_recall(
    records: Sequence[Mapping[str, Any]],
    query: str,
    limit: int,
    repo_root: Path | str,
    workspace_id: str,
    loader: Callable[[Sequence[str]], Sequence[Mapping[str, Any]]] | None = None,
    profile: str = "default",
) -> dict[str, Any]:
    """Score the bounded local window with every local tier."""

    terms = universal_text.distinctive_terms(query, limit=6)
    titles = [str(record.get("title") or "") for record in records]
    documents = [
        f"{title}\n{record.get('content') or ''}" for title, record in zip(titles, records)
    ]
    ranked = universal_text.rank(query, documents, titles=titles, limit=max(1, limit))
    hits: list[dict[str, Any]] = []
    matched_terms: list[str] = []
    for index, score, matched in ranked:
        record = records[index]
        if not str(record.get("content") or ""):
            continue
        hits.append(fallback_hit(record, score, "relative_lexical_ngram"))
        for term in matched:
            if term not in matched_terms:
                matched_terms.append(term)
    semantic = {"hits": [], "model": "", "blocks": []}
    if semantic_tier.query_uses_non_ascii_script(query) or not hits:
        # The semantic tier is needed when the question is written in another
        # script, or when the lexical tier found nothing at all. When the
        # lexical tier already answered, exact character matches stay in
        # charge and the model is not loaded for nothing.
        semantic = semantic_relaxed(
            records,
            query,
            limit,
            repo_root,
            workspace_id,
            loader=loader,
            profile=profile,
        )
    recent: list[dict[str, Any]] = []
    if not hits and not semantic["hits"]:
        recent = [
            {
                "record_id": str(record.get("record_id") or record.get("block_id") or ""),
                "title": str(record.get("title") or ""),
                "text": str(record.get("content") or "")[:200],
            }
            for record in records[:FALLBACK_RECENT_LIMIT]
        ]
    return {
        "hits": hits,
        "semantic_hits": semantic["hits"],
        "semantic_model": semantic["model"],
        "suggested_blocks": semantic["blocks"],
        "terms": terms,
        "matched_terms": matched_terms,
        "recent": recent,
    }


def apply_local_tier(
    result: dict[str, Any],
    *,
    name: str,
    arguments: Mapping[str, Any],
    records_provider: Callable[[], Sequence[Mapping[str, Any]]],
    repo_root: Path | str,
    workspace_id: str,
    record_loader: Callable[[Sequence[str]], Sequence[Mapping[str, Any]]] | None = None,
    profile: str = "default",
) -> dict[str, Any]:
    """Repair an empty, weak or failed retrieval with the local window.

    Native recall can fail for one query while the canonical index remains
    healthy. A dead end is worse than bounded local evidence, so the tier is
    applied to both ``recall`` and ``build_context`` and a failure is reported
    as degraded rather than hidden.
    """

    if not isinstance(result, dict) or result.get("fallback"):
        return result
    # The caller learns that the tier was considered, so a second layer does
    # not run it again with a second copy of the model.
    result["_local_tier_applied"] = True
    if not scope_allows(arguments):
        return result
    limit = max(1, int(arguments.get("limit") or (4 if name == "build_context" else 10)))
    query = str(arguments.get("query") or "")
    cross_script = semantic_tier.query_uses_non_ascii_script(query)
    semantic_ready = semantic_tier.semantic_ready(repo_root)
    if name == "recall":
        existing = [hit for hit in (result.get("hits") or []) if isinstance(hit, dict)]
        best_score = 0.0
        for hit in existing:
            try:
                best_score = max(best_score, float(hit.get("score") or 0.0))
            except (TypeError, ValueError):
                continue
        needs_fallback = (not result.get("ok")) or (not existing) or best_score < NATIVE_SCORE_FLOOR
    else:
        existing = [item for item in (result.get("items") or []) if isinstance(item, dict)]
        needs_fallback = (not result.get("ok")) or (not existing)
    # A question written in another script cannot be answered by matching the
    # characters of the stored text, so the local tier also runs for those
    # questions even when the keyword backend answered with something.
    needs_local = needs_fallback or (cross_script and semantic_ready)
    if not needs_local:
        return result

    degraded = not result.get("ok")
    if degraded:
        result["backend_error"] = str(result.get("error") or "")
        result["degraded"] = True
    result["relaxed_tried"] = True
    records = [record for record in records_provider() if record_allowed(record, arguments)]

    def filtered_loader(identifiers: Sequence[str]) -> list[Mapping[str, Any]]:
        loaded = record_loader(identifiers) if record_loader is not None else ()
        return [record for record in loaded if record_allowed(record, arguments)]

    relaxed = relaxed_recall(
        records,
        query,
        limit,
        repo_root,
        workspace_id,
        loader=filtered_loader if record_loader is not None else None,
        profile=profile,
    )
    lexical_hits = relaxed["hits"]
    semantic_hits = relaxed["semantic_hits"]
    if semantic_hits:
        result["semantic_model"] = relaxed["semantic_model"]
        if relaxed["suggested_blocks"]:
            result["suggested_blocks"] = relaxed["suggested_blocks"]
    # The semantic tier leads when the question is written in another script
    # or when the lexical tier found nothing at all; otherwise exact character
    # matches stay in charge.
    semantic_primary = bool(semantic_hits) and (cross_script or not lexical_hits)
    if semantic_primary:
        result["semantic"] = True
        result["relaxed"] = True
        result["relaxed_terms"] = relaxed["matched_terms"] or relaxed["terms"]
        result["fallback"] = "semantic"
        if name == "recall":
            result["hits"] = merge_tiered(semantic_hits, existing, limit)
        else:
            result["items"] = merge_context_items(semantic_hits, existing, limit)
            _render_context(result, arguments, query)
    elif lexical_hits:
        result["relaxed"] = True
        result["relaxed_terms"] = relaxed["matched_terms"] or relaxed["terms"]
        result["fallback"] = "unicode_ngram"
        if name == "recall":
            result["hits"] = merge_hits([existing, lexical_hits], limit)
            if semantic_hits:
                result["semantic"] = True
                result["hits"] = merge_tiered(result["hits"], semantic_hits, limit)
        else:
            result["items"] = merge_context_items(existing, lexical_hits, limit)
            _render_context(result, arguments, query)
    elif relaxed["recent"]:
        result["recent_fallback"] = relaxed["recent"]
    if degraded and (semantic_hits or lexical_hits):
        result["ok"] = True
        result.pop("error", None)
        result.pop("error_type", None)
    if name == "recall" and not result.get("hits"):
        result["hint"] = (
            "No hits from the backend, the script-neutral lexical fallback, or the "
            "multilingual semantic tier. Retry with a distinctive name, path, or "
            "technology, or call recent for the newest records."
        )
    if name == "build_context" and not result.get("items"):
        result["hint"] = (
            "No context items from the backend, the script-neutral lexical fallback, "
            "or the multilingual semantic tier. Use recall for a broader search or "
            "recent for the newest records."
        )
    return result


def _render_context(result: dict[str, Any], arguments: Mapping[str, Any], query: str) -> None:
    """Pack the recovered items into a prompt-safe block when asked for."""

    if not arguments.get("render_context") or not result.get("items"):
        return
    from um_mem.memory.context_payload import pack_evidence_payload

    packed = pack_evidence_payload(
        result["items"],
        max_tokens=int(arguments.get("max_tokens") or 2048),
        query=query,
    )
    if packed.rendered_context:
        result["rendered_context"] = packed.rendered_context
