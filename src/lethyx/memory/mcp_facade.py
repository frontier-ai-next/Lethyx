"""Project-scoped Lethyx facade used by MCP tools and Hermes integrations.

The functions in this module are plain JSON-friendly callables.  The MCP server
wraps them, and tests exercise them directly so the memory authority/isolation
rules stay independent of the transport.
"""

import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from lethyx import MemoryClient
from lethyx.contracts import EvidenceRef
from lethyx.memory.backends.base import MemoryHit
from lethyx.memory.backends.contracts import (
    MemoryBlockRecord,
    MemoryEdgeRecord,
    block_record_to_memory_hit,
)
from lethyx.memory.obsidian_export import export_backend_to_obsidian, export_records_to_obsidian

_SUPPRESSED_LIFECYCLES = {
    "quarantined",
    "superseded",
    "deprecated",
    "retracted",
    "archived",
}
_SECRET_PRIVACY_CLASSES = {
    "secret",
    "secret_pointer",
    "secret_pointer_only",
    "non-injectable",
    "non_injectable",
}
_PASSIVE_SURFACES = {"archive_hint", "reflection_candidate"}
_VERIFIED_TRUST = {
    "workspace_verified",
    "public_verified",
    "supervisor_verified",
    "mutation_verified",
    "hidden_verified",
    "adversarial_verified",
    "human_approved",
}


def _default_repo_root(repo_root: str | Path | None = None) -> Path:
    raw = str(repo_root or os.getenv("LETHYX_REPO_ROOT") or "").strip()
    if raw:
        return Path(raw).resolve()
    return Path.cwd().resolve()


def _default_workspace(workspace_id: str | None = None) -> str:
    return str(workspace_id or os.getenv("LETHYX_WORKSPACE") or "default").strip() or "default"


def _default_backend(backend: str | None = None) -> str:
    return str(backend or os.getenv("LETHYX_BACKEND") or "mempalace").strip() or "mempalace"


def _client(
    *, repo_root: str | Path | None, workspace_id: str | None, backend: str | None
) -> MemoryClient:
    return MemoryClient(
        repo_root=_default_repo_root(repo_root),
        workspace_id=_default_workspace(workspace_id),
        backend=_default_backend(backend),
    )


def _close_client(client: MemoryClient) -> None:
    backend = getattr(client, "_backend", None)
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _as_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _as_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _evidence_ref(
    value: Any, *, fallback_type: str = "mcp", fallback_producer: str = "lethyx_mcp"
) -> EvidenceRef:
    if isinstance(value, EvidenceRef):
        return value
    if isinstance(value, dict):
        return EvidenceRef.from_mapping(value)
    return EvidenceRef(ref_type=fallback_type, ref_id=str(value), produced_by=fallback_producer)


def _evidence_refs(values: Iterable[Any], *, fallback_type: str = "mcp") -> list[EvidenceRef]:
    return [_evidence_ref(value, fallback_type=fallback_type) for value in values]


def _safe_error(exc: BaseException) -> dict[str, Any]:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _record_authority(record: MemoryBlockRecord) -> str:
    if (
        record.surface == "directive"
        and record.lifecycle == "active"
        and record.trust_level in _VERIFIED_TRUST
    ):
        return "directive"
    return "non_directive"


def _record_in_scope(
    record: MemoryBlockRecord, *, workspace_id: str, include_global: bool = False
) -> bool:
    if record.lifecycle in _SUPPRESSED_LIFECYCLES:
        return False
    if record.workspace_id == workspace_id:
        return True
    if include_global and (
        (not record.workspace_id and record.scope in {"manager", "cross_run_durable"})
        or record.workspace_id in {"global", "shared"}
    ):
        return True
    return False


def _record_is_passive(record: MemoryBlockRecord) -> bool:
    raw = (
        str(
            record.metadata.get("memory_activity")
            or record.metadata.get("memory_role")
            or record.metadata.get("memory_block_role")
            or record.metadata.get("activity")
            or ""
        )
        .strip()
        .lower()
    )
    if raw == "passive":
        return True
    if raw == "active":
        return False
    return record.surface in _PASSIVE_SURFACES or record.tier == "cold"


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "did",
    "do",
    "does",
    "either",
    "for",
    "from",
    "have",
    "has",
    "in",
    "is",
    "of",
    "or",
    "run",
    "runs",
    "should",
    "the",
    "to",
    "use",
    "uses",
    "using",
    "what",
    "where",
    "which",
    "with",
    "location",
    "located",
}
_QUERY_RELATION_WORDS = {"use", "uses", "using", "run", "runs", "with", "have", "has"}
_QUERY_YES_NO_STARTERS = {"does", "do", "did", "is", "are", "can", "should"}


def _normalize_query_token(term: str) -> str:
    token = str(term or "").strip().lower()
    if not token or token in _QUERY_STOPWORDS:
        return ""
    if token in {"db", "sqlite"}:
        return "database"
    if token in {"repo", "repository"}:
        return "repo"
    if len(token) > 4 and token.endswith("s"):
        token = token[:-1]
    return token


def _raw_query_tokens(query: str) -> list[str]:
    return [token for token in re.split(r"\W+", str(query or "").lower()) if token]


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _raw_query_tokens(query):
        term = _normalize_query_token(raw)
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def _query_subject_terms(query: str) -> list[str]:
    raw_tokens = _raw_query_tokens(query)
    if not raw_tokens or raw_tokens[0] not in _QUERY_YES_NO_STARTERS:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_tokens[1:]:
        if raw in _QUERY_RELATION_WORDS:
            break
        term = _normalize_query_token(raw)
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def _query_anchor_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _raw_query_tokens(query):
        if "_" not in raw and "/" not in raw and not re.search(r"[A-Za-z]+[._-][A-Za-z0-9]", raw):
            continue
        term = _normalize_query_token(raw)
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def _haystack_terms(text: str) -> set[str]:
    terms = {_normalize_query_token(raw) for raw in re.split(r"\W+", str(text or "").lower())}
    terms.discard("")
    if "database" in terms or "sqlite" in str(text or "").lower():
        terms.add("database")
    return terms


def _record_haystack(record: MemoryBlockRecord) -> str:
    return " ".join(
        [
            record.title,
            record.content,
            record.kind,
            " ".join(record.tags),
            " ".join(str(value) for value in record.metadata.values()),
        ]
    )


def _record_matches_query(record: MemoryBlockRecord, query: str) -> bool:
    terms = _query_terms(query)
    if not terms:
        return True
    haystack_terms = _haystack_terms(_record_haystack(record))
    if all(term in haystack_terms for term in terms):
        return True
    subject_terms = _query_subject_terms(query)
    if subject_terms:
        return all(term in haystack_terms for term in subject_terms)
    anchor_terms = _query_anchor_terms(query)
    if anchor_terms:
        return all(term in haystack_terms for term in anchor_terms)
    coverage = sum(1 for term in terms if term in haystack_terms)
    threshold = max(1, min(3, (len(terms) + 1) // 2))
    return coverage >= threshold


_RECENT_SESSION_QUERY_TERMS = {
    "previous",
    "recent",
    "latest",
    "last",
    "remembered",
    "recalled",
    "session",
    "sessions",
    "context",
    "memory",
    "прошлая",
    "прошлой",
    "прошлую",
    "предыдущая",
    "предыдущей",
    "предыдущую",
    "последняя",
    "последней",
    "сессия",
    "сессии",
    "сессию",
    "контекст",
    "память",
    "вспомни",
    "вспомнил",
}


def _query_wants_recent_session_context(query: str) -> bool:
    terms = set(_query_terms(query))
    if not terms:
        return False
    return bool(terms & _RECENT_SESSION_QUERY_TERMS) and (
        "session" in terms
        or "sessions" in terms
        or "context" in terms
        or "memory" in terms
        or "сессия" in terms
        or "сессии" in terms
        or "сессию" in terms
        or "контекст" in terms
        or "память" in terms
    )


def _record_is_session_summary(record: MemoryBlockRecord) -> bool:
    block_id = str(record.block_id or "").lower()
    kind = str(record.kind or "").lower()
    tags = {str(tag).strip().lower() for tag in (record.tags or ()) if str(tag).strip()}
    return (
        kind == "session_summary"
        or block_id.startswith("session_summary")
        or "session-summary" in tags
    )


def _record_sort_key(record: MemoryBlockRecord) -> tuple[int, str, str]:
    lifecycle_rank = 1 if record.lifecycle == "active" else 0
    return (lifecycle_rank, record.updated_at or record.created_at, record.block_id)


_SENSITIVE_METADATA_RE = re.compile(
    r"(api[_-]?key|authorization|credential|password|secret|token)", re.I
)


def _is_secret_metadata(metadata: dict[str, Any]) -> bool:
    privacy_class = str(metadata.get("privacy_class") or "").strip().lower()
    risk_level = str(metadata.get("risk_level") or "").strip().lower()
    return privacy_class in _SECRET_PRIVACY_CLASSES or risk_level == "secret"


def _safe_text(text: str, metadata: dict[str, Any]) -> str:
    block_id = str(metadata.get("block_id") or "memory")
    if _is_secret_metadata(metadata):
        return f"[redacted non-exportable memory: {block_id}]"
    return text


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not _is_secret_metadata(metadata):
        return metadata
    return {
        key: value for key, value in metadata.items() if not _SENSITIVE_METADATA_RE.search(str(key))
    }


def _hit_from_record(
    record: MemoryBlockRecord, *, force_non_directive: bool = False
) -> dict[str, Any]:
    hit = block_record_to_memory_hit(record)
    return _hit_to_payload(hit, record=record, force_non_directive=force_non_directive)


def _hit_to_payload(
    hit: MemoryHit, *, record: MemoryBlockRecord | None = None, force_non_directive: bool = False
) -> dict[str, Any]:
    metadata = dict(hit.metadata or {})
    if record is not None:
        metadata["block_id"] = record.block_id
        metadata["workspace_id"] = record.workspace_id
        metadata["surface"] = record.surface
        metadata["lifecycle"] = record.lifecycle
        metadata["trust_level"] = record.trust_level
        metadata["source_backend"] = record.source_backend
        metadata["privacy_class"] = record.privacy_class
        metadata["risk_level"] = record.risk_level
    metadata = _safe_metadata(metadata)
    authority = (
        "non_directive"
        if force_non_directive
        else (_record_authority(record) if record is not None else "non_directive")
    )
    return {
        "text": _safe_text(str(hit.text or ""), metadata),
        "source": hit.source,
        "score": hit.score,
        "kind": hit.kind,
        "tags": list(hit.tags),
        "metadata": _as_jsonable(metadata),
        "surface": "supplemental_evidence" if force_non_directive else metadata.get("surface", ""),
        "authority": authority,
    }


def _payloads_from_records(
    records: Iterable[MemoryBlockRecord],
    *,
    max_tokens: int,
    force_non_directive: bool = False,
) -> list[dict[str, Any]]:
    budget = max(0, int(max_tokens or 0)) * 4
    if budget <= 0:
        return []
    out: list[dict[str, Any]] = []
    remaining = budget
    for record in records:
        payload = _hit_from_record(record, force_non_directive=force_non_directive)
        text = str(payload.get("text") or "")
        if len(text) > remaining:
            payload["text"] = text[:remaining]
        out.append(payload)
        remaining -= len(str(payload.get("text") or ""))
        if remaining <= 0:
            break
    return out


def _record_query_score(record: MemoryBlockRecord, query: str) -> tuple[int, int, int, str]:
    haystack = " ".join(
        [
            record.title,
            record.content,
            record.kind,
            " ".join(record.tags),
            " ".join(str(value) for value in record.metadata.values()),
        ]
    )
    haystack_terms = _haystack_terms(haystack)
    query_terms = _query_terms(query)
    coverage = sum(1 for term in query_terms if term in haystack_terms)
    exact_path_bonus = len(
        re.findall(r"[\w.-]+/[\w./:-]+|[\w.-]+\.[A-Za-z0-9]{1,8}", record.content or "")
    )
    return (
        coverage,
        exact_path_bonus,
        len(record.content or ""),
        record.updated_at or record.created_at,
    )


def _dedup_records(
    records: Iterable[MemoryBlockRecord], *, query: str = ""
) -> list[MemoryBlockRecord]:
    by_key: dict[str, MemoryBlockRecord] = {}
    order: list[str] = []
    for record in records:
        key = record.block_id or record.content
        current = by_key.get(key)
        if current is None:
            by_key[key] = record
            order.append(key)
            continue
        if _record_query_score(record, query) > _record_query_score(current, query):
            by_key[key] = record
    return [by_key[key] for key in order]


def _safe_records_from_backend(
    backend: Any,
    *,
    workspace_id: str,
    include_global: bool,
    include_passive: bool,
    query: str,
    limit: int,
) -> list[MemoryBlockRecord]:
    export_blocks = getattr(backend, "export_blocks", None)
    if not callable(export_blocks):
        return []
    records = [
        record
        for record in export_blocks()
        if isinstance(record, MemoryBlockRecord)
        and _record_in_scope(record, workspace_id=workspace_id, include_global=include_global)
        and (include_passive or not _record_is_passive(record))
        and _record_matches_query(record, query)
    ]
    return sorted(records, key=_record_sort_key, reverse=True)[:limit]


def _recent_session_records(
    backend: Any,
    *,
    workspace_id: str,
    include_global: bool,
    include_passive: bool,
    limit: int,
) -> list[MemoryBlockRecord]:
    export_blocks = getattr(backend, "export_blocks", None)
    if not callable(export_blocks):
        return []
    records = [
        record
        for record in export_blocks()
        if isinstance(record, MemoryBlockRecord)
        and _record_is_session_summary(record)
        and _record_in_scope(record, workspace_id=workspace_id, include_global=include_global)
        and (include_passive or not _record_is_passive(record))
    ]
    return sorted(records, key=_record_sort_key, reverse=True)[:limit]


def _safe_records_from_native_hits(
    client: MemoryClient,
    *,
    query: str,
    workspace_id: str,
    include_global: bool,
    include_passive: bool,
    tags: list[str] | tuple[str, ...] | None,
    max_tokens: int,
    limit: int,
) -> tuple[list[MemoryBlockRecord], dict[str, Any]]:
    backend = client.backend
    get_block = getattr(backend, "get_block", None)
    diagnostics: dict[str, Any] = {
        "native_count": 0,
        "native_error": "",
        "native_path": "",
    }
    if include_passive and hasattr(backend, "recall"):
        try:
            records = backend.recall(
                {
                    "query": query,
                    "workspace_id": workspace_id,
                    "tags": list(tags or []),
                    "limit": max(limit * 3, limit),
                    "max_tokens": max_tokens,
                    "include_passive": True,
                }
            )
            out = [
                record
                for record in records
                if isinstance(record, MemoryBlockRecord)
                and _record_in_scope(
                    record, workspace_id=workspace_id, include_global=include_global
                )
                and (include_passive or not _record_is_passive(record))
                and _record_matches_query(record, query)
            ]
            diagnostics["native_count"] = len(out)
            diagnostics["native_path"] = "backend.recall"
            return out, diagnostics
        except Exception as exc:
            diagnostics["native_error"] = str(exc)
            diagnostics["native_path"] = "backend.recall"
    if not callable(get_block):
        return [], diagnostics
    records: list[MemoryBlockRecord] = []
    try:
        hits = client.recall(query, limit=max(limit * 3, limit), tags=tags, max_tokens=max_tokens)
    except Exception as exc:
        diagnostics["native_error"] = str(exc)
        diagnostics["native_path"] = diagnostics.get("native_path") or "client.recall"
        return [], diagnostics
    for hit in hits:
        block_id = str(dict(hit.metadata or {}).get("block_id") or "")
        if not block_id:
            continue
        record = get_block(block_id)
        if not isinstance(record, MemoryBlockRecord):
            continue
        if (
            _record_in_scope(record, workspace_id=workspace_id, include_global=include_global)
            and (include_passive or not _record_is_passive(record))
            and _record_matches_query(record, query)
        ):
            records.append(record)
    diagnostics["native_count"] = len(records)
    diagnostics["native_path"] = diagnostics.get("native_path") or "client.recall"
    return records, diagnostics


def _recall_records(
    client: MemoryClient,
    *,
    query: str,
    limit: int,
    tags: list[str] | tuple[str, ...] | None,
    include_global: bool,
    include_passive: bool,
    max_tokens: int,
) -> tuple[list[MemoryBlockRecord], dict[str, Any]]:
    workspace_id = client.workspace_id
    backend = client.backend
    native, diagnostics = _safe_records_from_native_hits(
        client,
        query=query,
        workspace_id=workspace_id,
        include_global=include_global,
        include_passive=include_passive,
        tags=tags,
        max_tokens=max_tokens,
        limit=limit,
    )
    indexed = _safe_records_from_backend(
        backend,
        workspace_id=workspace_id,
        include_global=include_global,
        include_passive=include_passive,
        query=query,
        limit=max(limit * 3, limit),
    )
    diagnostics.update(
        {
            "index_used": bool(indexed),
            "index_count": len(indexed),
        }
    )
    records = _dedup_records([*native, *indexed], query=query)
    if _query_wants_recent_session_context(query):
        recent = _recent_session_records(
            backend,
            workspace_id=workspace_id,
            include_global=include_global,
            include_passive=include_passive,
            limit=max(limit * 3, limit),
        )
        diagnostics["recent_session_count"] = len(recent)
        records = _dedup_records([*records, *recent], query=query)
    else:
        diagnostics["recent_session_count"] = 0
    return records[:limit], diagnostics


def _recall_records_with_optional_global(
    client: MemoryClient,
    *,
    query: str,
    limit: int,
    tags: list[str] | tuple[str, ...] | None,
    include_global: bool,
    include_passive: bool,
    max_tokens: int,
    backend: str | None,
) -> tuple[list[MemoryBlockRecord], dict[str, Any]]:
    records, diagnostics = _recall_records(
        client,
        query=query,
        limit=limit,
        tags=tags,
        include_global=False,
        include_passive=include_passive,
        max_tokens=max_tokens,
    )
    diagnostics["global_count"] = 0
    if include_global and client.workspace_id not in {"global", "shared"}:
        global_client = _client(
            repo_root=client.repo_root,
            workspace_id="global",
            backend=backend,
        )
        try:
            global_records, global_diagnostics = _recall_records(
                global_client,
                query=query,
                limit=limit,
                tags=tags,
                include_global=False,
                include_passive=include_passive,
                max_tokens=max_tokens,
            )
            records = _dedup_records([*records, *global_records], query=query)
            diagnostics["global_count"] = len(global_records)
            diagnostics["global"] = global_diagnostics
        finally:
            _close_client(global_client)
    return records[:limit], diagnostics


def lethyx_health(
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        payload = dict(client.backend.health() or {})
        payload.setdefault("ok", False)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["repo_root"] = str(client.repo_root)
        return _as_jsonable(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["repo_root"] = str(client.repo_root)
        return payload
    finally:
        _close_client(client)


def lethyx_remember(
    content: str,
    title: str = "",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    kind: str = "observation",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    evidence_refs: list[Any] | None = None,
    scope: str = "workspace",
    lifecycle: str = "active",
    surface: str = "supplemental_evidence",
    trust_level: str = "agent_claim",
    tier: str = "warm",
    memory_id: str = "",
    run_id: str = "",
    phase_id: str = "",
    subtask_id: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        result = client.remember(
            content,
            title=title,
            kind=kind,
            tags=tags or [],
            metadata=metadata or {},
            evidence_refs=_evidence_refs(evidence_refs or [], fallback_type="remember"),
            scope=scope,
            lifecycle=lifecycle,
            surface=surface,
            trust_level=trust_level,
            tier=tier,
            memory_id=memory_id,
            run_id=run_id,
            phase_id=phase_id,
            subtask_id=subtask_id,
        )
        return {
            "ok": result.saved,
            "saved": result.saved,
            "memory_id": result.memory_id,
            "backend": result.backend or _default_backend(backend),
            "workspace_id": client.workspace_id,
            "raw": _as_jsonable(result.raw),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def lethyx_recall(
    query: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 10,
    tags: list[str] | None = None,
    include_global: bool = False,
    include_passive: bool = False,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        safe_limit = max(0, int(limit or 0))
        records, diagnostics = _recall_records_with_optional_global(
            client,
            query=query,
            limit=safe_limit,
            tags=tags or [],
            include_global=include_global,
            include_passive=include_passive,
            max_tokens=max_tokens,
            backend=backend,
        )
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            "include_global": include_global,
            "include_passive": include_passive,
            "diagnostics": _as_jsonable(diagnostics),
            "hits": _payloads_from_records(records, max_tokens=max_tokens),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        payload["hits"] = []
        return payload
    finally:
        _close_client(client)


def lethyx_build_context(
    query: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    limit: int = 8,
    include_global: bool = False,
    include_passive: bool = False,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        records, diagnostics = _recall_records_with_optional_global(
            client,
            query=query,
            limit=max(0, int(limit or 0)),
            tags=[],
            include_global=include_global,
            include_passive=include_passive,
            max_tokens=max_tokens,
            backend=backend,
        )
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            "surface": "supplemental_evidence",
            "authority": "non_directive",
            "diagnostics": _as_jsonable(diagnostics),
            "items": _payloads_from_records(
                records, max_tokens=max_tokens, force_non_directive=True
            ),
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload.update(
            {
                "workspace_id": client.workspace_id,
                "backend": _default_backend(backend),
                "surface": "supplemental_evidence",
                "authority": "non_directive",
                "items": [],
            }
        )
        return payload
    finally:
        _close_client(client)


def lethyx_promote(
    memory_id: str,
    reason: str,
    evidence_refs: list[Any],
    target_surface: str = "supplemental_evidence",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        payload = dict(
            client.promote(
                memory_id,
                reason=reason,
                evidence_refs=_evidence_refs(evidence_refs, fallback_type="promotion"),
                surface=target_surface,
            )
            or {}
        )
        payload.setdefault("ok", bool(payload.get("block_id")))
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = payload.get("backend") or _default_backend(backend)
        return _as_jsonable(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def lethyx_forget(
    memory_id: str,
    reason: str = "user_requested",
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        evidence = [
            EvidenceRef(
                ref_type="forget",
                ref_id=reason or "user_requested",
                produced_by="lethyx_mcp",
            )
        ]
        payload = dict(
            client.backend.update_lifecycle(
                memory_id,
                "retracted",
                reason=reason or "user_requested",
                evidence_refs=evidence,
                surface="archive_hint",
            )
            or {}
        )
        payload.setdefault("ok", bool(payload.get("block_id")))
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = payload.get("backend") or _default_backend(backend)
        return _as_jsonable(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def lethyx_add_edge(
    src_id: str,
    dst_id: str,
    edge_type: str,
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    reason: str = "",
    weight: float = 1.0,
    evidence_refs: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        edge = MemoryEdgeRecord(
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            workspace_id=client.workspace_id,
            reason=reason,
            weight=weight,
            evidence_refs=tuple(_evidence_refs(evidence_refs or [], fallback_type="edge")),
            metadata=metadata or {},
        )
        payload = dict(client.backend.add_edge(edge) or {})
        payload.setdefault("ok", bool(payload.get("src_id") and payload.get("dst_id")))
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = payload.get("backend") or _default_backend(backend)
        return _as_jsonable(payload)
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)


def lethyx_export_obsidian(
    workspace_id: str = "",
    backend: str = "",
    repo_root: str = "",
    vault_dir: str = "",
    project_title: str = "",
    include_global: bool = False,
) -> dict[str, Any]:
    client = _client(repo_root=repo_root, workspace_id=workspace_id, backend=backend)
    try:
        target_vault = str(vault_dir or os.getenv("LETHYX_OBSIDIAN_VAULT") or "").strip()
        if not target_vault:
            target_vault = str(client.repo_root / "exports" / "obsidian")
        if include_global and client.workspace_id not in {"global", "shared"}:
            blocks = list(getattr(client.backend, "export_blocks")())
            edges: list[MemoryEdgeRecord] = []
            edges_for = getattr(client.backend, "edges_for", None)
            if callable(edges_for):
                for block in blocks:
                    edges.extend(edges_for(block.block_id) or [])
            global_client = _client(
                repo_root=client.repo_root,
                workspace_id="global",
                backend=backend,
            )
            try:
                global_blocks = list(getattr(global_client.backend, "export_blocks")())
                blocks.extend(global_blocks)
                global_edges_for = getattr(global_client.backend, "edges_for", None)
                if callable(global_edges_for):
                    for block in global_blocks:
                        edges.extend(global_edges_for(block.block_id) or [])
            finally:
                _close_client(global_client)
            result = export_records_to_obsidian(
                blocks=blocks,
                edges=edges,
                vault_dir=target_vault,
                workspace_id=client.workspace_id,
                project_title=project_title,
                include_global=True,
            )
        else:
            result = export_backend_to_obsidian(
                backend=client.backend,
                vault_dir=target_vault,
                workspace_id=client.workspace_id,
                project_title=project_title,
                include_global=include_global,
            )
        return {
            "ok": True,
            "workspace_id": client.workspace_id,
            "backend": _default_backend(backend),
            "vault_dir": str(result.vault_dir),
            "index_path": str(result.index_path),
            "memory_overview_path": str(result.notebooklm_source_path),
            "blocks_exported": len(result.block_paths),
            "files_written": [str(path) for path in result.written_files],
        }
    except Exception as exc:
        payload = _safe_error(exc)
        payload["workspace_id"] = client.workspace_id
        payload["backend"] = _default_backend(backend)
        return payload
    finally:
        _close_client(client)
