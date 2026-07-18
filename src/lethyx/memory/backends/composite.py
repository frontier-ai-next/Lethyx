"""Production composite durable memory backend.

A composite backend is an Lethyx control-plane wrapper over concrete memory
technology blocks.  Each block owns one real backend adapter (MemPalace,
Hindsight, Mnemosyne, ...), has phase eligibility, and is marked as active or
passive memory.  Retrieval first chooses eligible blocks, then delegates into the
block's own native recall/search implementation.
"""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from lethyx.memory.backends.base import (
    DurableEvent,
    DurableLesson,
    MemoryHit,
    MemoryQuery,
    ReflectionCandidate,
    ReflectionQuery,
)
from lethyx.memory.backends.contracts import (
    BackendCapabilities,
    MemoryBlockRecord,
    MemoryEdgeRecord,
    MemoryUsageRecord,
    block_record_to_memory_hit,
    durable_event_to_block_record,
    durable_lesson_to_block_record,
)

MemoryBlockRole = Literal["active", "passive"]
_BLOCK_BACKENDS = {"mempalace", "hindsight", "mnemosyne"}
_BLOCK_ROLES = {"active", "passive"}
_PHASE_WILDCARDS = {"", "*", "all", "any"}
_PASSIVE_SURFACES = {"archive_hint", "reflection_candidate"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _enabled_flag(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _as_tuple(value: Any, *, default: tuple[str, ...] = ("*",)) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [item.strip().lower() for item in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(item).strip().lower() for item in value]
    else:
        parts = [str(value).strip().lower()]
    cleaned = tuple(item for item in parts if item)
    return cleaned or default


def _score(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


_RRF_K = 60.0


def _rank_fusion_score(*, rank: int, weight: float) -> float:
    rank = max(1, int(rank or 1))
    return (_score(weight) or 1.0) / (_RRF_K + float(rank))


def _limit_from_query(query: dict[str, Any]) -> int:
    try:
        return max(1, int(query.get("n") or query.get("limit") or 10))
    except (TypeError, ValueError):
        return 10


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raw = str(value or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _memory_activity_from_record(record: MemoryBlockRecord) -> str:
    raw = (
        str(
            record.metadata.get("memory_activity")
            or record.metadata.get("memory_role")
            or record.metadata.get("activity")
            or ""
        )
        .strip()
        .lower()
    )
    if raw in _BLOCK_ROLES:
        return raw
    if record.surface in _PASSIVE_SURFACES or record.tier == "cold":
        return "passive"
    return ""


def _requested_roles(query: dict[str, Any]) -> set[str]:
    raw = (
        str(query.get("memory_activity") or query.get("memory_role") or query.get("role") or "")
        .strip()
        .lower()
    )
    if raw in _BLOCK_ROLES:
        return {raw}
    if _truthy(query.get("include_passive")) or _truthy(query.get("passive_memory")):
        return {"active", "passive"}
    return {"active"}


def _query_payload(query: MemoryQuery | dict[str, Any]) -> dict[str, Any]:
    if isinstance(query, MemoryQuery):
        payload: dict[str, Any] = {
            "query": query.query,
            "workspace_id": query.workspace_id,
            "run_id": query.run_id,
            "phase_id": query.phase_id,
            "tags": list(query.tags),
            "limit": 10,
            "max_tokens": query.max_tokens,
            "budget": query.budget,
        }
    else:
        payload = dict(query)
    if "workspace_id" not in payload:
        payload["workspace_id"] = ""
    return payload


@dataclass(frozen=True)
class MemoryBackendBlock:
    """One concrete memory technology block in a composite backend."""

    block_id: str
    backend_id: str
    backend: Any
    role: MemoryBlockRole = "active"
    phases: tuple[str, ...] = ("*",)
    weight: float = 1.0
    read: bool = True
    write: bool = True
    enabled: bool = True

    def __post_init__(self) -> None:
        backend_id = str(self.backend_id or "").strip().lower()
        if backend_id not in _BLOCK_BACKENDS:
            raise ValueError(f"unknown memory block backend: {backend_id}")
        role = str(self.role or "active").strip().lower()
        if role not in _BLOCK_ROLES:
            role = "active"
        block_id = str(self.block_id or f"{backend_id}.{role}").strip()
        object.__setattr__(self, "backend_id", backend_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "phases", _as_tuple(self.phases))
        object.__setattr__(self, "weight", _score(self.weight) or 1.0)

    def matches_phase(self, phase_id: str) -> bool:
        phase = str(phase_id or "").strip().lower()
        if any(item in _PHASE_WILDCARDS for item in self.phases):
            return True
        return not phase or phase in self.phases

    def matches_record(self, record: MemoryBlockRecord) -> bool:
        if not self.enabled or not self.write or not self.matches_phase(record.phase_id):
            return False
        activity = _memory_activity_from_record(record)
        return not activity or activity == self.role

    def matches_query(self, query: dict[str, Any]) -> bool:
        if not self.enabled or not self.read:
            return False
        if self.role not in _requested_roles(query):
            return False
        return self.matches_phase(str(query.get("phase_id") or ""))


class CompositeMemoryBackend:
    """Route durable memory operations across real backend blocks."""

    def __init__(
        self,
        repo_root: Path | None,
        *,
        workspace_id: str = "",
        blocks: list[MemoryBackendBlock],
    ) -> None:
        if not blocks:
            raise ValueError("CompositeMemoryBackend requires at least one memory block")
        self._repo_root = Path(repo_root) if repo_root is not None else None
        self._workspace_id = workspace_id
        self._blocks = list(blocks)
        self._last_recall_errors: list[dict[str, Any]] = []

    @property
    def blocks(self) -> tuple[MemoryBackendBlock, ...]:
        return tuple(self._blocks)

    def _recall_error(
        self, block: MemoryBackendBlock, operation: str, exc: Exception
    ) -> dict[str, Any]:
        return {
            "memory_block_id": block.block_id,
            "backend": block.backend_id,
            "operation": operation,
            "error": str(exc),
        }

    def last_recall_diagnostics(self) -> dict[str, Any]:
        return {
            "degraded": bool(self._last_recall_errors),
            "errors": list(self._last_recall_errors),
        }

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id="composite",
            persistent=True,
            supports_embedded=True,
        )

    def health(self) -> dict[str, Any]:
        block_health: list[dict[str, Any]] = []
        ok = True
        strict_ok = True
        for block in self._blocks:
            try:
                health = (
                    dict(block.backend.health())
                    if hasattr(block.backend, "health")
                    else {"ok": True}
                )
            except Exception as exc:
                health = {"ok": False, "error": str(exc)}
            health.update(
                {
                    "memory_block_id": block.block_id,
                    "memory_block_role": block.role,
                    "memory_block_backend": block.backend_id,
                    "phases": list(block.phases),
                    "enabled": block.enabled,
                }
            )
            block_health.append(health)
            ok = ok and bool(health.get("ok", True))
            if health.get("strict_ok") is False or health.get("native_ok") is False:
                strict_ok = False
        return {
            "ok": ok,
            "backend": "composite",
            "implementation": "phase_routed_memory_blocks",
            "workspace_id": self._workspace_id,
            "selected": True,
            "native_ok": strict_ok,
            "strict_ok": strict_ok,
            "effective_mode": "composite",
            "memory_blocks": block_health,
            "last_recall": self.last_recall_diagnostics(),
        }

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]:
        results = []
        for block in self._blocks:
            if not hasattr(block.backend, "ensure_banks"):
                continue
            results.append(
                {
                    "memory_block_id": block.block_id,
                    "backend": block.backend_id,
                    "result": block.backend.ensure_banks(
                        workspace_id=workspace_id or self._workspace_id
                    ),
                }
            )
        return {"ok": True, "backend": "composite", "blocks": results}

    def _eligible_for_record(self, record: MemoryBlockRecord) -> list[MemoryBackendBlock]:
        return [block for block in self._blocks if block.matches_record(record)]

    def _eligible_for_query(self, query: dict[str, Any]) -> list[MemoryBackendBlock]:
        return [block for block in self._blocks if block.matches_query(query)]

    @staticmethod
    def _record_for_block(
        record: MemoryBlockRecord, block: MemoryBackendBlock
    ) -> MemoryBlockRecord:
        metadata = {
            **dict(record.metadata or {}),
            "memory_block_id": block.block_id,
            "memory_block_role": block.role,
            "memory_block_backend": block.backend_id,
        }
        return replace(record, source_backend=block.backend_id, metadata=metadata)

    @staticmethod
    def _hit_for_block(hit: MemoryHit, block: MemoryBackendBlock, *, rank: int = 1) -> MemoryHit:
        raw_score = _score(hit.score)
        fused_score = _rank_fusion_score(rank=rank, weight=block.weight)
        metadata = {
            "source_backend": block.backend_id,
            **dict(hit.metadata or {}),
            "memory_block_id": block.block_id,
            "memory_block_role": block.role,
            "memory_block_backend": block.backend_id,
            "memory_block_weight": block.weight,
            "memory_block_rank": max(1, int(rank or 1)),
            "raw_backend_score": raw_score,
            "memory_block_rrf_score": fused_score,
        }
        return MemoryHit(
            text=hit.text,
            source=hit.source or f"{block.backend_id}:{block.block_id}",
            score=fused_score,
            kind=hit.kind,
            tags=list(hit.tags),
            metadata=metadata,
        )

    @staticmethod
    def _block_for_backend(
        row: MemoryBlockRecord, block: MemoryBackendBlock, *, rank: int = 1
    ) -> MemoryBlockRecord:
        raw_score = _score((row.metadata or {}).get("retrieval_score"))
        fused_score = _rank_fusion_score(rank=rank, weight=block.weight)
        metadata = {
            **dict(row.metadata or {}),
            "memory_block_id": block.block_id,
            "memory_block_role": block.role,
            "memory_block_backend": block.backend_id,
            "memory_block_weight": block.weight,
            "memory_block_rank": max(1, int(rank or 1)),
            "raw_backend_score": raw_score,
            "memory_block_rrf_score": fused_score,
            "retrieval_score": fused_score,
        }
        return replace(row, source_backend=block.backend_id, metadata=metadata)

    @staticmethod
    def _hit_to_block(
        hit: MemoryHit, block: MemoryBackendBlock, query: dict[str, Any]
    ) -> MemoryBlockRecord:
        metadata = dict(hit.metadata or {})
        block_id = str(
            metadata.get("block_id") or hit.source or f"{block.block_id}:{hash(hit.text)}"
        )
        return MemoryBlockRecord(
            block_id=block_id,
            content=hit.text,
            kind=hit.kind or str(metadata.get("kind") or "observation"),
            lifecycle=str(metadata.get("lifecycle") or "active"),
            trust_level=str(metadata.get("trust_level") or "workspace_verified"),
            workspace_id=str(metadata.get("workspace_id") or query.get("workspace_id") or ""),
            run_id=str(metadata.get("run_id") or query.get("run_id") or ""),
            phase_id=str(metadata.get("phase_id") or query.get("phase_id") or ""),
            source_backend=block.backend_id,
            tags=tuple(hit.tags),
            metadata={
                **metadata,
                "native_score": hit.score,
                "memory_block_id": block.block_id,
                "memory_block_role": block.role,
                "memory_block_backend": block.backend_id,
            },
            verified=True,
        )

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        targets = self._eligible_for_record(record)
        results: list[dict[str, Any]] = []
        written = 0
        for block in targets:
            routed = self._record_for_block(record, block)
            try:
                result = dict(block.backend.retain_block(routed))
            except Exception as exc:
                result = {"ok": False, "error": str(exc), "backend": block.backend_id}
            result.update(
                {
                    "memory_block_id": block.block_id,
                    "memory_block_role": block.role,
                    "memory_block_backend": block.backend_id,
                }
            )
            if result.get("ok"):
                written += 1
            results.append(result)
        return {
            "ok": bool(targets) and all(bool(row.get("ok")) for row in results),
            "backend": "composite",
            "block_id": record.block_id,
            "n_written": written,
            "blocks": results,
        }

    def retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        grouped: dict[str, tuple[MemoryBackendBlock, list[MemoryBlockRecord]]] = {}
        record_results: list[dict[str, Any]] = []
        written = 0

        for record in records:
            targets = self._eligible_for_record(record)
            record_results.append(
                {
                    "block_id": record.block_id,
                    "targets": [block.block_id for block in targets],
                }
            )
            for block in targets:
                routed = self._record_for_block(record, block)
                key = block.block_id
                if key not in grouped:
                    grouped[key] = (block, [])
                grouped[key][1].append(routed)

        block_results: list[dict[str, Any]] = []
        for block, routed_records in grouped.values():
            if hasattr(block.backend, "retain_blocks"):
                try:
                    result = dict(block.backend.retain_blocks(routed_records))
                except Exception as exc:
                    result = {"ok": False, "error": str(exc), "backend": block.backend_id}
            else:
                child_results = []
                child_written = 0
                for routed in routed_records:
                    try:
                        row = dict(block.backend.retain_block(routed))
                    except Exception as exc:
                        row = {"ok": False, "error": str(exc), "backend": block.backend_id}
                    if row.get("ok"):
                        child_written += 1
                    child_results.append(row)
                result = {
                    "ok": all(bool(row.get("ok")) for row in child_results),
                    "backend": block.backend_id,
                    "n_written": child_written,
                    "rows": child_results,
                }
            result.update(
                {
                    "memory_block_id": block.block_id,
                    "memory_block_role": block.role,
                    "memory_block_backend": block.backend_id,
                }
            )
            if result.get("ok"):
                written += int(result.get("n_written") or len(routed_records))
            block_results.append(result)

        return {
            "ok": bool(grouped) and all(bool(row.get("ok")) for row in block_results),
            "backend": "composite",
            "n_written": written,
            "records": len(records),
            "record_targets": record_results,
            "blocks": block_results,
        }

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        for block in self._blocks:
            if hasattr(block.backend, "get_block"):
                found = block.backend.get_block(block_id)
                if found is not None:
                    return self._block_for_backend(found, block)
        return None

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        payload = _query_payload(query)
        limit = _limit_from_query(payload)
        rows: list[MemoryBlockRecord] = []
        errors: list[dict[str, Any]] = []
        for block in self._eligible_for_query(payload):
            if hasattr(block.backend, "recall"):
                try:
                    rows.extend(
                        self._block_for_backend(row, block, rank=rank)
                        for rank, row in enumerate(block.backend.recall(payload), start=1)
                    )
                    continue
                except Exception as exc:
                    errors.append(self._recall_error(block, "recall", exc))
            if hasattr(block.backend, "recall_evidence"):
                try:
                    rows.extend(
                        self._hit_to_block(
                            self._hit_for_block(hit, block, rank=rank), block, payload
                        )
                        for rank, hit in enumerate(block.backend.recall_evidence(payload), start=1)
                    )
                except Exception as exc:
                    errors.append(self._recall_error(block, "recall_evidence", exc))
        self._last_recall_errors = errors
        deduped: dict[str, MemoryBlockRecord] = {}
        for row in rows:
            key = row.block_id or row.content
            current = deduped.get(key)
            if current is None or _score(row.metadata.get("memory_block_rrf_score")) > _score(
                current.metadata.get("memory_block_rrf_score")
            ):
                deduped[key] = row
        return sorted(
            deduped.values(),
            key=lambda row: _score(row.metadata.get("memory_block_rrf_score")),
            reverse=True,
        )[:limit]

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        payload = _query_payload(query)
        limit = _limit_from_query(payload)
        hits: list[MemoryHit] = []
        errors: list[dict[str, Any]] = []
        for block in self._eligible_for_query(payload):
            if hasattr(block.backend, "recall_evidence"):
                try:
                    hits.extend(
                        self._hit_for_block(hit, block, rank=rank)
                        for rank, hit in enumerate(block.backend.recall_evidence(payload), start=1)
                    )
                    continue
                except Exception as exc:
                    errors.append(self._recall_error(block, "recall_evidence", exc))
            if hasattr(block.backend, "recall"):
                try:
                    hits.extend(
                        self._hit_for_block(block_record_to_memory_hit(row), block, rank=rank)
                        for rank, row in enumerate(block.backend.recall(payload), start=1)
                    )
                except Exception as exc:
                    errors.append(self._recall_error(block, "recall", exc))
        self._last_recall_errors = errors
        merged: dict[str, MemoryHit] = {}
        for hit in hits:
            key = str(hit.source or hit.metadata.get("block_id") or hit.text)
            current = merged.get(key)
            if current is None or _score(hit.score) > _score(current.score):
                merged[key] = hit
        ranked = sorted(merged.values(), key=lambda item: _score(item.score), reverse=True)
        return ranked[:limit]

    def update_lifecycle(
        self,
        block_id: str,
        lifecycle: str,
        *,
        reason: str = "",
        evidence_refs: list[Any] | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        results = []
        changed = 0
        for block in self._blocks:
            if not hasattr(block.backend, "update_lifecycle"):
                continue
            try:
                result = dict(
                    block.backend.update_lifecycle(
                        block_id,
                        lifecycle,
                        reason=reason,
                        evidence_refs=evidence_refs,
                        surface=surface,
                    )
                )
            except Exception as exc:
                result = {"ok": False, "error": str(exc), "backend": block.backend_id}
            if result.get("ok"):
                changed += 1
            result["memory_block_id"] = block.block_id
            results.append(result)
        return {"ok": changed > 0, "backend": "composite", "updated": changed, "blocks": results}

    def expire_scope(self, scope: str, scope_id: str = "", *, run_id: str = "") -> int:
        removed = 0
        for block in self._blocks:
            if hasattr(block.backend, "expire_scope"):
                removed += int(block.backend.expire_scope(scope, scope_id, run_id=run_id) or 0)
        return removed

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        results = []
        for block in self._blocks:
            if not hasattr(block.backend, "add_edge"):
                continue
            edge_for_block = replace(
                edge, metadata={**dict(edge.metadata), "memory_block_id": block.block_id}
            )
            results.append(block.backend.add_edge(edge_for_block))
        return {"ok": True, "backend": "composite", "blocks": results}

    def edges_for(self, block_id: str) -> list[MemoryEdgeRecord]:
        edges: list[MemoryEdgeRecord] = []
        for block in self._blocks:
            if hasattr(block.backend, "edges_for"):
                edges.extend(block.backend.edges_for(block_id))
        return edges

    def record_usage(self, usage: MemoryUsageRecord) -> dict[str, Any]:
        results = []
        for block in self._blocks:
            if hasattr(block.backend, "record_usage"):
                routed = replace(
                    usage, metadata={**dict(usage.metadata), "memory_block_id": block.block_id}
                )
                results.append(block.backend.record_usage(routed))
        return {"ok": True, "backend": "composite", "blocks": results}

    def value_stats(self, block_id: str) -> dict[str, Any]:
        stats = []
        for block in self._blocks:
            if hasattr(block.backend, "value_stats"):
                stats.append(
                    {
                        "memory_block_id": block.block_id,
                        "stats": block.backend.value_stats(block_id),
                    }
                )
        return {"backend": "composite", "blocks": stats}

    def export_blocks(self) -> list[MemoryBlockRecord]:
        rows: list[MemoryBlockRecord] = []
        for block in self._blocks:
            if hasattr(block.backend, "export_blocks"):
                rows.extend(
                    self._block_for_backend(row, block) for row in block.backend.export_blocks()
                )
        return rows

    def import_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        return self.retain_blocks(records)

    def compile_active_overlay(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> dict[str, Any]:
        directive_blocks: list[str] = []
        candidate_blocks: list[str] = []
        details: list[dict[str, Any]] = []
        for block in self._blocks:
            if block.role != "active" or not block.matches_phase(phase_id):
                continue
            if not hasattr(block.backend, "compile_active_overlay"):
                continue
            overlay = dict(
                block.backend.compile_active_overlay(
                    workspace_id=workspace_id or self._workspace_id,
                    run_id=run_id,
                    phase_id=phase_id,
                )
            )
            directive_blocks.extend(str(item) for item in overlay.get("directive_blocks") or [])
            candidate_blocks.extend(str(item) for item in overlay.get("candidate_blocks") or [])
            details.append(
                {"memory_block_id": block.block_id, "backend": block.backend_id, "overlay": overlay}
            )
        return {
            "backend": "composite",
            "directive_blocks": directive_blocks,
            "candidate_blocks": candidate_blocks,
            "memory_blocks": details,
        }

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for block in self._blocks:
            if block.role != "active" or not block.matches_phase(phase_id):
                continue
            if not hasattr(block.backend, "render_prompt_items"):
                continue
            for item in block.backend.render_prompt_items(
                workspace_id=workspace_id or self._workspace_id,
                run_id=run_id,
                phase_id=phase_id,
            ):
                items.append(
                    {
                        **dict(item),
                        "memory_block_id": block.block_id,
                        "memory_block_role": block.role,
                        "memory_block_backend": block.backend_id,
                    }
                )
        return items

    def retain_event(self, event: DurableEvent | Any) -> dict[str, Any]:
        return self.retain_block(durable_event_to_block_record(event, backend_id="composite"))

    def retain_lesson(self, lesson: DurableLesson | Any) -> dict[str, Any]:
        return self.retain_block(durable_lesson_to_block_record(lesson, backend_id="composite"))

    def reflect_candidates(
        self, query: ReflectionQuery | dict[str, Any]
    ) -> list[ReflectionCandidate]:
        payload = _query_payload(
            {
                "query": query.question
                if isinstance(query, ReflectionQuery)
                else dict(query).get("question", ""),
                "workspace_id": query.workspace_id
                if isinstance(query, ReflectionQuery)
                else dict(query).get("workspace_id", ""),
                "run_id": query.run_id
                if isinstance(query, ReflectionQuery)
                else dict(query).get("run_id", ""),
                "phase_id": query.phase_id
                if isinstance(query, ReflectionQuery)
                else dict(query).get("phase_id", "reflexion"),
                "include_passive": True,
            }
        )
        candidates: list[ReflectionCandidate] = []
        for block in self._eligible_for_query(payload):
            if hasattr(block.backend, "reflect_candidates"):
                candidates.extend(block.backend.reflect_candidates(query))
        return candidates

    def close(self) -> None:
        for block in self._blocks:
            if hasattr(block.backend, "close"):
                block.backend.close()


def memory_block_specs_from_env() -> list[dict[str, Any]]:
    raw = os.getenv("LETHYX_MEMORY_BLOCKS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, dict):
            parsed = parsed.get("blocks") or []
        if isinstance(parsed, list):
            specs = [dict(item) for item in parsed if isinstance(item, dict)]
            valid = [
                item
                for item in specs
                if str(item.get("backend") or item.get("backend_id") or "").strip().lower()
                in _BLOCK_BACKENDS
            ]
            if valid:
                return valid
    return [
        {
            "id": "mempalace.active",
            "backend": "mempalace",
            "role": "active",
            "phases": ["*"],
            "weight": 1.0,
        },
        {
            "id": "hindsight.passive",
            "backend": "hindsight",
            "role": "passive",
            "phases": ["research", "plan", "verify", "final_review", "reflexion"],
            "weight": 0.95,
        },
        {
            "id": "mnemosyne.passive",
            "backend": "mnemosyne",
            "role": "passive",
            "phases": ["execute", "verify", "final_review", "reflexion"],
            "weight": 0.9,
        },
    ]


def build_memory_blocks_from_specs(
    specs: list[dict[str, Any]],
    *,
    backend_factory: Any,
) -> list[MemoryBackendBlock]:
    blocks: list[MemoryBackendBlock] = []
    for spec in specs:
        backend_id = str(spec.get("backend") or spec.get("backend_id") or "").strip().lower()
        if backend_id not in _BLOCK_BACKENDS:
            continue
        enabled = not (
            spec.get("enabled") is False
            or str(spec.get("enabled") or "").strip().lower() in {"0", "false", "no", "off"}
        )
        if not enabled:
            continue
        role_raw = str(spec.get("role") or spec.get("memory_role") or "active").strip().lower()
        role: MemoryBlockRole = "passive" if role_raw == "passive" else "active"
        backend = spec.get("backend_instance") or backend_factory(backend_id)
        blocks.append(
            MemoryBackendBlock(
                block_id=str(spec.get("id") or spec.get("block_id") or f"{backend_id}.{role}"),
                backend_id=backend_id,
                backend=backend,
                role=role,
                phases=_as_tuple(spec.get("phases") or spec.get("phase_ids")),
                weight=_score(spec.get("weight")) or 1.0,
                read=_enabled_flag(spec.get("read")),
                write=_enabled_flag(spec.get("write")),
                enabled=enabled,
            )
        )
    return blocks
