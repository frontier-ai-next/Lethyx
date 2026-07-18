"""MemPalace durable backend adapter.

Public backend id is ``mempalace``. Physical Chroma/SQLite collection names may
still use the historical palace layout internally.
"""

import json
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

from lethyx.memory.backends.base import DurableEvent, DurableLesson, MemoryHit, MemoryQuery
from lethyx.memory.backends.contracts import (
    BackendCapabilities,
    MemoryBlockRecord,
    MemoryEdgeRecord,
    MemoryUsageRecord,
    block_record_to_memory_hit,
    durable_event_to_block_record,
    durable_lesson_to_block_record,
    memory_event_to_block_record,
)
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.palace.facade import MemPalace
from lethyx.memory.palace.tiers import Scope, Tier


def _index_path(repo_root: Path, workspace_id: str) -> Path:
    if workspace_id:
        return (
            repo_root
            / "workspaces"
            / workspace_id
            / ".memory"
            / "mempalace"
            / "LETHYX_index.sqlite"
        )
    return repo_root / ".Lethyx" / "memory" / "mempalace_LETHYX_index.sqlite"


def _store_for_block(record: MemoryBlockRecord) -> str:
    requested = str(record.metadata.get("palace_store") or "").strip()
    if requested:
        return requested
    if record.kind in {"durable", "verification_report"}:
        return "palace.durable"
    if record.kind in {"lesson", "bkb_rule"}:
        return "palace.lesson"
    if record.kind == "codeptr":
        return "palace.codeptr"
    if record.kind == "skill_index":
        return "palace.skill_index"
    if record.kind == "run_state" or record.surface == "current_run_state":
        return "palace.run"
    return "palace.idea"


def _record_to_palace_kwargs(record: MemoryBlockRecord) -> dict[str, Any]:
    source_backend = record.source_backend or "mempalace"
    return {
        "store": _store_for_block(record),
        "content": record.content,
        "tier": record.tier or Tier.WARM,
        "scope": record.scope or Scope.RUN_SCOPED,
        "tags": list(record.tags),
        "phase": record.phase_id or None,
        "subtask_id": record.subtask_id or None,
        "run_id": record.run_id or None,
        "source_path": str(
            record.metadata.get("source_path")
            or record.metadata.get("evidence_id")
            or "memory_block_record"
        ),
        "verified": record.verified,
        "kind": record.kind,
        "extra": {
            "title": record.title,
            "kind": record.kind,
            "lifecycle": record.lifecycle,
            "trust_level": record.trust_level,
            "surface": record.surface,
            "source_backend": source_backend,
            "storage_backend": "mempalace",
            "evidence_refs_json": json.dumps(
                [asdict(ref) for ref in record.evidence_refs],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "external_refs_json": json.dumps(
                record.external_refs, ensure_ascii=False, sort_keys=True
            ),
            "privacy_class": record.privacy_class,
            "risk_level": record.risk_level,
            "metadata_json": json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
            "evidence_id": str(record.metadata.get("evidence_id") or ""),
            "producer": record.producer,
        },
        "node_id": record.block_id,
    }


class MemPalaceBackend:
    def __init__(self, repo_root: Path, workspace_id: str = "") -> None:
        self._repo_root = Path(repo_root)
        self._workspace_id = workspace_id
        self._palace = MemPalace(self._repo_root, workspace_id or None)
        self._index = BackendSQLiteIndex(
            _index_path(self._repo_root, workspace_id),
            backend_id="mempalace",
            workspace_id=workspace_id,
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(backend_id="mempalace", persistent=True, supports_embedded=True)

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]:
        return {
            "ok": True,
            "backend": "mempalace",
            "workspace_id": workspace_id or self._workspace_id,
        }

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        native_record = record
        native_id = self._palace.add(**_record_to_palace_kwargs(native_record))
        record = replace(record, source_backend="mempalace")
        result = self._index.retain_block(record)
        return {
            **result,
            "store": _store_for_block(native_record),
            "native_id": native_id,
            "native_written": True,
        }

    def retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        """Batch-retain memory blocks into the native Palace store, then index metadata."""
        if not records:
            return {"ok": True, "n_written": 0, "native_written": 0}

        try:
            grouped: dict[str, dict[str, list[Any]]] = {}
            created_at = time.time()
            for record in records:
                kwargs = _record_to_palace_kwargs(record)
                store = str(kwargs["store"])
                tags = list(kwargs.get("tags") or [])
                node_id = str(kwargs.get("node_id") or record.block_id)
                metadata: dict[str, Any] = {
                    "id": node_id,
                    "store": store,
                    "tier": kwargs.get("tier") or Tier.WARM,
                    "scope": kwargs.get("scope") or Scope.RUN_SCOPED,
                    "tags": ",".join(str(tag) for tag in tags),
                    "phase": kwargs.get("phase") or "",
                    "subtask_id": kwargs.get("subtask_id") or "",
                    "verified": int(bool(kwargs.get("verified"))),
                    "source_path": kwargs.get("source_path") or "",
                    "workspace_id": self._workspace_id,
                    "run_id": kwargs.get("run_id") or "",
                    "created_at": created_at,
                }
                for key, value in dict(kwargs.get("extra") or {}).items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                bucket = grouped.setdefault(store, {"ids": [], "documents": [], "metadatas": []})
                bucket["ids"].append(node_id)
                bucket["documents"].append(str(kwargs.get("content") or ""))
                bucket["metadatas"].append(metadata)

            for store, payload in grouped.items():
                collection = self._palace._stores.chroma(store)
                collection.add(
                    ids=payload["ids"],
                    documents=payload["documents"],
                    metadatas=payload["metadatas"],
                )
        except Exception as exc:
            raise RuntimeError(f"mempalace native batch write failed: {exc}") from exc

        written = 0
        first_result: dict[str, Any] = {"ok": True}
        for native_record in records:
            row = replace(native_record, source_backend="mempalace")
            result = self._index.retain_block(row)
            first_result = result if written == 0 else first_result
            if result.get("ok"):
                written += 1
        return {**first_result, "n_written": written, "native_written": len(records)}

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self._index.get_block(block_id, workspace_id=self._workspace_id)

    @staticmethod
    def _tags_from_any(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(tag.strip() for tag in value.split(",") if tag.strip())
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(str(tag).strip() for tag in value if str(tag).strip())
        return ()

    @staticmethod
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

    @staticmethod
    def _json_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return list(value)
        raw = str(value or "").strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list(parsed) if isinstance(parsed, list) else []

    def _palace_node_to_block(self, node: dict[str, Any]) -> MemoryBlockRecord:
        metadata = self._json_dict(node.get("metadata_json"))
        metadata.update(
            {
                "native_backend": "mempalace",
                "native_score": node.get("score"),
                "palace_store": node.get("store"),
                "source_path": node.get("source_path") or metadata.get("source_path") or "",
            }
        )
        external_refs = self._json_dict(node.get("external_refs_json"))
        return MemoryBlockRecord(
            block_id=str(node.get("id") or metadata.get("block_id") or ""),
            content=str(node.get("content") or ""),
            title=str(metadata.get("title") or ""),
            kind=str(node.get("kind") or metadata.get("kind") or "observation"),
            lifecycle=str(node.get("lifecycle") or metadata.get("lifecycle") or "active"),
            trust_level=str(
                node.get("trust_level") or metadata.get("trust_level") or "workspace_verified"
            ),
            scope=str(node.get("scope") or metadata.get("scope") or "run_scoped"),
            tier=str(node.get("tier") or metadata.get("tier") or "warm"),
            surface=str(node.get("surface") or metadata.get("surface") or "supplemental_evidence"),
            workspace_id=str(node.get("workspace_id") or self._workspace_id),
            run_id=str(node.get("run_id") or metadata.get("run_id") or ""),
            phase_id=str(node.get("phase") or metadata.get("phase_id") or ""),
            subtask_id=str(node.get("subtask_id") or metadata.get("subtask_id") or ""),
            producer=str(node.get("producer") or metadata.get("producer") or "mempalace"),
            source_backend="mempalace",
            tags=self._tags_from_any(node.get("tags") or metadata.get("tags")),
            evidence_refs=self._json_list(node.get("evidence_refs_json")),
            external_refs=external_refs,
            privacy_class=str(metadata.get("privacy_class") or "normal"),
            risk_level=str(metadata.get("risk_level") or "normal"),
            metadata=metadata,
            verified=bool(node.get("verified", metadata.get("verified", True))),
        )

    @staticmethod
    def _record_allowed_for_query(record: MemoryBlockRecord, query: dict[str, Any]) -> bool:
        if record.lifecycle in {
            "quarantined",
            "superseded",
            "retracted",
            "deprecated",
            "archived",
        }:
            return False
        if not bool(query.get("include_candidates", False)) and record.lifecycle == "candidate":
            return False
        workspace_id = str(query.get("workspace_id") or "")
        if workspace_id and record.workspace_id and record.workspace_id != workspace_id:
            return False
        run_id = str(query.get("run_id") or "")
        if run_id and record.run_id != run_id:
            return False
        phase_id = str(query.get("phase_id") or "")
        if phase_id and record.phase_id != phase_id:
            return False
        subtask_id = str(query.get("subtask_id") or "")
        if subtask_id and record.subtask_id != subtask_id:
            return False
        current_run_id = str(query.get("current_run_id") or "")
        if (
            current_run_id
            and record.surface == "current_run_state"
            and record.run_id
            and record.run_id != current_run_id
        ):
            return False
        return True

    @staticmethod
    def _query_payload(query: MemoryQuery | dict[str, Any]) -> dict[str, Any]:
        if isinstance(query, dict):
            return dict(query)
        if is_dataclass(query):
            return asdict(query)
        return dict(getattr(query, "__dict__", {}) or {})

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        payload = self._query_payload(query)
        text = str(payload.get("query") or "").strip()
        try:
            limit = max(1, int(payload.get("limit") or payload.get("n") or 10))
        except (TypeError, ValueError):
            limit = 10
        if text:
            rows = self._palace.search(
                text,
                n=limit,
                tags_any=list(self._tags_from_any(payload.get("tags"))) or None,
            )
        else:
            rows = self._palace.list_all(n=limit)
        return [
            block
            for block in (self._palace_node_to_block(dict(row)) for row in rows)
            if self._record_allowed_for_query(block, payload)
        ][:limit]

    def update_lifecycle(
        self,
        block_id: str,
        lifecycle: str,
        *,
        reason: str = "",
        evidence_refs: list[Any] | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        return self._index.update_lifecycle(
            block_id, lifecycle, reason=reason, evidence_refs=evidence_refs, surface=surface
        )

    def expire_scope(self, scope: str, scope_id: str = "", *, run_id: str = "") -> int:
        return self._index.expire_scope(scope, scope_id, run_id=run_id)

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        return self._index.add_edge(edge)

    def edges_for(self, block_id: str) -> list[MemoryEdgeRecord]:
        return self._index.edges_for(block_id)

    def record_usage(self, usage: MemoryUsageRecord) -> dict[str, Any]:
        return self._index.record_usage(usage)

    def value_stats(self, block_id: str) -> dict[str, Any]:
        return self._index.value_stats(block_id)

    def export_blocks(self) -> list[MemoryBlockRecord]:
        return self._index.export_blocks()

    def import_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        return self._index.import_blocks(records)

    def compile_active_overlay(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> dict[str, Any]:
        return self._index.compile_active_overlay(
            workspace_id=workspace_id or self._workspace_id, run_id=run_id, phase_id=phase_id
        )

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> list[dict[str, Any]]:
        return self._index.render_prompt_items(
            workspace_id=workspace_id or self._workspace_id, run_id=run_id, phase_id=phase_id
        )

    def retain_event(self, event: DurableEvent | dict[str, Any] | Any) -> dict[str, Any]:
        data = event if isinstance(event, dict) else getattr(event, "__dict__", {})
        if "memory_kind" in data:
            record = memory_event_to_block_record(event, backend_id="mempalace")
        else:
            record = durable_event_to_block_record(event, backend_id="mempalace")
        return self.retain_block(record)

    def retain_lesson(self, lesson: DurableLesson | dict[str, Any] | Any) -> dict[str, Any]:
        data = lesson if isinstance(lesson, dict) else getattr(lesson, "__dict__", {})
        if "memory_kind" in data:
            record = memory_event_to_block_record(lesson, backend_id="mempalace")
        else:
            record = durable_lesson_to_block_record(lesson, backend_id="mempalace")
        return self.retain_block(record)

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        return [block_record_to_memory_hit(row) for row in self.recall(query)]

    def reflect_candidates(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def health(self) -> dict[str, Any]:
        try:
            native = dict(self._palace.health())
        except Exception as exc:
            native = {"ok": False, "error": str(exc)}
        native_ok = bool(native.get("ok", False)) and not bool(native.get("volatile_stub"))
        reason = (
            ""
            if native_ok
            else str(native.get("error") or native.get("stores_fail") or "native_unavailable")
        )
        return {
            "ok": native_ok,
            "backend": "mempalace",
            "implementation": "mempalace_chroma",
            "selected": True,
            "native_ok": native_ok,
            "strict_ok": native_ok,
            "effective_mode": "native" if native_ok else "unavailable",
            "reason": reason,
            "native": native,
        }

    def close(self) -> None:
        self._palace.close()
        self._index.close()


# Backward import compatibility for old tests/importers during migration.
CanonicalMemoryBackend = MemPalaceBackend
