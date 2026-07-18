"""Mnemosyne selected durable backend adapter.

The optional upstream package can be present for richer native behavior.  When
native Mnemosyne is enabled and importable, this adapter writes benchmark/runtime
blocks into Mnemosyne's own BEAM SQLite store and retrieves through its native
hybrid recall.  A backend-owned SQLite index is still maintained for Lethyx's
neutral contracts (metadata round-trip, lifecycle records, export/import), but
native recall does not read from that parity index.
"""

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    durable_event_to_block_record,
    durable_lesson_to_block_record,
)
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


def _enabled() -> bool:
    return os.getenv("LETHYX_MNEMOSYNE_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class MnemosyneBackend:
    def __init__(
        self,
        repo_root: Path,
        workspace_id: str = "",
        data_dir: Path | None = None,
        enabled: bool | None = None,
    ):
        self.repo_root = Path(repo_root)
        self.workspace_id = workspace_id
        self.enabled = _enabled() if enabled is None else enabled
        self.available = importlib.util.find_spec("mnemosyne") is not None
        self.data_dir = data_dir or Path(
            os.getenv("LETHYX_MNEMOSYNE_DATA_DIR")
            or self.repo_root / "workspaces" / (workspace_id or "default") / ".memory" / "mnemosyne"
        )
        self._index = BackendSQLiteIndex(
            self.data_dir / "LETHYX_index.sqlite",
            backend_id="mnemosyne",
            workspace_id=workspace_id,
        )
        self._native_memory: Any | None = None

    @classmethod
    def from_env(cls, repo_root: Path, workspace_id: str = "") -> "MnemosyneBackend":
        return cls(repo_root=repo_root, workspace_id=workspace_id)

    def _native_ok(self) -> bool:
        return bool(self.enabled and self.available)

    def _require_native(self) -> None:
        if not self.enabled:
            raise RuntimeError("native Mnemosyne is disabled")
        if not self.available:
            raise RuntimeError("native Mnemosyne dependency is not installed")

    def _native(self) -> Any:
        if not self._native_ok():
            self._require_native()
        if self._native_memory is None:
            from mnemosyne import Mnemosyne

            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._native_memory = Mnemosyne(
                session_id=self.workspace_id or "default",
                db_path=self.data_dir / "mnemosyne.db",
            )
        return self._native_memory

    @staticmethod
    def _record_evidence_id(record: MemoryBlockRecord) -> str:
        return str(
            record.metadata.get("evidence_id")
            or record.metadata.get("source_path")
            or record.external_refs.get("source_path")
            or record.external_refs.get("evidence_id")
            or record.block_id
        )

    @staticmethod
    def _importance(record: MemoryBlockRecord) -> float:
        raw = record.metadata.get("importance")
        if raw is None:
            return 0.8 if record.verified else 0.5
        try:
            return max(0.0, min(float(raw), 1.0))
        except (TypeError, ValueError):
            return 0.8 if record.verified else 0.5

    @staticmethod
    def _native_scope(record: MemoryBlockRecord) -> str:
        if record.scope in {"manager", "workspace", "cross_run_durable"}:
            return "global"
        return "session"

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _native_metadata_for(self, native_id: str) -> dict[str, Any]:
        native = self._native()
        try:
            row = native.get(native_id)
        except Exception:
            row = None
        if not isinstance(row, dict):
            return {}
        raw = row.get("metadata_json") or row.get("metadata") or {}
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return dict(raw) if isinstance(raw, dict) else {}

    def _native_row_to_block(self, row: Any) -> MemoryBlockRecord:
        item = dict(row) if isinstance(row, dict) else {}
        native_id = str(item.get("id") or "")
        metadata = self._native_metadata_for(native_id) if native_id else {}
        evidence_id = str(metadata.get("evidence_id") or item.get("source") or native_id)
        block_id = str(metadata.get("block_id") or metadata.get("record_id") or native_id)
        merged_metadata = {
            **metadata,
            "evidence_id": evidence_id,
            "mnemosyne_native_id": native_id,
            "native_backend": "mnemosyne",
            "native_score": item.get("score"),
            "native_tier": item.get("tier"),
            "native_source": item.get("source"),
        }
        return MemoryBlockRecord(
            block_id=block_id,
            content=str(item.get("content") or ""),
            title=str(metadata.get("title") or ""),
            kind=str(metadata.get("kind") or "observation"),
            lifecycle=str(metadata.get("lifecycle") or "active"),
            trust_level=str(metadata.get("trust_level") or "workspace_verified"),
            scope=str(metadata.get("scope") or "run_scoped"),
            tier=str(metadata.get("tier") or "warm"),
            surface=str(metadata.get("surface") or "supplemental_evidence"),
            workspace_id=str(metadata.get("workspace_id") or self.workspace_id),
            run_id=str(metadata.get("run_id") or ""),
            phase_id=str(metadata.get("phase_id") or ""),
            subtask_id=str(metadata.get("subtask_id") or ""),
            producer=str(metadata.get("producer") or "mnemosyne"),
            source_backend="mnemosyne",
            tags=tuple(metadata.get("tags") or ()),
            external_refs=dict(metadata.get("external_refs") or {}),
            privacy_class=str(metadata.get("privacy_class") or "normal"),
            risk_level=str(metadata.get("risk_level") or "normal"),
            token_budget=int(metadata.get("token_budget") or 0),
            metadata=merged_metadata,
            verified=bool(metadata.get("verified", True)),
            created_at=str(metadata.get("created_at") or item.get("timestamp") or ""),
            updated_at=str(metadata.get("updated_at") or item.get("timestamp") or ""),
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id="mnemosyne",
            persistent=True,
            supports_embedded=True,
            requires_external_service=False,
        )

    def health(self) -> dict[str, Any]:
        native_ok = self._native_ok()
        effective_mode = "native" if native_ok else "unavailable"
        reason = ""
        if not self.enabled:
            reason = "native_disabled"
        elif not self.available:
            reason = "dependency_unavailable"
        return {
            "ok": native_ok,
            "backend": "mnemosyne",
            "selected": True,
            "enabled": self.enabled,
            "available": self.available,
            "native_ok": native_ok,
            "strict_ok": native_ok,
            "effective_mode": effective_mode,
            "reason": reason,
            "parity_ok": False,
            "operational_ok": native_ok,
            "data_dir": str(self.data_dir),
            "workspace_id": self.workspace_id,
            "provider_prompt_injection_used": False,
            "implementation": "mnemosyne_memory_beam" if native_ok else "unavailable",
            "operational_note": ""
            if native_ok
            else (
                "Mnemosyne native layer is unavailable; the selected "
                "Mnemosyne backend fails closed instead of using the "
                "Lethyx SQLite parity index as a recall implementation."
            ),
        }

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._native()
        return {
            "ok": True,
            "backend": "mnemosyne",
            "data_dir": str(self.data_dir),
            "workspace_id": workspace_id or self.workspace_id,
            "native": True,
        }

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        evidence_id = self._record_evidence_id(record)
        metadata = {
            **dict(record.metadata or {}),
            "block_id": record.block_id,
            "evidence_id": evidence_id,
            "title": record.title,
            "kind": record.kind,
            "lifecycle": record.lifecycle,
            "trust_level": record.trust_level,
            "scope": record.scope,
            "tier": record.tier,
            "surface": record.surface,
            "workspace_id": record.workspace_id or self.workspace_id,
            "run_id": record.run_id,
            "phase_id": record.phase_id,
            "subtask_id": record.subtask_id,
            "producer": record.producer,
            "tags": list(record.tags),
            "external_refs": dict(record.external_refs),
            "privacy_class": record.privacy_class,
            "risk_level": record.risk_level,
            "token_budget": record.token_budget,
            "verified": record.verified,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        native_id = self._native().remember(
            record.content,
            source=evidence_id,
            importance=self._importance(record),
            metadata=metadata,
            scope=self._native_scope(record),
            valid_until=record.expires_at or None,
            trust_tier=record.trust_level or None,
        )
        indexed = replace(
            record,
            source_backend="mnemosyne",
            metadata={
                **metadata,
                "mnemosyne_native_id": native_id,
                "native_backend": "mnemosyne",
            },
            external_refs={
                **dict(record.external_refs),
                "mnemosyne_native_id": native_id,
            },
        )
        self._index.retain_block(indexed)
        return {
            "ok": True,
            "backend": "mnemosyne",
            "native": True,
            "block_id": record.block_id,
            "memory_id": native_id,
        }

    def retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        n_written = 0
        for record in records:
            result = self.retain_block(record)
            if result.get("ok"):
                n_written += 1
        return {
            "ok": True,
            "backend": "mnemosyne",
            "native": self._native_ok(),
            "n_written": n_written,
        }

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self._index.get_block(block_id, workspace_id=self.workspace_id)

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        if isinstance(query, MemoryQuery):
            payload = {
                "query": query.query,
                "workspace_id": query.workspace_id,
                "run_id": query.run_id,
                "phase_id": query.phase_id,
                "limit": 10,
                "token_budget": query.max_tokens,
            }
        else:
            payload = dict(query)
        n = int(payload.get("n") or payload.get("limit") or 10)
        rows = self._native().recall(str(payload.get("query") or ""), top_k=n)
        return [self._native_row_to_block(row) for row in rows]

    def update_lifecycle(
        self,
        block_id: str,
        lifecycle: str,
        *,
        reason: str = "",
        evidence_refs: list[Any] | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        if self._native_ok() and lifecycle in {
            "quarantined",
            "superseded",
            "deprecated",
            "retracted",
            "archived",
        }:
            stored = self._index.get_block(block_id, workspace_id=self.workspace_id)
            native_id = ""
            if stored is not None:
                native_id = str(
                    stored.metadata.get("mnemosyne_native_id")
                    or stored.external_refs.get("mnemosyne_native_id")
                    or ""
                )
            if native_id:
                try:
                    self._native().invalidate(native_id)
                except Exception:
                    pass
        return self._index.update_lifecycle(
            block_id,
            lifecycle,
            reason=reason,
            evidence_refs=evidence_refs,
            surface=surface,
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
        target_records = [
            replace(record, workspace_id=self.workspace_id) if self.workspace_id else record
            for record in records
        ]
        stats = self.retain_blocks(target_records)
        return {
            "ok": True,
            "imported": stats.get("n_written", 0),
            "n_imported": stats.get("n_written", 0),
        }

    def compile_active_overlay(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> dict[str, Any]:
        return self._index.compile_active_overlay(
            workspace_id=workspace_id or self.workspace_id,
            run_id=run_id,
            phase_id=phase_id,
        )

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = "", phase_id: str = ""
    ) -> list[dict[str, Any]]:
        return self._index.render_prompt_items(
            workspace_id=workspace_id or self.workspace_id,
            run_id=run_id,
            phase_id=phase_id,
        )

    def retain_event(self, event: DurableEvent | Any) -> dict[str, Any]:
        return self.retain_block(durable_event_to_block_record(event, backend_id="mnemosyne"))

    def retain_lesson(self, lesson: DurableLesson | Any) -> dict[str, Any]:
        return self.retain_block(durable_lesson_to_block_record(lesson, backend_id="mnemosyne"))

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for row in self.recall(query):
            evidence_id = str(row.metadata.get("evidence_id") or row.block_id)
            hits.append(
                MemoryHit(
                    text=row.content,
                    source=evidence_id,
                    score=self._score(row.metadata.get("native_score")),
                    kind=row.kind,
                    tags=list(row.tags),
                    metadata={
                        "block_id": row.block_id,
                        "source_backend": "mnemosyne",
                        **dict(row.metadata),
                    },
                )
            )
        return hits

    def reflect_candidates(
        self, query: ReflectionQuery | dict[str, Any]
    ) -> list[ReflectionCandidate]:
        return []

    def close(self) -> None:
        if self._native_memory is not None:
            connections = []
            conn = getattr(self._native_memory, "conn", None)
            if conn is not None:
                connections.append(conn)
            beam = getattr(self._native_memory, "beam", None)
            beam_conn = getattr(beam, "conn", None)
            if beam_conn is not None and beam_conn not in connections:
                connections.append(beam_conn)
            for conn in connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._native_memory = None
        self._index.close()
