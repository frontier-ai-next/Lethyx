"""Optional Hindsight archive backend.

Lethyx calls retain/recall/reflect explicitly. This module never installs a
Hindsight LLM wrapper and is never imported by the proactive prompt compiler.
"""

import importlib.util
import asyncio
import inspect
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from lethyx.contracts import json_ready
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
from lethyx.memory.hindsight import banks
from lethyx.memory.hindsight.candidates import (
    BKB_CANDIDATE_SCHEMA,
    build_reflection_question,
    parse_reflection_candidates,
)
from lethyx.memory.hindsight.config import HindsightConfig
from lethyx.memory.hindsight.errors import (
    HindsightPolicyError,
    HindsightUnavailableError,
)
from lethyx.memory.hindsight.mapping import (
    derived_tags,
    normalize_metadata,
    normalize_tags,
    stable_hash,
)
from lethyx.memory.hindsight.payloads import (
    build_event_payload,
    build_lesson_payload,
    document_id_for_event,
    document_id_for_lesson,
)
from lethyx.memory.kernel.telemetry import record_memory_event

_VERIFIED_TRUST = {
    "workspace_verified",
    "public_verified",
    "supervisor_verified",
    "mutation_verified",
    "hidden_verified",
    "adversarial_verified",
}
_ALLOWED_EVENT_KINDS = {
    "phase_completed",
    "phase_failed",
    "verification_report",
    "durable_promotion",
    "bkb_patch_accepted",
    "bkb_patch_rejected",
    "architecture_decision",
    "run_summary",
}


def _now_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(getattr(value, "__dict__"))
    return {}


def _lesson_from_any(value: DurableLesson | Any) -> DurableLesson:
    if isinstance(value, DurableLesson):
        return value
    data = _as_dict(value)
    evidence = data.get("evidence_refs") or ()
    return DurableLesson(
        lesson_id=str(
            data.get("lesson_id")
            or data.get("id")
            or data.get("event_id")
            or stable_hash(data)[:24]
        ),
        kind=str(data.get("kind") or data.get("memory_kind") or "lesson"),
        title=str(data.get("title") or data.get("kind") or "Verified lesson"),
        content=str(data.get("content") or ""),
        workspace_id=str(data.get("workspace_id") or ""),
        run_id=str(data.get("run_id") or ""),
        phase_id=str(data.get("phase_id") or data.get("phase") or ""),
        trust_level=str(data.get("trust_level") or "public_verified"),
        evidence_refs=[json_ready(ref) for ref in evidence],
        tags=list(data.get("tags") or []),
        metadata=dict(data.get("metadata") or {}),
    )


def _event_from_any(value: DurableEvent | Any) -> DurableEvent:
    if isinstance(value, DurableEvent):
        return value
    data = _as_dict(value)
    evidence = data.get("evidence_refs") or ()
    return DurableEvent(
        event_id=str(data.get("event_id") or data.get("id") or stable_hash(data)[:24]),
        kind=str(data.get("kind") or data.get("memory_kind") or "durable_promotion"),
        content=str(data.get("content") or ""),
        workspace_id=str(data.get("workspace_id") or ""),
        run_id=str(data.get("run_id") or ""),
        phase_id=str(data.get("phase_id") or data.get("phase") or ""),
        subtask_id=str(data.get("subtask_id") or ""),
        agent=str(data.get("agent") or data.get("agent_kind") or ""),
        trust_level=str(data.get("trust_level") or "workspace_verified"),
        evidence_refs=[json_ready(ref) for ref in evidence],
        tags=list(data.get("tags") or []),
        metadata=dict(data.get("metadata") or {}),
    )


class HindsightBackend:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        workspace_id: str = "",
        config: HindsightConfig | None = None,
        client: Any | None = None,
        bank_id: str = "",
    ) -> None:
        self._repo_root = repo_root
        self._workspace_id = workspace_id
        self.config = config or HindsightConfig.from_env()
        self._provided_client = client
        self._client_instance: Any | None = None
        self._default_bank_id = bank_id or (
            banks.workspace_bank_id(workspace_id) if workspace_id else banks.MANAGER_BANK_ID
        )
        index_root = (
            Path(repo_root) / "workspaces" / workspace_id / ".memory" / "hindsight"
            if repo_root is not None and workspace_id
            else Path(repo_root or Path.cwd()) / ".Lethyx" / "memory" / "hindsight"
        )
        self._index = BackendSQLiteIndex(
            index_root / "LETHYX_index.sqlite",
            backend_id="hindsight",
            workspace_id=workspace_id,
        )

    @classmethod
    def from_env(cls, *, repo_root: Path, workspace_id: str = "") -> "HindsightBackend":
        return cls(
            repo_root=repo_root,
            workspace_id=workspace_id,
            config=HindsightConfig.from_env(),
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id="hindsight",
            persistent=True,
            requires_external_service=not self.config.embedded,
            supports_embedded=True,
        )

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        record = replace(record, source_backend="hindsight")
        native_allowed = record.verified and record.lifecycle == "active"
        if not native_allowed:
            raise HindsightPolicyError(
                "Hindsight retain_block requires active, verified records for native storage"
            )
        self._require_enabled()
        bank_id = self._bank_for_workspace(record.workspace_id)
        response = self._retain(
            bank_id=bank_id,
            content=record.content,
            context=f"Lethyx memory block: {record.kind}",
            document_id=f"Lethyx:block:{record.block_id}",
            metadata=self._block_native_metadata(record),
            tags=list(record.tags),
            retain_async=self.config.retain_async,
        )
        native_result = self._normalize_retain_response(response)
        result = self._index.retain_block(record)
        return {**result, **native_result, "bank_id": bank_id, "native_written": True}

    @staticmethod
    def _block_native_metadata(record: MemoryBlockRecord) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in {
                **record.metadata,
                "block_id": record.block_id,
                "record_id": record.block_id,
                "kind": record.kind,
                "workspace_id": record.workspace_id,
                "run_id": record.run_id,
                "phase_id": record.phase_id,
                "surface": record.surface,
                "lifecycle": record.lifecycle,
                "trust_level": record.trust_level,
            }.items()
            if value not in (None, "")
        }

    def retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        """Batch-retain benchmark memory blocks into native Hindsight, then index metadata."""
        if not records:
            return {"ok": True, "n_written": 0, "native_written": 0}
        inactive = [r.block_id for r in records if not (r.verified and r.lifecycle == "active")]
        if inactive:
            raise HindsightPolicyError(
                "Hindsight retain_blocks requires active, verified records for native storage: "
                + ", ".join(inactive[:5])
            )
        self._require_enabled()
        bank_id = self._bank_for_workspace(records[0].workspace_id)
        client = self._client()
        retain_batch = getattr(client, "retain_batch", None)
        if retain_batch is None:
            nested = getattr(client, "client", None)
            retain_batch = getattr(nested, "retain_batch", None) if nested is not None else None
        if retain_batch is None:
            raise HindsightUnavailableError("Hindsight client has no retain_batch API")
        total_native = 0
        chunk_size = 200
        for offset in range(0, len(records), chunk_size):
            chunk = records[offset : offset + chunk_size]
            items = [
                {
                    "content": record.content,
                    "context": f"Lethyx memory block: {record.kind}",
                    "document_id": f"Lethyx:block:{record.block_id}",
                    "metadata": self._block_native_metadata(record),
                    "tags": list(record.tags),
                }
                for record in chunk
            ]
            retain_batch(
                bank_id=bank_id,
                items=items,
                retain_async=self.config.retain_async,
            )
            total_native += len(items)

        written = 0
        first_result: dict[str, Any] = {"ok": True}
        for record in records:
            row = replace(record, source_backend="hindsight")
            result = self._index.retain_block(row)
            first_result = result if written == 0 else first_result
            if result.get("ok"):
                written += 1
        return {
            **first_result,
            "n_written": written,
            "native_written": total_native,
            "bank_id": bank_id,
        }

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self._index.get_block(block_id, workspace_id=self._workspace_id)

    @staticmethod
    def _query_payload(query: MemoryQuery | dict[str, Any]) -> tuple[MemoryQuery, dict[str, Any]]:
        if isinstance(query, MemoryQuery):
            q = query
            payload = {
                "query": query.query,
                "workspace_id": query.workspace_id,
                "run_id": query.run_id,
                "phase_id": query.phase_id,
                "tags": list(query.tags),
                "limit": 10,
                "token_budget": query.max_tokens,
            }
            return q, payload
        data = dict(query)
        q = MemoryQuery(
            query=str(data.get("query") or ""),
            workspace_id=str(data.get("workspace_id") or ""),
            run_id=str(data.get("run_id") or ""),
            phase_id=str(data.get("phase_id") or ""),
            tags=list(data.get("tags") or []),
            max_tokens=int(data.get("max_tokens") or data.get("token_budget") or 2048),
            budget=str(data.get("budget") or "low"),
        )
        return q, data

    def _recall_api(self) -> Any | None:
        client = self._client()
        recall_api = getattr(client, "recall", None)
        if recall_api is None:
            memories = getattr(client, "memories", None)
            recall_api = getattr(memories, "recall", None) if memories is not None else None
        return recall_api

    def _native_recall_hits(self, q: MemoryQuery) -> list[MemoryHit]:
        self._require_enabled()
        recall_api = self._recall_api()
        if recall_api is None:
            raise HindsightUnavailableError("Hindsight client has no recall API")
        bank_id = self._bank_for_workspace(q.workspace_id)
        response = recall_api(
            bank_id=bank_id,
            query=q.query,
            max_tokens=q.max_tokens,
            budget=q.budget,
            tags=q.tags or None,
        )
        rows = getattr(response, "results", response)
        if isinstance(rows, dict):
            rows = rows.get("results") or []
        hits: list[MemoryHit] = []
        for row in rows or []:
            data = dict(row) if isinstance(row, dict) else _as_dict(row)
            metadata = dict(data.get("metadata") or {})
            score = data.get("score")
            source = str(
                data.get("source") or data.get("id") or metadata.get("block_id") or "hindsight"
            )
            metadata.update(
                {
                    "source_backend": "hindsight",
                    "native_backend": "hindsight",
                    "native_score": score,
                    "bank_id": bank_id,
                }
            )
            hits.append(
                MemoryHit(
                    text=str(data.get("text") or data.get("content") or ""),
                    source=source,
                    score=score,
                    kind=str(data.get("kind") or metadata.get("kind") or ""),
                    tags=list(data.get("tags") or metadata.get("tags") or []),
                    metadata=metadata,
                )
            )
        return hits

    def _native_hit_to_block(self, hit: MemoryHit, q: MemoryQuery) -> MemoryBlockRecord:
        metadata = dict(hit.metadata or {})
        block_id = str(metadata.get("block_id") or metadata.get("record_id") or hit.source or "")
        if not block_id:
            block_id = stable_hash({"source": hit.source, "text": hit.text})[:24]
        return MemoryBlockRecord(
            block_id=block_id,
            content=hit.text,
            title=str(metadata.get("title") or ""),
            kind=hit.kind or str(metadata.get("kind") or "observation"),
            lifecycle=str(metadata.get("lifecycle") or "active"),
            trust_level=str(metadata.get("trust_level") or "workspace_verified"),
            scope=str(metadata.get("scope") or "run_scoped"),
            tier=str(metadata.get("tier") or "warm"),
            surface=str(metadata.get("surface") or "supplemental_evidence"),
            workspace_id=str(metadata.get("workspace_id") or q.workspace_id or self._workspace_id),
            run_id=str(metadata.get("run_id") or q.run_id),
            phase_id=str(metadata.get("phase_id") or q.phase_id),
            subtask_id=str(metadata.get("subtask_id") or ""),
            producer=str(metadata.get("producer") or "hindsight"),
            source_backend="hindsight",
            tags=tuple(hit.tags),
            external_refs=dict(metadata.get("external_refs") or {}),
            privacy_class=str(metadata.get("privacy_class") or "normal"),
            risk_level=str(metadata.get("risk_level") or "normal"),
            metadata=metadata,
            verified=True,
        )

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        q, _ = self._query_payload(query)
        hits = self._native_recall_hits(q)
        return [self._native_hit_to_block(hit, q) for hit in hits]

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

    def _dependency_available(self) -> bool:
        if self._provided_client is not None:
            return True
        module_name = "hindsight" if self.config.embedded else "hindsight_client"
        return importlib.util.find_spec(module_name) is not None

    def _create_client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        if self.config.embedded:
            from hindsight import HindsightEmbedded  # type: ignore[import-untyped]

            kwargs: dict[str, Any] = {
                "profile": self.config.profile,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "llm_api_key": os.environ.get("HINDSIGHT_API_LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY"),
            }
            llm_base_url = (
                os.environ.get("HINDSIGHT_API_LLM_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LLM_BASE_URL")
            )
            if llm_base_url:
                kwargs["llm_base_url"] = llm_base_url
            return HindsightEmbedded(**kwargs)
        from hindsight_client import Hindsight  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "base_url": self.config.base_url,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        return Hindsight(**kwargs)

    def _client(self) -> Any:
        if self._client_instance is None:
            self._client_instance = self._create_client()
        return self._client_instance

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise HindsightUnavailableError("Hindsight is disabled")
        if not self._dependency_available():
            raise HindsightUnavailableError("Hindsight client dependency is not installed")

    def _bank_for_workspace(self, workspace_id: str = "") -> str:
        ws = workspace_id or self._workspace_id
        return banks.workspace_bank_id(ws) if ws else banks.MANAGER_BANK_ID

    def _record(
        self,
        *,
        event_type: str,
        workspace_id: str = "",
        run_id: str = "",
        phase_id: str = "",
        status: str = "",
        error: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        if self._repo_root is None:
            return
        record_memory_event(
            self._repo_root,
            event_type=event_type,
            workspace_id=workspace_id or self._workspace_id,
            run_id=run_id,
            phase_id=phase_id,
            backend="hindsight",
            status=status,
            error=error,
            data=data or {},
        )

    def health(self) -> dict[str, Any]:
        base = {
            "backend": "hindsight",
            "selected": True,
            "workspace_id": self._workspace_id,
            "provider_prompt_injection_used": False,
            "implementation": "hindsight_native",
        }
        if not self.config.enabled:
            return {
                **base,
                "ok": False,
                "enabled": False,
                "native_ok": False,
                "strict_ok": False,
                "effective_mode": "unavailable",
                "reason": "native_disabled",
                "mode": self.config.backend_mode,
                "operational_note": (
                    "Hindsight native client is disabled; selected Hindsight "
                    "backend fails closed instead of using a SQLite sidecar."
                ),
            }
        if not self._dependency_available():
            return {
                **base,
                "ok": False,
                "enabled": True,
                "native_ok": False,
                "strict_ok": False,
                "effective_mode": "unavailable",
                "reason": "dependency_unavailable",
                "embedded": self.config.embedded,
                "mode": self.config.backend_mode,
                "operational_note": (
                    "Hindsight native dependency is unavailable; selected "
                    "Hindsight backend fails closed instead of using a SQLite sidecar."
                ),
            }
        try:
            if not self.config.embedded and self._provided_client is None:
                request = Request(f"{self.config.base_url.rstrip('/')}/health")
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    if not (200 <= int(getattr(response, "status", 0) or 0) < 300):
                        raise HindsightUnavailableError(
                            f"Hindsight health endpoint returned HTTP {getattr(response, 'status', 'unknown')}"
                        )
                return {
                    **base,
                    "ok": True,
                    "enabled": True,
                    "native_ok": True,
                    "strict_ok": True,
                    "effective_mode": "native",
                    "base_url": self.config.base_url,
                    "embedded": self.config.embedded,
                    "mode": self.config.backend_mode,
                    "bank_id": self._default_bank_id,
                }
            client = self._client()
            client_banks = getattr(client, "banks", None)
            probe = None
            if client_banks is not None:
                probe = getattr(client_banks, "list", None) or getattr(
                    client_banks, "list_banks", None
                )
            if probe is None:
                raise HindsightUnavailableError("Hindsight client has no bank health probe API")
            kwargs: dict[str, Any] = {}
            try:
                signature = inspect.signature(probe)
            except (TypeError, ValueError):
                signature = None
            if signature is not None and "_request_timeout" in signature.parameters:
                kwargs["_request_timeout"] = self.config.timeout_seconds
            result = probe(**kwargs)
            if inspect.isawaitable(result):
                asyncio.run(result)
            return {
                **base,
                "ok": True,
                "enabled": True,
                "native_ok": True,
                "strict_ok": True,
                "effective_mode": "native",
                "base_url": self.config.base_url,
                "embedded": self.config.embedded,
                "mode": self.config.backend_mode,
                "bank_id": self._default_bank_id,
            }
        except Exception as exc:
            reason = (
                "native_health_probe_unavailable"
                if isinstance(exc, HindsightUnavailableError)
                else "native_health_check_failed"
            )
            return {
                **base,
                "ok": False,
                "enabled": True,
                "native_ok": False,
                "strict_ok": False,
                "effective_mode": "unavailable",
                "base_url": self.config.base_url,
                "embedded": self.config.embedded,
                "mode": self.config.backend_mode,
                "native_error": str(exc),
                "reason": reason,
                "operational_note": (
                    "Hindsight native health check failed; selected Hindsight "
                    "backend fails closed instead of using a SQLite sidecar."
                ),
            }

    def close(self) -> None:
        client = self._client_instance
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                close()
            self._client_instance = None
        self._index.close()

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]:
        self._require_enabled()
        return banks.ensure_banks(self._client(), workspace_id=workspace_id or self._workspace_id)

    def _retain(self, **kwargs: Any) -> Any:
        client = self._client()
        if hasattr(client, "retain"):
            return client.retain(**kwargs)
        memories = getattr(client, "memories", None)
        if memories is not None and hasattr(memories, "retain"):
            return memories.retain(**kwargs)
        raise HindsightUnavailableError("Hindsight client has no retain API")

    @staticmethod
    def _normalize_retain_response(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            result = dict(response)
        else:
            result = {
                key: getattr(response, key)
                for key in ("id", "document_id", "status")
                if hasattr(response, key)
            }
        result.setdefault("ok", True)
        result.setdefault("backend", "hindsight")
        return result

    def retain_event(self, event: DurableEvent | Any) -> dict[str, Any]:
        durable = _event_from_any(event)
        if durable.kind not in _ALLOWED_EVENT_KINDS:
            raise HindsightPolicyError(f"Hindsight retain_event disallows kind {durable.kind!r}")
        if durable.trust_level in {"untrusted", "agent_reported", "agent_claim"}:
            raise HindsightPolicyError("Hindsight retain_event requires verified trust")
        self._require_enabled()
        self.ensure_banks(workspace_id=durable.workspace_id)
        bank_id = self._bank_for_workspace(durable.workspace_id)
        tags = normalize_tags(
            durable.tags
            + derived_tags(
                kind=durable.kind,
                workspace_id=durable.workspace_id,
                run_id=durable.run_id,
                phase_id=durable.phase_id,
                subtask_id=durable.subtask_id,
                agent=durable.agent,
                trust_level=str(durable.trust_level),
                scope="workspace" if durable.workspace_id else "manager",
            )
        )
        metadata = normalize_metadata(
            {
                **durable.metadata,
                "LETHYX_id": durable.event_id,
                "workspace_id": durable.workspace_id,
                "run_id": durable.run_id,
                "phase_id": durable.phase_id,
                "subtask_id": durable.subtask_id,
                "agent": durable.agent,
                "kind": durable.kind,
                "trust_level": durable.trust_level,
                "source_hash": durable.metadata.get("source_hash") or stable_hash(durable.content),
            },
            durable.evidence_refs,
        )
        start = time.perf_counter()
        response = self._retain(
            bank_id=bank_id,
            content=build_event_payload(durable),
            context=f"Lethyx durable event: {durable.kind}",
            timestamp=durable.occurred_at or None,
            document_id=document_id_for_event(durable),
            metadata=metadata,
            tags=tags,
            retain_async=self.config.retain_async,
        )
        result = self._normalize_retain_response(response)
        sidecar_result = self._index.retain_block(
            durable_event_to_block_record(durable, backend_id="hindsight")
        )
        result = {**sidecar_result, **result}
        result["latency_ms"] = _now_ms(start)
        result["bank_id"] = bank_id
        self._record(
            event_type="hindsight_retain_success",
            workspace_id=durable.workspace_id,
            run_id=durable.run_id,
            phase_id=durable.phase_id,
            status="ok",
            data={
                "kind": durable.kind,
                "latency_ms": result["latency_ms"],
                "bank_id": bank_id,
            },
        )
        return result

    def retain_lesson(self, lesson: DurableLesson | Any) -> dict[str, Any]:
        durable = _lesson_from_any(lesson)
        if str(durable.trust_level) not in _VERIFIED_TRUST:
            raise HindsightPolicyError("Hindsight retain_lesson requires verified trust level")
        if not durable.evidence_refs:
            raise HindsightPolicyError("Hindsight retain_lesson requires evidence_refs")
        self._require_enabled()
        self.ensure_banks(workspace_id=durable.workspace_id)
        bank_id = self._bank_for_workspace(durable.workspace_id)
        scope = "workspace" if durable.workspace_id else "manager"
        tags = normalize_tags(
            durable.tags
            + derived_tags(
                kind=durable.kind,
                workspace_id=durable.workspace_id,
                run_id=durable.run_id,
                phase_id=durable.phase_id,
                trust_level=str(durable.trust_level),
                scope=scope,
                store="bkb" if "bkb" in durable.kind else "palace.lesson",
            )
        )
        metadata = normalize_metadata(
            {
                **durable.metadata,
                "LETHYX_id": durable.lesson_id,
                "workspace_id": durable.workspace_id,
                "run_id": durable.run_id,
                "phase_id": durable.phase_id,
                "kind": durable.kind,
                "trust_level": durable.trust_level,
                "source_hash": durable.metadata.get("source_hash") or stable_hash(durable.content),
            },
            durable.evidence_refs,
        )
        start = time.perf_counter()
        response = self._retain(
            bank_id=bank_id,
            content=build_lesson_payload(durable),
            context=f"Lethyx verified lesson: {durable.kind}",
            timestamp=None,
            document_id=document_id_for_lesson(durable),
            metadata=metadata,
            tags=tags,
            retain_async=self.config.retain_async,
        )
        result = self._normalize_retain_response(response)
        sidecar_result = self._index.retain_block(
            durable_lesson_to_block_record(durable, backend_id="hindsight")
        )
        result = {**sidecar_result, **result}
        result["latency_ms"] = _now_ms(start)
        result["bank_id"] = bank_id
        self._record(
            event_type="hindsight_retain_success",
            workspace_id=durable.workspace_id,
            run_id=durable.run_id,
            phase_id=durable.phase_id,
            status="ok",
            data={
                "kind": durable.kind,
                "latency_ms": result["latency_ms"],
                "bank_id": bank_id,
            },
        )
        return result

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        q, _ = self._query_payload(query)
        return self._native_recall_hits(q)

    @staticmethod
    def tag_supplemental_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tagged: list[dict[str, Any]] = []
        for hit in hits:
            row = dict(hit)
            row.setdefault("surface", "supplemental_evidence")
            row.setdefault("directive", False)
            row.setdefault("source_backend", "hindsight")
            tagged.append(row)
        return tagged

    def reflect_candidates(
        self, query: ReflectionQuery | dict[str, Any]
    ) -> list[ReflectionCandidate]:
        if not self.config.enabled or not self.config.reflect_enabled:
            return []
        self._require_enabled()
        if isinstance(query, ReflectionQuery):
            q = query
        else:
            data = dict(query)
            q = ReflectionQuery(
                question=str(data.get("question") or ""),
                workspace_id=str(data.get("workspace_id") or ""),
                run_id=str(data.get("run_id") or ""),
                phase_id=str(data.get("phase_id") or "reflexion"),
                tags=list(data.get("tags") or []),
                max_candidates=int(data.get("max_candidates") or 3),
                budget=str(data.get("budget") or "mid"),
            )
        bank_id = self._bank_for_workspace(q.workspace_id)
        client = self._client()
        if not hasattr(client, "reflect"):
            return []
        question = q.question or build_reflection_question(
            max_candidates=q.max_candidates or self.config.max_candidates
        )
        kwargs = {
            "bank_id": bank_id,
            "question": question,
            "budget": q.budget,
            "tags": q.tags or None,
            "response_schema": BKB_CANDIDATE_SCHEMA,
        }
        start = time.perf_counter()
        try:
            response = client.reflect(**kwargs)
        except TypeError:
            kwargs.pop("response_schema", None)
            response = client.reflect(**kwargs)
        candidates = parse_reflection_candidates(
            response,
            bank_id=bank_id,
            max_candidates=min(q.max_candidates, self.config.max_candidates),
            budget=q.budget,
        )
        self._record(
            event_type="hindsight_reflect_success",
            workspace_id=q.workspace_id,
            run_id=q.run_id,
            phase_id=q.phase_id,
            status="ok",
            data={
                "candidate_count": len(candidates),
                "latency_ms": _now_ms(start),
                "bank_id": bank_id,
            },
        )
        return candidates
