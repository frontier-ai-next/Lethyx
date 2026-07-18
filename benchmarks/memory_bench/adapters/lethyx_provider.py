"""Lethyx memory provider adapter for memory_bench.

Wraps Lethyx's three durable backends (mempalace / hindsight / mnemosyne)
behind the MemoryProvider / MemoryStore protocol.

Design constraints (from plan):
- Read-only dependency: we only import from lethyx.memory.backends.*
  No write to Lethyx source files, no directive/promotion surfaces.
- All records are written as supplemental-evidence DurableEvents with
  scope "run_scoped" and trust_level "workspace_verified".
- Each namespace() call creates a fresh workspace_id (= ns slug) so episodes
  are fully isolated.
"""

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from memory_bench.core.protocols import ProviderCapabilities
from memory_bench.core.registry import register_provider
from memory_bench.core.types import MemoryRecord, RetrievedItem

log = logging.getLogger(__name__)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_env_for_native_clients() -> None:
    try:
        from lethyx.env import get_llm_env_config, load_env

        load_env()
        raw = get_llm_env_config()
        if isinstance(raw, dict):
            api_key = str(raw.get("api_key") or "")
            base_url = str(raw.get("base_url") or "")
            model = str(raw.get("model") or "")
        elif isinstance(raw, (tuple, list)) and len(raw) >= 3:
            model, api_key, base_url = (str(raw[0] or ""), str(raw[1] or ""), str(raw[2] or ""))
        else:
            api_key = base_url = model = ""
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            os.environ.setdefault("HINDSIGHT_API_LLM_API_KEY", api_key)
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", base_url)
            os.environ.setdefault("HINDSIGHT_API_LLM_BASE_URL", base_url)
        if model:
            os.environ.setdefault("LETHYX_HINDSIGHT_LLM_MODEL", model)
            os.environ.setdefault("HINDSIGHT_API_LLM_MODEL", model)
    except Exception:
        log.debug("unable to load .env for native memory clients", exc_info=True)


def _close_quietly(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        log.debug("resource close failed", exc_info=True)


def _native_health_extra(backend: Any, backend_mode: str) -> dict[str, Any]:
    try:
        health = backend.health()
    except Exception as exc:
        return {
            "backend": backend_mode,
            "native_ok": False,
            "strict_ok": False,
            "effective_mode": "unavailable",
            "native_reason": str(exc),
        }
    native_ok = bool(health.get("native_ok") or health.get("strict_ok") is True)
    if backend_mode == "mempalace" and bool(health.get("ok")):
        native_ok = True
    effective_mode = str(health.get("effective_mode") or backend_mode)
    return {
        "backend": backend_mode,
        "native_ok": native_ok,
        "strict_ok": bool(health.get("strict_ok", native_ok)),
        "effective_mode": effective_mode,
        "native_reason": str(health.get("reason") or health.get("error") or ""),
        "implementation": str(health.get("implementation") or ""),
    }


def _fail_if_native_requested_but_not_strict(backend: Any, backend_mode: str) -> None:
    mode = str(backend_mode or "").strip().lower()
    extra = _native_health_extra(backend, mode)
    sidecar_modes = {"sqlite_sidecar", "sqlite_parity", "unavailable"}
    effective_mode = str(extra.get("effective_mode") or "").strip().lower()
    native_mode_reported = effective_mode and effective_mode not in sidecar_modes
    if native_mode_reported and (
        extra.get("native_ok") or extra.get("strict_ok") or effective_mode == mode
    ):
        return
    label = mode[:1].upper() + mode[1:]
    raise RuntimeError(
        f"native {label} benchmark requested, but native {label} is not "
        f"available (effective_mode={extra.get('effective_mode')}, "
        f"reason={extra.get('native_reason') or 'unknown'}). "
        "This run would not exercise the real backend, so it is not an honest "
        "native ablation. Install/enable the native backend before running this provider."
    )


# ---------------------------------------------------------------------------
# Store implementation
# ---------------------------------------------------------------------------


class _LethyxStore:
    """One isolated episode store backed by a single Lethyx durable backend."""

    def __init__(
        self,
        backend: Any,
        workspace_id: str,
        backend_mode: str,
        *,
        native: bool = False,
        phase_id: str = "memory_bench",
        include_passive: bool = False,
        memory_activity: str = "",
    ) -> None:
        self._backend = backend
        self._workspace_id = workspace_id
        self._backend_mode = backend_mode
        self._native = native
        self._phase_id = str(phase_id or "memory_bench")
        self._include_passive = include_passive
        self._memory_activity = str(memory_activity or "").strip().lower()
        self._n_written = 0

    # -- write ---------------------------------------------------------------

    def _block_for_record(self, record: MemoryRecord) -> Any:
        from lethyx.contracts import EvidenceRef
        from lethyx.memory.backends.contracts import MemoryBlockRecord

        evidence_ref = EvidenceRef(
            ref_type="artifact",
            ref_id=record.evidence_id or record.id or f"memory_bench:{self._workspace_id}",
            produced_by="harness",
            phase=self._phase_id,
        )
        return MemoryBlockRecord(
            block_id=record.id or f"mb_{uuid.uuid4().hex[:12]}",
            content=record.content,
            title="memory_bench dialog turn",
            kind=record.kind or "message",
            lifecycle="active",
            trust_level="workspace_verified",
            scope="run_scoped",
            tier="warm",
            surface="supplemental_evidence",
            workspace_id=self._workspace_id,
            run_id=record.session_id,
            phase_id=self._phase_id,
            producer=record.role or "benchmark",
            tags=tuple(
                list(record.tags)
                + ([f"session:{record.session_id}"] if record.session_id else [])
                + ([f"role:{record.role}"] if record.role else [])
            ),
            evidence_refs=(evidence_ref,),
            metadata={
                "bench_kind": record.kind,
                "evidence_id": record.evidence_id,
                "source_path": record.evidence_id or record.id,
                "ts": record.ts,
                **({"memory_activity": self._memory_activity} if self._memory_activity else {}),
                **record.metadata,
            },
            verified=True,
            created_at=record.ts,
            updated_at=record.ts,
        )

    def write(self, record: MemoryRecord) -> None:
        """Write one MemoryRecord as a backend-neutral MemoryBlockRecord."""
        try:
            self._backend.retain_block(self._block_for_record(record))
            self._n_written += 1
        except Exception as exc:
            raise RuntimeError(
                f"Lethyx {self._backend_mode} retain_block failed; no parity fallback is used"
            ) from exc

    def write_many(self, records: list[MemoryRecord]) -> None:
        if self._native and hasattr(self._backend, "retain_blocks"):
            try:
                blocks = [self._block_for_record(record) for record in records]
                result = self._backend.retain_blocks(blocks)
                self._n_written += int(result.get("n_written", len(blocks)))
                return
            except Exception as exc:
                raise RuntimeError(
                    f"Lethyx {self._backend_mode} retain_blocks failed; no parity fallback is used"
                ) from exc
        for record in records:
            self.write(record)

    # -- retrieve ------------------------------------------------------------

    def _retrieve_native_mempalace(
        self,
        query: str,
        *,
        k: int,
        max_tokens: int = 0,
    ) -> list[RetrievedItem]:
        palace = getattr(self._backend, "_palace", None)
        if palace is None or not hasattr(palace, "search"):
            return []
        hits = palace.search(
            query=query,
            stores=["palace.idea", "palace.run", "palace.lesson", "palace.durable"],
            n=k,
        )
        out: list[RetrievedItem] = []
        used_chars = 0
        budget = max_tokens * 4 if max_tokens > 0 else 0
        for i, hit in enumerate(hits or []):
            if not isinstance(hit, dict):
                continue
            text = str(hit.get("content") or hit.get("text") or "")
            if budget and out and used_chars + len(text) > budget:
                break
            if budget and not out and len(text) > budget:
                break
            source_id = str(hit.get("evidence_id") or hit.get("source_path") or hit.get("id") or "")
            try:
                score = float(hit.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out.append(
                RetrievedItem(
                    text=text,
                    score=score,
                    source_id=source_id,
                    rank=i,
                    metadata={**dict(hit), "native_backend": "mempalace"},
                )
            )
            used_chars += len(text)
        return out

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        max_tokens: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        """Query the backend via native retrieval when requested, otherwise recall_evidence."""
        try:
            if self._native and self._backend_mode == "mempalace":
                return self._retrieve_native_mempalace(
                    query,
                    k=k,
                    max_tokens=max_tokens,
                )
            query_dict: dict[str, Any] = {
                "query": query,
                "n": k,
                "workspace_id": self._workspace_id,
                "phase_id": self._phase_id,
            }
            if self._include_passive:
                query_dict["include_passive"] = True
            if max_tokens:
                query_dict["max_tokens"] = max_tokens
            if filters:
                query_dict.update(filters)

            hits = self._backend.recall_evidence(query_dict)
            return [
                RetrievedItem(
                    text=str(hit.text),
                    score=float(hit.score or 0.0),
                    source_id=str(
                        hit.metadata.get("evidence_id")
                        or hit.metadata.get("block_id")
                        or hit.source
                        or ""
                    ),
                    rank=i,
                    metadata=dict(hit.metadata or {}),
                )
                for i, hit in enumerate(hits[:k])
            ]
        except Exception as exc:
            raise RuntimeError(
                f"Lethyx {self._backend_mode} recall_evidence failed; no parity fallback is used"
            ) from exc

    # -- update --------------------------------------------------------------

    def update(self, record_id: str, **changes: Any) -> bool:
        """Update lifecycle of an existing block (knowledge-update benchmarks)."""
        try:
            lifecycle = str(changes.get("lifecycle") or "active")
            reason = str(changes.get("reason") or "memory_bench_update")
            self._backend.update_lifecycle(record_id, lifecycle, reason=reason)
            return True
        except Exception as exc:
            log.debug("update_lifecycle(%s) failed: %s", record_id, exc)
            return False

    # -- stats / close -------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "workspace_id": self._workspace_id,
            "backend": self._backend_mode,
            "native_requested": self._native,
            "n_written": self._n_written,
        }

    def close(self) -> None:
        try:
            self._backend.close()
        except Exception:
            pass


class _CompositeLethyxStore:
    """Benchmark-only composite store that fans out to selected Lethyx backends.

    This is separate from the production backend factory, which selects exactly
    one durable backend.  The composite is an ablation target for measuring
    whether a multi-backend wrapper adds value over each backend alone.
    """

    def __init__(self, stores: dict[str, Any]) -> None:
        self._stores = dict(stores)

    def write(self, record: MemoryRecord) -> None:
        for store in self._stores.values():
            store.write(record)

    def write_many(self, records: list[MemoryRecord]) -> None:
        for store in self._stores.values():
            if hasattr(store, "write_many"):
                store.write_many(records)
            else:
                for record in records:
                    store.write(record)

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        max_tokens: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        merged: dict[str, RetrievedItem] = {}
        for backend_order, (backend_name, store) in enumerate(self._stores.items()):
            for backend_rank, hit in enumerate(
                store.retrieve(query, k=k, max_tokens=max_tokens, filters=filters),
                start=1,
            ):
                key = hit.source_id or hit.text
                fusion_score = 1.0 / (60.0 + float(backend_rank))
                metadata = {
                    **dict(hit.metadata or {}),
                    "composite_backend": backend_name,
                    "composite_backend_order": backend_order,
                    "composite_backend_rank": backend_rank,
                    "raw_backend_score": float(hit.score or 0.0),
                    "composite_rank_fusion_score": fusion_score,
                }
                candidate = RetrievedItem(
                    text=hit.text,
                    score=fusion_score,
                    source_id=hit.source_id,
                    rank=hit.rank,
                    metadata=metadata,
                )
                existing = merged.get(key)
                if existing is None or candidate.score > existing.score:
                    merged[key] = candidate
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return [
            RetrievedItem(
                text=item.text,
                score=item.score,
                source_id=item.source_id,
                rank=i,
                metadata=item.metadata,
            )
            for i, item in enumerate(ranked[:k])
        ]

    def update(self, record_id: str, **changes: Any) -> bool:
        saved = False
        for store in self._stores.values():
            saved = bool(store.update(record_id, **changes)) or saved
        return saved

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "Lethyx-composite",
            "children": {name: store.stats() for name, store in self._stores.items()},
        }

    def close(self) -> None:
        for store in self._stores.values():
            try:
                store.close()
            except Exception:
                log.debug("composite child close failed", exc_info=True)


class _RawMemPalaceStore:
    """Direct MemPalace facade provider, bypassing the typed backend sidecar."""

    def __init__(self, repo_root: Path, workspace_id: str) -> None:
        from lethyx.memory.palace.facade import MemPalace

        self._workspace_id = workspace_id
        self._palace = MemPalace(Path(repo_root), workspace_id)
        self._n_written = 0

    def write(self, record: MemoryRecord) -> None:
        from lethyx.memory.palace.tiers import Scope, Tier

        try:
            self._palace.add(
                store="palace.idea",
                content=record.content,
                tier=Tier.WARM,
                scope=Scope.RUN_SCOPED,
                tags=list(record.tags)
                + ([f"session:{record.session_id}"] if record.session_id else [])
                + ([f"role:{record.role}"] if record.role else []),
                phase="memory_bench",
                run_id=record.session_id,
                source_path=record.evidence_id or record.id,
                verified=True,
                kind=record.kind or "message",
                node_id=record.id or f"raw_mp_{uuid.uuid4().hex[:12]}",
                extra={
                    "evidence_id": record.evidence_id or record.id,
                    "bench_kind": record.kind,
                },
            )
            self._n_written += 1
        except Exception as exc:
            log.warning(
                "raw MemPalace write failed (ws=%s, record=%s): %s",
                self._workspace_id,
                record.id,
                exc,
            )

    def write_many(self, records: list[MemoryRecord]) -> None:
        records = [record for record in records if record.content]
        if not records:
            return
        try:
            collection = self._palace._stores.chroma("palace.idea")
            ids = [record.id or f"raw_mp_{uuid.uuid4().hex[:12]}" for record in records]
            now = time.time()
            documents = [record.content for record in records]
            metadatas = []
            for record in records:
                metadatas.append(
                    {
                        "id": record.id or "",
                        "store": "palace.idea",
                        "tier": "warm",
                        "scope": "run_scoped",
                        "tags": ",".join(
                            list(record.tags)
                            + ([f"session:{record.session_id}"] if record.session_id else [])
                            + ([f"role:{record.role}"] if record.role else [])
                        ),
                        "phase": "memory_bench",
                        "subtask_id": "",
                        "verified": 1,
                        "source_path": record.evidence_id or record.id,
                        "workspace_id": self._workspace_id,
                        "run_id": record.session_id,
                        "created_at": now,
                        "kind": record.kind or "message",
                        "evidence_id": record.evidence_id or record.id,
                        "bench_kind": record.kind,
                    }
                )
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            self._n_written += len(records)
        except Exception as exc:
            log.warning(
                "raw MemPalace batch write failed (ws=%s, n=%d): %s; falling back to row writes",
                self._workspace_id,
                len(records),
                exc,
            )
            for record in records:
                self.write(record)

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        max_tokens: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        try:
            hits = self._palace.search(
                query=query,
                stores=["palace.idea", "palace.run", "palace.lesson", "palace.durable"],
                n=k,
            )
        except Exception as exc:
            log.warning(
                "raw MemPalace search failed (ws=%s, query=%r): %s",
                self._workspace_id,
                query[:80],
                exc,
            )
            return []
        out: list[RetrievedItem] = []
        used_chars = 0
        budget = max_tokens * 4 if max_tokens > 0 else 0
        for i, hit in enumerate(hits):
            text = str(hit.get("content") or "")
            if budget and out and used_chars + len(text) > budget:
                break
            if budget and not out and len(text) > budget:
                break
            source_id = str(hit.get("source_path") or hit.get("evidence_id") or hit.get("id") or "")
            try:
                score = float(hit.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            out.append(
                RetrievedItem(
                    text=text,
                    score=score,
                    source_id=source_id,
                    rank=i,
                    metadata=dict(hit),
                )
            )
            used_chars += len(text)
        return out

    def update(self, record_id: str, **changes: Any) -> bool:
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "raw-mempalace",
            "workspace_id": self._workspace_id,
            "n_written": self._n_written,
        }

    def close(self) -> None:
        try:
            self._palace.close()
        except Exception as exc:
            log.debug("raw MemPalace close failed: %s", exc)


class _RawHindsightStore:
    """Direct Hindsight client provider, bypassing the Lethyx SQLite sidecar."""

    def __init__(self, client: Any, workspace_id: str) -> None:
        self._client = client
        self._workspace_id = workspace_id
        self._bank_id = f"mb:raw:{workspace_id}"
        self._n_written = 0
        self._ensure_bank()

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            return dict(dumped) if isinstance(dumped, dict) else {}
        if hasattr(value, "dict"):
            dumped = value.dict()
            return dict(dumped) if isinstance(dumped, dict) else {}
        if hasattr(value, "__dict__"):
            return dict(getattr(value, "__dict__"))
        return {}

    @staticmethod
    def _api(client: Any, name: str) -> Any | None:
        if hasattr(client, name):
            return getattr(client, name)
        for attr in ("memories", "client"):
            nested = getattr(client, attr, None)
            if nested is not None and hasattr(nested, name):
                return getattr(nested, name)
        return None

    def _ensure_bank(self) -> None:
        banks_api = getattr(self._client, "banks", None)
        if banks_api is None:
            nested_client = getattr(self._client, "client", None)
            banks_api = getattr(nested_client, "banks", None)
        if banks_api is None:
            log.debug("raw Hindsight client has no banks API")
            return
        try:
            if hasattr(banks_api, "create"):
                banks_api.create(
                    bank_id=self._bank_id,
                    name=f"memory_bench raw Hindsight {self._workspace_id}",
                    mission=(
                        "Store benchmark dialogue turns verbatim and retrieve the "
                        "minimal evidence needed to answer benchmark probes."
                    ),
                )
        except Exception as exc:
            text = str(exc).lower()
            if "exist" not in text and "already" not in text:
                log.debug("raw Hindsight bank create failed: %s", exc)
        try:
            if hasattr(banks_api, "set_mission"):
                banks_api.set_mission(
                    bank_id=self._bank_id,
                    mission="Retrieve benchmark evidence verbatim for memory_bench probes.",
                )
        except Exception as exc:
            log.debug("raw Hindsight bank mission update failed: %s", exc)

    def write(self, record: MemoryRecord) -> None:
        retain = self._api(self._client, "retain")
        if retain is None:
            log.warning("raw Hindsight client has no retain API")
            return
        metadata = {
            "evidence_id": record.evidence_id or record.id,
            "record_id": record.id,
            "session_id": record.session_id,
            "role": record.role,
            "kind": record.kind or "message",
        }
        try:
            retain(
                bank_id=self._bank_id,
                content=record.content,
                context="memory_bench dialogue record",
                document_id=record.id or f"raw_hs_{uuid.uuid4().hex[:12]}",
                metadata={key: str(value) for key, value in metadata.items() if value},
                tags=list(record.tags)
                + ([f"session:{record.session_id}"] if record.session_id else [])
                + ([f"role:{record.role}"] if record.role else []),
                retain_async=False,
            )
            self._n_written += 1
        except Exception as exc:
            log.warning(
                "raw Hindsight retain failed (ws=%s, record=%s): %s",
                self._workspace_id,
                record.id,
                exc,
            )

    def write_many(self, records: list[MemoryRecord]) -> None:
        records = [record for record in records if record.content]
        if not records:
            return
        client = getattr(self._client, "client", self._client)
        retain_batch = getattr(client, "retain_batch", None)
        if retain_batch is None:
            for record in records:
                self.write(record)
            return
        try:
            items = []
            for record in records:
                metadata = {
                    "evidence_id": record.evidence_id or record.id,
                    "record_id": record.id,
                    "session_id": record.session_id,
                    "role": record.role,
                    "kind": record.kind or "message",
                }
                items.append(
                    {
                        "content": record.content,
                        "context": "memory_bench dialogue record",
                        "document_id": record.id or f"raw_hs_{uuid.uuid4().hex[:12]}",
                        "metadata": {key: str(value) for key, value in metadata.items() if value},
                        "tags": list(record.tags)
                        + ([f"session:{record.session_id}"] if record.session_id else [])
                        + ([f"role:{record.role}"] if record.role else []),
                    }
                )
            retain_batch(bank_id=self._bank_id, items=items, retain_async=False)
            self._n_written += len(items)
        except Exception as exc:
            log.warning(
                "raw Hindsight batch retain failed (ws=%s, n=%d): %s; falling back to row writes",
                self._workspace_id,
                len(records),
                exc,
            )
            for record in records:
                self.write(record)

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        max_tokens: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        recall = self._api(self._client, "recall")
        if recall is None:
            log.warning("raw Hindsight client has no recall API")
            return []
        try:
            response = recall(
                bank_id=self._bank_id,
                query=query,
                max_tokens=max_tokens or 4096,
                budget=str((filters or {}).get("budget") or "low"),
            )
        except Exception as exc:
            log.warning(
                "raw Hindsight recall failed (ws=%s, query=%r): %s",
                self._workspace_id,
                query[:80],
                exc,
            )
            return []
        data = self._as_dict(response)
        rows = data.get("results") if data else response
        if isinstance(rows, dict):
            rows = rows.get("results") or []
        out: list[RetrievedItem] = []
        used_chars = 0
        budget_chars = max_tokens * 4 if max_tokens > 0 else 0
        for i, row in enumerate(list(rows or [])[:k]):
            item = self._as_dict(row)
            metadata = dict(item.get("metadata") or {})
            text = str(item.get("text") or item.get("content") or item.get("document") or "")
            if budget_chars and out and used_chars + len(text) > budget_chars:
                break
            if budget_chars and not out and len(text) > budget_chars:
                break
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            source_id = str(
                metadata.get("evidence_id")
                or item.get("evidence_id")
                or item.get("document_id")
                or item.get("id")
                or item.get("source")
                or ""
            )
            out.append(
                RetrievedItem(
                    text=text,
                    score=score,
                    source_id=source_id,
                    rank=i,
                    metadata={**metadata, **item, "raw_backend": "hindsight"},
                )
            )
            used_chars += len(text)
        return out

    def update(self, record_id: str, **changes: Any) -> bool:
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "raw-hindsight",
            "workspace_id": self._workspace_id,
            "bank_id": self._bank_id,
            "n_written": self._n_written,
        }

    def close(self) -> None:
        # The client is provider-owned and reused across episode namespaces.
        return


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class LethyxMemoryProvider:
    """Memory provider backed by one of the three Lethyx durable backends.

    Registered as 'Lethyx' in the memory_bench registry.
    Each namespace() call instantiates a fresh backend with a unique workspace_id
    so no state leaks between episodes.

    Configuration (via ProviderSpec.params):
        backend:  "mempalace" | "hindsight" | "mnemosyne"  (default: "mempalace")
        repo_root: path to repo root (default: auto-detected)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = dict(params or {})
        self._backend_mode = str(self._params.get("backend") or "mempalace").strip().lower()
        self._native = _truthy(self._params.get("native") or self._params.get("use_native"))
        raw_root = str(self._params.get("repo_root") or "").strip()
        if raw_root:
            self._repo_root = Path(raw_root)
        else:
            # Auto-detect: walk up from this file to find pyproject.toml
            candidate = Path(__file__).resolve()
            for _ in range(6):
                candidate = candidate.parent
                if (candidate / "pyproject.toml").is_file():
                    self._repo_root = candidate
                    break
            else:
                self._repo_root = Path.cwd()
        log.debug(
            "LethyxMemoryProvider: backend=%s, repo_root=%s",
            self._backend_mode,
            self._repo_root,
        )

    # Override name for display using the backend mode
    @property
    def name(self) -> str:  # type: ignore[override]
        return f"Lethyx-{self._backend_mode}"

    def capabilities(self) -> ProviderCapabilities:
        """Map Lethyx BackendCapabilities to framework ProviderCapabilities."""
        try:
            from lethyx.memory.backends.factory import create_durable_backend

            backend = create_durable_backend(
                self._repo_root,
                workspace_id=f"_caps_{uuid.uuid4().hex[:8]}",
                mode=self._backend_mode,
            )
            bc = backend.capabilities()
            health_extra = _native_health_extra(backend, self._backend_mode)
            _close_quietly(backend)
            return ProviderCapabilities(
                persistent=bool(getattr(bc, "persistent", True)),
                supports_update=True,  # update_lifecycle is always available
                supports_delete=False,
                supports_temporal_filter=False,
                supports_graph=bool(getattr(bc, "graph_edges", False)),
                supports_namespacing=True,
                memory_as_tool=False,
                token_accounting=False,
                extra={
                    "LETHYX_backend": self._backend_mode,
                    "native_requested": self._native,
                    **health_extra,
                    "native_honesty": (
                        "native"
                        if not self._native or health_extra.get("native_ok")
                        else "requested_but_unavailable"
                    ),
                },
            )
        except Exception as exc:
            log.warning("capabilities() probe failed: %s — returning defaults", exc)
            return ProviderCapabilities(
                persistent=True,
                supports_update=True,
                extra={"LETHYX_backend": self._backend_mode, "native_requested": self._native},
            )

    def namespace(self, ns: str) -> _LethyxStore:
        """Create a fresh isolated store for the given namespace string."""
        from lethyx.memory.backends.factory import create_durable_backend

        # Sanitise namespace to a valid workspace_id (alphanumeric + hyphens)
        import re

        ws_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", ns)[:64]
        ws_id = f"mb_{ws_id}_{uuid.uuid4().hex[:6]}"

        if self._native and self._backend_mode == "hindsight":
            _load_env_for_native_clients()
            from dataclasses import replace
            from lethyx.memory.backends.hindsight import HindsightBackend
            from lethyx.memory.hindsight.config import HindsightConfig

            raw_embedded = self._params.get(
                "embedded",
                os.getenv("LETHYX_HINDSIGHT_EMBEDDED", "false"),
            )
            base_config = HindsightConfig.from_env()
            raw_timeout = self._params.get("timeout_seconds") or self._params.get("timeout")
            raw_retain_async = self._params.get("retain_async")
            config = replace(
                base_config,
                enabled=True,
                embedded=_truthy(raw_embedded),
                timeout_seconds=float(raw_timeout) if raw_timeout else base_config.timeout_seconds,
                llm_provider=str(
                    self._params.get("llm_provider")
                    or os.getenv("LETHYX_HINDSIGHT_LLM_PROVIDER")
                    or "openai"
                ),
                llm_model=str(
                    self._params.get("llm_model")
                    or os.getenv("LETHYX_HINDSIGHT_LLM_MODEL")
                    or os.getenv("LLM_MODEL")
                    or "gemma-4-31B-it"
                ),
                profile=str(
                    self._params.get("profile")
                    or os.getenv("LETHYX_HINDSIGHT_PROFILE")
                    or "memory-bench-Lethyx-native"
                ),
                retain_async=(
                    base_config.retain_async
                    if raw_retain_async is None
                    else _truthy(raw_retain_async)
                ),
            )
            backend = HindsightBackend(repo_root=self._repo_root, workspace_id=ws_id, config=config)
        else:
            backend = create_durable_backend(
                self._repo_root,
                workspace_id=ws_id,
                mode=self._backend_mode,
            )
        if self._native:
            try:
                _fail_if_native_requested_but_not_strict(backend, self._backend_mode)
            except Exception:
                _close_quietly(backend)
                raise
        return _LethyxStore(backend, ws_id, self._backend_mode, native=self._native)


class CompositeLethyxMemoryProvider:
    """Composite Lethyx provider.

    The historical default keeps the benchmark-only fanout ablation.  When the
    suite passes ``production=true`` it instantiates Lethyx's production
    ``CompositeMemoryBackend``: phase-routed active/passive blocks, each backed
    by a real memory technology adapter and queried through that backend's own
    recall/search path.
    """

    name = "lethyx-composite"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = dict(params or {})
        raw_modes = self._params.get("backends") or ("mempalace", "hindsight", "mnemosyne")
        if isinstance(raw_modes, str):
            modes = tuple(part.strip().lower() for part in raw_modes.split(",") if part.strip())
        else:
            modes = tuple(str(part).strip().lower() for part in raw_modes)
        self._modes = modes or ("mempalace", "hindsight", "mnemosyne")
        self._native = _truthy(self._params.get("native") or self._params.get("use_native"))
        self._production = _truthy(
            self._params.get("production")
            or self._params.get("phase_routed")
            or self._params.get("production_composite")
        )
        self._phase_id = str(self._params.get("phase_id") or "memory_bench")
        self._include_passive = _truthy(self._params.get("include_passive") or self._production)
        raw_root = str(self._params.get("repo_root") or "").strip()
        if raw_root:
            self._repo_root = Path(raw_root)
        else:
            candidate = Path(__file__).resolve()
            for _ in range(6):
                candidate = candidate.parent
                if (candidate / "pyproject.toml").is_file():
                    self._repo_root = candidate
                    break
            else:
                self._repo_root = Path.cwd()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            persistent=True,
            supports_update=True,
            supports_delete=False,
            supports_temporal_filter=False,
            supports_graph=True,
            supports_namespacing=True,
            memory_as_tool=False,
            token_accounting=False,
            extra={
                "LETHYX_backends": ",".join(self._modes),
                "native_requested": self._native,
                "production_composite": self._production,
                "phase_id": self._phase_id,
                "include_passive": self._include_passive,
            },
        )

    def _child_store_for(self, mode: str, ns: str) -> _LethyxStore:
        provider = LethyxMemoryProvider(
            {
                "backend": mode,
                "repo_root": str(self._repo_root),
                "native": self._native,
                "embedded": self._params.get(
                    "embedded", self._params.get("hindsight_embedded", False)
                ),
                "profile": self._params.get("profile"),
                "llm_provider": self._params.get("llm_provider"),
            }
        )
        return provider.namespace(ns)

    def _production_block_specs(self) -> list[dict[str, Any]]:
        raw = self._params.get("memory_blocks") or self._params.get("blocks")
        if raw:
            try:
                parsed = json.loads(str(raw)) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, dict):
                parsed = parsed.get("blocks") or []
            if isinstance(parsed, list):
                return [dict(item) for item in parsed if isinstance(item, dict)]

        specs: list[dict[str, Any]] = []
        for mode in self._modes:
            role = "active" if mode == "mempalace" else "passive"
            specs.append(
                {
                    "id": f"{mode}.{role}",
                    "backend": mode,
                    "role": role,
                    "phases": [self._phase_id],
                    "weight": 1.0,
                }
            )
        return specs

    def _production_namespace(self, ns: str) -> _LethyxStore:
        from lethyx.memory.backends.composite import (
            CompositeMemoryBackend,
            MemoryBackendBlock,
            build_memory_blocks_from_specs,
        )

        ws_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", ns)[:64]
        ws_id = f"mb_composite_{ws_id}_{uuid.uuid4().hex[:6]}"

        def backend_factory(backend_id: str) -> Any:
            child = self._child_store_for(backend_id, ns)
            return child._backend

        blocks = build_memory_blocks_from_specs(
            self._production_block_specs(),
            backend_factory=backend_factory,
        )
        blocks = [block for block in blocks if isinstance(block, MemoryBackendBlock)]
        backend = CompositeMemoryBackend(self._repo_root, workspace_id=ws_id, blocks=blocks)
        return _LethyxStore(
            backend,
            ws_id,
            "composite",
            native=self._native,
            phase_id=self._phase_id,
            include_passive=self._include_passive,
        )

    def namespace(self, ns: str) -> _CompositeLethyxStore | _LethyxStore:
        if self._production:
            return self._production_namespace(ns)

        stores: dict[str, Any] = {}
        for mode in self._modes:
            stores[mode] = self._child_store_for(mode, ns)
        return _CompositeLethyxStore(stores)


class RawMemPalaceProvider:
    """Direct MemPalace facade baseline, without the Lethyx backend sidecar."""

    name = "raw-mempalace"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = dict(params or {})
        raw_root = str(self._params.get("repo_root") or "").strip()
        if raw_root:
            self._repo_root = Path(raw_root)
        else:
            candidate = Path(__file__).resolve()
            for _ in range(6):
                candidate = candidate.parent
                if (candidate / "pyproject.toml").is_file():
                    self._repo_root = candidate
                    break
            else:
                self._repo_root = Path.cwd()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            persistent=True,
            supports_update=False,
            supports_delete=False,
            supports_temporal_filter=False,
            supports_graph=True,
            supports_namespacing=True,
            memory_as_tool=False,
            token_accounting=False,
            extra={"raw_backend": "mempalace"},
        )

    def namespace(self, ns: str) -> _RawMemPalaceStore:
        ws_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", ns)[:64]
        ws_id = f"raw_mp_{ws_id}_{uuid.uuid4().hex[:6]}"
        return _RawMemPalaceStore(self._repo_root, ws_id)


class _RawMnemosyneStore:
    """Direct Mnemosyne provider, bypassing Lethyx's typed parity index."""

    def __init__(self, repo_root: Path, workspace_id: str) -> None:
        from mnemosyne import Mnemosyne

        self._workspace_id = workspace_id
        self._data_dir = Path(repo_root) / ".memory_bench_raw" / "mnemosyne" / workspace_id
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._memory = Mnemosyne(
            session_id=workspace_id,
            db_path=self._data_dir / "mnemosyne.db",
        )
        self._n_written = 0

    @staticmethod
    def _metadata_from_row(row: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(row, dict):
            return {}
        raw = row.get("metadata") or row.get("metadata_json") or {}
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return dict(parsed) if isinstance(parsed, dict) else {}
        return dict(raw) if isinstance(raw, dict) else {}

    def _metadata_for_native_id(self, native_id: str) -> dict[str, Any]:
        if not native_id:
            return {}
        try:
            return self._metadata_from_row(self._memory.get(native_id))
        except Exception:
            log.debug("raw Mnemosyne metadata fetch failed", exc_info=True)
            return {}

    def write(self, record: MemoryRecord) -> None:
        metadata = {
            "evidence_id": record.evidence_id or record.id,
            "record_id": record.id,
            "session_id": record.session_id,
            "role": record.role,
            "kind": record.kind or "message",
            "tags": list(record.tags),
        }
        try:
            self._memory.remember(
                record.content,
                source=record.evidence_id or record.id or "memory_bench",
                importance=float(record.metadata.get("importance", 0.8) or 0.8),
                metadata={key: value for key, value in metadata.items() if value},
                scope="session",
            )
            self._n_written += 1
        except Exception as exc:
            log.warning(
                "raw Mnemosyne write failed (ws=%s, record=%s): %s",
                self._workspace_id,
                record.id,
                exc,
            )

    def write_many(self, records: list[MemoryRecord]) -> None:
        for record in records:
            if record.content:
                self.write(record)

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        max_tokens: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedItem]:
        try:
            rows = self._memory.recall(query, top_k=k)
        except Exception as exc:
            log.warning(
                "raw Mnemosyne recall failed (ws=%s, query=%r): %s",
                self._workspace_id,
                query[:80],
                exc,
            )
            return []

        out: list[RetrievedItem] = []
        used_chars = 0
        budget_chars = max_tokens * 4 if max_tokens > 0 else 0
        for i, row in enumerate(list(rows or [])[:k]):
            item = dict(row) if isinstance(row, dict) else {}
            text = str(item.get("content") or item.get("text") or "")
            if budget_chars and out and used_chars + len(text) > budget_chars:
                break
            if budget_chars and not out and len(text) > budget_chars:
                break
            native_id = str(item.get("id") or "")
            metadata = self._metadata_for_native_id(native_id)
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            source_id = str(
                metadata.get("evidence_id")
                or item.get("source")
                or metadata.get("record_id")
                or native_id
                or ""
            )
            out.append(
                RetrievedItem(
                    text=text,
                    score=score,
                    source_id=source_id,
                    rank=i,
                    metadata={**metadata, **item, "raw_backend": "mnemosyne"},
                )
            )
            used_chars += len(text)
        return out

    def update(self, record_id: str, **changes: Any) -> bool:
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "raw-mnemosyne",
            "workspace_id": self._workspace_id,
            "n_written": self._n_written,
        }

    def close(self) -> None:
        connections = []
        conn = getattr(self._memory, "conn", None)
        if conn is not None:
            connections.append(conn)
        beam = getattr(self._memory, "beam", None)
        beam_conn = getattr(beam, "conn", None)
        if beam_conn is not None and beam_conn not in connections:
            connections.append(beam_conn)
        for conn in connections:
            try:
                conn.close()
            except Exception:
                log.debug("raw Mnemosyne close failed", exc_info=True)


class RawMnemosyneProvider:
    """Direct Mnemosyne baseline, without Lethyx wrapper metadata/indexing."""

    name = "raw-mnemosyne"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = dict(params or {})
        raw_root = str(self._params.get("repo_root") or "").strip()
        if raw_root:
            self._repo_root = Path(raw_root)
        else:
            candidate = Path(__file__).resolve()
            for _ in range(6):
                candidate = candidate.parent
                if (candidate / "pyproject.toml").is_file():
                    self._repo_root = candidate
                    break
            else:
                self._repo_root = Path.cwd()

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            persistent=True,
            supports_update=False,
            supports_delete=False,
            supports_temporal_filter=False,
            supports_graph=False,
            supports_namespacing=True,
            memory_as_tool=False,
            token_accounting=False,
            extra={"raw_backend": "mnemosyne"},
        )

    def namespace(self, ns: str) -> _RawMnemosyneStore:
        ws_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", ns)[:64]
        ws_id = f"raw_mn_{ws_id}_{uuid.uuid4().hex[:6]}"
        return _RawMnemosyneStore(self._repo_root, ws_id)


class RawHindsightProvider:
    """Direct Hindsight embedded-client baseline, without the Lethyx sidecar."""

    name = "raw-hindsight"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = dict(params or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            _load_env_for_native_clients()
            raw_embedded = self._params.get(
                "embedded",
                os.getenv("MEMORY_BENCH_RAW_HINDSIGHT_EMBEDDED", ""),
            )
            embedded = str(raw_embedded).strip().lower() in {"1", "true", "yes", "on"}
            base_url = (
                self._params.get("base_url")
                or os.getenv("HINDSIGHT_BASE_URL")
                or os.getenv("HINDSIGHT_API_BASE_URL")
                or os.getenv("LETHYX_HINDSIGHT_BASE_URL")
            )
            if not embedded and not base_url:
                raise RuntimeError(
                    "raw Hindsight requires base_url/HINDSIGHT_BASE_URL or "
                    "embedded=true; embedded startup can be slow, so it is opt-in"
                )
            if embedded:
                from hindsight import HindsightEmbedded

                kwargs: dict[str, Any] = {
                    "profile": str(self._params.get("profile") or "memory-bench-raw"),
                    "llm_provider": str(
                        self._params.get("llm_provider")
                        or os.getenv("HINDSIGHT_API_LLM_PROVIDER")
                        or os.getenv("MEMORY_BENCH_HINDSIGHT_LLM_PROVIDER")
                        or "openai"
                    ),
                    "llm_model": str(
                        self._params.get("llm_model")
                        or os.getenv("HINDSIGHT_API_LLM_MODEL")
                        or os.getenv("MEMORY_BENCH_HINDSIGHT_LLM_MODEL")
                        or os.getenv("LLM_MODEL")
                        or "gemma-4-31B-it"
                    ),
                    "llm_api_key": str(
                        self._params.get("llm_api_key")
                        or os.getenv("HINDSIGHT_API_LLM_API_KEY")
                        or os.getenv("OPENAI_API_KEY")
                        or ""
                    ),
                }
                llm_base_url = self._params.get("llm_base_url") or os.getenv("OPENAI_BASE_URL")
                if llm_base_url:
                    kwargs["llm_base_url"] = str(llm_base_url)
                self._client = HindsightEmbedded(**kwargs)
            else:
                from hindsight_client import Hindsight

                kwargs = {"base_url": str(base_url)}
                timeout = self._params.get("timeout") or os.getenv("HINDSIGHT_TIMEOUT_SECONDS")
                if timeout:
                    kwargs["timeout"] = float(timeout)
                api_key = self._params.get("api_key") or os.getenv("HINDSIGHT_API_KEY")
                if api_key:
                    kwargs["api_key"] = str(api_key)
                self._client = Hindsight(**kwargs)
        return self._client

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            persistent=True,
            supports_update=False,
            supports_delete=False,
            supports_temporal_filter=False,
            supports_graph=False,
            supports_namespacing=True,
            memory_as_tool=False,
            token_accounting=False,
            extra={"raw_backend": "hindsight"},
        )

    def namespace(self, ns: str) -> _RawHindsightStore:
        ws_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", ns)[:64]
        ws_id = f"raw_hs_{ws_id}_{uuid.uuid4().hex[:6]}"
        return _RawHindsightStore(self._get_client(), ws_id)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _lethyx_factory(params: dict[str, Any] | None = None) -> LethyxMemoryProvider:
    return LethyxMemoryProvider(params)


# Register with the framework registry.
# We register three named entries (one per backend) so suite.toml can say
# adapter = "lethyx" with backend = "mempalace" or just use the pre-named aliases.
try:
    register_provider("lethyx")(_lethyx_factory)
    register_provider("lethyx-mempalace")(
        lambda params=None: LethyxMemoryProvider({**(params or {}), "backend": "mempalace"})
    )
    register_provider("lethyx-hindsight")(
        lambda params=None: LethyxMemoryProvider({**(params or {}), "backend": "hindsight"})
    )
    register_provider("lethyx-mnemosyne")(
        lambda params=None: LethyxMemoryProvider({**(params or {}), "backend": "mnemosyne"})
    )
    register_provider("lethyx-composite")(lambda params=None: CompositeLethyxMemoryProvider(params))
    register_provider("raw-mempalace")(lambda params=None: RawMemPalaceProvider(params))
    register_provider("raw-mnemosyne")(lambda params=None: RawMnemosyneProvider(params))
    register_provider("raw-hindsight")(lambda params=None: RawHindsightProvider(params))
except Exception:
    pass
