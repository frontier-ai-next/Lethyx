"""Mnemosyne selected durable backend adapter.

The optional upstream package can be present for richer native behavior.  When
native Mnemosyne is enabled and importable, this adapter writes durable runtime
blocks into Mnemosyne's own BEAM SQLite store and retrieves through its native
hybrid recall.  A backend-owned SQLite index is still maintained for UM-MEM's
neutral contracts (metadata round-trip, lifecycle records, export/import), but
native recall does not read from that parity index.
"""

import hashlib
import importlib.util
import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from typing import Any

from um_mem.contracts import (
    ContextualUtilitySummary,
    EvidenceRef,
    MemoryBlockRecord,
    MemoryDecisionEvent,
    MemoryDecisionTrace,
    MemoryUsageEvent,
    OutcomeCreditEvent,
    ProspectiveObligation,
)
from um_mem.memory.backends.base import (
    MemoryHit,
    MemoryQuery,
    NativeProjectionPending,
    ReconcileReport,
)
from um_mem.memory.backends.contracts import (
    BackendCapabilities,
    CanonicalHeadConflict,
    CanonicalHeadReservation,
    MemoryEdgeRecord,
    MemoryIndexNode,
    MemoryUsageRecord,
    NativePartitionIdentity,
    block_from_json,
    block_record_to_memory_hit,
    block_to_json,
)
from um_mem.memory.backends.query_policy import (
    NATIVE_PROJECTION_REBUILD_REQUIRED,
    ensure_record_is_storable,
    merge_native_block_results,
    native_block_source,
    native_primary_block_id,
    normalize_query_payload,
    record_id_sequence,
)
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex, WriteOutboxEntry
from um_mem.memory.dependencies import DependencyValidation
from um_mem.memory.mutation import (
    AmendmentCommitReceipt,
    CanonicalMutationIdentity,
    MutationDecision,
    PublicationCommitReceipt,
)
from um_mem.memory.paths import MANAGER_WORKSPACE_ID, workspace_memory_root
from um_mem.memory.prospective import ProspectiveTriggerContext, ProspectiveTriggerDecision

_LOG = logging.getLogger(__name__)


_NATIVE_IMPORT_LOCK = threading.RLock()
MNEMOSYNE_PARTITION_PROJECTION_MARKER = "mnemosyne_session_partition_v2"
#: Mnemosyne supplements its own recall with MEMORIA rows. One of them is a
#: context string and the others are duplicate views of working-memory rows.
#: They are extraction artifacts without a canonical UM-MEM identity, so they
#: are never hydrated as evidence.
MNEMOSYNE_MEMORIA_TIERS = frozenset({"memoria", "memoria_source"})


def _configure_native_embeddings(repo_root: Path | None = None) -> None:
    """Keep the native backend from loading a second embedding model.

    The local retrieval tier owns the semantic path. Mnemosyne would otherwise
    load a model of its own for dense recall, which means a second copy in
    resident memory and a second download on a fresh machine. The native side
    therefore stays keyword-only unless an operator asks for native dense
    recall explicitly with ``MNEMOSYNE_EMBEDDING_MODEL``.
    """

    if "MNEMOSYNE_EMBEDDING_MODEL" in os.environ:
        # The operator asked for native dense recall: respect that choice.
        return
    os.environ.setdefault("MNEMOSYNE_NO_EMBEDDINGS", "1")


def _enabled() -> bool:
    return os.getenv("UM_MEM_MNEMOSYNE_ENABLED", "1").strip().lower() in {
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
        reconcile_pending_writes: bool = True,
    ):
        self.repo_root = Path(repo_root)
        self.workspace_id = workspace_id
        self.enabled = _enabled() if enabled is None else enabled
        self.available = importlib.util.find_spec("mnemosyne") is not None
        self.data_dir = data_dir or Path(
            os.getenv("UM_MEM_MNEMOSYNE_DATA_DIR")
            or workspace_memory_root(self.repo_root, workspace_id or MANAGER_WORKSPACE_ID)
            / "mnemosyne"
        )
        self._index = BackendSQLiteIndex(
            self.data_dir / "UM_MEM_index.sqlite",
            backend_id="mnemosyne",
            workspace_id=workspace_id,
            repo_root=self.repo_root,
        )
        self._native_memory: Any | None = None
        self._native_partitions: dict[str, Any] = {}
        if reconcile_pending_writes:
            self._index.replay_pending_canonical_heads()
        self._index.configure_native_projection(
            self._project_durable_record, start=reconcile_pending_writes
        )

    @classmethod
    def from_env(
        cls,
        repo_root: Path,
        workspace_id: str = "",
        *,
        reconcile_pending_writes: bool = True,
    ) -> "MnemosyneBackend":
        return cls(
            repo_root=repo_root,
            workspace_id=workspace_id,
            reconcile_pending_writes=reconcile_pending_writes,
        )

    @staticmethod
    def _native_surface_ready(native: Any) -> bool:
        return all(
            callable(getattr(native, name, None))
            for name in ("remember", "recall", "get", "get_all_memories")
        )

    def _construct_native(self, session_partition_id: str) -> Any:
        if (self.data_dir / "mnemosyne.db").exists():
            raise ValueError(NATIVE_PROJECTION_REBUILD_REQUIRED)
        partition_dir = (
            self.data_dir
            / "partitions"
            / hashlib.sha256(session_partition_id.encode("utf-8")).hexdigest()
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        with _NATIVE_IMPORT_LOCK:
            previous = os.environ.get("MNEMOSYNE_DATA_DIR")
            os.environ["MNEMOSYNE_DATA_DIR"] = str(partition_dir)
            try:
                # The embedding defaults have to be in place before Mnemosyne
                # reads them at import time.
                _configure_native_embeddings(self.repo_root)
                from mnemosyne import Mnemosyne  # type: ignore[import-untyped]

                return Mnemosyne(
                    session_id=session_partition_id,
                    db_path=partition_dir / "mnemosyne.db",
                )
            finally:
                if previous is None:
                    os.environ.pop("MNEMOSYNE_DATA_DIR", None)
                else:
                    os.environ["MNEMOSYNE_DATA_DIR"] = previous

    def _native_ok(self) -> bool:
        if not self.enabled or not self.available:
            return False
        try:
            return self._native_surface_ready(
                self._native(
                    self.native_partition_identity(
                        "__health__",
                        workspace_id=self.workspace_id or "default",
                        catalog_revision_id="__health__",
                        backend_partition_id="__health__",
                    )
                )
            )
        except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error):
            _LOG.debug("Mnemosyne native initialization unavailable", exc_info=True)
            return False

    def _require_native(self) -> None:
        if not self.enabled:
            raise RuntimeError("native Mnemosyne is disabled")
        if not self.available:
            raise RuntimeError("native Mnemosyne dependency is not installed")

    @staticmethod
    def native_partition_identity(
        primary_block_id: str,
        *,
        workspace_id: str,
        catalog_revision_id: str,
        backend_partition_id: str,
    ) -> NativePartitionIdentity:
        block_id = str(primary_block_id).strip()
        workspace = str(workspace_id).strip()
        catalog_revision = str(catalog_revision_id).strip()
        if not block_id:
            raise ValueError("Mnemosyne native session partition requires a primary block id")
        if not workspace:
            raise ValueError("Mnemosyne native session partition requires a workspace id")
        if not catalog_revision:
            raise ValueError("Mnemosyne native session partition requires a catalog revision")
        partition_key = str(backend_partition_id or "").strip()
        if not partition_key:
            raise ValueError("Mnemosyne native session partition requires backend_partition_id")
        workspace_digest = hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:16]
        partition_digest = hashlib.sha256(partition_key.encode("utf-8")).hexdigest()[:16]
        return NativePartitionIdentity(
            backend_id="mnemosyne",
            workspace_id=workspace,
            primary_block_id=block_id,
            catalog_revision_id=catalog_revision,
            partition_id=(
                f"um-mem:{workspace_digest}:partition:{partition_digest}:block:{block_id}"
            ),
            isolation_class="physical",
            partition_mode="native_session_partition",
        )

    @classmethod
    def _native_session_partition_id(
        cls,
        primary_block_id: str,
        *,
        workspace_id: str,
        catalog_revision_id: str,
        backend_partition_id: str,
    ) -> str:
        return cls.native_partition_identity(
            primary_block_id=primary_block_id,
            workspace_id=workspace_id,
            catalog_revision_id=catalog_revision_id,
            backend_partition_id=backend_partition_id,
        ).partition_id

    @classmethod
    def _record_partition_identity(
        cls,
        record: MemoryBlockRecord,
    ) -> NativePartitionIdentity:
        membership = record.block_membership
        if membership is None:
            raise ValueError("Mnemosyne native partition requires typed block membership")
        return cls.native_partition_identity(
            workspace_id=record.workspace_id,
            primary_block_id=membership.primary_block_id,
            catalog_revision_id=membership.catalog_revision_id,
            backend_partition_id=membership.backend_partition_id,
        )

    def _native(self, partition: NativePartitionIdentity) -> Any:
        if not self.enabled or not self.available:
            self._require_native()
        if self._native_memory is not None:
            native = self._native_memory
        else:
            session_partition_id = partition.partition_id
            native = self._native_partitions.get(session_partition_id)
            if native is None:
                native = self._construct_native(session_partition_id)
                self._native_partitions[session_partition_id] = native
        if not self._native_surface_ready(native):
            raise RuntimeError("native Mnemosyne API surface is unavailable")
        return native

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
    def _native_source_identity(record: MemoryBlockRecord) -> str:
        return f"um-mem:{record.workspace_id}:{record.block_id}"

    @staticmethod
    def _native_block_source(record: MemoryBlockRecord) -> str:
        return native_block_source(native_primary_block_id(record))

    @staticmethod
    def _native_idempotency_key(record: MemoryBlockRecord, commit_revision: int) -> str:
        content_hash = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        material = f"{record.workspace_id}\0{record.block_id}\0{commit_revision}\0{content_hash}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _provider_temporal_tags(content: str) -> str:
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", content)
        tagged = content
        if dates:
            tagged = f"{tagged} [DATES: {', '.join(dates)}]"
        durations = re.findall(r"\b\d+\s(?:days|weeks|months|years)\b", tagged, re.IGNORECASE)
        if durations:
            tagged = f"{tagged} [DURATIONS: {', '.join(durations)}]"
        return tagged

    @classmethod
    def _native_content_envelope(
        cls,
        record: MemoryBlockRecord,
        commit_revision: int,
    ) -> str:
        return f"{record.content}\n[UMM:{cls._native_idempotency_key(record, commit_revision)}]"

    @classmethod
    def _native_expected_content(
        cls,
        record: MemoryBlockRecord,
        commit_revision: int,
    ) -> str:
        return cls._native_content_envelope(record, commit_revision)

    @classmethod
    def _native_content_variants(
        cls,
        record: MemoryBlockRecord,
        commit_revision: int,
    ) -> set[str]:
        """Accepted stored contents across Mnemosyne versions.

        Before Mnemosyne 3.x the provider appended ``[DATES: ...]`` /
        ``[DURATIONS: ...]`` to the stored text.  Current releases keep those
        annotations in metadata (``_dates`` / ``_durations``) and store the
        caller's text verbatim, so both shapes are valid for rows written by
        UM-MEM.
        """

        envelope = cls._native_content_envelope(record, commit_revision)
        return {envelope, cls._provider_temporal_tags(envelope)}

    @staticmethod
    def _importance(record: MemoryBlockRecord) -> float:
        return 0.8 if record.verification_status == "verified" else 0.5

    @staticmethod
    def _native_scope(record: MemoryBlockRecord) -> str:
        del record
        return "session"

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _tags_from_any(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(tag.strip() for tag in value.split(",") if tag.strip())
        if isinstance(value, (list, tuple, set, frozenset)):
            return tuple(str(tag).strip() for tag in value if str(tag).strip())
        return ()

    def _native_row_to_block(self, row: Any) -> MemoryBlockRecord:
        item = dict(row) if isinstance(row, dict) else {}
        raw_metadata = item.get("metadata_json") or item.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        elif isinstance(raw_metadata, str):
            try:
                parsed_metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as exc:
                raise ValueError("mnemosyne native metadata is invalid") from exc
            metadata = parsed_metadata if isinstance(parsed_metadata, dict) else {}
        else:
            metadata = {}
        raw = metadata.get("canonical_record_json")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("mnemosyne native canonical record is missing")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("mnemosyne native canonical record is invalid") from exc
        try:
            record = block_from_json(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("mnemosyne native canonical record is invalid") from exc
        native_block_id = str(metadata.get("block_id") or "")
        if native_block_id and native_block_id != record.block_id:
            raise ValueError("mnemosyne native canonical record id mismatch")
        try:
            commit_revision = int(str(metadata.get("commit_revision") or ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("mnemosyne native commit revision is invalid") from exc
        if str(metadata.get("workspace_id") or "") != record.workspace_id:
            raise ValueError("mnemosyne native canonical record workspace mismatch")
        if str(metadata.get("source_identity") or "") != self._native_source_identity(record):
            raise ValueError("mnemosyne native source identity mismatch")
        projection_marker = str(metadata.get("projection_marker") or "")
        if projection_marker != MNEMOSYNE_PARTITION_PROJECTION_MARKER:
            raise ValueError(NATIVE_PROJECTION_REBUILD_REQUIRED)
        primary_block_id = native_primary_block_id(record)
        if str(metadata.get("primary_block_id") or "") != primary_block_id:
            raise ValueError("mnemosyne native primary block mismatch")
        session_partition_id = self._record_partition_identity(record).partition_id
        if (
            str(metadata.get("catalog_revision_id") or "")
            != self._record_partition_identity(record).catalog_revision_id
        ):
            raise ValueError("mnemosyne native catalog revision mismatch")
        if str(metadata.get("session_partition_id") or "") != session_partition_id:
            raise ValueError("mnemosyne native session partition mismatch")
        row_session_id = str(item.get("session_id") or "")
        if row_session_id and row_session_id != session_partition_id:
            raise ValueError("mnemosyne native row escaped its session partition")
        if str(item.get("source") or "") != native_block_source(primary_block_id):
            raise ValueError("mnemosyne native block source mismatch")
        content = item.get("content")
        if not isinstance(content, str) or content not in self._native_content_variants(
            record, commit_revision
        ):
            raise ValueError("mnemosyne native canonical record content mismatch")
        return record

    def _overlay_native_sidecar_contract(
        self,
        record: MemoryBlockRecord,
        query: dict[str, Any] | None = None,
    ) -> MemoryBlockRecord | None:
        return self._index.overlay_native_hit(
            record,
            workspace_id=self.workspace_id,
            query=query,
        )

    def capabilities(self) -> BackendCapabilities:
        health = self.health()
        native_ok = health.get("native_ok") is True and health.get("effective_mode") == "native"
        return BackendCapabilities.effective(
            "mnemosyne",
            native_ok=native_ok,
            implemented=frozenset(
                {
                    "persistent",
                    "semantic_recall",
                    "lifecycle_mutation",
                    "workspace_isolation",
                    "run_subtask_isolation",
                    "value_tracking",
                    "export_import",
                    "exact_record_read",
                    "scoped_search",
                    "revision_history",
                    "revision_compare_and_swap",
                    "recoverable_writes",
                    "staged_topology_evolution",
                }
            ),
            supports_embedded=True,
        )

    def _existing_native_partitions(
        self,
        *,
        logical_block_ids: tuple[str, ...] = (),
        profile_id: str = "",
    ) -> tuple[NativePartitionIdentity, ...]:
        selected_order = {block_id: index for index, block_id in enumerate(logical_block_ids)}
        return tuple(
            sorted(
                (
                    self.native_partition_identity(
                        primary_block_id=block_id,
                        workspace_id=self.workspace_id,
                        catalog_revision_id=catalog_revision_id,
                        backend_partition_id=partition_id,
                    )
                    for block_id, catalog_revision_id, partition_id in self._index.native_partition_memberships(
                        workspace_id=self.workspace_id,
                        profile_id=profile_id,
                        logical_block_ids=logical_block_ids,
                    )
                ),
                key=lambda item: (
                    selected_order.get(item.primary_block_id, len(selected_order)),
                    item.primary_block_id,
                    item.catalog_revision_id,
                    item.partition_id,
                ),
            )
        )

    def _partition_projection_ready(self) -> tuple[bool, str]:
        for partition in self._existing_native_partitions():
            native = self._native(partition)
            get_all_memories = getattr(native, "get_all_memories", None)
            get_memory = getattr(native, "get", None)
            if not callable(get_all_memories) or not callable(get_memory):
                return False, "native_partition_surface_unavailable"
            for value in get_all_memories() or ():
                row = dict(value) if isinstance(value, dict) else {}
                native_id = str(row.get("id") or "")
                if not native_id:
                    return False, "native_partition_row_missing_id"
                detail = get_memory(native_id)
                if not isinstance(detail, dict):
                    return False, "native_partition_detail_missing"
                raw_metadata = detail.get("metadata") or detail.get("metadata_json") or {}
                if isinstance(raw_metadata, str):
                    try:
                        raw_metadata = json.loads(raw_metadata)
                    except json.JSONDecodeError:
                        return False, "native_partition_metadata_invalid"
                if not isinstance(raw_metadata, dict):
                    return False, "native_partition_metadata_missing"
                if (
                    str(raw_metadata.get("projection_marker") or "")
                    != MNEMOSYNE_PARTITION_PROJECTION_MARKER
                ):
                    return False, NATIVE_PROJECTION_REBUILD_REQUIRED
        return True, ""

    def health(self) -> dict[str, Any]:
        return self._health(verify_native_partitions=False)

    def block_health(
        self, logical_block_ids: tuple[str, ...], *, profile_id: str
    ) -> dict[str, Any]:
        if not profile_id or not 1 <= len(logical_block_ids) <= 201:
            raise ValueError("block health requires a profile and 1-201 selected block IDs")
        partitions = tuple(
            (
                self.native_partition_identity(
                    workspace_id=self.workspace_id,
                    primary_block_id=block_id,
                    catalog_revision_id=revision_id,
                    backend_partition_id=partition_id,
                ),
                partition_id,
            )
            for block_id, revision_id, partition_id in self._index.native_partition_memberships(
                workspace_id=self.workspace_id,
                logical_block_ids=logical_block_ids,
                profile_id=profile_id,
            )
        )
        return self._index.partition_health(
            partitions,
            profile_id=profile_id,
            partition_factory=lambda block, revision, raw: self.native_partition_identity(
                primary_block_id=block,
                workspace_id=self.workspace_id,
                catalog_revision_id=revision,
                backend_partition_id=raw,
            ),
        )

    def preflight_health(self) -> dict[str, Any]:
        return self._health(verify_native_partitions=True)

    def _health(self, *, verify_native_partitions: bool) -> dict[str, Any]:
        partitions = self._existing_native_partitions()
        native_ok = self._native_ok()
        effective_mode = "native" if native_ok else "unavailable"
        reason = ""
        if not self.enabled:
            reason = "native_disabled"
        elif not self.available:
            reason = "dependency_unavailable"
        elif (self.data_dir / "mnemosyne.db").exists():
            reason = NATIVE_PROJECTION_REBUILD_REQUIRED
        elif not native_ok:
            reason = "native_api_unavailable"
        partition_ready = native_ok
        partition_reason = reason
        if native_ok and verify_native_partitions:
            try:
                partition_ready, partition_reason = self._partition_projection_ready()
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                partition_reason = f"native_partition_check_failed:{type(exc).__name__}"
        metric_health = (
            "ready"
            if native_ok and partition_ready
            else f"unavailable:{reason or partition_reason}"
        )
        return {
            **self._index.operational_revisions(),
            "projection": self._index.native_projection_health(),
            "strict_partition_validation": verify_native_partitions,
            "ok": native_ok,
            "backend": "mnemosyne",
            "selected": True,
            "enabled": self.enabled,
            "available": self.available,
            "native_ok": native_ok,
            "strict_ok": native_ok and partition_ready,
            "native_block_scoping_supported": native_ok and partition_ready,
            "native_block_scoping_reason": (
                "native_session_partition" if partition_ready else partition_reason
            ),
            "native_partition_isolation_class": "physical",
            "native_partition_mode": "native_session_partition",
            "native_partition_count": len(partitions),
            "native_partitions": [partition.to_mapping() for partition in partitions],
            "block_metrics": [
                self._index.partition_metrics(
                    partition,
                    health=metric_health,
                ).to_mapping()
                for partition in partitions
            ],
            "effective_mode": effective_mode,
            "reason": reason or partition_reason,
            "parity_ok": False,
            "operational_ok": native_ok,
            "data_dir": str(self.data_dir),
            "workspace_id": self.workspace_id,
            "provider_prompt_injection_used": False,
            "implementation": "mnemosyne_memory_beam" if native_ok else "unavailable",
            "operational_note": (
                "Primary UM-MEM blocks are isolated in separate native Mnemosyne databases; "
                "multi-block recall merges native ranks round by round."
            )
            if native_ok and partition_ready
            else (
                "Existing Mnemosyne native projections require a clean partition rebuild."
                if native_ok and partition_reason == NATIVE_PROJECTION_REBUILD_REQUIRED
                else (
                    "Mnemosyne native partition health is unavailable; strict block scoping "
                    "fails closed."
                )
            )
            if native_ok
            else (
                "Mnemosyne native layer is unavailable; the selected "
                "Mnemosyne backend fails closed instead of using the "
                "UM-MEM SQLite parity index as a recall implementation."
            ),
        }

    def ensure_banks(self, *, workspace_id: str = "") -> dict[str, Any]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._native(
            self.native_partition_identity(
                "__health__",
                workspace_id=self.workspace_id or "default",
                catalog_revision_id="__health__",
                backend_partition_id="__health__",
            )
        )
        return {
            "ok": True,
            "backend": "mnemosyne",
            "data_dir": str(self.data_dir),
            "workspace_id": workspace_id or self.workspace_id,
            "native": True,
        }

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        with self._index.projection_gate(record):
            return self._retain_block(record)

    def _retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        started_at = time.perf_counter()
        ensure_record_is_storable(record)
        outbox = self._index.open_write_outbox(
            record=record,
            store="mnemosyne",
        )
        record = self._index.write_outbox_record(outbox)
        result = self._index.retain_block(record)
        native_id = self._project_write_outbox(outbox, record)
        partition = self._record_partition_identity(record)
        self._index.record_partition_operation(
            partition,
            operation="write",
            latency_ms=(time.perf_counter() - started_at) * 1_000,
            record_bytes=len(
                json.dumps(
                    block_to_json(record),
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ),
            run_id=record.run_id,
            profile_id=record.profile_id,
        )
        return {
            **result,
            "ok": True,
            "backend": "mnemosyne",
            "native": True,
            "block_id": record.block_id,
            "memory_id": native_id,
            "commit_revision": outbox.commit_revision,
        }

    def prepare_canonical_write(self, record: MemoryBlockRecord) -> dict[str, Any]:
        ensure_record_is_storable(record)
        entry = self._index.open_write_outbox(record=record, store="mnemosyne")
        return {"ok": True, "commit_revision": entry.commit_revision}

    def enqueue_native_projection(self, record: MemoryBlockRecord) -> dict[str, Any]:
        return self._index.enqueue_native_projection(record)

    def canonical_write_gate(self) -> AbstractContextManager[None]:
        return self._index.canonical_write_gate()

    def atomic_canonical_write_gate(self) -> AbstractContextManager[None]:
        return self._index.atomic_canonical_write_gate()

    def wake_native_projection(self) -> None:
        self._index.wake_native_projection()

    def _project_durable_record(self, record: MemoryBlockRecord) -> None:
        self._require_native()
        entry = self._index.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None:
            raise ValueError("native projection outbox is missing")
        self._project_write_outbox(entry, record)

    def stage_topology_projection(self, record: MemoryBlockRecord) -> None:
        entry = self._index.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None or entry.status != "committed":
            raise NativeProjectionPending("native_projection_pending")
        self._require_native()
        with self._index.projection_gate(record):
            row = self._native_row_for_source(record, entry.commit_revision)
            if row is None:
                native_id = self._native(self._record_partition_identity(record)).remember(
                    self._native_content_envelope(record, entry.commit_revision),
                    source=self._native_block_source(record),
                    importance=self._importance(record),
                    metadata=self._native_projection_metadata(record, entry.commit_revision),
                    scope=self._native_scope(record),
                    valid_until=record.expires_at or None,
                    trust_tier=record.trust_level or None,
                )
                if not isinstance(native_id, str) or not native_id.strip():
                    raise RuntimeError("Mnemosyne topology projection returned no native ID")
                row = self._native_row_for_source(record, entry.commit_revision)
            if row is None or self._native_row_to_block(row) != record:
                raise RuntimeError("staged Mnemosyne topology projection is invalid")

    def _project_write_outbox(
        self,
        entry: WriteOutboxEntry,
        record: MemoryBlockRecord,
    ) -> str:
        with self._index.projection_gate(record):
            return self._project_write_outbox_locked(entry, record)

    def _project_write_outbox_locked(
        self, entry: WriteOutboxEntry, record: MemoryBlockRecord
    ) -> str:
        block_source = self._native_block_source(record)
        actual = self._index.write_outbox_entry(
            block_id=entry.block_id,
            commit_revision=entry.commit_revision,
        )
        if actual is None:
            raise RuntimeError("selected backend write journal is unavailable")
        stored_native_id = self._index.write_outbox_native_id(entry.commit_revision)
        if actual.status == "committed":
            if not stored_native_id:
                raise RuntimeError("Mnemosyne committed write has no immutable native reference")
            projected, error = self._native_record_for_write_outbox(entry)
            if error or projected != record:
                raise RuntimeError(
                    "Mnemosyne committed native projection validation failed"
                    + (f": {error}" if error else "")
                )
            return stored_native_id
        if actual.status != "pending":
            raise RuntimeError("selected backend write journal is not replayable")

        if stored_native_id:
            native_id = stored_native_id
            projected, error = self._native_record_for_write_outbox(entry)
            if error or projected != record:
                raise RuntimeError(
                    "Mnemosyne native projection validation failed"
                    + (f": {error}" if error else "")
                )
        else:
            row = (
                self._native_row_for_source(record, entry.commit_revision) if actual.error else None
            )
            if row is None:
                self._index.mark_write_outbox_projection_started(entry.commit_revision)
                native_id = self._native(self._record_partition_identity(record)).remember(
                    self._native_content_envelope(record, entry.commit_revision),
                    source=block_source,
                    importance=self._importance(record),
                    metadata=self._native_projection_metadata(record, entry.commit_revision),
                    scope=self._native_scope(record),
                    valid_until=record.expires_at or None,
                    trust_tier=record.trust_level or None,
                )
                if not isinstance(native_id, str) or not native_id.strip():
                    raise RuntimeError("Mnemosyne native write returned no memory id")
            else:
                native_id = str(row.get("id") or "")
                if not native_id:
                    raise RuntimeError("Mnemosyne native source row has no memory id")
            self._index.set_write_outbox_native_ref(entry.commit_revision, native_id)
        projected, error = self._native_record_for_write_outbox(entry)
        if error or projected != record:
            raise RuntimeError(
                "Mnemosyne native projection validation failed" + (f": {error}" if error else "")
            )
        self._index.complete_write_outbox(entry.commit_revision)
        return native_id

    def _native_projection_metadata(
        self, record: MemoryBlockRecord, commit_revision: int
    ) -> dict[str, Any]:
        partition = self._record_partition_identity(record)
        return {
            **{
                key: value
                for key, value in dict(record.metadata or {}).items()
                if key != "importance"
            },
            "block_id": record.block_id,
            "commit_revision": commit_revision,
            "evidence_id": self._record_evidence_id(record),
            "source_identity": self._native_source_identity(record),
            "primary_block_id": native_primary_block_id(record),
            "projection_marker": MNEMOSYNE_PARTITION_PROJECTION_MARKER,
            "session_partition_id": partition.partition_id,
            "catalog_revision_id": partition.catalog_revision_id,
            "native_idempotency_key": self._native_idempotency_key(record, commit_revision),
            "canonical_content_sha256": hashlib.sha256(record.content.encode("utf-8")).hexdigest(),
            "title": record.title,
            "kind": record.kind,
            "status": record.status,
            "retention": record.retention,
            "delivery_policy": record.delivery_policy,
            "temperature": record.temperature,
            "authority": record.authority,
            "trust_level": record.trust_level,
            "scope": record.scope,
            "workspace_id": record.workspace_id or self.workspace_id,
            "run_id": record.run_id,
            "subtask_id": record.subtask_id,
            "producer": record.producer,
            "tags": list(record.tags),
            "external_refs": dict(record.external_refs),
            "sensitivity": record.sensitivity,
            "content_mode": record.content_mode,
            "operational_risk": record.operational_risk,
            "revision_of": record.revision_of,
            "supersedes_ids": list(record.supersedes_ids),
            "verification_status": record.verification_status,
            "verification_methods": list(record.verification_methods),
            "promotion_decision": record.promotion_decision,
            "expires_at": record.expires_at,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "canonical_record_json": json.dumps(
                block_to_json(record), ensure_ascii=False, sort_keys=True
            ),
        }

    def _native_row_for_source(
        self,
        record: MemoryBlockRecord,
        commit_revision: int,
    ) -> dict[str, Any] | None:
        primary_block_id = native_primary_block_id(record)
        native = self._native(self._record_partition_identity(record))
        get_all_memories = getattr(native, "get_all_memories", None)
        if not callable(get_all_memories):
            raise RuntimeError("Mnemosyne native summary API is unavailable")
        get_memory = getattr(native, "get", None)
        if not callable(get_memory):
            raise RuntimeError("Mnemosyne native detail API is unavailable")
        rows: Any = get_all_memories()
        source_identity = self._native_source_identity(record)
        block_source = self._native_block_source(record)
        matches: list[dict[str, Any]] = []
        for value in rows or []:
            row = dict(value) if isinstance(value, dict) else {}
            if str(row.get("source") or "") != block_source:
                continue
            native_id = str(row.get("id") or "")
            if not native_id:
                raise RuntimeError("Mnemosyne native source summary has no memory id")
            detail = get_memory(native_id)
            if not isinstance(detail, dict):
                raise RuntimeError("Mnemosyne native source detail is missing")
            if str(detail.get("source") or "") != block_source:
                raise RuntimeError("Mnemosyne native block source detail mismatch")
            metadata = detail.get("metadata") or detail.get("metadata_json") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    raise RuntimeError("Mnemosyne native source detail metadata is invalid")
            if not isinstance(metadata, dict):
                raise RuntimeError("Mnemosyne native source detail metadata is unavailable")
            if str(metadata.get("block_id") or "") == record.block_id and str(
                metadata.get("workspace_id") or ""
            ) == (record.workspace_id or self.workspace_id):
                if str(metadata.get("commit_revision") or "") != str(commit_revision):
                    raise RuntimeError(
                        "Mnemosyne native source identity has a conflicting revision"
                    )
                if str(metadata.get("source_identity") or "") != source_identity:
                    raise RuntimeError("Mnemosyne native source identity is invalid")
                if str(metadata.get("primary_block_id") or "") != primary_block_id:
                    raise RuntimeError("Mnemosyne native primary block is invalid")
                if str(metadata.get("projection_marker") or "") != (
                    MNEMOSYNE_PARTITION_PROJECTION_MARKER
                ):
                    raise RuntimeError(NATIVE_PROJECTION_REBUILD_REQUIRED)
                expected_partition = self._record_partition_identity(record).partition_id
                if str(metadata.get("session_partition_id") or "") != expected_partition:
                    raise RuntimeError("Mnemosyne native session partition is invalid")
                matches.append(detail)
            elif str(metadata.get("source_identity") or "") == source_identity:
                raise RuntimeError("Mnemosyne native source detail metadata mismatches record")
        if len(matches) > 1:
            raise RuntimeError("Mnemosyne native source identity is ambiguous")
        return matches[0] if matches else None

    def retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        if not records:
            return {"ok": True, "n_written": 0, "native_written": 0}
        for record in records:
            ensure_record_is_storable(record)
        grouped: dict[str, list[MemoryBlockRecord]] = {}
        for record in records:
            grouped.setdefault(
                self._record_partition_identity(record).partition_id,
                [],
            ).append(record)
        if len(grouped) == 1:
            results = [self.retain_block(record) for record in records]
        else:

            def retain_group(group: list[MemoryBlockRecord]) -> list[dict[str, Any]]:
                try:
                    return [self.retain_block(record) for record in group]
                finally:
                    self._index.close_thread_connection()

            with ThreadPoolExecutor(
                max_workers=min(4, len(grouped)),
                thread_name_prefix="um-mem-mnemosyne-retain",
            ) as executor:
                grouped_results = list(executor.map(retain_group, grouped.values()))
            results = [result for group_results in grouped_results for result in group_results]
        n_written = sum(1 for result in results if result.get("ok"))
        return {
            "ok": True,
            "backend": "mnemosyne",
            "native": self._native_ok(),
            "n_written": n_written,
        }

    def get_block(
        self,
        block_id: str,
        *,
        workspace_id: str = "",
    ) -> MemoryBlockRecord | None:
        return self._index.get_block(
            block_id,
            workspace_id=str(workspace_id or self.workspace_id),
        )

    def exact_reference_block_ids(self, reference: str, *, workspace_id: str) -> tuple[str, ...]:
        return self._index.exact_reference_block_ids(reference, workspace_id=workspace_id)

    def scoped_record_ids(
        self,
        *,
        workspace_id: str,
        profile_id: str,
        scope_key: str,
        logical_block_id: str = "",
    ) -> tuple[str, ...]:
        return self._index.scoped_record_ids(
            workspace_id=workspace_id,
            profile_id=profile_id,
            scope_key=scope_key,
            logical_block_id=logical_block_id,
        )

    def validate_dependencies(
        self,
        record_id: str,
        *,
        workspace_id: str = "",
        profile_id: str = "default",
        as_of: str = "",
        max_nodes: int = 256,
        max_edges: int = 1024,
    ) -> DependencyValidation:
        return self._index.validate_dependencies(
            record_id,
            workspace_id=workspace_id,
            profile_id=profile_id,
            as_of=as_of,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def record_ids_for_logical_blocks(
        self,
        logical_block_ids: tuple[str, ...] | list[str],
        *,
        workspace_id: str = "",
        profile_id: str = "",
        historical: bool = False,
    ) -> tuple[str, ...]:
        return self._index.record_ids_for_logical_blocks(
            logical_block_ids,
            workspace_id=str(workspace_id or self.workspace_id),
            profile_id=profile_id,
            historical=historical,
        )

    def logical_block_descriptor_snapshot(
        self, logical_block_id: str, *, workspace_id: str, profile_id: str
    ) -> dict[str, Any]:
        return self._index.logical_block_descriptor_snapshot(
            logical_block_id, workspace_id=workspace_id, profile_id=profile_id
        )

    def creation_operation_receipt(
        self, operation_id: str, *, workspace_id: str
    ) -> dict[str, Any] | None:
        return self._index.creation_operation_receipt(operation_id, workspace_id=workspace_id)

    def descriptor_sync_gate(self, logical_block_id: str) -> AbstractContextManager[None]:
        return self._index.descriptor_sync_gate(logical_block_id)

    def _normal_write_visible(self, block_id: str) -> bool:
        return self._index.get_block(block_id, workspace_id=self.workspace_id) is not None

    def write_outbox_status(self, block_id: str) -> str | None:
        return self._index.write_outbox_status(block_id)

    def latest_write_outbox_entry(
        self, *, block_id: str, workspace_id: str = ""
    ) -> WriteOutboxEntry | None:
        return self._index.latest_write_outbox_entry(
            block_id=block_id, workspace_id=workspace_id or self.workspace_id
        )

    def _reconcile_pending_write_outbox(self) -> None:
        pending = self._index.pending_write_outbox()
        if not pending:
            return
        for entry in pending:
            try:
                record = self._index.write_outbox_record(entry)
                if self._index.canonical_head_status(record.block_id) not in {None, "committed"}:
                    continue
                self._project_write_outbox(entry, record)
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                raise RuntimeError("selected backend pending write reconcile failed") from exc

    def reconcile(self, *, dry_run: bool = True) -> ReconcileReport:
        initial_head_entries = self._index.reconcile_canonical_head_reservations()
        initial_write_entries = self._index.reconcile_write_outbox()
        initial_head_ids = {entry.commit_revision for entry in initial_head_entries}
        initial_write_ids = {entry.commit_revision for entry in initial_write_entries}
        initial_heads, initial_writes = self._index._reconcile_entry_statuses(
            canonical_head_revisions=initial_head_ids,
            write_revisions=initial_write_ids,
        )
        pending_heads = sum(initial_heads.get(key) == "pending" for key in initial_head_ids)
        pending_writes = sum(initial_writes.get(key) == "pending" for key in initial_write_ids)
        inventory_count = len(initial_head_ids) + len(initial_write_ids)
        initial_failed = sum(initial_heads.get(key) == "failed" for key in initial_head_ids) + sum(
            initial_writes.get(key) == "failed" for key in initial_write_ids
        )
        initial_unresolved = sum(
            initial_heads.get(key) not in {"committed", "failed"} for key in initial_head_ids
        ) + sum(initial_writes.get(key) not in {"committed", "failed"} for key in initial_write_ids)
        initial_missing = sum(key not in initial_heads for key in initial_head_ids) + sum(
            key not in initial_writes for key in initial_write_ids
        )
        if dry_run:
            return ReconcileReport(
                backend="mnemosyne",
                dry_run=True,
                pending_canonical_heads=pending_heads,
                pending_write_outbox=pending_writes,
                repaired=0,
                remaining=initial_failed + initial_unresolved,
                failures=initial_failed + initial_missing,
                result_code=(
                    "failed"
                    if initial_failed + initial_missing
                    else ("inventory" if inventory_count else "clean")
                ),
                reason_code=(
                    "reconcile_failures"
                    if initial_failed + initial_missing
                    else ("pending_work" if inventory_count else "no_pending_work")
                ),
            )
        stage_failures = 0
        try:
            self._index.replay_pending_canonical_heads()
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            stage_failures += 1
        try:
            self._reconcile_pending_write_outbox()
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            stage_failures += 1
        final_heads, final_writes = self._index._reconcile_entry_statuses(
            canonical_head_revisions=initial_head_ids,
            write_revisions=initial_write_ids,
        )
        repaired = sum(final_heads.get(key) == "committed" for key in initial_head_ids) + sum(
            final_writes.get(key) == "committed" for key in initial_write_ids
        )
        failed = sum(final_heads.get(key) == "failed" for key in initial_head_ids) + sum(
            final_writes.get(key) == "failed" for key in initial_write_ids
        )
        unresolved = sum(
            final_heads.get(key) not in {"committed", "failed"} for key in initial_head_ids
        ) + sum(final_writes.get(key) not in {"committed", "failed"} for key in initial_write_ids)
        failures = failed + unresolved + (1 if stage_failures and not (failed + unresolved) else 0)
        remaining = failed + unresolved
        return ReconcileReport(
            backend="mnemosyne",
            dry_run=False,
            pending_canonical_heads=pending_heads,
            pending_write_outbox=pending_writes,
            repaired=repaired,
            remaining=remaining,
            failures=failures,
            result_code=("failed" if failures else ("repaired" if repaired else "clean")),
            reason_code=(
                "reconcile_failures"
                if failures
                else ("reconcile_complete" if repaired else "no_pending_work")
            ),
        )

    def _native_record_for_write_outbox(
        self,
        entry: WriteOutboxEntry,
    ) -> tuple[MemoryBlockRecord | None, str]:
        if entry.store != "mnemosyne":
            return None, "native_store_mismatch"
        native_id = self._index.write_outbox_native_id(entry.commit_revision)
        if not native_id:
            return None, "native_reference_missing"
        try:
            staged = self._index.write_outbox_record(entry)
            row = self._native(self._record_partition_identity(staged)).get(native_id)
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            return None, "native_projection_unavailable"
        if not isinstance(row, dict):
            return None, "native_projection_missing"
        raw_metadata = row.get("metadata_json") or row.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        elif isinstance(raw_metadata, str):
            try:
                parsed = json.loads(raw_metadata)
            except json.JSONDecodeError:
                return None, "native_metadata_invalid"
            if not isinstance(parsed, dict):
                return None, "native_metadata_invalid"
            metadata = parsed
        else:
            return None, "native_metadata_missing"
        content = row.get("content")
        if (
            str(row.get("id") or "") != native_id
            or not isinstance(content, str)
            or str(metadata.get("block_id") or "") != entry.block_id
            or str(metadata.get("workspace_id") or "") != entry.workspace_id
            or str(metadata.get("commit_revision") or "") != str(entry.commit_revision)
        ):
            return None, "native_projection_mismatch"
        if staged.workspace_id != entry.workspace_id:
            return None, "sidecar_staged_record_workspace_mismatch"
        try:
            record = self._native_row_to_block(row)
        except ValueError:
            return None, "native_canonical_record_invalid"
        if (
            record.block_id != entry.block_id
            or record.workspace_id != entry.workspace_id
            or record != staged
        ):
            return None, "native_record_mismatch"
        return staged, ""

    def retain_index_node(self, node: MemoryIndexNode) -> dict[str, Any]:
        return self._index.retain_index_node(node)

    def get_index_node(self, node_id: str, *, workspace_id: str = "") -> MemoryIndexNode | None:
        return self._index.get_index_node(node_id, workspace_id=workspace_id or self.workspace_id)

    def select_index_nodes(
        self,
        *,
        workspace_id: str = "",
        node_type: str = "",
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryIndexNode]:
        return self._index.select_index_nodes(
            workspace_id=workspace_id or self.workspace_id,
            node_type=node_type,
            tags=tags,
            limit=limit,
        )

    def search_index_nodes(
        self,
        *,
        workspace_id: str = "",
        terms: list[str] | tuple[str, ...] = (),
        node_types: list[str] | tuple[str, ...] = (),
        limit: int = 128,
    ) -> list[MemoryIndexNode]:
        return self._index.search_index_nodes(
            workspace_id=workspace_id or self.workspace_id,
            terms=terms,
            node_types=node_types,
            limit=limit,
        )

    def export_index_nodes(self) -> list[MemoryIndexNode]:
        return self._index.export_index_nodes()

    def create_prospective_obligation(
        self,
        obligation: ProspectiveObligation,
        *,
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        return self._index.create_prospective_obligation(
            obligation,
            event_id=event_id,
            idempotency_key=idempotency_key,
        )

    def get_prospective_obligation(self, obligation_id: str) -> ProspectiveObligation | None:
        return self._index.get_prospective_obligation(obligation_id)

    def transition_prospective_obligation(
        self,
        obligation_id: str,
        *,
        expected_version: int,
        expected_status: str,
        status: str,
        evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef],
        satisfaction_evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef] = (),
        satisfaction_verifier: str = "",
        satisfaction_verifier_result: str = "",
        event_id: str,
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        return self._index.transition_prospective_obligation(
            obligation_id,
            expected_version=expected_version,
            expected_status=expected_status,
            status=status,
            evidence_refs=evidence_refs,
            satisfaction_evidence_refs=satisfaction_evidence_refs,
            satisfaction_verifier=satisfaction_verifier,
            satisfaction_verifier_result=satisfaction_verifier_result,
            event_id=event_id,
            idempotency_key=idempotency_key,
        )

    def prospective_trigger_candidates(
        self,
        context: ProspectiveTriggerContext,
        *,
        workspace_id: str = "",
        limit: int = 50,
    ) -> list[ProspectiveTriggerDecision]:
        return self._index.prospective_trigger_candidates(
            context,
            workspace_id=workspace_id or self.workspace_id,
            limit=limit,
        )

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        payload = normalize_query_payload(query, default_workspace_id=self.workspace_id)
        with self._index.topology_snapshot(str(payload.get("profile_id") or "default")):
            return self._recall_payload(payload)

    def _recall_payload(self, payload: dict[str, Any]) -> list[MemoryBlockRecord]:
        if payload["retrieval_mode"] == "hybrid":
            raise ValueError("hybrid retrieval is unsupported by mnemosyne")
        payload = self._index.apply_exact_query(payload)
        if (
            str(payload.get("temporal_mode") or "current") == "historical"
            or payload.get("as_of")
            or payload.get("known_as_of")
        ):
            return self._index.recall(payload)
        limit = int(payload.get("limit") or 10)
        if record_id_sequence(payload):
            return [
                record
                for record in self._index.recall(payload)
                if self._normal_write_visible(record.block_id)
            ][:limit]
        selected_blocks = tuple(str(value) for value in payload.get("logical_block_ids") or ())
        if selected_blocks:
            partitions = self._existing_native_partitions(
                logical_block_ids=selected_blocks,
                profile_id=str(payload.get("profile_id") or "default"),
            )
        elif payload.get("search_all") is True:
            partitions = self._existing_native_partitions(
                profile_id=str(payload.get("profile_id") or "default"),
            )
        else:
            raise ValueError(
                "Mnemosyne native recall requires logical_block_ids or search_all=True"
            )
        pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]] = []
        for partition in partitions:
            logical_block_id = partition.primary_block_id
            for record_id in self._index.record_ids_for_logical_blocks(
                [logical_block_id],
                workspace_id=self.workspace_id,
                profile_id=str(payload.get("profile_id") or "default"),
            ):
                if self._index.write_outbox_status(record_id) == "pending":
                    continue
            native = self._native(partition)
            get_memory = getattr(native, "get", None)
            if not callable(get_memory):
                raise RuntimeError("Mnemosyne native detail API is unavailable")
            source = native_block_source(logical_block_id)
            started_at = time.perf_counter()
            rows: list[dict[str, Any]] = []
            native_success = False
            try:
                rows = native.recall(
                    str(payload.get("query") or ""),
                    top_k=limit,
                    source=source,
                )
                native_success = True
            finally:
                self._index.record_partition_operation(
                    partition,
                    operation="recall",
                    latency_ms=(time.perf_counter() - started_at) * 1_000,
                    hit_count=len(rows),
                    success=native_success,
                    search_mode="scoped" if selected_blocks else "search_all",
                    run_id=str(payload.get("run_id") or ""),
                    profile_id=str(payload.get("profile_id") or ""),
                )
            pool: list[tuple[MemoryBlockRecord, float | None]] = []
            pool_index: dict[str, int] = {}
            for row in rows:
                summary = dict(row) if isinstance(row, dict) else {}
                native_id = str(summary.get("id") or "")
                if not native_id:
                    raise ValueError("mnemosyne native recall row has no memory id")
                memoria = str(
                    summary.get("tier") or ""
                ) in MNEMOSYNE_MEMORIA_TIERS or native_id.startswith("memoria_")
                if memoria:
                    # Mnemosyne supplements its own recall with MEMORIA rows: a
                    # context string and duplicate views of working-memory
                    # rows. They are extraction artifacts, not UM-MEM records,
                    # so they are never hydrated as evidence. A duplicate view
                    # points at the row it mirrors.
                    mirrored = str(summary.get("source_memory_id") or "")
                    if not mirrored:
                        continue
                    native_id = mirrored
                detail = get_memory(native_id)
                if not isinstance(detail, dict):
                    if memoria:
                        continue
                    raise ValueError("mnemosyne native recall detail is missing")
                hydrated = dict(detail)
                if "score" in summary:
                    hydrated["score"] = summary["score"]
                native_record = self._native_row_to_block(hydrated)
                if native_primary_block_id(native_record) != logical_block_id:
                    if memoria:
                        continue
                    raise ValueError("mnemosyne native scoped row escaped its primary block")
                record = self._overlay_native_sidecar_contract(native_record, payload)
                if record is None:
                    continue
                if not self._normal_write_visible(record.block_id):
                    raise ValueError("mnemosyne native record is not visible in its sidecar")
                if not self._index.record_matches_temporal(record, payload):
                    continue
                raw_score = summary.get("score")
                try:
                    score = float(raw_score) if raw_score is not None else None
                except (TypeError, ValueError) as exc:
                    raise ValueError("mnemosyne native score is invalid") from exc
                position = pool_index.get(native_id)
                if position is None:
                    pool.append((record, score))
                    pool_index[native_id] = len(pool) - 1
                elif score is not None and (
                    pool[position][1] is None or score > float(pool[position][1] or 0.0)
                ):
                    pool[position] = (record, score)
            pools.append((logical_block_id, pool))
        return merge_native_block_results(pools, limit=limit, score_mode="partition_rank")

    def reserve_canonical_head(
        self,
        *,
        workspace_id: str,
        canonical_key: str,
        predecessor_id: str,
        successor_id: str,
        successor_status: str,
        successor_authority: str,
        mutation_action: str = "insert",
        rollback_target_id: str = "",
        mutation: MutationDecision,
    ) -> CanonicalHeadReservation:
        return self._index.reserve_canonical_head(
            workspace_id=workspace_id,
            canonical_key=canonical_key,
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            successor_status=successor_status,
            successor_authority=successor_authority,
            mutation_action=mutation_action,
            rollback_target_id=rollback_target_id,
            mutation=mutation,
        )

    def complete_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        mutation: MutationDecision,
    ) -> dict[str, Any]:
        entry = self._index.latest_write_outbox_entry(
            block_id=mutation.record.block_id,
            workspace_id=reservation.workspace_id,
        )
        if entry is None or entry.status != "pending" or entry.store != "mnemosyne":
            raise CanonicalHeadConflict("canonical_head_write_journal_missing")
        return self._index.complete_canonical_head(reservation, mutation)

    def fail_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        *,
        error: str,
    ) -> None:
        self._index.fail_canonical_head(reservation, error=error)

    def canonical_history(
        self,
        *,
        canonical_key: str,
        workspace_id: str = "",
    ) -> list[MemoryBlockRecord]:
        return self._index.canonical_history(
            canonical_key=canonical_key,
            workspace_id=workspace_id or self.workspace_id,
        )

    def publication_receipt(self, receipt_id: str) -> PublicationCommitReceipt | None:
        return self._index.publication_receipt(receipt_id)

    def amendment_receipt(self, receipt_id: str) -> AmendmentCommitReceipt | None:
        return self._index.amendment_receipt(receipt_id)

    def record_publication_receipt(self, receipt: PublicationCommitReceipt) -> None:
        self._index.record_publication_receipt(receipt)

    def canonical_mutation_identity(
        self,
        *,
        successor_id: str,
        workspace_id: str = "",
    ) -> CanonicalMutationIdentity | None:
        return self._index.canonical_mutation_identity(
            successor_id=successor_id,
            workspace_id=workspace_id or self.workspace_id,
        )

    def expire_scope(self, scope: str, scope_id: str = "", *, run_id: str = "") -> int:
        return self._index.expire_scope(scope, scope_id, run_id=run_id)

    def add_edge(self, edge: MemoryEdgeRecord) -> dict[str, Any]:
        return self._index.add_edge(edge)

    def edges_for(self, block_id: str) -> list[MemoryEdgeRecord]:
        return self._index.edges_for(block_id)

    def edges_between(self, record_ids: tuple[str, ...], *, limit: int) -> list[MemoryEdgeRecord]:
        return self._index.edges_between(record_ids, limit=limit)

    def export_edges(self) -> list[MemoryEdgeRecord]:
        return self._index.export_edges()

    def reverse_provenance_sources(self, derived_id: str) -> tuple[str, ...]:
        return self._index.reverse_provenance_sources(derived_id)

    def rebuild_memory_indexes(self, *, catalog: Any) -> dict[str, Any]:
        return self._index.rebuild_memory_indexes(catalog=catalog)

    def refresh_memory_indexes(self, block_id: str) -> dict[str, Any]:
        return self._index.refresh_memory_indexes(block_id)

    def record_usage(self, usage: MemoryUsageRecord) -> dict[str, Any]:
        return self._index.record_usage(usage)

    def record_memory_decision(self, event: MemoryDecisionEvent) -> dict[str, Any]:
        return self._index.record_memory_decision(event)

    def get_memory_decision(self, decision_id: str) -> MemoryDecisionEvent | None:
        return self._index.get_memory_decision(decision_id)

    def record_memory_usage_event(self, event: MemoryUsageEvent) -> dict[str, Any]:
        return self._index.record_memory_usage_event(event)

    def memory_decision_trace(self, decision_id: str) -> MemoryDecisionTrace:
        return self._index.memory_decision_trace(decision_id)

    def record_outcome_credit(self, event: OutcomeCreditEvent) -> dict[str, Any]:
        return self._index.record_outcome_credit(event)

    def outcome_credits(self, decision_id: str) -> tuple[OutcomeCreditEvent, ...]:
        return self._index.outcome_credits(decision_id)

    def outcome_credits_for_memory(self, memory_id: str) -> tuple[OutcomeCreditEvent, ...]:
        return self._index.outcome_credits_for_memory(memory_id)

    def contextual_utility_summary(self, **kwargs: Any) -> ContextualUtilitySummary:
        return self._index.contextual_utility_summary(**kwargs)

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

    def compile_active_overlay(self, *, workspace_id: str = "", run_id: str = "") -> dict[str, Any]:
        return self._index.compile_active_overlay(
            workspace_id=workspace_id or self.workspace_id,
            run_id=run_id,
        )

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = ""
    ) -> list[dict[str, Any]]:
        return self._index.render_prompt_items(
            workspace_id=workspace_id or self.workspace_id,
            run_id=run_id,
        )

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for row in self.recall(query):
            evidence_id = str(row.metadata.get("evidence_id") or row.block_id)
            hit = block_record_to_memory_hit(row)
            hits.append(
                MemoryHit(
                    text=hit.text,
                    source=evidence_id or hit.source,
                    score=hit.score,
                    kind=hit.kind,
                    tags=hit.tags,
                    metadata={
                        **hit.metadata,
                        "source_backend": "mnemosyne",
                    },
                )
            )
        return hits

    def record_rendered_tokens(
        self,
        rendered_tokens_by_record_id: dict[str, int],
    ) -> None:
        for record_id, rendered_tokens in rendered_tokens_by_record_id.items():
            record = self._index.get_block(
                record_id,
                workspace_id=self.workspace_id,
            )
            if record is None or record.block_membership is None:
                continue
            self._index.record_partition_operation(
                self._record_partition_identity(record),
                operation="render",
                rendered_tokens=rendered_tokens,
            )

    def close(self) -> None:
        self._index.stop_native_projection()
        native_instances = [
            value
            for value in (self._native_memory, *self._native_partitions.values())
            if value is not None
        ]
        connections: dict[int, Any] = {}
        for native in {id(value): value for value in native_instances}.values():
            native_connection = getattr(native, "conn", None)
            beam_connection = getattr(getattr(native, "beam", None), "conn", None)
            if native_connection is not None:
                connections[id(native_connection)] = native_connection
            if beam_connection is not None:
                connections[id(beam_connection)] = beam_connection
            close = getattr(native, "close", None)
            if callable(close):
                close()
        for connection in connections.values():
            try:
                connection.close()
            except sqlite3.ProgrammingError:
                pass
        if connections:
            from mnemosyne.core import beam as mnemosyne_beam  # type: ignore[import-untyped]
            from mnemosyne.core import memory as mnemosyne_memory  # type: ignore[import-untyped]

            for state in (mnemosyne_memory._thread_local, mnemosyne_beam._thread_local):
                connection = getattr(state, "conn", None)
                if connection is not None and id(connection) in connections:
                    state.conn = None
                    state.db_path = ""
        self._native_memory = None
        self._native_partitions.clear()
        self._index.close()
