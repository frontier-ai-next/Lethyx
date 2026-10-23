"""Optional Hindsight archive backend.

UM-MEM calls retain and recall explicitly. This module never installs a
Hindsight LLM wrapper.
"""

import asyncio
import hashlib
import importlib.util
import inspect
import json
import time
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from um_mem.contracts import (
    ContextualUtilitySummary,
    EvidenceRef,
    MemoryBlockRecord,
    MemoryDecisionEvent,
    MemoryDecisionTrace,
    MemoryUsageEvent,
    OutcomeCreditEvent,
    ProspectiveObligation,
    json_ready,
)
from um_mem.contracts.memory import has_decisive_evidence
from um_mem.contracts.models import APPROVED_TRUST_LEVELS
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
    NATIVE_BLOCK_PROJECTION_MARKER,
    NATIVE_PROJECTION_REBUILD_REQUIRED,
    RESERVED_NATIVE_BLOCK_TAG_PREFIX,
    ensure_record_is_storable,
    merge_native_block_results,
    native_block_tag,
    native_primary_block_id,
    normalize_query_payload,
    record_id_sequence,
)
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex, WriteOutboxEntry
from um_mem.memory.dependencies import DependencyValidation
from um_mem.memory.hindsight import banks
from um_mem.memory.hindsight.config import HindsightConfig
from um_mem.memory.hindsight.errors import (
    HindsightPolicyError,
    HindsightUnavailableError,
)
from um_mem.memory.kernel.telemetry import record_memory_event
from um_mem.memory.mutation import (
    AmendmentCommitReceipt,
    CanonicalMutationIdentity,
    MutationDecision,
    PublicationCommitReceipt,
)
from um_mem.memory.paths import MANAGER_WORKSPACE_ID, workspace_memory_root
from um_mem.memory.prospective import ProspectiveTriggerContext, ProspectiveTriggerDecision

_BLOCK_TIMESTAMP_KEYS = (
    "occurred_at",
    "event_time",
    "timestamp",
    "ts",
    "session_date_time",
    "session_datetime",
    "session_date",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


async def _await_native_result(value: Awaitable[object]) -> object:
    return await value


class HindsightBackend:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        workspace_id: str = "",
        config: HindsightConfig | None = None,
        client: Any | None = None,
        bank_id: str = "",
        reconcile_pending_writes: bool = True,
    ) -> None:
        self._repo_root = repo_root
        self._workspace_id = workspace_id
        self.config = config or HindsightConfig.from_env()
        self._provided_client = client
        self._client_instance: Any | None = None
        self._ensured_partition_banks: set[str] = set()
        self._default_bank_id = bank_id or (
            banks.workspace_bank_id(workspace_id) if workspace_id else banks.MANAGER_BANK_ID
        )
        index_root = (
            workspace_memory_root(Path(repo_root), workspace_id or MANAGER_WORKSPACE_ID)
            / "hindsight"
            if repo_root is not None
            else Path.cwd() / ".UM-MEM" / "memory" / "hindsight"
        )
        self._index = BackendSQLiteIndex(
            index_root / "UM_MEM_index.sqlite",
            backend_id="hindsight",
            workspace_id=workspace_id,
            repo_root=Path(repo_root) if repo_root is not None else None,
        )
        if reconcile_pending_writes:
            self._index.replay_pending_canonical_heads()
        self._index.configure_native_projection(
            self._project_durable_record, start=reconcile_pending_writes
        )

    @classmethod
    def from_env(
        cls,
        *,
        repo_root: Path,
        workspace_id: str = "",
        reconcile_pending_writes: bool = True,
    ) -> "HindsightBackend":
        return cls(
            repo_root=repo_root,
            workspace_id=workspace_id,
            config=HindsightConfig.from_env(),
            reconcile_pending_writes=reconcile_pending_writes,
        )

    def capabilities(self) -> BackendCapabilities:
        health = self.health()
        native_ok = health.get("native_ok") is True and health.get("effective_mode") == "native"
        client = self._client() if native_ok else None
        memories = getattr(client, "memories", None)
        retain_api = getattr(client, "retain", None)
        if not callable(retain_api):
            retain_api = getattr(memories, "retain", None)
        recall_api = getattr(client, "recall", None)
        if not callable(recall_api):
            recall_api = getattr(memories, "recall", None)
        retain_ok = callable(retain_api)
        recall_ok = callable(recall_api)
        implemented: set[str] = set()
        if retain_ok:
            implemented.update(
                {
                    "persistent",
                    "lifecycle_mutation",
                    "value_tracking",
                    "export_import",
                    "exact_record_read",
                    "revision_history",
                    "revision_compare_and_swap",
                    "recoverable_writes",
                    "staged_topology_evolution",
                }
            )
        if recall_ok:
            implemented.add("semantic_recall")
        if retain_ok and recall_ok:
            implemented.update({"workspace_isolation", "run_subtask_isolation", "scoped_search"})
        return BackendCapabilities.effective(
            "hindsight",
            native_ok=native_ok,
            implemented=frozenset(implemented),
            requires_external_service=not self.config.embedded,
            supports_embedded=self.config.embedded,
        )

    @staticmethod
    def native_partition_identity(
        *,
        workspace_id: str,
        primary_block_id: str,
        catalog_revision_id: str,
        backend_partition_id: str,
    ) -> NativePartitionIdentity:
        workspace = str(workspace_id or "").strip()
        block_id = str(primary_block_id or "").strip()
        catalog_revision = str(catalog_revision_id or "").strip()
        partition_key = str(backend_partition_id or "").strip()
        if not workspace or not block_id or not catalog_revision or not partition_key:
            raise ValueError(
                "Hindsight native partition requires workspace, primary block, and catalog revision"
            )
        material = json.dumps(
            [workspace, partition_key],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
        return NativePartitionIdentity(
            backend_id="hindsight",
            workspace_id=workspace,
            primary_block_id=block_id,
            catalog_revision_id=catalog_revision,
            partition_id=f"um-mem-bank-{digest}",
            isolation_class="physical",
            partition_mode="native_bank",
        )

    @classmethod
    def _record_partition_identity(
        cls,
        record: MemoryBlockRecord,
    ) -> NativePartitionIdentity:
        membership = record.block_membership
        if membership is None:
            raise ValueError("Hindsight native partition requires typed block membership")
        return cls.native_partition_identity(
            workspace_id=record.workspace_id,
            primary_block_id=membership.primary_block_id,
            catalog_revision_id=membership.catalog_revision_id,
            backend_partition_id=membership.backend_partition_id,
        )

    def _existing_native_partitions(
        self,
        *,
        logical_block_ids: tuple[str, ...] = (),
        profile_id: str = "",
    ) -> tuple[NativePartitionIdentity, ...]:
        selected_order = {block_id: index for index, block_id in enumerate(logical_block_ids)}
        identities = {
            partition
            for block_id, revision_id, partition_id in self._index.native_partition_memberships(
                workspace_id=self._workspace_id,
                profile_id=profile_id,
                logical_block_ids=logical_block_ids,
            )
            for partition in (
                self.native_partition_identity(
                    workspace_id=self._workspace_id,
                    primary_block_id=block_id,
                    catalog_revision_id=revision_id,
                    backend_partition_id=partition_id,
                ),
            )
            if any(
                (record := self._index.get_block(record_id, workspace_id=self._workspace_id))
                is not None
                and self._record_partition_identity(record) == partition
                and self._native_projection_allowed(record)
                for record_id in self._index.record_ids_for_logical_blocks(
                    (block_id,), workspace_id=self._workspace_id, profile_id=profile_id
                )
            )
        }
        return tuple(
            sorted(
                identities,
                key=lambda item: (
                    selected_order.get(item.primary_block_id, len(selected_order)),
                    item.primary_block_id,
                    item.catalog_revision_id,
                    item.partition_id,
                ),
            )
        )

    def _ensure_partition_bank(self, partition: NativePartitionIdentity) -> None:
        if partition.partition_id in self._ensured_partition_banks:
            return
        if not self.config.embedded and self._provided_client is None:
            self._external_json(
                f"/v1/default/banks/{quote(partition.partition_id, safe='')}",
                method="PUT",
                payload={
                    "name": f"UM-MEM block {partition.primary_block_id}",
                    "retain_mission": "",
                    "reflect_mission": "",
                },
            )
            self._ensured_partition_banks.add(partition.partition_id)
            return
        create_bank = getattr(self._client(), "create_bank", None)
        if callable(create_bank):
            result = create_bank(
                bank_id=partition.partition_id,
                name=f"UM-MEM block {partition.primary_block_id}",
                retain_mission="",
                reflect_mission="",
            )
            if inspect.isawaitable(result):
                asyncio.run(_await_native_result(result))
        self._ensured_partition_banks.add(partition.partition_id)

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        with self._index.projection_gate(record):
            return self._retain_block(record)

    def _retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        started_at = time.perf_counter()
        self._validate_caller_tags(record)
        ensure_record_is_storable(record)
        self._require_enabled()
        if self.config.retain_async and self._native_projection_allowed(record):
            raise HindsightUnavailableError(
                "Hindsight normal writes require synchronous native acknowledgement"
            )
        outbox = self._index.open_write_outbox(
            record=record,
            store="hindsight",
        )
        record = self._index.write_outbox_record(outbox)
        native_allowed = self._native_projection_allowed(record)
        bank_id = self._record_partition_identity(record).partition_id
        if not native_allowed:
            result = self._index.retain_block(record)
            if outbox.status == "pending":
                self._index.complete_write_outbox(outbox.commit_revision)
            return {
                **result,
                "backend": "hindsight",
                "bank_id": bank_id,
                "native_written": False,
                "native_skipped_reason": "record_not_native_recall_eligible",
            }
        result = self._index.retain_block(record)
        native_result = self._project_write_outbox(outbox, record, allow_unverified=True)
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
            **native_result,
            "bank_id": bank_id,
            "native_written": True,
            "commit_revision": outbox.commit_revision,
        }

    @staticmethod
    def _native_projection_allowed(record: MemoryBlockRecord) -> bool:
        accepted_published = (
            record.status == "accepted"
            and (
                (
                    record.trust_level in APPROVED_TRUST_LEVELS
                    and record.verification_status == "verified"
                )
                or (
                    record.authority == "advisory"
                    and record.trust_level == "agent_reported"
                    and record.verification_status == "evidence_backed"
                )
            )
            and bool(record.verification_methods)
            and has_decisive_evidence(record.evidence_refs)
            and record.promotion_decision == "accepted"
        )
        advisory_candidate = (
            record.status == "candidate"
            and record.retention == "durable"
            and record.delivery_policy == "on_demand"
            and record.authority == "advisory"
            and record.trust_level == "agent_reported"
            and record.verification_status == "unverified"
            and not record.verification_methods
            and record.promotion_decision == "not_requested"
        )
        return accepted_published or advisory_candidate

    def prepare_canonical_write(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self._validate_caller_tags(record)
        ensure_record_is_storable(record)
        entry = self._index.open_write_outbox(record=record, store="hindsight")
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
        entry = self._index.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None:
            raise ValueError("native projection outbox is missing")
        if not self._native_projection_allowed(record):
            self._index.complete_write_outbox(entry.commit_revision)
            return
        self._require_enabled()
        self._project_write_outbox(entry, record)

    def stage_topology_projection(self, record: MemoryBlockRecord) -> None:
        entry = self._index.latest_write_outbox_entry(
            block_id=record.block_id, workspace_id=record.workspace_id
        )
        if entry is None or entry.status != "committed":
            raise NativeProjectionPending("native_projection_pending")
        if not self._native_projection_allowed(record):
            raise HindsightPolicyError("record is ineligible for native topology projection")
        self._require_enabled()
        with self._index.projection_gate(record):
            projected, error = self._native_record_for_projection(record, entry.commit_revision)
            if error == "native_projection_missing":
                partition = self._record_partition_identity(record)
                self._ensure_partition_bank(partition)
                response = self._retain(
                    bank_id=partition.partition_id,
                    content=record.content,
                    context=self._block_native_context(record),
                    timestamp=self._block_event_timestamp(record),
                    document_id=f"UM-MEM:block:{record.block_id}",
                    metadata=self._block_native_metadata(
                        record, commit_revision=entry.commit_revision
                    ),
                    tags=[native_block_tag(native_primary_block_id(record)), *record.tags],
                    retain_async=False,
                )
                if not self._native_write_acknowledged(response):
                    raise HindsightUnavailableError(
                        "Hindsight topology retain was not acknowledged"
                    )
                projected, error = self._native_record_for_projection(record, entry.commit_revision)
            if error or projected != record:
                raise HindsightUnavailableError(
                    "Hindsight topology projection validation failed"
                    + (f": {error}" if error else "")
                )

    @staticmethod
    def _block_native_metadata(
        record: MemoryBlockRecord, *, commit_revision: int | None = None
    ) -> dict[str, str]:
        primary_block_id = native_primary_block_id(record)
        partition = HindsightBackend._record_partition_identity(record)
        native = {
            key: str(value)
            for key, value in dict(record.metadata or {}).items()
            if key != "importance"
            and value not in (None, "")
            and isinstance(value, (str, int, float, bool))
        }
        native.update(
            {
                "block_id": record.block_id,
                "record_id": record.block_id,
                "title": record.title,
                "kind": record.kind,
                "workspace_id": record.workspace_id,
                "primary_block_id": primary_block_id,
                "projection_marker": NATIVE_BLOCK_PROJECTION_MARKER,
                "native_partition_id": partition.partition_id,
                "catalog_revision_id": partition.catalog_revision_id,
                "bank_id": partition.partition_id,
                "run_id": record.run_id,
                "subtask_id": record.subtask_id,
                "producer": record.producer,
                "scope": record.scope,
                "status": record.status,
                "retention": record.retention,
                "delivery_policy": record.delivery_policy,
                "temperature": record.temperature,
                "authority": record.authority,
                "trust_level": record.trust_level,
                "verification_status": record.verification_status,
                "verification_methods": json.dumps(list(record.verification_methods)),
                "promotion_decision": record.promotion_decision,
                "sensitivity": record.sensitivity,
                "content_mode": record.content_mode,
                "operational_risk": record.operational_risk,
                "revision_of": record.revision_of,
                "expires_at": record.expires_at,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "record_metadata_json": json.dumps(
                    json_ready(record.metadata), ensure_ascii=False, sort_keys=True
                ),
                "canonical_record_json": json.dumps(
                    block_to_json(record), ensure_ascii=False, sort_keys=True
                ),
                "tags_json": json.dumps(list(record.tags), ensure_ascii=False),
                "evidence_refs_json": json.dumps(
                    json_ready(record.evidence_refs), ensure_ascii=False, sort_keys=True
                ),
                "external_refs_json": json.dumps(
                    json_ready(record.external_refs), ensure_ascii=False, sort_keys=True
                ),
                "supersedes_ids_json": json.dumps(list(record.supersedes_ids), ensure_ascii=False),
            }
        )
        if commit_revision is not None:
            native["commit_revision"] = str(commit_revision)
        return {key: value for key, value in native.items() if value not in (None, "")}

    @staticmethod
    def _validate_caller_tags(record: MemoryBlockRecord) -> None:
        if any(tag.startswith(RESERVED_NATIVE_BLOCK_TAG_PREFIX) for tag in record.tags):
            raise ValueError("caller tags cannot use the reserved um_mem:block: prefix")

    @staticmethod
    def _block_raw_event_timestamp(record: MemoryBlockRecord) -> Any:
        for key in _BLOCK_TIMESTAMP_KEYS:
            value = record.metadata.get(key)
            if value not in (None, ""):
                return value
        return None

    @staticmethod
    def _normalize_block_event_timestamp(value: Any) -> str | int | None:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return "unset"
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        raw = str(value).strip()
        if not raw:
            return None
        if raw.lower() == "unset":
            return "unset"
        candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            datetime.fromisoformat(candidate)
        except ValueError:
            return "unset"
        return raw

    @classmethod
    def _block_event_timestamp(cls, record: MemoryBlockRecord) -> str | int | None:
        return cls._normalize_block_event_timestamp(cls._block_raw_event_timestamp(record))

    @classmethod
    def _block_native_context(cls, record: MemoryBlockRecord) -> str:
        context = f"UM-MEM memory block: {record.kind}"
        timestamp = cls._block_event_timestamp(record)
        raw_timestamp = cls._block_raw_event_timestamp(record)
        if timestamp and timestamp != "unset":
            return f"{context}\nevent_time: {timestamp}"
        if raw_timestamp not in (None, ""):
            return f"{context}\nevent_time_raw: {raw_timestamp}\nevent_time: unset"
        return context

    def retain_blocks(self, records: list[MemoryBlockRecord]) -> dict[str, Any]:
        """Retain records with the same durable-first contract as single writes."""
        if not records:
            return {"ok": True, "n_written": 0, "native_written": 0}
        for record in records:
            ensure_record_is_storable(record)
        inactive = [
            record.block_id
            for record in records
            if not (record.status == "accepted" and self._native_projection_allowed(record))
        ]
        if inactive:
            raise HindsightPolicyError(
                "Hindsight retain_blocks requires accepted, evidence-backed records for native storage: "
                + ", ".join(inactive[:5])
            )
        grouped: dict[str, list[MemoryBlockRecord]] = {}
        for record in records:
            grouped.setdefault(self._record_partition_identity(record).partition_id, []).append(
                record
            )
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
                thread_name_prefix="um-mem-hindsight-retain",
            ) as executor:
                grouped_results = list(executor.map(retain_group, grouped.values()))
            results = [result for group_results in grouped_results for result in group_results]
        written = sum(1 for result in results if result.get("ok"))
        first_result = results[0] if results else {"ok": True, "backend": "hindsight"}
        native_written = sum(int(bool(result.get("native_written"))) for result in results)
        return {
            **first_result,
            "n_written": written,
            "native_written": native_written,
        }

    def get_block(
        self,
        block_id: str,
        *,
        workspace_id: str = "",
    ) -> MemoryBlockRecord | None:
        return self._index.get_block(
            block_id,
            workspace_id=str(workspace_id or self._workspace_id),
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
            workspace_id=str(workspace_id or self._workspace_id),
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
        return self._index.get_block(block_id, workspace_id=self._workspace_id) is not None

    def write_outbox_status(self, block_id: str) -> str | None:
        return self._index.write_outbox_status(block_id)

    def latest_write_outbox_entry(
        self, *, block_id: str, workspace_id: str = ""
    ) -> WriteOutboxEntry | None:
        return self._index.latest_write_outbox_entry(
            block_id=block_id, workspace_id=workspace_id or self._workspace_id
        )

    def _reconcile_pending_write_outbox(self) -> None:
        pending = self._index.pending_write_outbox()
        if not pending:
            return
        self._require_enabled()
        for entry in pending:
            try:
                record = self._index.write_outbox_record(entry)
                if self._index.canonical_head_status(record.block_id) not in {None, "committed"}:
                    continue
                if self._native_projection_allowed(record):
                    self._project_write_outbox(entry, record)
                else:
                    self._index.retain_block(record)
                    self._index.complete_write_outbox(entry.commit_revision)
            except (OSError, RuntimeError, ValueError, HindsightUnavailableError) as exc:
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
                backend="hindsight",
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
        except (OSError, RuntimeError, ValueError):
            stage_failures += 1
        try:
            self._reconcile_pending_write_outbox()
        except (OSError, RuntimeError, ValueError, HindsightUnavailableError):
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
            backend="hindsight",
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

    def _project_write_outbox(
        self,
        entry: WriteOutboxEntry,
        record: MemoryBlockRecord,
        *,
        allow_unverified: bool = False,
    ) -> dict[str, Any]:
        with self._index.projection_gate(record):
            return self._project_write_outbox_locked(
                entry, record, allow_unverified=allow_unverified
            )

    def _project_write_outbox_locked(
        self,
        entry: WriteOutboxEntry,
        record: MemoryBlockRecord,
        *,
        allow_unverified: bool = False,
    ) -> dict[str, Any]:
        document_id = f"UM-MEM:block:{record.block_id}"
        native_record, error = self._native_record_for_write_outbox(entry)
        response: Any | None = None
        if error in {"native_projection_missing", "native_document_lookup_unavailable"}:
            partition = self._record_partition_identity(record)
            self._ensure_partition_bank(partition)
            response = self._retain(
                bank_id=partition.partition_id,
                content=record.content,
                context=self._block_native_context(record),
                timestamp=self._block_event_timestamp(record),
                document_id=document_id,
                metadata=self._block_native_metadata(record, commit_revision=entry.commit_revision),
                tags=[native_block_tag(native_primary_block_id(record)), *record.tags],
                retain_async=False,
            )
            if not self._native_write_acknowledged(response):
                raise HindsightUnavailableError(
                    "Hindsight native write returned a negative acknowledgement"
                )
            if error == "native_projection_missing":
                native_record, error = self._native_record_for_write_outbox(entry)
            else:
                native_record = record
                error = ""
        if error:
            raise HindsightUnavailableError(
                f"Hindsight native projection validation failed: {error}"
            )
        if native_record != record:
            raise HindsightUnavailableError("Hindsight native projection does not match payload")
        actual = self._index.write_outbox_entry(
            block_id=entry.block_id,
            commit_revision=entry.commit_revision,
        )
        if actual is None:
            raise RuntimeError("selected backend write journal is unavailable")
        if actual.status == "pending":
            self._index.set_write_outbox_native_ref(entry.commit_revision, document_id)
            self._index.complete_write_outbox(entry.commit_revision)
        elif actual.status != "committed":
            raise RuntimeError("selected backend write journal is not replayable")
        result = self._normalize_retain_response(response or {"ok": True})
        result.setdefault("document_id", document_id)
        if response is None:
            result["replayed"] = True
        return result

    def _native_record_for_write_outbox(
        self,
        entry: WriteOutboxEntry,
    ) -> tuple[MemoryBlockRecord | None, str]:
        if entry.store != "hindsight":
            return None, "native_store_mismatch"
        try:
            staged = self._index.write_outbox_record(entry)
        except ValueError:
            return None, "sidecar_staged_record_invalid"
        return self._native_record_for_projection(staged, entry.commit_revision)

    def _native_record_for_projection(
        self, staged: MemoryBlockRecord, commit_revision: int
    ) -> tuple[MemoryBlockRecord | None, str]:
        partition = self._record_partition_identity(staged)
        bank_id = partition.partition_id
        try:
            if not self.config.embedded and self._provided_client is None:
                document_id = f"UM-MEM:block:{staged.block_id}"
                headers = {"Accept": "application/json"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                request = Request(
                    f"{self.config.base_url.rstrip('/')}/v1/default/banks/"
                    f"{quote(bank_id, safe='')}/documents/{quote(document_id, safe='')}",
                    headers=headers,
                )
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    document = json.loads(response.read().decode("utf-8"))
            else:
                documents = getattr(self._client(), "documents", None)
                get_document = getattr(documents, "get_document", None)
                if get_document is None:
                    return None, "native_document_lookup_unavailable"
                document = get_document(
                    bank_id=bank_id,
                    document_id=f"UM-MEM:block:{staged.block_id}",
                )
                if inspect.isawaitable(document):
                    document = asyncio.run(_await_native_result(document))
        except Exception as exc:
            if getattr(exc, "status", None) == 404 or (
                isinstance(exc, HTTPError) and exc.code == 404
            ):
                return None, "native_projection_missing"
            raise
        data = _as_dict(document)
        if not data or not str(data.get("id") or ""):
            return None, "native_projection_missing"
        metadata = data.get("document_metadata")
        if not isinstance(metadata, dict):
            return None, "native_metadata_missing"
        content = data.get("original_text")
        if (
            str(data.get("id") or "") != f"UM-MEM:block:{staged.block_id}"
            or str(data.get("bank_id") or "") != bank_id
            or not isinstance(content, str)
            or str(metadata.get("block_id") or "") != staged.block_id
            or str(metadata.get("workspace_id") or "") != staged.workspace_id
            or str(metadata.get("commit_revision") or "") != str(commit_revision)
            or str(metadata.get("native_partition_id") or "") != partition.partition_id
            or str(metadata.get("catalog_revision_id") or "") != partition.catalog_revision_id
        ):
            return None, "native_projection_mismatch"
        try:
            record = self._native_hit_to_block(
                MemoryHit(
                    text=content,
                    source=str(data.get("id") or ""),
                    score=None,
                    kind=str(metadata.get("kind") or "observation"),
                    tags=list(data.get("tags") or []),
                    metadata=metadata,
                ),
                MemoryQuery(query="", workspace_id=staged.workspace_id),
            )
        except ValueError:
            return None, "native_canonical_record_invalid"
        if (
            record.block_id != staged.block_id
            or record.workspace_id != staged.workspace_id
            or record.content != content
        ):
            return None, "native_record_mismatch"
        return record, ""

    def _require_current_block_projection(self, logical_block_id: str) -> None:
        for record_id in self._index.record_ids_for_logical_blocks(
            (logical_block_id,),
            workspace_id=self._workspace_id,
        ):
            current = self._index.get_block(record_id, workspace_id=self._workspace_id)
            if current is None or not self._native_projection_allowed(current):
                continue
            entry = self._index.latest_write_outbox_entry(
                block_id=record_id,
                workspace_id=self._workspace_id,
            )
            if entry is not None and entry.status == "pending":
                raise NativeProjectionPending("native_projection_pending")
            if (
                entry is None
                or entry.status != "committed"
                or self._index.write_outbox_native_id(entry.commit_revision) is None
            ):
                raise RuntimeError(NATIVE_PROJECTION_REBUILD_REQUIRED)
            expected = self._index.write_outbox_record(entry)
            if current.block_membership != expected.block_membership:
                projected, error = self._native_record_for_projection(
                    current, entry.commit_revision
                )
                expected = current
            else:
                projected, error = self._native_record_for_write_outbox(entry)
            if error or projected != expected:
                raise RuntimeError(NATIVE_PROJECTION_REBUILD_REQUIRED)

    def retain_index_node(self, node: MemoryIndexNode) -> dict[str, Any]:
        return self._index.retain_index_node(node)

    def get_index_node(self, node_id: str, *, workspace_id: str = "") -> MemoryIndexNode | None:
        return self._index.get_index_node(node_id, workspace_id=workspace_id or self._workspace_id)

    def select_index_nodes(
        self,
        *,
        workspace_id: str = "",
        node_type: str = "",
        tags: list[str] | tuple[str, ...] | set[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryIndexNode]:
        return self._index.select_index_nodes(
            workspace_id=workspace_id or self._workspace_id,
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
            workspace_id=workspace_id or self._workspace_id,
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
            workspace_id=workspace_id or self._workspace_id,
            limit=limit,
        )

    @staticmethod
    def _query_payload(
        query: MemoryQuery | dict[str, Any], *, default_workspace_id: str = ""
    ) -> tuple[MemoryQuery, dict[str, Any]]:
        data = normalize_query_payload(query, default_workspace_id=default_workspace_id)
        if data["retrieval_mode"] == "hybrid":
            raise ValueError("hybrid retrieval is unsupported by hindsight")
        q = MemoryQuery(
            query=str(data.get("query") or ""),
            workspace_id=str(data.get("workspace_id") or default_workspace_id),
            run_id=str(data.get("run_id") or ""),
            tags=list(data.get("tags") or []),
            max_tokens=int(data.get("max_tokens") or 2048),
            limit=int(data.get("limit") or 10),
        )
        return q, data

    def _recall_api(self) -> Any | None:
        client = self._client()
        recall_api = getattr(client, "recall", None)
        if recall_api is None:
            memories = getattr(client, "memories", None)
            recall_api = getattr(memories, "recall", None) if memories is not None else None
        return recall_api

    def _native_recall_hits(
        self,
        q: MemoryQuery,
        *,
        partition: NativePartitionIdentity,
    ) -> list[MemoryHit]:
        self._require_enabled()
        bank_id = partition.partition_id
        native_tags = [native_block_tag(partition.primary_block_id)]
        request: dict[str, Any] = {
            "query": q.query,
            "max_tokens": q.max_tokens,
            "budget": "low",
            "tags": native_tags,
            "tags_match": "all_strict",
        }
        if not self.config.embedded and self._provided_client is None:
            response = self._external_json(
                f"/v1/default/banks/{quote(bank_id, safe='')}/memories/recall",
                method="POST",
                payload=request,
            )
        else:
            recall_api = self._recall_api()
            if recall_api is None:
                raise HindsightUnavailableError("Hindsight client has no recall API")
            response = recall_api(bank_id=bank_id, **request)
        rows = getattr(response, "results", response)
        if isinstance(rows, dict):
            rows = rows.get("results") or []
        hits: list[MemoryHit] = []
        for row in rows or []:
            data = dict(row) if isinstance(row, dict) else _as_dict(row)
            metadata = dict(data.get("metadata") or {})
            if (
                str(metadata.get("projection_marker") or "") != NATIVE_BLOCK_PROJECTION_MARKER
                or not str(metadata.get("canonical_record_json") or "").strip()
            ):
                # Hindsight may return derived observation facts alongside the
                # source document fact. Only the latter carries UM-MEM's typed
                # projection and can be restored without inventing identity.
                continue
            document_id = str(data.get("document_id") or "").strip()
            if document_id:
                metadata["native_document_id"] = document_id
            if data.get("timestamp") not in (None, ""):
                metadata.setdefault("native_timestamp", data.get("timestamp"))
            score = data.get("score")
            source = str(
                document_id
                or data.get("source")
                or data.get("id")
                or metadata.get("block_id")
                or "hindsight"
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
                    tags=list(
                        self._tags_from_any(
                            data.get("tags") or metadata.get("tags_json") or metadata.get("tags")
                        )
                    ),
                    metadata=metadata,
                )
            )
        return hits

    @staticmethod
    def _json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        try:
            parsed = json.loads(str(value or ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _json_list(value: Any) -> list[Any]:
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            parsed = json.loads(str(value or ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
        return list(parsed) if isinstance(parsed, list) else []

    @classmethod
    def _tags_from_any(cls, value: Any) -> tuple[str, ...]:
        parsed = cls._json_list(value)
        if parsed:
            return tuple(str(tag).strip() for tag in parsed if str(tag).strip())
        if isinstance(value, str):
            return tuple(tag.strip() for tag in value.split(",") if tag.strip())
        return ()

    @staticmethod
    def _bool_from_any(value: Any, *, default: bool = False) -> bool:
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _int_from_any(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _native_hit_to_block(self, hit: MemoryHit, q: MemoryQuery) -> MemoryBlockRecord:
        transport_metadata = dict(hit.metadata or {})
        raw = transport_metadata.get("canonical_record_json")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("hindsight native canonical record is missing")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("hindsight native canonical record is invalid") from exc
        try:
            record = block_from_json(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("hindsight native canonical record is invalid") from exc
        native_block_id = str(transport_metadata.get("block_id") or "")
        if native_block_id and native_block_id != record.block_id:
            raise ValueError("hindsight native canonical record id mismatch")
        native_document_id = str(transport_metadata.get("native_document_id") or "")
        expected_document_id = f"UM-MEM:block:{record.block_id}"
        if native_document_id and native_document_id != expected_document_id:
            raise ValueError("hindsight native document id mismatch")
        if not native_document_id and hit.text != record.content:
            raise ValueError("hindsight native canonical record content mismatch")
        if str(transport_metadata.get("projection_marker") or "") != (
            NATIVE_BLOCK_PROJECTION_MARKER
        ):
            raise ValueError(NATIVE_PROJECTION_REBUILD_REQUIRED)
        primary_block_id = native_primary_block_id(record)
        if str(transport_metadata.get("primary_block_id") or "") != primary_block_id:
            raise ValueError("hindsight native primary block mismatch")
        if native_block_tag(primary_block_id) not in hit.tags:
            raise ValueError("hindsight native protected block tag is missing")
        partition = self._record_partition_identity(record)
        if str(transport_metadata.get("native_partition_id") or "") != partition.partition_id:
            raise ValueError("hindsight native partition identity mismatch")
        if (
            str(transport_metadata.get("catalog_revision_id") or "")
            != partition.catalog_revision_id
        ):
            raise ValueError("hindsight native catalog revision mismatch")
        if str(transport_metadata.get("bank_id") or "") != partition.partition_id:
            raise ValueError("hindsight native row escaped its bank partition")
        return record

    def _overlay_native_sidecar_contract(
        self,
        record: MemoryBlockRecord,
        query: dict[str, Any] | None = None,
    ) -> MemoryBlockRecord | None:
        return self._index.overlay_native_hit(
            record,
            workspace_id=self._workspace_id,
            query=query,
        )

    def recall(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryBlockRecord]:
        q, payload = self._query_payload(query, default_workspace_id=self._workspace_id)
        with self._index.topology_snapshot(str(payload.get("profile_id") or "default")):
            return self._recall_payload(q, payload)

    def _recall_payload(self, q: MemoryQuery, payload: dict[str, Any]) -> list[MemoryBlockRecord]:
        payload = self._index.apply_exact_query(payload)
        if (
            str(payload.get("temporal_mode") or "current") == "historical"
            or payload.get("as_of")
            or payload.get("known_as_of")
        ):
            return self._index.recall(payload)
        if record_id_sequence(payload):
            return [
                record
                for record in self._index.recall(payload)
                if self._normal_write_visible(record.block_id)
            ]
        limit = max(1, int(payload.get("limit") or 10))
        selected_blocks = tuple(str(value) for value in payload.get("logical_block_ids") or ())
        if not selected_blocks and payload.get("search_all") is not True:
            raise ValueError(
                "Hindsight native recall requires logical_block_ids or search_all=True"
            )
        partitions = self._existing_native_partitions(
            logical_block_ids=selected_blocks,
            profile_id=str(payload.get("profile_id") or "default"),
        )
        for partition in partitions:
            self._require_current_block_projection(partition.primary_block_id)
        pools: list[tuple[str, list[tuple[MemoryBlockRecord, float | None]]]] = []
        for partition in partitions:
            logical_block_id = partition.primary_block_id
            started_at = time.perf_counter()
            hits: list[MemoryHit] = []
            native_success = False
            try:
                hits = self._native_recall_hits(q, partition=partition)
                native_success = True
            finally:
                self._index.record_partition_operation(
                    partition,
                    operation="recall",
                    latency_ms=(time.perf_counter() - started_at) * 1_000,
                    hit_count=len(hits),
                    success=native_success,
                    search_mode="scoped" if selected_blocks else "search_all",
                    run_id=str(payload.get("run_id") or ""),
                    profile_id=str(payload.get("profile_id") or ""),
                )
            pool: list[tuple[MemoryBlockRecord, float | None]] = []
            for hit in hits:
                native_record = self._native_hit_to_block(hit, q)
                if logical_block_id and native_primary_block_id(native_record) != logical_block_id:
                    raise ValueError("hindsight native scoped row escaped its primary block")
                record = self._overlay_native_sidecar_contract(native_record, payload)
                if record is None:
                    continue
                if not self._normal_write_visible(record.block_id):
                    raise ValueError("hindsight native record is not visible in its sidecar")
                if not self._index.record_matches_temporal(record, payload):
                    continue
                pool.append((record, hit.score))
                if len(pool) >= limit:
                    break
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
        if entry is None or entry.status != "pending" or entry.store != "hindsight":
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
            workspace_id=workspace_id or self._workspace_id,
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
            workspace_id=workspace_id or self._workspace_id,
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
        return self._index.import_blocks(records)

    def compile_active_overlay(self, *, workspace_id: str = "", run_id: str = "") -> dict[str, Any]:
        return self._index.compile_active_overlay(
            workspace_id=workspace_id or self._workspace_id, run_id=run_id
        )

    def render_prompt_items(
        self, *, workspace_id: str = "", run_id: str = ""
    ) -> list[dict[str, Any]]:
        return self._index.render_prompt_items(
            workspace_id=workspace_id or self._workspace_id, run_id=run_id
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
            from hindsight import HindsightEmbedded  # type: ignore[import-not-found, import-untyped]

            kwargs: dict[str, Any] = {
                "profile": self.config.profile,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "llm_api_key": self.config.api_key,
            }
            llm_base_url = self.config.llm_base_url
            if llm_base_url:
                kwargs["llm_base_url"] = llm_base_url
            return HindsightEmbedded(**kwargs)
        from hindsight_client import Hindsight  # type: ignore[import-not-found, import-untyped]

        client_kwargs: dict[str, Any] = {
            "base_url": self.config.base_url,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.api_key:
            client_kwargs["api_key"] = self.config.api_key
        return Hindsight(**client_kwargs)

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
            backend="hindsight",
            status=status,
            error=error,
            data=data or {},
        )

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
                    workspace_id=self._workspace_id,
                    primary_block_id=block_id,
                    catalog_revision_id=revision_id,
                    backend_partition_id=partition_id,
                ),
                partition_id,
            )
            for block_id, revision_id, partition_id in self._index.native_partition_memberships(
                workspace_id=self._workspace_id,
                logical_block_ids=logical_block_ids,
                profile_id=profile_id,
            )
        )
        return self._index.partition_health(
            partitions,
            profile_id=profile_id,
            partition_factory=lambda block, revision, raw: self.native_partition_identity(
                primary_block_id=block,
                workspace_id=self._workspace_id,
                catalog_revision_id=revision,
                backend_partition_id=raw,
            ),
        )

    def _health(self, *, verify_native_partitions: bool) -> dict[str, Any]:
        partitions = self._existing_native_partitions()

        def partition_health(status: str) -> dict[str, Any]:
            return {
                "native_block_scoping_supported": status == "ready",
                "native_block_scoping_reason": ("native_bank" if status == "ready" else status),
                "native_partition_isolation_class": "physical",
                "native_partition_mode": "native_bank",
                "native_partition_count": len(partitions),
                "native_partitions": [partition.to_mapping() for partition in partitions],
                "block_metrics": [
                    self._index.partition_metrics(
                        partition,
                        health=status,
                    ).to_mapping()
                    for partition in partitions
                ],
            }

        base = {
            **self._index.operational_revisions(),
            "projection": self._index.native_projection_health(),
            "strict_partition_validation": verify_native_partitions,
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
                **partition_health("unavailable:native_disabled"),
                "operational_note": (
                    "Hindsight native client is disabled; selected Hindsight "
                    "backend fails closed instead of using a SQLite sidecar."
                ),
            }

        if (
            self.config.embedded
            and self.config.llm_provider.strip().casefold() not in {"ollama", "lmstudio"}
            and not self.config.api_key.strip()
        ):
            reason = "embedded_llm_credentials_missing"
            return {
                **base,
                "ok": False,
                "enabled": True,
                "native_ok": False,
                "strict_ok": False,
                "effective_mode": "unavailable",
                "base_url": self.config.base_url,
                "embedded": True,
                "mode": self.config.backend_mode,
                "reason": reason,
                **partition_health(f"unavailable:{reason}"),
                "operational_note": (
                    "Embedded Hindsight requires an LLM API key for the selected remote "
                    "provider; configure it in the foreground process environment."
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
                **partition_health("unavailable:dependency_unavailable"),
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
                for partition in partitions if verify_native_partitions else ():
                    self._require_current_block_projection(partition.primary_block_id)
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
                    **partition_health("ready"),
                }
            client = self._client()
            if self.config.embedded:
                # HindsightEmbedded starts its managed daemon lazily when the
                # native client is first used. Current clients expose a private
                # startup hook and a public readiness property; the property-only
                # path keeps compatibility with earlier embedded releases.
                ensure_started = getattr(client, "_ensure_started", None)
                if callable(ensure_started):
                    ensure_started()
                else:
                    native_client = getattr(client, "client", None)
                    if native_client is None:
                        raise HindsightUnavailableError(
                            "Hindsight embedded client has no startup API"
                        )
                if getattr(client, "is_running", False) is not True:
                    raise HindsightUnavailableError(
                        "Hindsight embedded client did not reach a running state"
                    )
                for partition in partitions if verify_native_partitions else ():
                    self._require_current_block_projection(partition.primary_block_id)
                return {
                    **base,
                    "ok": True,
                    "enabled": True,
                    "native_ok": True,
                    "strict_ok": True,
                    "effective_mode": "native",
                    "base_url": self.config.base_url,
                    "embedded": True,
                    "mode": self.config.backend_mode,
                    **partition_health("ready"),
                }
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
                asyncio.run(_await_native_result(result))
            for partition in partitions if verify_native_partitions else ():
                self._require_current_block_projection(partition.primary_block_id)
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
                **partition_health("ready"),
            }
        except (
            HindsightUnavailableError,
            HTTPError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
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
                **partition_health(f"unavailable:{reason}"),
                "operational_note": (
                    "Hindsight native health check failed; selected Hindsight "
                    "backend fails closed instead of using a SQLite sidecar."
                ),
            }

    def preflight_health(self) -> dict[str, Any]:
        """Verify that a selected external service can perform retain extraction.

        The ordinary health check stays cheap because runtime memory calls use it
        frequently. Preflight deliberately performs one read-only Hindsight
        extraction against an ephemeral empty bank, then removes that exact bank.
        """

        health = self._health(verify_native_partitions=True)
        if not (
            health.get("ok") is True
            and health.get("native_ok") is True
            and health.get("strict_ok") is True
        ):
            return health
        if self.config.embedded or self._provided_client is not None:
            return {**health, "llm_write_ready": True}
        try:
            self._probe_external_retain_readiness()
        except HTTPError as error:
            detail = error.read(4_096).decode("utf-8", errors="replace")
            reason = self._llm_preflight_failure_reason(error.code, detail)
            return {
                **health,
                "ok": False,
                "native_ok": False,
                "strict_ok": False,
                "effective_mode": "unavailable",
                "reason": reason,
                "llm_write_ready": False,
                "operational_note": (
                    "Hindsight is reachable but its retain LLM is not ready; "
                    "Preflight fails before an agent can stage memory."
                ),
            }
        except (HindsightUnavailableError, OSError, RuntimeError, TypeError, ValueError):
            return {
                **health,
                "ok": False,
                "native_ok": False,
                "strict_ok": False,
                "effective_mode": "unavailable",
                "reason": "native_llm_readiness_failed",
                "llm_write_ready": False,
                "operational_note": (
                    "Hindsight retain readiness could not be confirmed; Preflight "
                    "fails closed without creating user memory."
                ),
            }
        return {
            **health,
            "reason": "native_llm_write_ready",
            "llm_write_ready": True,
        }

    def _probe_external_retain_readiness(self) -> None:
        bank_id = f"umbrella-preflight-{uuid4().hex}"
        bank_path = f"/v1/default/banks/{quote(bank_id, safe='')}"
        created = False
        probe_succeeded = False
        try:
            self._external_json(
                bank_path,
                method="PUT",
                payload={
                    "name": "Umbrella readiness probe",
                    "retain_mission": banks.WORKSPACE_RETAIN_MISSION,
                    "reflect_mission": banks.WORKSPACE_REFLECT_MISSION,
                },
            )
            created = True
            result = self._external_json(
                f"{bank_path}/memories/dry-run-extract",
                method="POST",
                payload={
                    "content": "Umbrella readiness probe for one temporary memory fact.",
                    "context": "Read-only preflight; do not persist.",
                },
            )
            if not isinstance(result.get("facts"), list) or not isinstance(
                result.get("usage"), dict
            ):
                raise HindsightUnavailableError(
                    "Hindsight dry-run extraction returned an invalid readiness response"
                )
            probe_succeeded = True
        finally:
            if created:
                try:
                    self._external_json(bank_path, method="DELETE")
                except (HTTPError, OSError, RuntimeError, TypeError, ValueError) as cleanup_error:
                    if probe_succeeded:
                        raise HindsightUnavailableError(
                            "Hindsight readiness bank cleanup failed"
                        ) from cleanup_error

    def _external_json(
        self,
        path: str,
        *,
        method: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        body = (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {"accept": "application/json"}
        if body is not None:
            headers["content-type"] = "application/json"
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            f"{self.config.base_url.rstrip('/')}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            status = int(getattr(response, "status", 0) or 0)
            if not 200 <= status < 300:
                raise HindsightUnavailableError(
                    f"Hindsight native endpoint returned HTTP {status or 'unknown'}"
                )
            raw = response.read()
        decoded = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(decoded, dict):
            raise HindsightUnavailableError(
                "Hindsight native endpoint returned a non-object response"
            )
        return {str(key): value for key, value in decoded.items()}

    @staticmethod
    def _llm_preflight_failure_reason(status: int, detail: str) -> str:
        normalized = detail.casefold()
        if status in {401, 403} or any(
            marker in normalized
            for marker in (
                "authenticationerror",
                "authentication error",
                "invalid proxy server token",
                "invalid api key",
                "unauthorized",
            )
        ):
            return "native_llm_authentication_failed"
        if status == 404 or "not configured" in normalized:
            return "native_llm_not_configured"
        return "native_llm_unavailable"

    def close(self) -> None:
        self._index.stop_native_projection()
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
        if not self.config.embedded and self._provided_client is None:
            # SDK sync methods share an aiohttp session bound to the first caller's
            # event loop. Projection workers and readers have independent threads.
            # Use the same native HTTP boundary as exact document reads instead.
            item = dict(kwargs)
            bank_id = str(item.pop("bank_id"))
            retain_async = item.pop("retain_async", False)
            timestamp = item.get("timestamp")
            if isinstance(timestamp, datetime):
                item["timestamp"] = timestamp.isoformat()
            return self._external_json(
                f"/v1/default/banks/{quote(bank_id, safe='')}/memories",
                method="POST",
                payload={"items": [item], "async": retain_async},
            )
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

    @staticmethod
    def _native_write_acknowledged(response: Any) -> bool:
        if isinstance(response, dict):
            if "success" in response:
                return response["success"] is True
            return response.get("ok") is True
        if hasattr(response, "success"):
            return response.success is True
        return getattr(response, "ok", None) is True

    def recall_evidence(self, query: MemoryQuery | dict[str, Any]) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for row in self.recall(query):
            hit = block_record_to_memory_hit(row)
            hits.append(
                MemoryHit(
                    text=hit.text,
                    source=f"hindsight:{row.block_id}",
                    score=hit.score,
                    kind=hit.kind,
                    tags=hit.tags,
                    metadata={
                        **hit.metadata,
                        "source_backend": "hindsight",
                        "native_written": True,
                        "sidecar_only": False,
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
                workspace_id=self._workspace_id,
            )
            if record is None or record.block_membership is None:
                continue
            self._index.record_partition_operation(
                self._record_partition_identity(record),
                operation="render",
                rendered_tokens=rendered_tokens,
            )

    @staticmethod
    def tag_supplemental_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tagged: list[dict[str, Any]] = []
        for hit in hits:
            row = dict(hit)
            row.setdefault("delivery_policy", "on_demand")
            row.setdefault("authority", "advisory")
            row.setdefault("source_backend", "hindsight")
            tagged.append(row)
        return tagged
