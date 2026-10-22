import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from um_mem import MemoryClient
from um_mem.contracts import (
    EvidenceRef,
    MemoryBlockRecord,
    OriginRef,
    OriginSegment,
    RecordBlockMembership,
    TurnChildClaim,
)
from um_mem.contracts.memory import CURRENT_SCHEMA_VERSION
from um_mem.memory.backends.base import MemoryHit, MemoryQuery
from um_mem.memory.backends.contracts import block_to_json
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import (
    MemPalaceBackend,
    _record_partition_identity,
    _record_to_native_payload,
    _store_for_block,
)
from um_mem.memory.backends.mnemosyne import (
    MNEMOSYNE_PARTITION_PROJECTION_MARKER,
    MnemosyneBackend,
)
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.catalog import compile_semantic_catalog
from um_mem.memory.blocks.registry import load_memory_blocks
from um_mem.memory.hindsight.config import HindsightConfig
from um_mem.memory.kernel.models import block_record_to_event
from um_mem.memory.kernel.writer import _write_memory_event
from um_mem.memory.mutation import plan_memory_mutation

_CATALOG_REVISION = "test-catalog-revision"


def test_reopened_native_mempalace_recalls_exact_content_from_the_ninth_leaf(
    tmp_path: Path,
) -> None:
    backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    root = client.block_map()[0]["block_id"]
    leaves = [
        client.create_leaf_block(
            name=f"Area {index}",
            synopsis="Anonymous reference area",
            parent_id=root,
            reason="test recall coverage",
        )
        for index in range(9)
    ]
    catalog = compile_semantic_catalog(load_memory_blocks(tmp_path, workspace_id="ws", strict=True))
    query = "The reference signal is a violet triangle with seven dotted edges."
    try:
        for index, leaf in enumerate(leaves):
            backend.retain_block(
                _block(
                    f"fact-{index}",
                    repo_root=tmp_path,
                    content=query
                    if index == 8
                    else f"Archived inventory item {index} belongs to the storage register.",
                    block_membership=RecordBlockMembership(
                        primary_block_id=leaf["block_id"], catalog_revision_id=catalog.revision_id
                    ),
                )
            )
    finally:
        backend.close()
    reopened = MemoryClient(tmp_path, workspace_id="ws", backend="mempalace")
    try:
        result = reopened.build_context(
            query, logical_block_ids=[leaf["block_id"] for leaf in leaves], max_tokens=384, limit=8
        )
        assert result
        assert result[0]["metadata"]["block_id"] == "fact-8"
        assert result[0]["text"] == query
        assert len(result) <= 8
    finally:
        reopened.backend.close()


def _block(
    block_id: str,
    *,
    repo_root: Path | None = None,
    **overrides: Any,
) -> MemoryBlockRecord:
    catalog_revision = (
        compile_semantic_catalog(
            load_memory_blocks(repo_root, workspace_id="ws", strict=True)
        ).revision_id
        if repo_root
        else _CATALOG_REVISION
    )
    data: dict[str, Any] = {
        "block_id": block_id,
        "revision_id": f"rev_test_{block_id}",
        "content": f"{block_id} native memory",
        "title": block_id,
        "kind": "observation",
        "status": "accepted",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "advisory",
        "trust_level": "workspace_verified",
        "scope": "workspace",
        "verification_status": "verified",
        "verification_methods": ("test_run",),
        "promotion_decision": "accepted",
        "workspace_id": "ws",
        "run_id": "run-new",
        "producer": "test",
        "tags": ("wanted",),
        "evidence_refs": (
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"native-{block_id}",
                produced_by="verifier",
                producer_id=f"verifier:{block_id}",
            ),
        ),
        "block_membership": RecordBlockMembership(
            primary_block_id="project_ideas",
            catalog_revision_id=catalog_revision,
        ),
        "metadata": {},
        "created_at": "2024-01-02T03:04:05Z",
        "updated_at": "2024-02-03T04:05:06Z",
    }
    data.update(overrides)
    return MemoryBlockRecord(**data)


def _child_claims() -> tuple[TurnChildClaim, ...]:
    return (
        TurnChildClaim(
            claim_type="user_intent",
            source_role="user",
            content="Use native child claim transport.",
            parent_session_id="session-alpha",
            origin_segments=(
                OriginSegment(
                    claim_id="user-intent:session-alpha",
                    origin=OriginRef(
                        channel="user_message",
                        principal_id="session-alpha",
                        integrity_class="reported",
                    ),
                    integrity_ceiling="reported",
                    independence_domain="hermes-session:session-alpha:user",
                    evidence_refs=(
                        EvidenceRef(
                            ref_type="artifact",
                            ref_id="hermes-session:session-alpha",
                            produced_by="agent",
                            producer_id="hermes:session-alpha",
                        ),
                    ),
                ),
            ),
            temperature="hot",
            tags=("intent",),
        ),
    )


class _FakeHindsightClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.documents = self

    def recall(self, **kwargs: Any) -> list[dict[str, Any]]:
        tags = list(kwargs.get("tags") or ())
        bank_id = str(kwargs.get("bank_id") or "")
        return [
            row
            for row in self.rows
            if str(row.get("bank_id") or "") == bank_id
            and all(tag in list(row.get("tags") or ()) for tag in tags)
        ]

    def retain(self, **kwargs: Any) -> dict[str, Any]:
        self.rows.append(
            {
                "id": str(kwargs["document_id"]),
                "bank_id": str(kwargs.get("bank_id") or ""),
                "text": str(kwargs["content"]),
                "score": 1.0,
                "tags": list(kwargs["tags"]),
                "metadata": dict(kwargs["metadata"]),
            }
        )
        return {"ok": True}

    def get_document(self, *, bank_id: str, document_id: str) -> dict[str, Any] | None:
        for row in self.rows:
            if str(row.get("id") or "") != document_id or str(row.get("bank_id") or "") != bank_id:
                continue
            return {
                "id": document_id,
                "bank_id": bank_id,
                "original_text": str(row.get("text") or ""),
                "document_metadata": dict(row.get("metadata") or {}),
                "tags": list(row.get("tags") or ()),
            }
        return None


class _FakeMnemosyneMemory:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        metadata_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.metadata_by_id = dict(metadata_by_id or {})
        self.remembered: list[dict[str, Any]] = []
        for value in rows or []:
            native_id = str(value["id"])
            metadata = self.metadata_by_id.setdefault(native_id, {})
            self.rows.append(
                {
                    "id": native_id,
                    "content": MnemosyneBackend._provider_temporal_tags(
                        str(value.get("content") or "")
                    ),
                    "source": str(
                        value.get("source")
                        or f"um_mem:block:{metadata.get('logical_block_id') or 'project_ideas'}"
                    ),
                    "score": value.get("score", 1.0),
                    "metadata": metadata,
                }
            )

    def remember(self, content: str, **kwargs: Any) -> str:
        native_id = f"native-{len(self.remembered) + 1}"
        tagged_content = MnemosyneBackend._provider_temporal_tags(content)
        metadata = dict(kwargs.get("metadata") or {})
        self.remembered.append({"id": native_id, "content": tagged_content, **kwargs})
        self.metadata_by_id[native_id] = metadata
        self.rows.append(
            {
                "id": native_id,
                "content": tagged_content,
                "source": str(kwargs.get("source") or ""),
                "score": 1.0,
                "metadata": metadata,
            }
        )
        return native_id

    def get_all_memories(self) -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "content": str(row["content"])[:500],
                "source": row["source"],
                "timestamp": self.metadata_by_id[row["id"]].get("created_at", ""),
            }
            for row in self.rows
        ]

    def get(self, native_id: str) -> dict[str, Any] | None:
        row = next((row for row in self.rows if row["id"] == native_id), None)
        return dict(row) if row is not None else None

    def recall(
        self,
        _query: str,
        *,
        top_k: int,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": row["id"],
                "content": str(row["content"])[:500],
                "source": row["source"],
                "score": row["score"],
            }
            for row in [
                value for value in self.rows if source is None or value["source"] == source
            ][:top_k]
        ]


def _native_metadata(block_id: str, **overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "block_id": block_id,
        "workspace_id": "ws",
        "run_id": "run-new",
        "scope": "workspace",
        "status": "accepted",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "advisory",
        "trust_level": "workspace_verified",
        "verification_status": "verified",
        "verification_methods": ["test_run"],
        "promotion_decision": "accepted",
        "logical_block_id": "project_ideas",
        "primary_block_id": "project_ideas",
        "projection_marker": "block_partition_v1",
        "tags": ["wanted"],
    }
    metadata.update(overrides)
    metadata["primary_block_id"] = metadata["logical_block_id"]
    metadata["source_identity"] = f"um-mem:{metadata['workspace_id']}:{block_id}"
    record = _block(
        block_id,
        status=str(metadata["status"]),
        retention=str(metadata["retention"]),
        delivery_policy=str(metadata["delivery_policy"]),
        temperature=str(metadata["temperature"]),
        authority=str(metadata["authority"]),
        trust_level=str(metadata["trust_level"]),
        scope=str(metadata["scope"]),
        verification_status=str(metadata["verification_status"]),
        verification_methods=tuple(metadata["verification_methods"]),
        promotion_decision=str(metadata["promotion_decision"]),
        workspace_id=str(metadata["workspace_id"]),
        run_id=str(metadata["run_id"]),
        tags=tuple(metadata["tags"]),
        block_membership=RecordBlockMembership(
            primary_block_id=str(metadata.get("logical_block_id") or "project_ideas"),
            catalog_revision_id=_CATALOG_REVISION,
        ),
    )
    metadata["canonical_record_json"] = json.dumps(
        block_to_json(record), ensure_ascii=False, sort_keys=True
    )
    partition = HindsightBackend._record_partition_identity(record)
    metadata["native_partition_id"] = partition.partition_id
    metadata["catalog_revision_id"] = partition.catalog_revision_id
    metadata["bank_id"] = partition.partition_id
    return metadata


def _native_rows() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("allowed", _native_metadata("allowed")),
        ("quarantined", _native_metadata("quarantined", status="quarantined")),
        (
            "passive",
            _native_metadata("passive", retention="archive", temperature="cold"),
        ),
        (
            "candidate",
            _native_metadata(
                "candidate",
                status="candidate",
                delivery_policy="reflection_only",
                trust_level="agent_reported",
                verification_status="unverified",
                verification_methods=[],
                promotion_decision="not_requested",
            ),
        ),
        ("wrong-workspace", _native_metadata("wrong-workspace", workspace_id="other")),
        ("wrong-run", _native_metadata("wrong-run", run_id="run-old")),
        (
            "stale-current",
            _native_metadata("stale-current", run_id="run-old", delivery_policy="working_set"),
        ),
        ("wrong-logical", _native_metadata("wrong-logical", logical_block_id="other")),
        ("wrong-tags", _native_metadata("wrong-tags", tags=["other"])),
        ("not-selected", _native_metadata("not-selected")),
    ]


def _seed_exact_sidecar(
    backend: HindsightBackend | MnemosyneBackend,
    rows: list[tuple[str, dict[str, Any]]],
) -> None:
    # Synthetic fixture rows are committed explicitly below; the production
    # projection worker must not race that outbox completion. An empty fixture
    # can still be used for live writes, so leave its worker running.
    if rows:
        backend._index.stop_native_projection()
    for block_id, metadata in rows:
        record = _block(
            block_id,
            status=str(metadata.get("status") or "accepted"),
            retention=str(metadata.get("retention") or "durable"),
            delivery_policy=str(metadata.get("delivery_policy") or "on_demand"),
            temperature=str(metadata.get("temperature") or "warm"),
            authority=str(metadata.get("authority") or "advisory"),
            trust_level=str(metadata.get("trust_level") or "agent_reported"),
            scope=str(metadata.get("scope") or "workspace"),
            verification_status=str(metadata.get("verification_status") or "unverified"),
            verification_methods=tuple(metadata.get("verification_methods") or ()),
            promotion_decision=str(metadata.get("promotion_decision") or "not_requested"),
            workspace_id=str(metadata.get("workspace_id") or ""),
            run_id=str(metadata.get("run_id") or ""),
            tags=tuple(metadata.get("tags") or ()),
            block_membership=RecordBlockMembership(
                primary_block_id=str(metadata.get("logical_block_id") or "project_ideas"),
                catalog_revision_id=_CATALOG_REVISION,
            ),
        )
        store = "hindsight" if isinstance(backend, HindsightBackend) else "mnemosyne"
        entry = backend._index.open_write_outbox(record=record, store=store)
        backend._index.retain_block(record)
        backend._index.complete_write_outbox(entry.commit_revision)


def _query(selected: list[str]) -> dict[str, Any]:
    return {
        "query": "native memory",
        "workspace_id": "ws",
        "run_id": "run-new",
        "logical_block_ids": ["project_ideas"],
        "record_ids": selected,
        "scope_chain": ["workspace:ws"],
        "tags": ["wanted"],
        "limit": 20,
    }


def _hindsight_backend(tmp_path: Path, rows: list[tuple[str, dict[str, Any]]]) -> HindsightBackend:
    client_rows = [
        {
            "id": block_id,
            "bank_id": str(metadata["native_partition_id"]),
            "text": f"{block_id} native memory",
            "score": 1.0,
            "tags": [
                f"um_mem:block:{metadata.get('logical_block_id') or 'project_ideas'}",
                *list(metadata.get("tags") or []),
            ],
            "metadata": metadata,
        }
        for block_id, metadata in rows
    ]
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=_FakeHindsightClient(client_rows),
    )
    _seed_exact_sidecar(backend, rows)
    return backend


def test_hindsight_recall_ignores_derived_fact_and_restores_source_document(
    tmp_path: Path,
) -> None:
    metadata = _native_metadata("allowed")
    partition_id = str(metadata["native_partition_id"])
    protected_tag = "um_mem:block:project_ideas"
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="ws",
        config=HindsightConfig(enabled=True),
        client=_FakeHindsightClient(
            [
                {
                    "id": "derived-observation",
                    "bank_id": partition_id,
                    "text": "A derived semantic observation without source metadata.",
                    "tags": [protected_tag, "wanted"],
                    "metadata": {},
                },
                {
                    "id": "derived-experience",
                    "document_id": "UM-MEM:block:allowed",
                    "bank_id": partition_id,
                    "text": "Hindsight may reformulate the retained document as an experience.",
                    "tags": [protected_tag, "wanted"],
                    "metadata": metadata,
                },
            ]
        ),
    )
    query = MemoryQuery(query="native memory", workspace_id="ws", run_id="run-new")
    try:
        hits = backend._native_recall_hits(
            query,
            partition=backend.native_partition_identity(
                workspace_id="ws",
                primary_block_id="project_ideas",
                catalog_revision_id=_CATALOG_REVISION,
                backend_partition_id=RecordBlockMembership(
                    primary_block_id="project_ideas", catalog_revision_id=_CATALOG_REVISION
                ).backend_partition_id,
            ),
        )
        restored = backend._native_hit_to_block(hits[0], query)
    finally:
        backend.close()

    assert len(hits) == 1
    assert hits[0].source == "UM-MEM:block:allowed"
    assert restored.block_id == "allowed"
    assert restored.content == "allowed native memory"


def _mnemosyne_backend(tmp_path: Path, rows: list[tuple[str, dict[str, Any]]]) -> MnemosyneBackend:
    mnemosyne_rows = []
    for block_id, raw_metadata in rows:
        metadata = dict(raw_metadata)
        primary_block_id = str(metadata["primary_block_id"])
        metadata["projection_marker"] = MNEMOSYNE_PARTITION_PROJECTION_MARKER
        metadata["session_partition_id"] = MnemosyneBackend._native_session_partition_id(
            primary_block_id,
            workspace_id=str(metadata["workspace_id"]),
            catalog_revision_id=_CATALOG_REVISION,
            backend_partition_id=RecordBlockMembership(
                primary_block_id=primary_block_id, catalog_revision_id=_CATALOG_REVISION
            ).backend_partition_id,
        )
        metadata["catalog_revision_id"] = _CATALOG_REVISION
        mnemosyne_rows.append((block_id, metadata))
    native_rows = [
        {"id": block_id, "content": f"{block_id} native memory", "score": 1.0}
        for block_id, _ in mnemosyne_rows
    ]
    backend = MnemosyneBackend(repo_root=tmp_path, workspace_id="ws", enabled=True)
    backend.available = True
    backend._native_memory = _FakeMnemosyneMemory(native_rows, dict(mnemosyne_rows))
    _seed_exact_sidecar(backend, rows)
    return backend


def test_hindsight_keeps_trusted_correction_pending_until_host_approval(tmp_path: Path) -> None:
    backend = _hindsight_backend(tmp_path, [])
    try:
        initial = plan_memory_mutation(
            _block(
                "hindsight-history",
                repo_root=tmp_path,
                content="Original native memory",
                title="Native history",
            ),
            existing=None,
        )
        written = _write_memory_event(
            tmp_path,
            block_record_to_event(initial.record),
            workspace_id="ws",
            backend_instance=backend,
            mutation=initial,
        )
        assert written.saved is True
        client = MemoryClient(repo_root=tmp_path, workspace_id="ws", backend_instance=backend)
        corrected = client.correction(
            initial.record.block_id,
            "Corrected native memory",
            reason="verified correction",
            requested_block_id="project_ideas",
        )
        assert corrected["ok"] is True
        assert corrected["pending_host_approval"] is True
        assert [
            hit.text
            for hit in client.recall(
                "native memory",
                logical_block_ids=["project_ideas"],
                run_id="run-new",
                limit=5,
            )
        ] == ["Original native memory"]
        retracted = client.forget(initial.record.block_id, reason="withdrawn")
        assert retracted["ok"] is True
        assert (
            client.recall(
                "native memory",
                logical_block_ids=["project_ideas"],
                run_id="run-new",
                limit=5,
            )
            == []
        )
    finally:
        backend.close()


@pytest.mark.parametrize("backend_factory", [_hindsight_backend, _mnemosyne_backend])
def test_native_backends_apply_uniform_query_post_filters(tmp_path: Path, backend_factory) -> None:
    rows = _native_rows()
    backend = backend_factory(tmp_path, rows)
    selected = [block_id for block_id, _ in rows if block_id != "not-selected"]
    try:
        recalled = backend.recall(_query(selected))
    finally:
        backend.close()

    assert {row.block_id for row in recalled} == {
        "allowed",
        "wrong-run",
        "wrong-tags",
        "stale-current",
    }


@pytest.mark.parametrize("backend_factory", [_hindsight_backend, _mnemosyne_backend])
def test_native_backends_apply_source_horizon_and_candidate_gate(
    tmp_path: Path, backend_factory
) -> None:
    rows = _native_rows()[:4]
    backend = backend_factory(tmp_path, rows)
    query = _query([block_id for block_id, _ in rows])
    query.update({"source_horizon": "durable", "include_candidates": True})
    try:
        recalled = backend.recall(query)
        archived = backend.recall({**query, "source_horizon": "archive"})
    finally:
        backend.close()

    assert {row.block_id for row in recalled} == {"allowed", "candidate"}
    assert {row.block_id for row in archived} == {"allowed", "passive", "candidate"}


@pytest.mark.parametrize("backend_factory", [_hindsight_backend, _mnemosyne_backend])
def test_native_backends_keep_run_metadata_advisory_and_do_not_invent_it(
    tmp_path: Path, backend_factory
) -> None:
    rows = [("missing-run", _native_metadata("missing-run", run_id=""))]
    backend = backend_factory(tmp_path, rows)
    try:
        recalled = backend.recall(_query(["missing-run"]))
    finally:
        backend.close()

    assert [record.block_id for record in recalled] == ["missing-run"]
    assert recalled[0].run_id == ""


def test_sqlite_accepts_direct_memory_query(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="ws")
    try:
        index.retain_block(_block("direct-query"))
        rows = index.recall(
            MemoryQuery(
                query="direct query native memory",
                workspace_id="ws",
                run_id="run-new",
                tags=["wanted"],
                scope_chain=["workspace:ws"],
            )
        )
    finally:
        index.close()

    assert [row.block_id for row in rows] == ["direct-query"]


@pytest.mark.parametrize("backend_name", ["mempalace", "hindsight", "mnemosyne"])
def test_selected_backends_apply_exact_symbol_index_before_native_recall(
    tmp_path: Path,
    backend_name: str,
) -> None:
    if backend_name == "mempalace":
        backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    elif backend_name == "hindsight":
        backend = _hindsight_backend(tmp_path, [])
    else:
        backend = _mnemosyne_backend(tmp_path, [])
    try:
        hit = _block(
            "exact-symbol-hit",
            metadata={"symbols": ["ReleaseController.publish"]},
        )
        miss = _block(
            "exact-symbol-miss",
            metadata={"symbols": ["ReleaseController.preview"]},
        )
        backend.retain_block(hit)
        backend.retain_block(miss)

        recalled = backend.recall(
            {
                "query": "native memory",
                "workspace_id": "ws",
                "run_id": "run-new",
                "scope_chain": ["workspace:ws"],
                "symbols": ["ReleaseController.publish"],
                "limit": 10,
            }
        )
        absent = backend.recall(
            {
                "query": "native memory",
                "workspace_id": "ws",
                "run_id": "run-new",
                "scope_chain": ["workspace:ws"],
                "symbols": ["ReleaseController.missing"],
                "limit": 10,
            }
        )
    finally:
        backend.close()

    assert [row.block_id for row in recalled] == [hit.block_id]
    assert absent == []


def test_mempalace_internal_store_derives_from_canonical_record() -> None:
    codeptr = _block("codeptr", kind="codeptr")
    lesson = _block("lesson", kind="lesson")
    observation = _block("observation")

    assert _store_for_block(codeptr) == "palace.codeptr"
    assert _store_for_block(lesson) == "palace.lesson"
    assert _store_for_block(observation) == "palace.idea"


def test_caller_metadata_importance_does_not_control_native_governance(
    tmp_path: Path,
) -> None:
    record = _block("caller-importance", metadata={"importance": 0.01})
    hindsight = _hindsight_backend(tmp_path, [])
    try:
        hindsight_metadata = hindsight._block_native_metadata(record)
    finally:
        hindsight.close()

    assert "importance" not in hindsight_metadata
    assert MnemosyneBackend._importance(record) == MnemosyneBackend._importance(
        _block("default-importance")
    )


def test_hindsight_native_metadata_roundtrip_preserves_timestamp_and_metadata(
    tmp_path: Path,
) -> None:
    backend = _hindsight_backend(tmp_path, [])
    record = _block(
        "roundtrip",
        metadata={
            "ts": "2023-05-25T12:00:00Z",
            "nested": {"answer": 42},
        },
    )
    native_metadata = backend._block_native_metadata(record)
    hit = MemoryHit(
        text=record.content,
        source=record.block_id,
        score=1.0,
        tags=[f"um_mem:block:{record.block_membership.primary_block_id}", *record.tags],
        metadata=native_metadata,
    )
    try:
        restored = backend._native_hit_to_block(
            hit,
            MemoryQuery(
                query="roundtrip",
                workspace_id="ws",
                run_id="run-new",
                scope_chain=["workspace:ws"],
            ),
        )
    finally:
        backend.close()

    assert restored.run_id == "run-new"
    assert restored.created_at == record.created_at
    assert restored.updated_at == record.updated_at
    assert restored.metadata["ts"] == "2023-05-25T12:00:00Z"
    assert restored.metadata["nested"] == {"answer": 42}
    assert restored.source_backend == record.source_backend


@pytest.mark.parametrize(
    ("backend_name", "batch"),
    [("hindsight", False), ("hindsight", True), ("mempalace", False), ("mnemosyne", False)],
)
def test_integrity_publication_is_written_and_recalled_natively(
    tmp_path: Path,
    backend_name: str,
    batch: bool,
) -> None:
    record = _block(
        "integrity-publication",
        trust_level="agent_reported",
        verification_status="evidence_backed",
        metadata={"verification_scope": "integrity"},
    )
    if backend_name == "hindsight":
        backend = _hindsight_backend(tmp_path, [])
    elif backend_name == "mempalace":
        backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    else:
        backend = _mnemosyne_backend(tmp_path, [])
    try:
        if batch:
            assert isinstance(backend, HindsightBackend)
            result = backend.retain_blocks([record])
        else:
            result = backend.retain_block(record)
        assert result["ok"] is True
        if isinstance(backend, HindsightBackend):
            assert result["native_written"] == 1
        recalled = backend.recall(
            MemoryQuery(
                query="integrity-publication native memory",
                workspace_id="ws",
                tags=["wanted"],
                scope_chain=["workspace:ws"],
                logical_block_ids=["project_ideas"],
            )
        )
        assert len(recalled) == 1
        assert replace(recalled[0], metadata=record.metadata) == record
        assert recalled[0].verification_status == "evidence_backed"
        assert recalled[0].authority == "advisory"
    finally:
        backend.close()


def test_mnemosyne_native_metadata_roundtrip_preserves_timestamp_and_metadata(
    tmp_path: Path,
) -> None:
    backend = _mnemosyne_backend(tmp_path, [])
    record = _block(
        "roundtrip",
        metadata={
            "ts": "2023-05-25T12:00:00Z",
            "nested": {"answer": 42},
        },
    )
    try:
        backend.retain_block(record)
        [restored] = backend.recall(
            MemoryQuery(
                query="roundtrip native memory",
                workspace_id="ws",
                run_id="run-new",
                tags=["wanted"],
                scope_chain=["workspace:ws"],
                search_all=True,
            )
        )
    finally:
        backend.close()

    assert restored.created_at == record.created_at
    assert restored.updated_at == record.updated_at
    assert restored.metadata["ts"] == "2023-05-25T12:00:00Z"
    assert restored.metadata["nested"] == {"answer": 42}
    assert restored.source_backend == record.source_backend


@pytest.mark.parametrize("backend_name", ["mempalace", "hindsight", "mnemosyne"])
def test_native_canonical_transport_preserves_identity_fields(
    tmp_path: Path,
    backend_name: str,
) -> None:
    record = _block(
        "identity-roundtrip",
        kind="lesson",
        stage_id="stage-canonical",
        subtask_id="subtask-current",
        source_backend="source-system",
        external_refs={"source": "external-source"},
        sensitivity="public",
        content_mode="pointer_only",
        operational_risk="high",
        revision_of="identity-predecessor",
        supersedes_ids=("identity-predecessor",),
        expires_at="2099-02-03T04:05:06Z",
        child_claims=_child_claims(),
        metadata={"nested": {"answer": 42}},
        session_id="session-current",
        task_id="task-current",
        parent_session_id="session-parent",
        artifact_id="artifact-current",
        artifact_revision="artifact-revision",
        valid_from="2024-01-02T03:04:05Z",
        valid_to="2099-01-03T03:04:05Z",
        project_id="project-current",
        repository_id="repository-current",
        checkout=str(tmp_path / "checkout"),
        worktree=str(tmp_path / "worktree"),
        branch="feature/canonical-transport",
        revision="abc123",
        dirty_state="dirty",
        hard_policy_version=7,
        semantic_catalog_version="catalog-current",
    )
    query = MemoryQuery(
        query="identity-roundtrip native memory",
        workspace_id="ws",
        run_id="run-new",
        tags=["wanted"],
        scope_chain=["workspace:ws"],
        search_all=True,
    )
    if backend_name == "mempalace":
        backend = MemPalaceBackend(tmp_path, workspace_id="ws")
        try:
            backend.retain_block(record)
            native = backend._native.collection(
                _store_for_block(record),
                partition=_record_partition_identity(record),
            ).get(ids=[record.block_id])
            restored = backend._native_node_to_block(
                {
                    **dict(native.metadatas[0]),
                    "id": str(native.ids[0]),
                    "content": str(native.documents[0]),
                }
            )
            recalled = backend.recall(query)
        finally:
            backend.close()
    elif backend_name == "hindsight":
        backend = _hindsight_backend(tmp_path, [])
        try:
            backend.retain_block(record)
            [native_hit] = backend._native_recall_hits(
                query,
                partition=backend._record_partition_identity(record),
            )
            restored = backend._native_hit_to_block(native_hit, query)
            recalled = backend.recall(query)
        finally:
            backend.close()
    else:
        backend = _mnemosyne_backend(tmp_path, [])
        try:
            result = backend.retain_block(record)
            native = backend._native_memory
            assert native is not None
            row = next(item for item in native.rows if item["id"] == result["memory_id"])
            restored = backend._native_row_to_block(row)
            recalled = backend.recall(query)
        finally:
            backend.close()

    assert restored == record
    assert len(recalled) == 1
    assert replace(recalled[0], metadata=record.metadata) == record


@pytest.mark.parametrize("backend_name", ["mempalace", "hindsight", "mnemosyne"])
def test_backend_rejects_noncurrent_schema_before_native_or_outbox_write(
    tmp_path: Path,
    backend_name: str,
) -> None:
    if backend_name == "mempalace":
        backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    elif backend_name == "hindsight":
        backend = _hindsight_backend(tmp_path, [])
    else:
        backend = _mnemosyne_backend(tmp_path, [])
    try:
        with pytest.raises(ValueError, match="schema_version"):
            backend.retain_block(_block("wrong-schema", schema_version=CURRENT_SCHEMA_VERSION + 1))

        assert backend._index.get_block("wrong-schema", workspace_id="ws") is None
        assert backend._index.pending_write_outbox() == []
    finally:
        backend.close()


def test_mempalace_native_child_claim_transport_round_trip_and_fails_closed_when_invalid() -> None:
    record = _block("mempalace-child-claim", child_claims=_child_claims())
    _, metadata = _record_to_native_payload(record)
    backend = MemPalaceBackend.__new__(MemPalaceBackend)
    backend._workspace_id = "ws"
    node = {
        "id": record.block_id,
        "content": record.content,
        "temperature": record.temperature,
        "scope": record.scope,
        "tags": list(record.tags),
        "workspace_id": record.workspace_id,
        "run_id": record.run_id,
        **metadata,
    }

    restored = backend._native_node_to_block(node)
    serialized = json.loads(metadata["canonical_record_json"])
    invalid_payload = {**serialized, "child_claims": {"invalid": True}}
    invalid_node = {
        **node,
        "canonical_record_json": json.dumps(invalid_payload, ensure_ascii=False, sort_keys=True),
    }

    assert serialized["child_claims"][0]["claim_type"] == "user_intent"
    assert restored.child_claims == record.child_claims
    assert "child_claims_json" not in restored.metadata
    with pytest.raises(ValueError, match="native canonical record"):
        backend._native_node_to_block(invalid_node)


def test_mempalace_real_native_child_claim_transport_reaches_parser(tmp_path: Path) -> None:
    backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    record = _block("mempalace-real-child-claim", child_claims=_child_claims())
    try:
        backend.retain_block(record)
        native = backend._native.collection(
            _store_for_block(record),
            partition=_record_partition_identity(record),
        ).get(ids=[record.block_id])
        node = {
            **dict(native.metadatas[0]),
            "id": str(native.ids[0]),
            "content": str(native.documents[0]),
        }

        serialized = json.loads(node["canonical_record_json"])
        assert serialized["child_claims"][0]["claim_type"] == "user_intent"
        assert backend._native_node_to_block(node).child_claims == record.child_claims
    finally:
        backend.close()


def test_hindsight_native_child_claim_transport_round_trip_and_fails_closed_when_invalid(
    tmp_path: Path,
) -> None:
    backend = _hindsight_backend(tmp_path, [])
    record = _block("hindsight-child-claim", child_claims=_child_claims())
    native_metadata = backend._block_native_metadata(record)
    hit = MemoryHit(
        text=record.content,
        source=record.block_id,
        score=1.0,
        tags=[f"um_mem:block:{record.block_membership.primary_block_id}", *record.tags],
        metadata=native_metadata,
    )
    serialized = json.loads(native_metadata["canonical_record_json"])
    invalid_payload = {**serialized, "child_claims": {"invalid": True}}
    invalid_metadata = {
        **native_metadata,
        "canonical_record_json": json.dumps(invalid_payload, ensure_ascii=False, sort_keys=True),
    }
    try:
        restored = backend._native_hit_to_block(
            hit,
            MemoryQuery(query="child claim", workspace_id="ws", run_id="run-new"),
        )

        assert serialized["child_claims"][0]["claim_type"] == "user_intent"
        assert restored.child_claims == record.child_claims
        assert "child_claims_json" not in restored.metadata
        with pytest.raises(ValueError, match="native canonical record"):
            backend._native_hit_to_block(
                MemoryHit(
                    text=record.content,
                    source=record.block_id,
                    score=1.0,
                    tags=[
                        f"um_mem:block:{record.block_membership.primary_block_id}",
                        *record.tags,
                    ],
                    metadata=invalid_metadata,
                ),
                MemoryQuery(query="child claim", workspace_id="ws", run_id="run-new"),
            )
    finally:
        backend.close()


def test_mnemosyne_native_child_claim_transport_round_trip_and_fails_closed_when_invalid(
    tmp_path: Path,
) -> None:
    backend = _mnemosyne_backend(tmp_path, [])
    record = _block("mnemosyne-child-claim", child_claims=_child_claims())
    try:
        result = backend.retain_block(record)
        native = backend._native_memory
        assert native is not None
        native_id = result["memory_id"]
        metadata = native.metadata_by_id[native_id]
        row = next(item for item in native.rows if item["id"] == native_id)

        restored = backend._native_row_to_block(row)

        serialized = json.loads(metadata["canonical_record_json"])
        assert serialized["child_claims"][0]["claim_type"] == "user_intent"
        assert restored.child_claims == record.child_claims
        assert "child_claims_json" not in restored.metadata
        metadata["canonical_record_json"] = json.dumps(
            {**serialized, "child_claims": {"invalid": True}},
            ensure_ascii=False,
            sort_keys=True,
        )
        with pytest.raises(ValueError, match="native canonical record"):
            backend._native_row_to_block(row)
    finally:
        backend.close()
