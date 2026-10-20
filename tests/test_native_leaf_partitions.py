import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from chromadb.errors import InternalError as ChromaInternalError

from lethyx.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from lethyx.memory.backends.contracts import NativePartitionIdentity, block_from_json
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mempalace import MemPalaceBackend, _NativeMemPalace
from lethyx.memory.backends.mnemosyne import MnemosyneBackend


def _record(
    record_id: str,
    primary_block_id: str,
    *,
    catalog_revision_id: str = "catalog:revision-a",
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=record_id,
        revision_id=f"revision:{record_id}",
        content=f"content for {record_id}",
        title=record_id,
        kind="observation",
        status="accepted",
        retention="durable",
        delivery_policy="on_demand",
        temperature="warm",
        authority="advisory",
        trust_level="workspace_verified",
        scope="workspace",
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        workspace_id="workspace-alpha",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"partition:{record_id}",
                produced_by="verifier",
                producer_id="native-partition-test",
            ),
        ),
        block_membership=RecordBlockMembership(
            primary_block_id=primary_block_id,
            catalog_revision_id=catalog_revision_id,
        ),
    )


def test_mnemosyne_native_leaf_databases_prevent_foreign_top_k_starvation(tmp_path: Path) -> None:
    pytest.importorskip("mnemosyne")
    backend = MnemosyneBackend(tmp_path, workspace_id="workspace-alpha")
    try:
        target = backend._native(
            backend.native_partition_identity(
                "personal",
                workspace_id="workspace-alpha",
                catalog_revision_id="catalog:revision-a",
                backend_partition_id=RecordBlockMembership(
                    primary_block_id="personal", catalog_revision_id="catalog:revision-a"
                ).backend_partition_id,
            )
        )
        foreign = backend._native(
            backend.native_partition_identity(
                "research",
                workspace_id="workspace-alpha",
                catalog_revision_id="catalog:revision-a",
                backend_partition_id=RecordBlockMembership(
                    primary_block_id="research", catalog_revision_id="catalog:revision-a"
                ).backend_partition_id,
            )
        )
        assert target.db_path != foreign.db_path
        identity = target.remember("Caroline research about adoption agencies", source="target")
        for number in range(60):
            foreign.remember(
                f"Caroline research about adoption agencies foreign {number}", source="foreign"
            )
        assert target.get(identity)
        assert len(target.get_all_memories()) == 1
        assert len(foreign.get_all_memories()) == 60
        recalled = target.recall("Caroline adoption agencies", top_k=1)
        assert recalled and recalled[0]["source"] == "target"
    finally:
        backend.close()


def test_mnemosyne_records_keep_native_partition_after_catalog_change(tmp_path: Path) -> None:
    pytest.importorskip("mnemosyne")
    backend = MnemosyneBackend(tmp_path, workspace_id="workspace-alpha", enabled=True)
    first = _record("first", "personal", catalog_revision_id="catalog:revision-a")
    second = _record("second", "personal", catalog_revision_id="catalog:revision-b")
    try:
        backend.retain_block(first)
        backend.retain_block(second)
        first_partition = backend._record_partition_identity(first)
        second_partition = backend._record_partition_identity(second)
        assert first_partition.partition_id == second_partition.partition_id
        assert len(backend._native(first_partition).get_all_memories()) == 2
    finally:
        backend.close()
    reopened = MnemosyneBackend(tmp_path, workspace_id="workspace-alpha", enabled=True)
    try:
        assert reopened.get_block(first.block_id) == first
        assert reopened.get_block(second.block_id) == second
        assert (
            len(reopened._native(reopened._record_partition_identity(first)).get_all_memories())
            == 2
        )
    finally:
        reopened.close()


def test_mnemosyne_discovers_scoped_partitions_without_exporting_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = MnemosyneBackend(
        tmp_path, workspace_id="workspace-alpha", enabled=False, reconcile_pending_writes=False
    )
    first = _record("first", "personal", catalog_revision_id="catalog:revision-a")
    second = _record("second", "personal", catalog_revision_id="catalog:revision-b")
    foreign = replace(_record("foreign", "research"), profile_id="another-profile")
    try:
        for record in (first, second, foreign):
            backend._index.retain_block(record)

        def fail_export() -> None:
            raise AssertionError("full record export used")

        monkeypatch.setattr(
            backend._index,
            "export_blocks",
            fail_export,
        )
        partitions = backend._existing_native_partitions(profile_id=first.profile_id)
        assert [(item.primary_block_id, item.catalog_revision_id) for item in partitions] == [
            ("personal", "catalog:revision-a"),
            ("personal", "catalog:revision-b"),
        ]
        assert partitions[0].partition_id == partitions[1].partition_id
        assert [
            item.primary_block_id
            for item in backend._existing_native_partitions(
                logical_block_ids=("research",), profile_id="another-profile"
            )
        ] == ["research"]
    finally:
        backend.close()


def test_mnemosyne_legacy_shared_database_requires_rebuild_without_deleting_history(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "native"
    data_dir.mkdir()
    original = b"old native database"
    (data_dir / "mnemosyne.db").write_bytes(original)
    backend = MnemosyneBackend(
        tmp_path, workspace_id="workspace-alpha", data_dir=data_dir, reconcile_pending_writes=False
    )
    try:
        health = backend.health()
        assert not health["native_ok"]
        assert health["reason"] == "native_projection_rebuild_required"
        with pytest.raises(ValueError, match="native_projection_rebuild_required"):
            backend._construct_native("native-leaf-partition")
        assert (data_dir / "mnemosyne.db").read_bytes() == original
    finally:
        backend.close()


def test_hindsight_semantic_leaf_bank_has_neutral_extraction_mission(tmp_path: Path) -> None:
    created: list[dict[str, object]] = []
    client = SimpleNamespace(create_bank=lambda **kwargs: created.append(kwargs))
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="workspace-alpha",
        client=client,
        reconcile_pending_writes=False,
    )
    try:
        partition = backend.native_partition_identity(
            primary_block_id="personal",
            workspace_id="workspace-alpha",
            catalog_revision_id="catalog:revision-a",
            backend_partition_id=RecordBlockMembership(
                primary_block_id="personal", catalog_revision_id="catalog:test"
            ).backend_partition_id,
        )
        backend._ensure_partition_bank(partition)
        assert created[0]["retain_mission"] == ""
        assert created[0]["reflect_mission"] == ""
    finally:
        backend.close()


def test_mempalace_native_partition_identity_is_unique_for_each_canonical_axis() -> None:
    workspace_a = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:revision-a",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="code_resources", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    block_b = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="verified_lessons",
        catalog_revision_id="catalog:revision-a",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="verified_lessons", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    revision_b = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:revision-b",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="code_resources", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    workspace_b = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-beta",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:revision-a",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="code_resources", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )

    assert (
        len(
            {
                workspace_a.partition_id,
                block_b.partition_id,
                revision_b.partition_id,
                workspace_b.partition_id,
            }
        )
        == 3
    )
    assert workspace_a.partition_id == revision_b.partition_id
    assert workspace_a == NativePartitionIdentity(
        backend_id="mempalace",
        workspace_id="workspace-alpha",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:revision-a",
        partition_id=workspace_a.partition_id,
        isolation_class="physical",
        partition_mode="native_collection_family",
    )


@pytest.mark.parametrize(
    ("backend", "expected_mode"),
    [
        (HindsightBackend, "native_bank"),
        (MnemosyneBackend, "native_session_partition"),
    ],
)
def test_native_partition_identity_survives_catalog_revision_for_every_provider(
    backend: type[HindsightBackend] | type[MnemosyneBackend],
    expected_mode: str,
) -> None:
    first = backend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:revision-a",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="code_resources", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    changed = backend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:revision-b",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="code_resources", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )

    assert first.partition_id == changed.partition_id
    assert first.isolation_class == "physical"
    assert first.partition_mode == expected_mode


class _FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any]]] = {}
        self.query_calls: list[dict[str, Any]] = []

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: Any = None,
    ) -> None:
        for record_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
            strict=True,
        ):
            self.rows[record_id] = (document, dict(metadata))

    def count(self) -> int:
        return len(self.rows)

    def query(self, **kwargs: Any) -> SimpleNamespace:
        self.query_calls.append(dict(kwargs))
        values = list(self.rows.items())[: int(kwargs["n_results"])]
        return SimpleNamespace(
            ids=[[record_id for record_id, _value in values]],
            documents=[[value[0] for _record_id, value in values]],
            metadatas=[[value[1] for _record_id, value in values]],
            distances=[[0.1 for _record_id, _value in values]],
        )

    def get(self, **kwargs: Any) -> SimpleNamespace:
        ids = kwargs.get("ids")
        values = (
            [(record_id, self.rows[record_id]) for record_id in ids if record_id in self.rows]
            if ids is not None
            else list(self.rows.items())[: int(kwargs.get("limit") or len(self.rows))]
        )
        return SimpleNamespace(
            ids=[record_id for record_id, _value in values],
            documents=[value[0] for _record_id, value in values],
            metadatas=[value[1] for _record_id, value in values],
        )


class _FakeChromaBackend:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}
        self.collection_calls: list[str] = []

    def get_collection(
        self,
        *,
        palace: object,
        collection_name: str,
        create: bool,
        options: dict[str, Any],
    ) -> _FakeCollection:
        del palace, create, options
        self.collection_calls.append(collection_name)
        return self.collections.setdefault(collection_name, _FakeCollection())

    def health(self, palace: object) -> SimpleNamespace:
        del palace
        return SimpleNamespace(ok=True, detail="ready")

    def close_palace(self, palace: object) -> None:
        del palace

    def close(self) -> None:
        return None


def test_mempalace_scoped_recall_reads_only_selected_native_collection_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_backend = _FakeChromaBackend()
    monkeypatch.setattr(
        "lethyx.memory.backends.mempalace.ChromaBackend",
        lambda: fake_backend,
    )
    native = _NativeMemPalace(tmp_path, "workspace-alpha")
    code = _record("code-1", "code_resources")
    lesson = _record("lesson-1", "verified_lessons")
    native.add_record(code, commit_revision=1)
    native.add_record(lesson, commit_revision=2)
    code_partition = MemPalaceBackend.native_partition_identity(
        workspace_id=code.workspace_id,
        primary_block_id=code.block_membership.primary_block_id,
        catalog_revision_id=code.block_membership.catalog_revision_id,
        backend_partition_id=code.block_membership.backend_partition_id,
    )

    before = len(fake_backend.collection_calls)
    rows = native.rows(
        query="code",
        limit=5,
        partition=code_partition,
    )
    scoped_calls = fake_backend.collection_calls[before:]

    assert [row["id"] for row in rows] == ["code-1"]
    assert scoped_calls
    assert all(code_partition.collection_prefix in name for name in scoped_calls)
    assert all("verified_lessons" not in name for name in scoped_calls)
    queried = [call for name in scoped_calls for call in fake_backend.collections[name].query_calls]
    assert queried
    assert all("where" not in call for call in queried)
    assert block_from_json(json.loads(rows[0]["canonical_record_json"])) == code


def test_current_recall_opens_only_partitions_with_eligible_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = _FakeChromaBackend()
    monkeypatch.setattr("lethyx.memory.backends.mempalace.ChromaBackend", lambda: native)
    backend = MemPalaceBackend(tmp_path, workspace_id="workspace-alpha")
    candidate = replace(_record("candidate", "project_ideas"), status="candidate")
    retired = replace(
        _record("retired", "project_ideas", catalog_revision_id="catalog:retired"),
        status="superseded",
    )
    current = _record("current", "project_ideas", catalog_revision_id="catalog:current")
    query = {
        "query": "content",
        "profile_id": "default",
        "logical_block_ids": ["project_ideas"],
        "scope_chain": ["workspace:workspace-alpha"],
        "source_horizon": "durable",
    }
    try:
        backend.retain_block(candidate)
        backend.retain_block(retired)
        native.collection_calls.clear()
        assert backend.recall(query) == []
        assert native.collection_calls == []

        backend.retain_block(current)
        native.collection_calls.clear()
        assert [row.block_id for row in backend.recall(query)] == ["current"]
        expected = backend.native_partition_identity(
            workspace_id=current.workspace_id,
            primary_block_id="project_ideas",
            catalog_revision_id="catalog:current",
            backend_partition_id=RecordBlockMembership(
                primary_block_id="project_ideas", catalog_revision_id="catalog:test"
            ).backend_partition_id,
        ).collection_prefix
        assert native.collection_calls
        assert all(expected in name for name in native.collection_calls)
        assert backend.get_block("candidate") == candidate
        assert backend.get_block("retired") == retired
    finally:
        backend.close()


def test_mempalace_scoped_recall_skips_stores_absent_from_the_canonical_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_backend = _FakeChromaBackend()
    monkeypatch.setattr(
        "lethyx.memory.backends.mempalace.ChromaBackend",
        lambda: fake_backend,
    )
    native = _NativeMemPalace(tmp_path, "workspace-alpha")
    record = _record("idea-1", "project_ideas")
    native.add_record(record, commit_revision=1)
    partition = MemPalaceBackend.native_partition_identity(
        workspace_id=record.workspace_id,
        primary_block_id=record.block_membership.primary_block_id,
        catalog_revision_id=record.block_membership.catalog_revision_id,
        backend_partition_id=record.block_membership.backend_partition_id,
    )

    before = len(fake_backend.collection_calls)
    rows = native.rows(
        query="content",
        limit=5,
        partition=partition,
        rebuild_records=((record, 1),),
    )
    scoped_calls = fake_backend.collection_calls[before:]

    assert [row["id"] for row in rows] == [record.block_id]
    assert scoped_calls
    assert all(name.endswith("palace_idea") for name in scoped_calls)


def test_mempalace_reopens_once_for_transient_missing_hnsw_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Collection:
        def __init__(self, *, fail_query: bool, populated: bool) -> None:
            self.fail_query = fail_query
            self.populated = populated

        def count(self) -> int:
            return 1 if self.populated else 0

        def query(self, **_kwargs: Any) -> SimpleNamespace:
            if self.fail_query:
                self.fail_query = False
                raise ChromaInternalError(
                    "Error creating hnsw segment reader: Nothing found on disk"
                )
            return SimpleNamespace(
                ids=[["record-1"]],
                documents=[["recovered content"]],
                metadatas=[[{"logical_block_id": "block-a"}]],
                distances=[[0.1]],
            )

    class Backend:
        def __init__(self, *, fail_query: bool) -> None:
            self.fail_query = fail_query
            self.closed = False

        def get_collection(self, *, collection_name: str, **_kwargs: Any) -> Collection:
            populated = collection_name.endswith("palace_idea")
            return Collection(fail_query=self.fail_query and populated, populated=populated)

        def close_palace(self, _palace: object) -> None:
            self.closed = True

        def close(self) -> None:
            self.closed = True

    backends: list[Backend] = []

    def backend_factory() -> Backend:
        backend = Backend(fail_query=not backends)
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        "lethyx.memory.backends.mempalace.ChromaBackend",
        backend_factory,
    )
    native = _NativeMemPalace(tmp_path, "workspace-alpha")
    partition = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="block-a",
        catalog_revision_id="catalog:test",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="block-a", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )

    rows = native.rows(query="target", limit=1, partition=partition)

    assert [row["id"] for row in rows] == ["record-1"]
    assert len(backends) == 2
    assert backends[0].closed is True


def test_mempalace_rebuilds_one_persistently_missing_hnsw_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Collection:
        def __init__(self, backend: "Backend", name: str) -> None:
            self.backend = backend
            self.name = name

        def count(self) -> int:
            return 1 if self.name.endswith("palace_idea") else 0

        def query(self, **_kwargs: Any) -> SimpleNamespace:
            if not self.backend.rebuilt:
                raise ChromaInternalError(
                    "Error creating hnsw segment reader: Nothing found on disk"
                )
            return SimpleNamespace(
                ids=[["record-1"]],
                documents=[["recovered content"]],
                metadatas=[[{"logical_block_id": "block-a"}]],
                distances=[[0.1]],
            )

        def get(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                ids=["record-1"],
                documents=["recovered content"],
                metadatas=[{"logical_block_id": "block-a"}],
                embeddings=[[0.1, 0.2]],
            )

        def add(self, **kwargs: Any) -> None:
            self.backend.added = kwargs
            self.backend.rebuilt = True

    class Backend:
        def __init__(self) -> None:
            self.rebuilt = False
            self.closed = False
            self.deleted: list[tuple[str, str]] = []
            self.added: dict[str, Any] | None = None

        def get_collection(self, *, collection_name: str, **_kwargs: Any) -> Collection:
            return Collection(self, collection_name)

        def delete_collection(self, palace_path: str, collection_name: str) -> None:
            self.deleted.append((palace_path, collection_name))

        def close_palace(self, _palace: object) -> None:
            self.closed = True

        def close(self) -> None:
            self.closed = True

    backends: list[Backend] = []

    def backend_factory() -> Backend:
        backend = Backend()
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        "lethyx.memory.backends.mempalace.ChromaBackend",
        backend_factory,
    )
    native = _NativeMemPalace(tmp_path, "workspace-alpha")
    partition = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="block-a",
        catalog_revision_id="catalog:test",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="block-a", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    rows = native.rows(query="target", limit=1, partition=partition)

    assert [row["id"] for row in rows] == ["record-1"]
    assert len(backends) == 2
    assert backends[1].deleted
    assert backends[1].added == {
        "ids": ["record-1"],
        "documents": ["recovered content"],
        "metadatas": [{"logical_block_id": "block-a"}],
        "embeddings": [[0.1, 0.2]],
    }


def test_mempalace_repairs_missing_hnsw_reader_raised_while_counting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Collection:
        def __init__(self, backend: "Backend", name: str) -> None:
            self.backend = backend
            self.name = name

        def count(self) -> int:
            if self.name.endswith("palace_idea") and not self.backend.rebuilt:
                raise ChromaInternalError(
                    "Error creating hnsw segment reader: Nothing found on disk"
                )
            return 1 if self.name.endswith("palace_idea") else 0

        def query(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                ids=[["record-1"]],
                documents=[["recovered content"]],
                metadatas=[[{"logical_block_id": "block-a"}]],
                distances=[[0.1]],
            )

        def get(self, **_kwargs: Any) -> SimpleNamespace:
            if not self.backend.rebuilt:
                raise ChromaInternalError(
                    "Error creating hnsw segment reader: Nothing found on disk"
                )
            return SimpleNamespace(
                ids=["record-1"],
                documents=["recovered content"],
                metadatas=[{"logical_block_id": "block-a"}],
                embeddings=[[0.1, 0.2]],
            )

        def add(self, **kwargs: Any) -> None:
            self.backend.added = kwargs
            self.backend.rebuilt = True

    class Backend:
        def __init__(self) -> None:
            self.rebuilt = False
            self.closed = False
            self.deleted: list[tuple[str, str]] = []
            self.added: dict[str, Any] | None = None

        def get_collection(self, *, collection_name: str, **_kwargs: Any) -> Collection:
            return Collection(self, collection_name)

        def delete_collection(self, palace_path: str, collection_name: str) -> None:
            self.deleted.append((palace_path, collection_name))

        def close_palace(self, _palace: object) -> None:
            self.closed = True

        def close(self) -> None:
            self.closed = True

    backends: list[Backend] = []

    def backend_factory() -> Backend:
        backend = Backend()
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        "lethyx.memory.backends.mempalace.ChromaBackend",
        backend_factory,
    )
    native = _NativeMemPalace(tmp_path, "workspace-alpha")
    partition = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="block-a",
        catalog_revision_id="catalog:test",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="block-a", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    record = _record("record-1", "block-a", catalog_revision_id="catalog:test")

    rows = native.rows(
        query="target",
        limit=1,
        partition=partition,
        rebuild_records=((record, 7),),
    )

    assert [row["id"] for row in rows] == ["record-1"]
    assert len(backends) == 2
    assert backends[0].closed is True
    assert backends[1].deleted
    assert backends[1].added is not None
    assert backends[1].added["ids"] == ["record-1"]
    assert "embeddings" not in backends[1].added


def test_real_mempalace_semantic_recall_survives_cold_reopen_of_one_record_partition(
    tmp_path: Path,
) -> None:
    palace_root = tmp_path / "palace-cold-reopen"
    record = _record("cold-record-1", "cold_reopen_leaf")
    assert record.block_membership is not None
    partition = MemPalaceBackend.native_partition_identity(
        workspace_id=record.workspace_id,
        primary_block_id=record.block_membership.primary_block_id,
        catalog_revision_id=record.block_membership.catalog_revision_id,
        backend_partition_id=record.block_membership.backend_partition_id,
    )
    writer = _NativeMemPalace(palace_root, "workspace-alpha")
    try:
        writer.add_record(record, commit_revision=1)
    finally:
        writer.close()

    reader = _NativeMemPalace(palace_root, "workspace-alpha")
    try:
        rows = reader.rows(
            query="reusable cold reopen evidence",
            limit=5,
            partition=partition,
        )
    finally:
        reader.close()

    assert [row["id"] for row in rows] == [record.block_id]
    assert block_from_json(json.loads(rows[0]["canonical_record_json"])) == record


def test_real_mempalace_materializes_distinct_leaf_collection_families(
    tmp_path: Path,
) -> None:
    native = _NativeMemPalace(tmp_path / "palace", "workspace-alpha")
    code = _record("real-code-1", "code_resources")
    lesson = _record("real-lesson-1", "verified_lessons")
    code_partition = MemPalaceBackend.native_partition_identity(
        workspace_id=code.workspace_id,
        primary_block_id=code.block_membership.primary_block_id,
        catalog_revision_id=code.block_membership.catalog_revision_id,
        backend_partition_id=code.block_membership.backend_partition_id,
    )
    lesson_partition = MemPalaceBackend.native_partition_identity(
        workspace_id=lesson.workspace_id,
        primary_block_id=lesson.block_membership.primary_block_id,
        catalog_revision_id=lesson.block_membership.catalog_revision_id,
        backend_partition_id=lesson.block_membership.backend_partition_id,
    )
    try:
        native.add_record(code, commit_revision=1)
        native.add_record(lesson, commit_revision=2)
        code_rows = native.rows(query="", limit=10, partition=code_partition)
        lesson_rows = native.rows(query="", limit=10, partition=lesson_partition)
        code_collection = native.collection("palace.idea", partition=code_partition)
        lesson_collection = native.collection("palace.idea", partition=lesson_partition)
        code_collection_name = str(code_collection._collection.name)
        lesson_collection_name = str(lesson_collection._collection.name)
    finally:
        native.close()

    assert code_partition.partition_id != lesson_partition.partition_id
    assert code_collection_name != lesson_collection_name
    assert code_partition.collection_prefix in code_collection_name
    assert lesson_partition.collection_prefix in lesson_collection_name
    assert [row["id"] for row in code_rows] == ["real-code-1"]
    assert [row["id"] for row in lesson_rows] == ["real-lesson-1"]


def test_real_mempalace_keeps_leaf_collection_after_catalog_change(tmp_path: Path) -> None:
    palace_root = tmp_path / "palace-catalog-change"
    first = _record("real-first", "stable_leaf", catalog_revision_id="catalog:revision-a")
    second = _record("real-second", "stable_leaf", catalog_revision_id="catalog:revision-b")
    first_partition = MemPalaceBackend.native_partition_identity(
        workspace_id=first.workspace_id,
        primary_block_id=first.block_membership.primary_block_id,
        catalog_revision_id=first.block_membership.catalog_revision_id,
        backend_partition_id=first.block_membership.backend_partition_id,
    )
    second_partition = MemPalaceBackend.native_partition_identity(
        workspace_id=second.workspace_id,
        primary_block_id=second.block_membership.primary_block_id,
        catalog_revision_id=second.block_membership.catalog_revision_id,
        backend_partition_id=second.block_membership.backend_partition_id,
    )
    assert first_partition.partition_id == second_partition.partition_id
    writer = _NativeMemPalace(palace_root, "workspace-alpha")
    try:
        writer.add_record(first, commit_revision=1)
        writer.add_record(second, commit_revision=2)
    finally:
        writer.close()
    reader = _NativeMemPalace(palace_root, "workspace-alpha")
    try:
        rows = reader.rows(query="", limit=10, partition=second_partition)
        collection = reader.collection("palace.idea", partition=second_partition)
        assert collection.count() == 2
    finally:
        reader.close()
    assert {row["id"] for row in rows} == {first.block_id, second.block_id}


def test_real_mempalace_scales_to_fifteen_isolated_leaf_collection_families(
    tmp_path: Path,
) -> None:
    native = _NativeMemPalace(tmp_path / "palace-scale", "workspace-alpha")
    records = [
        _record(f"real-scale-{index:02d}", f"materialized_leaf_{index:02d}") for index in range(15)
    ]
    collection_names: list[str] = []
    try:
        for commit_revision, record in enumerate(records, 1):
            native.add_record(record, commit_revision=commit_revision)
            assert record.block_membership is not None
            partition = MemPalaceBackend.native_partition_identity(
                workspace_id=record.workspace_id,
                primary_block_id=record.block_membership.primary_block_id,
                catalog_revision_id=record.block_membership.catalog_revision_id,
                backend_partition_id=record.block_membership.backend_partition_id,
            )
            collection = native.collection("palace.idea", partition=partition)
            collection_names.append(str(collection._collection.name))
            rows = native.rows(query="", limit=10, partition=partition)
            assert [row["id"] for row in rows] == [record.block_id]
    finally:
        native.close()

    assert len(set(collection_names)) == 15
