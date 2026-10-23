import pytest

from um_mem import MemoryClient
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.backends.mnemosyne import MnemosyneBackend


def test_expansion_requests_health_only_for_the_visible_page(tmp_path):
    class Backend:
        def health(self):
            raise AssertionError("navigation must not call global/native readiness")

        def block_health(self, logical_block_ids, *, profile_id):
            self.observed = logical_block_ids, profile_id
            return {
                "ok": True,
                "readiness": "not_probed",
                "native_partitions": [],
                "block_metrics": [],
            }

    backend = Backend()
    client = MemoryClient(tmp_path, workspace_id="health", backend_instance=backend)
    root = client.block_map()[0]["block_id"]
    children = [
        client.create_leaf_block(name=str(i), synopsis="Evidence", parent_id=root, reason="test")
        for i in range(3)
    ]
    page = client.expand_block(root, limit=1)
    assert backend.observed == ((root, children[0]["block_id"]), "default")
    assert page["root"]["diagnostics_status"] == "not_probed"


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, MnemosyneBackend, HindsightBackend])
def test_scoped_health_does_not_open_native_provider_or_unrelated_partitions(
    tmp_path, monkeypatch, backend_type
):
    backend = backend_type(repo_root=tmp_path, workspace_id="health")
    try:
        requested = []
        metrics = []

        def partitions(*, workspace_id, logical_block_ids=(), profile_id=""):
            assert workspace_id == "health"
            requested.append((logical_block_ids, profile_id))
            assert logical_block_ids == ("selected",)
            assert profile_id == "reviewer"
            return (("selected", "catalog:test", "selected-partition"),)

        def forbidden(*args, **kwargs):
            raise AssertionError("navigation must not probe native service")

        original_metrics = backend._index.partition_metrics

        def record_metrics(partition, **kwargs):
            metrics.append(partition.primary_block_id)
            return original_metrics(partition, **kwargs)

        monkeypatch.setattr(backend._index, "native_partition_memberships", partitions)
        monkeypatch.setattr(backend, "health", forbidden)
        monkeypatch.setattr(backend._index, "native_projection_health", forbidden)
        monkeypatch.setattr(backend._index, "partition_metrics", record_metrics)
        if backend_type is HindsightBackend:
            monkeypatch.setattr(backend, "_client", forbidden)
        result = backend.block_health(("selected",), profile_id="reviewer")
        assert result["readiness"] == "not_probed"
        assert result["block_metrics"][0]["health"] == "not_probed"
        assert metrics == ["selected"]
        assert requested == [(("selected",), "reviewer")] * 2
        with pytest.raises(ValueError):
            backend.block_health((), profile_id="reviewer")
    finally:
        backend.close()


def test_zvec_scoped_health_uses_only_selected_persisted_observations(tmp_path, monkeypatch):
    from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
    from um_mem.memory.backends.zvec import ZvecBackend

    backend = object.__new__(ZvecBackend)
    BackendSQLiteIndex.__init__(
        backend, tmp_path / "zvec.sqlite", backend_id="zvec", workspace_id="health"
    )
    try:
        calls = []

        def memberships(**kwargs):
            calls.append(kwargs)
            return (("selected", "catalog:test", "partition:test"),)

        monkeypatch.setattr(backend, "native_partition_memberships", memberships)
        result = backend.block_health(("selected",), profile_id="reviewer")
        assert (
            calls
            == [
                {
                    "workspace_id": "health",
                    "profile_id": "reviewer",
                    "logical_block_ids": ("selected",),
                }
            ]
            * 2
        )
        assert result["block_metrics"][0]["primary_block_id"] == "selected"
        assert result["readiness"] == "not_probed"
    finally:
        BackendSQLiteIndex.close(backend)


def test_partition_observations_keep_profile_and_span_catalog_revisions(tmp_path):
    from um_mem.contracts import MemoryBlockRecord, RecordBlockMembership
    from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex

    index = BackendSQLiteIndex(
        tmp_path / "counts.sqlite", backend_id="mempalace", workspace_id="health"
    )

    def identity_for(block_id, revision, partition_id):
        return MemPalaceBackend.native_partition_identity(
            primary_block_id=block_id,
            workspace_id="health",
            catalog_revision_id=revision,
            backend_partition_id=partition_id,
        )

    try:
        for record_id, profile, revision, partition_id in (
            ("a1", "alpha", "catalog:old", "partition-a"),
            ("a2", "alpha", "catalog:new", "partition-a"),
            ("a3", "alpha", "catalog:old", "partition-c"),
            ("a0", "alpha", "catalog:old", "retired-partition"),
            ("b1", "beta", "catalog:old", "partition-b"),
        ):
            index.retain_block(
                MemoryBlockRecord(
                    block_id=record_id,
                    revision_id=f"rev:{record_id}",
                    content=record_id,
                    workspace_id="health",
                    profile_id=profile,
                    status="superseded" if record_id == "a0" else "candidate",
                    scope="workspace",
                    block_membership=RecordBlockMembership(
                        primary_block_id="shared_leaf",
                        catalog_revision_id=revision,
                        backend_partition_id=partition_id,
                    ),
                )
            )
            identity = MemPalaceBackend.native_partition_identity(
                primary_block_id="shared_leaf",
                workspace_id="health",
                catalog_revision_id=revision,
                backend_partition_id=partition_id,
            )
            index.record_partition_operation(
                identity, operation="recall", hit_count=1, latency_ms=5.0
            )
        alpha = MemPalaceBackend.native_partition_identity(
            primary_block_id="shared_leaf",
            workspace_id="health",
            catalog_revision_id="catalog:new",
            backend_partition_id="partition-a",
        )
        result = index.partition_health(
            ((alpha, "partition-a"),), profile_id="alpha", partition_factory=identity_for
        )
        assert result["block_metrics"][0]["record_count"] == 2
        assert result["block_metrics"][0]["recall_count"] == 2
        for raw_id, count in (("partition-c", 1), ("retired-partition", 0)):
            identity = identity_for("shared_leaf", "catalog:old", raw_id)
            result = index.partition_health(
                ((identity, raw_id),), profile_id="alpha", partition_factory=identity_for
            )
            assert result["block_metrics"][0]["record_count"] == count
            assert result["block_metrics"][0]["recall_count"] == 1
        beta = MemPalaceBackend.native_partition_identity(
            primary_block_id="shared_leaf",
            workspace_id="health",
            catalog_revision_id="catalog:old",
            backend_partition_id="partition-b",
        )
        result = index.partition_health(
            ((beta, "partition-b"),), profile_id="beta", partition_factory=identity_for
        )
        assert result["block_metrics"][0]["record_count"] == 1
        assert result["block_metrics"][0]["recall_count"] == 1
        with pytest.raises(ValueError, match="profile"):
            index.partition_health(
                ((beta, "partition-b"),), profile_id="alpha", partition_factory=identity_for
            )
        with pytest.raises(ValueError, match="identity"):
            index.partition_health(
                ((beta, "partition-a"),), profile_id="alpha", partition_factory=identity_for
            )
        with pytest.raises(ValueError, match="identity"):
            index.partition_health(
                ((alpha, "partition-b"), (alpha, "partition-a")),
                profile_id="alpha",
                partition_factory=identity_for,
            )
    finally:
        index.close()


def test_revived_source_does_not_count_records_owned_by_split_targets(tmp_path):
    from um_mem.memory.catalog_store import BlockEvolutionTarget

    client = MemoryClient(tmp_path, workspace_id="revived", backend="mnemosyne")
    try:
        source = client.create_leaf_block(
            name="Source",
            synopsis="Original source",
            parent_id=client.block_map()[0]["block_id"],
            reason="test",
        )
        saved = client.remember("Source evidence", requested_block_id=source["block_id"])
        assert client.projection_barrier(saved.memory_id)["ok"]
        split = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget(
                    name="Target", synopsis="New owner", record_ids=(saved.memory_id,)
                ),
            ),
            reason="move ownership without rewriting evidence",
        )
        client.revive_block(
            source["block_id"],
            expected_revision=split["source"]["revision"],
            reason="reuse empty source",
        )
        view = client.expand_block(source["block_id"])["root"]
        assert view["record_count"] == 0
        assert view["metrics"]["record_count"] == 0
        assert (
            client.expand_block(split["targets"][0]["block_id"])["root"]["metrics"]["record_count"]
            == 1
        )
    finally:
        client.backend.close()
