from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time

import pytest

from lethyx import MemoryClient
from lethyx.memory.catalog_store import BlockEvolutionTarget, BlockMergeSource


def test_zvec_long_projection_batches_and_resumes_after_partial_write(monkeypatch) -> None:
    from lethyx.memory.backends.zvec import ZvecBackend

    backend = object.__new__(ZvecBackend)
    backend._zvec = SimpleNamespace(Doc=lambda **values: SimpleNamespace(**values))
    stored = {}
    batch_sizes = []
    embedded = []
    fail_next = [True]

    def upsert(documents):
        assert len(documents) <= 1024, "native Zvec write batch limit"
        batch_sizes.append(len(documents))
        if stored and fail_next[0]:
            fail_next[0] = False
            raise OSError("interrupted between native batches")
        stored.update((document.id, document) for document in documents)
        return [SimpleNamespace(ok=lambda: True) for _ in documents]

    def embed(texts):
        embedded.extend(texts)
        return [[1.0] * 384 for _ in texts]

    collection = SimpleNamespace(
        fetch=lambda ids: {key: stored[key] for key in ids if key in stored},
        upsert=upsert,
        flush=lambda: None,
    )
    monkeypatch.setattr(backend, "_partition", lambda record: None)
    monkeypatch.setattr(backend, "_collection", lambda partition, create: collection)
    monkeypatch.setattr(backend, "_embed", embed)
    record = SimpleNamespace(block_id="long-source", content="exact source " * 40000)
    chunks = backend._chunks(record)
    assert len(chunks) > 1024
    with pytest.raises(OSError, match="between native batches"):
        backend._write_native(record, 17)
    completed_before_retry = len(stored)
    embedded.clear()
    assert backend._write_native(record, 17) == chunks[0][0]
    assert len(embedded) == len(chunks) - completed_before_retry
    assert len(stored) == len(chunks) and max(batch_sizes) <= 1024
    assert all(stored[key].fields["content"] == text for key, _, _, text in chunks)
    embedded.clear()
    backend._write_native(record, 17)
    assert not embedded


def test_zvec_native_leaf_recall_survives_reopen_and_keeps_full_source(tmp_path: Path) -> None:
    pytest.importorskip("zvec")
    client = MemoryClient(tmp_path, workspace_id="zvec-case", backend="zvec")
    root_id = client.block_map()[0]["block_id"]
    target = client.create_leaf_block(
        name="Pelican expedition",
        synopsis="Pelican measurements",
        parent_id=root_id,
        reason="Agent-selected independent subject",
    )["block_id"]
    foreign = client.create_leaf_block(
        name="Falcon expedition",
        synopsis="Falcon measurements",
        parent_id=root_id,
        reason="Agent-selected separate subject",
    )["block_id"]
    full_source = ("Routine packing inventory. " * 180) + "Pelican cargo weighs 74 kilograms."
    saved = client.remember(
        full_source,
        requested_block_id=target,
        write_mode="lossless_atomic",
        source_id="pelican-source",
    )
    distractor = client.remember("Falcon cargo weighs 999 kilograms.", requested_block_id=foreign)
    assert saved.saved and distractor.saved
    assert client.projection_barrier(saved.memory_id)["ok"]
    assert client.projection_barrier(distractor.memory_id)["ok"]
    assert client.backend.capabilities().semantic_recall
    first_partition = client.backend.native_partition_identity(
        workspace_id="zvec-case",
        primary_block_id=target,
        catalog_revision_id="catalog-a",
        backend_partition_id=client.backend.get_block(
            saved.memory_id
        ).block_membership.backend_partition_id,
    )
    second_partition = client.backend.native_partition_identity(
        workspace_id="zvec-case",
        primary_block_id=foreign,
        catalog_revision_id="catalog-b",
        backend_partition_id=client.backend.get_block(
            distractor.memory_id
        ).block_membership.backend_partition_id,
    )
    assert first_partition.partition_id != second_partition.partition_id
    client.backend.close()

    reopened = MemoryClient(tmp_path, workspace_id="zvec-case", session_id="next", backend="zvec")
    try:
        exact = reopened.backend.get_block(saved.memory_id)
        assert exact is not None and exact.content == full_source
        hits = reopened.backend.recall(
            {
                "query": "How much does Pelican cargo weigh?",
                "workspace_id": "zvec-case",
                "profile_id": "default",
                "scope_chain": ["workspace:zvec-case"],
                "logical_block_ids": [target],
                "include_candidates": True,
                "limit": 4,
            }
        )
        assert [hit.block_id for hit in hits] == [saved.memory_id]
        assert hits[0].content == full_source
    finally:
        reopened.backend.close()


def test_zvec_split_merge_publishes_only_new_native_leaf_partitions(tmp_path: Path) -> None:
    pytest.importorskip("zvec")
    client = MemoryClient(tmp_path, workspace_id="zvec-evolution", backend="zvec")
    root_id = client.block_map()[0]["block_id"]
    source = client.create_leaf_block(
        name="Observations", synopsis="Two topics", parent_id=root_id, reason="agent placement"
    )
    first = client.remember(
        "Pelican cargo weighs 74 kilograms.", requested_block_id=source["block_id"]
    )
    second = client.remember(
        "Falcon cargo weighs 999 kilograms.", requested_block_id=source["block_id"]
    )
    try:
        assert client.projection_barrier(first.memory_id)["ok"]
        assert client.projection_barrier(second.memory_id)["ok"]
        split = client.split_block(
            source["block_id"],
            expected_revision=source["revision"],
            targets=(
                BlockEvolutionTarget("Pelican", "Pelican facts", (first.memory_id,)),
                BlockEvolutionTarget("Falcon", "Falcon facts", (second.memory_id,)),
            ),
            reason="agent separates independent topics",
            operation_id="zvec-split",
        )
        a, b = split["targets"]
        assert split["generation_status"] == "published"
        assert {
            block_id
            for block_id, _, _ in client.backend.native_partition_memberships(
                workspace_id="zvec-evolution", profile_id="default"
            )
        } == {a["block_id"], b["block_id"]}
        assert [
            hit.metadata["block_id"]
            for hit in client.recall(
                "Pelican cargo weighs 74 kilograms",
                logical_block_ids=(a["block_id"],),
                include_candidates=True,
            )
        ] == [first.memory_id]
        merged = client.merge_blocks(
            (
                BlockMergeSource(a["block_id"], a["revision"]),
                BlockMergeSource(b["block_id"], b["revision"]),
            ),
            target=BlockEvolutionTarget(
                "Expeditions", "Both expeditions", (first.memory_id, second.memory_id)
            ),
            parent_id=root_id,
            reason="agent reunites related topics",
            operation_id="zvec-merge",
        )
        assert merged["generation_status"] == "published"
        merged_id = merged["target"]["block_id"]
        assert {
            hit.metadata["block_id"]
            for hit in client.recall(
                "cargo kilograms",
                logical_block_ids=(merged_id,),
                include_candidates=True,
                retrieval_mode="hybrid",
            )
        } == {first.memory_id, second.memory_id}
    finally:
        client.backend.close()


def test_zvec_durable_ack_precedes_native_projection(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("zvec")
    client = MemoryClient(tmp_path, workspace_id="zvec-ack", backend="zvec")
    root_id = client.block_map()[0]["block_id"]
    leaf = client.create_leaf_block(
        name="Timing", synopsis="Timing facts", parent_id=root_id, reason="agent placement"
    )["block_id"]
    original = client.backend._write_native

    def slow_projection(record, revision):
        time.sleep(1.0)
        return original(record, revision)

    monkeypatch.setattr(client.backend, "_write_native", slow_projection)
    try:
        started = time.monotonic()
        result = client.remember("The cargo was checked at 09:17.", requested_block_id=leaf)
        elapsed = time.monotonic() - started
        assert result.saved and elapsed < 0.8
        assert client.backend.get_block(result.memory_id) is not None
        assert (
            client.projection_barrier(result.memory_id, timeout_seconds=0)["projection_status"]
            == "pending"
        )
        assert client.projection_barrier(result.memory_id, timeout_seconds=30)["ok"]
    finally:
        client.backend.close()


def test_zvec_parallel_writes_share_one_workspace_without_loss(tmp_path: Path) -> None:
    pytest.importorskip("zvec")
    owner = MemoryClient(tmp_path, workspace_id="zvec-concurrent", backend="zvec")
    root_id = owner.block_map()[0]["block_id"]
    leaves = [
        owner.create_leaf_block(
            name=f"Topic {index}",
            synopsis=f"Facts {index}",
            parent_id=root_id,
            reason="agent placement",
        )["block_id"]
        for index in range(2)
    ]

    def write(index: int) -> str:
        client = MemoryClient(tmp_path, workspace_id="zvec-concurrent", backend="zvec")
        try:
            result = client.remember(
                f"Fact number {index} has exact value {index * 17}.",
                requested_block_id=leaves[index % 2],
                source_id=f"source-{index}",
                write_mode="lossless_atomic",
                operation_id=f"parallel-{index}",
            )
            assert result.saved
            assert client.projection_barrier(result.memory_id)["ok"]
            return result.memory_id
        finally:
            client.backend.close()

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(write, range(4)))
        assert len(set(ids)) == 4
        for index, record_id in enumerate(ids):
            record = owner.backend.get_block(record_id)
            assert record is not None
            assert record.content == f"Fact number {index} has exact value {index * 17}."
    finally:
        owner.backend.close()
