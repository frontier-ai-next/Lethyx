import json
import hashlib
import multiprocessing
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lethyx import MemoryClient
from lethyx.contracts import RecordBlockMembership
from lethyx.memory.backends.contracts import CanonicalHeadConflict
from lethyx.memory.backends.mempalace import MemPalaceBackend, _NativeMemPalace
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.blocks.catalog import block_descriptor_node_id
from lethyx.memory.kernel.telemetry import read_memory_events


def test_mempalace_embeddings_overlap_outside_native_commit_gate(tmp_path, monkeypatch):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    saved = client.remember("embedding source", requested_block_id="stable_project_facts")
    record = client.read_exact_record(record_id=saved.memory_id)
    index.close()
    native = _NativeMemPalace(tmp_path / "palace", "ws")
    embedding_barrier = threading.Barrier(2)
    commit_lock = threading.Lock()
    added = []

    def embed(documents):
        assert documents == [record.content]
        embedding_barrier.wait(3)
        return [[1.0, 0.0]]

    class Collection:
        def get(self, **_kwargs):
            return SimpleNamespace(ids=[], documents=[], metadatas=[])

        def add(self, **kwargs):
            assert commit_lock.acquire(blocking=False), "native commits overlapped"
            try:
                assert kwargs["embeddings"] == [[1.0, 0.0]]
                time.sleep(0.02)
                added.extend(kwargs["ids"])
            finally:
                commit_lock.release()

    monkeypatch.setattr("lethyx.memory.backends.mempalace.get_embedding_function", lambda: embed)
    monkeypatch.setattr(native, "collection", lambda *args, **kwargs: Collection())
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    native.add_record,
                    replace(record, block_id=f"record-{i}"),
                    commit_revision=i + 1,
                )
                for i in range(2)
            ]
            for future in futures:
                future.result(timeout=5)
        assert sorted(added) == ["record-0", "record-1"]
    finally:
        native.close()


def test_durable_ack_does_not_wait_for_native_and_barrier_is_bounded(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")

    def project(record):
        entered.set()
        assert release.wait(10)
        entry = index.latest_write_outbox_entry(block_id=record.block_id)
        index.complete_write_outbox(entry.commit_revision)

    index.configure_native_projection(project)
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    delayed_release: threading.Timer | None = None
    try:
        ack_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            write = executor.submit(
                client.remember,
                "exact source",
                requested_block_id="stable_project_facts",
                operation_id="source-1",
            )
            result = write.result(timeout=3)
        assert time.perf_counter() - ack_started < 1.8
        assert result.saved and result.raw["durable_status"] == "committed"
        assert result.raw["projection_status"] == "pending"
        assert entered.wait(3)
        assert client.read_exact_record(record_id=result.memory_id).content == "exact source"
        barrier = client.projection_barrier(
            result.memory_id, commit_revision=result.raw["commit_revision"], timeout_seconds=0.05
        )
        assert barrier["ok"] is False and barrier["projection_status"] == "pending"
        assert barrier["commit_revision"] == result.raw["commit_revision"]
        assert barrier["projection_revision"] == 0
        delayed_release = threading.Timer(2.0, release.set)
        delayed_release.start()
        assert client.projection_barrier(
            result.memory_id, commit_revision=result.raw["commit_revision"], timeout_seconds=5
        )["ok"]
        replay = client.remember(
            "exact source", requested_block_id="stable_project_facts", operation_id="source-1"
        )
        assert replay.memory_id == result.memory_id
    finally:
        release.set()
        if delayed_release is not None:
            delayed_release.join(timeout=3)
        index.close()


def test_partition_projection_overlaps_but_same_leaf_is_ordered(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    first = client.remember("seed", requested_block_id="stable_project_facts")
    seed = client.read_exact_record(record_id=first.memory_id)
    left_entered = threading.Event()
    right_entered = threading.Event()
    release = threading.Event()
    seen = []

    def project(record):
        if record.block_id == "left-1":
            left_entered.set()
            assert release.wait(10)
        if record.block_id == "right-1":
            right_entered.set()
        seen.append(record.block_id)
        index.complete_write_outbox(
            index.latest_write_outbox_entry(block_id=record.block_id).commit_revision
        )

    try:
        for record_id, leaf in (("left-1", "left"), ("left-2", "left"), ("right-1", "right")):
            record = replace(
                seed,
                block_id=record_id,
                canonical_key="",
                revision_id=f"revision:{record_id}",
                block_membership=RecordBlockMembership(
                    primary_block_id=leaf,
                    catalog_revision_id=seed.block_membership.catalog_revision_id,
                ),
            )
            index.open_write_outbox(record=record, store="test")
            index.retain_block(record)
        index.configure_native_projection(project)
        assert left_entered.wait(3) and right_entered.wait(3)
        assert "left-2" not in seen
        release.set()
        deadline = time.monotonic() + 5
        while index.write_outbox_status("left-2") != "committed" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert index.write_outbox_status("left-2") == "committed"
        assert seen.index("left-1") < seen.index("left-2")
    finally:
        release.set()
        index.close()


def test_barrier_never_accepts_unknown_record_or_unrelated_revision(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    try:
        with pytest.raises(ValueError, match="was not found"):
            client.projection_barrier("absent", commit_revision=123, timeout_seconds=0)
        saved = client.remember("source", requested_block_id="stable_project_facts")
        result = client.projection_barrier(saved.memory_id, commit_revision=123, timeout_seconds=0)
        assert result["ok"] is False
    finally:
        index.close()


def test_stop_projection_waits_before_native_resources_can_close(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    entered, release, native_closed = threading.Event(), threading.Event(), threading.Event()

    def project(record):
        entered.set()
        assert release.wait(5)
        assert not native_closed.is_set(), "native resources closed during projection"
        entry = index.latest_write_outbox_entry(block_id=record.block_id)
        index.complete_write_outbox(entry.commit_revision)

    def close_native():
        index.stop_native_projection()
        native_closed.set()

    index.configure_native_projection(project)
    closer = threading.Thread(target=close_native)
    try:
        saved = client.remember(
            "retained during shutdown", requested_block_id="stable_project_facts"
        )
        assert entered.wait(3)
        closer.start()
        assert not native_closed.wait(0.2)
        release.set()
        closer.join(5)
        assert not closer.is_alive()
        assert index.write_outbox_status(saved.memory_id) == "committed"
        assert not any(worker.is_alive() for worker in index._projection_workers)
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(5)
        index.close()
    reopened = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    try:
        assert reopened.get_block(saved.memory_id).content == "retained during shutdown"
        assert not reopened.pending_write_outbox()
    finally:
        reopened.close()


def test_shutdown_leaves_unstarted_same_leaf_work_recoverable(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    seed = client.read_exact_record(
        record_id=client.remember("seed", requested_block_id="stable_project_facts").memory_id
    )
    entered, release = threading.Event(), threading.Event()
    calls = []
    for number in (1, 2):
        record = replace(
            seed, block_id=f"queued-{number}", canonical_key="", revision_id=f"r-{number}"
        )
        index.open_write_outbox(record=record, store="test")
        index.retain_block(record)

    def project(record):
        calls.append(record.block_id)
        entered.set()
        assert release.wait(5)
        index.complete_write_outbox(
            index.latest_write_outbox_entry(block_id=record.block_id).commit_revision
        )

    index.configure_native_projection(project)
    closer = threading.Thread(target=index.stop_native_projection)
    try:
        assert entered.wait(3)
        closer.start()
        assert index._projection_stop.wait(3)
        release.set()
        closer.join(5)
        assert not closer.is_alive()
        assert calls == ["queued-1"]
        assert index.write_outbox_status("queued-2") == "pending"
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(5)
        index.close()
    reopened = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    try:
        entry = reopened.latest_write_outbox_entry(block_id="queued-2")
        assert entry.status == "pending"
        assert reopened.write_outbox_record(entry).content == "seed"
    finally:
        reopened.close()


def test_exhausted_projection_requires_attention_without_losing_durable_record(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    failures = []

    def unavailable(record):
        failures.append(record.block_id)
        raise ConnectionError("native service unavailable")

    index.configure_native_projection(unavailable)
    try:
        saved = client.remember("exact evidence", requested_block_id="stable_project_facts")
        result = client.projection_barrier(
            saved.memory_id, commit_revision=saved.raw["commit_revision"], timeout_seconds=8
        )
        assert result["reason_code"] == "native_projection_attention"
        assert result["attention_required"] is True and result["retryable"] is False
        assert len(failures) == 3
        assert index.pending_write_outbox(), "operator repair must remain possible"
        assert client.read_exact_record(record_id=saved.memory_id).content == "exact evidence"
    finally:
        index.close()


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, MnemosyneBackend])
def test_real_native_partitions_overlap_after_durable_ack(tmp_path, monkeypatch, backend_type):
    backend = backend_type(tmp_path, workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    root = client.block_map()[0]["block_id"]
    leaves = [
        client.create_leaf_block(
            name=f"Area {i}", synopsis="Model-selected area", parent_id=root, reason="parallel test"
        )["block_id"]
        for i in range(4)
    ]
    entered = [threading.Event() for _ in leaves]
    release = threading.Event()
    original = backend._project_write_outbox

    def delayed(entry, record, **kwargs):
        entered[leaves.index(record.block_membership.primary_block_id)].set()
        assert release.wait(15)
        return original(entry, record, **kwargs)

    monkeypatch.setattr(backend, "_project_write_outbox", delayed)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    client.remember,
                    f"Exact fact {i}",
                    requested_block_id=leaf,
                    operation_id=f"source-{i}",
                    write_mode="lossless_atomic",
                    source_id=f"turn-{i}",
                )
                for i, leaf in enumerate(leaves)
            ]
            results = [future.result(timeout=5) for future in futures]
        assert all(
            result.saved and result.raw["projection_status"] == "pending" for result in results
        )
        assert all(event.wait(5) for event in entered)
        for i, result in enumerate(results):
            assert client.read_exact_record(record_id=result.memory_id).content == f"Exact fact {i}"
        release.set()
        for result in results:
            assert client.projection_barrier(
                result.memory_id, commit_revision=result.raw["commit_revision"], timeout_seconds=30
            )["ok"]
        assert not backend._index.pending_write_outbox()
    finally:
        release.set()
        backend.close()
    events = [
        event
        for event in read_memory_events(tmp_path)
        if event["event_type"] == "native_projection"
    ]
    assert len(events) == len(leaves)
    assert all(event["status"] == "ok" for event in events)
    reopened = backend_type(tmp_path, workspace_id="ws")
    try:
        client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=reopened)
        assert [client.read_exact_record(record_id=r.memory_id).content for r in results] == [
            f"Exact fact {i}" for i in range(len(leaves))
        ]
    finally:
        reopened.close()


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_independent_connections_append_to_one_shared_leaf_without_loss(tmp_path, workers):
    path = tmp_path / "index.sqlite"
    bootstrap = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    MemoryClient(tmp_path, workspace_id="ws", backend_instance=bootstrap).block_map()
    bootstrap.close()

    def write(i):
        index = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
        try:
            client = MemoryClient(
                tmp_path, workspace_id="ws", session_id=f"session-{i}", backend_instance=index
            )
            result = client.remember(
                f"  fact {i}\r\n",
                requested_block_id="stable_project_facts",
                operation_id=f"op-{i}",
                source_id=f"source-{i}",
                write_mode="lossless_atomic",
            )
            assert result.saved
            return result.memory_id
        finally:
            index.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        ids = list(executor.map(write, range(16)))
    assert len(set(ids)) == 16
    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    try:
        assert len(reopened.export_blocks()) == 16
        descriptor = reopened.get_index_node(
            block_descriptor_node_id("ws", "stable_project_facts"), workspace_id="ws"
        )
        assert descriptor is not None
        assert descriptor.metadata["coverage"]["canonical_record_count"] == 16
        assert [reopened.get_block(record_id).content for record_id in ids] == [
            f"  fact {i}\r\n" for i in range(16)
        ]
    finally:
        reopened.close()


def test_failed_projection_is_recoverable_after_reopen(tmp_path):
    path = tmp_path / "index.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    failed = threading.Event()

    def unavailable(record):
        failed.set()
        raise ConnectionError("native offline")

    index.configure_native_projection(unavailable)
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    saved = client.remember(
        "survives native failure", requested_block_id="stable_project_facts", operation_id="recover"
    )
    assert saved.saved and failed.wait(3)
    index.close()
    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    try:
        reopened.configure_native_projection(
            lambda record: reopened.complete_write_outbox(
                reopened.latest_write_outbox_entry(block_id=record.block_id).commit_revision
            )
        )
        client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=reopened)
        assert client.projection_barrier(
            saved.memory_id, commit_revision=saved.raw["commit_revision"], timeout_seconds=5
        )["ok"]
        assert (
            client.read_exact_record(record_id=saved.memory_id).content == "survives native failure"
        )
        assert len(reopened.export_blocks()) == 1
    finally:
        reopened.close()


def test_concurrent_amendment_has_one_winner(tmp_path):
    path = tmp_path / "index.sqlite"
    index = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    saved = client.remember("version one", requested_block_id="stable_project_facts")
    revision = client.read_exact_record(record_id=saved.memory_id).revision_id
    index.close()
    start = threading.Barrier(2)

    def amend(i):
        backend = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
        try:
            client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
            start.wait(timeout=5)
            try:
                return client.amend(
                    saved.memory_id,
                    expected_revision=revision,
                    content=f"version {i}",
                    reason="explicit correction",
                    requested_block_id="stable_project_facts",
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        finally:
            backend.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(amend, [2, 3]))
    assert sum(bool(result.get("ok")) for result in results) == 1
    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    try:
        assert len(reopened.export_blocks()) == 2
        assert not reopened.pending_write_outbox()
    finally:
        reopened.close()


def _ack_then_wait(path, acknowledgements, entered, release):
    index = BackendSQLiteIndex(path / "index.sqlite", backend_id="test", workspace_id="ws")

    def project(record):
        entered.set()
        release.wait(30)

    index.configure_native_projection(project)
    client = MemoryClient(path, workspace_id="ws", backend_instance=index)
    saved = client.remember(
        " exact bytes\r\n",
        requested_block_id="stable_project_facts",
        operation_id="process-death",
        write_mode="lossless_atomic",
        source_id="turn",
    )
    acknowledgements.put((saved.saved, saved.memory_id, saved.raw["commit_revision"]))
    release.wait(30)
    index.close()


def _crash_before_canonical_commit(path):
    index = BackendSQLiteIndex(path / "index.sqlite", backend_id="test", workspace_id="ws")

    def crash_after_prepare(*_args, **_kwargs):
        assert index.pending_write_outbox()
        os._exit(37)

    index.complete_canonical_head = crash_after_prepare
    MemoryClient(path, workspace_id="ws", backend_instance=index).remember(
        "precommit crash source",
        requested_block_id="stable_project_facts",
        operation_id="precommit-crash",
    )


def test_precommit_process_death_rolls_back_rows_and_replays_same_operation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=_crash_before_canonical_commit, args=(tmp_path,))
    child.start()
    child.join(timeout=10)
    try:
        assert child.exitcode == 37
        index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
        try:
            assert index.pending_write_outbox() == []
            assert index.pending_canonical_head_reservations() == []
            assert index.export_blocks() == []
            assert (
                index._conn.execute("SELECT COUNT(*) FROM canonical_mutation_receipts").fetchone()[
                    0
                ]
                == 0
            )
            saved = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index).remember(
                "precommit crash source",
                requested_block_id="stable_project_facts",
                operation_id="precommit-crash",
            )
            assert saved.saved is True
            assert len(index.export_blocks()) == 1
            assert index.pending_write_outbox() == []
            operation_hash = hashlib.sha256(b"precommit-crash").hexdigest()
            rows = index._conn.execute(
                "SELECT payload_json FROM canonical_mutation_receipts WHERE receipt_id=?",
                (f"creation:{operation_hash}",),
            ).fetchall()
            assert len(rows) == 1
            assert json.loads(rows[0][0])["record_id"] == saved.memory_id
        finally:
            index.close()
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)


def test_acknowledged_write_survives_process_death_before_projection(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    context = multiprocessing.get_context("spawn")
    acknowledgements, entered, release = context.Queue(), context.Event(), context.Event()
    child = context.Process(
        target=_ack_then_wait, args=(tmp_path, acknowledgements, entered, release)
    )
    child.start()
    try:
        saved, record_id, revision = acknowledgements.get(timeout=15)
        assert saved and entered.wait(5)
        child.terminate()
        child.join(timeout=10)
        assert not child.is_alive()
        index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
        try:
            index.configure_native_projection(
                lambda record: index.complete_write_outbox(
                    index.latest_write_outbox_entry(block_id=record.block_id).commit_revision
                )
            )
            client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
            assert client.projection_barrier(
                record_id, commit_revision=revision, timeout_seconds=5
            )["ok"]
            assert client.read_exact_record(record_id=record_id).content == " exact bytes\r\n"
            assert len(index.export_blocks()) == 1
            assert not index.pending_write_outbox()
        finally:
            index.close()
    finally:
        if child.is_alive():
            child.terminate()
        child.join(timeout=10)
        acknowledgements.close()


def test_projection_worker_connections_return_to_plateau(tmp_path) -> None:
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")

    def project(record):
        entry = index.latest_write_outbox_entry(block_id=record.block_id)
        assert entry is not None
        index.complete_write_outbox(entry.commit_revision)

    index.configure_native_projection(project)
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    baseline = len(index._connections)
    try:
        for ordinal in range(20):
            saved = client.remember(f"fact {ordinal}", requested_block_id="stable_project_facts")
            assert saved.saved
            assert client.projection_barrier(saved.memory_id, timeout_seconds=5)["ok"]
        deadline = time.monotonic() + 5
        while len(index._connections) > baseline + 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        index.close()


def _drain_shared_index(path, start):
    index = BackendSQLiteIndex(path / "index.sqlite", backend_id="test", workspace_id="ws")

    def project(record):
        with sqlite3.connect(path / "native.sqlite", timeout=10) as native:
            native.execute("INSERT INTO calls VALUES (?)", (record.block_id,))
        time.sleep(0.02)
        index.complete_write_outbox(
            index.latest_write_outbox_entry(block_id=record.block_id).commit_revision
        )

    assert start.wait(15)
    index.configure_native_projection(project)
    deadline = time.monotonic() + 15
    while index.pending_write_outbox() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not index.pending_write_outbox()
    index.close()


def test_two_processes_project_shared_outbox_once_and_in_order(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    saved = client.remember("seed", requested_block_id="stable_project_facts")
    seed = index.get_block(saved.memory_id)
    for i in range(8):
        record = replace(
            seed, block_id=f"record-{i}", canonical_key="", revision_id=f"revision-{i}"
        )
        index.open_write_outbox(record=record, store="test")
        index.retain_block(record)
    index.close()
    with sqlite3.connect(tmp_path / "native.sqlite") as native:
        native.execute("CREATE TABLE calls(record_id TEXT)")
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    children = [
        context.Process(target=_drain_shared_index, args=(tmp_path, start)) for _ in range(2)
    ]
    try:
        for child in children:
            child.start()
        start.set()
        for child in children:
            child.join(timeout=25)
            assert child.exitcode == 0
        with sqlite3.connect(tmp_path / "native.sqlite") as native:
            assert [
                row[0] for row in native.execute("SELECT record_id FROM calls ORDER BY rowid")
            ] == [f"record-{i}" for i in range(8)]
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)


def test_rejected_canonical_head_does_not_poison_leaf_projection(tmp_path, monkeypatch):
    index = BackendSQLiteIndex(tmp_path / "index.sqlite", backend_id="test", workspace_id="ws")
    index.configure_native_projection(
        lambda record: index.complete_write_outbox(
            index.latest_write_outbox_entry(block_id=record.block_id).commit_revision
        )
    )
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
    original = index.complete_canonical_head

    def reject(*args, **kwargs):
        raise CanonicalHeadConflict("race lost")

    try:
        monkeypatch.setattr(index, "complete_canonical_head", reject)
        rejected = client.remember("rejected", requested_block_id="stable_project_facts")
        assert not rejected.saved
        monkeypatch.setattr(index, "complete_canonical_head", original)
        saved = client.remember("accepted", requested_block_id="stable_project_facts")
        assert saved.saved and client.projection_barrier(saved.memory_id, timeout_seconds=5)["ok"]
        assert not index.pending_write_outbox()
    finally:
        index.close()


def test_exhausted_projection_retries_are_visible_and_explicitly_repairable(tmp_path, monkeypatch):
    backend = MnemosyneBackend(tmp_path, workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    original = backend._project_write_outbox

    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(backend, "_project_write_outbox", unavailable)
    try:
        saved = client.remember("durable", requested_block_id="stable_project_facts")
        deadline = time.monotonic() + 10
        while (
            backend._index.native_projection_health()["attention"] == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        health = backend.health()["projection"]
        assert health["attention"] == 1 and health["recovery"] == "reconcile"
        assert client.projection_barrier(saved.memory_id, timeout_seconds=0)["ok"] is False
        monkeypatch.setattr(backend, "_project_write_outbox", original)
        assert backend.reconcile(dry_run=False).remaining == 0
        assert client.projection_barrier(saved.memory_id, timeout_seconds=0)["ok"]
        assert backend.health()["projection"]["attention"] == 0
    finally:
        backend.close()


@pytest.mark.parametrize("explicit_reconcile", [False, True])
def test_recovery_does_not_interfere_with_live_canonical_writer(
    tmp_path, monkeypatch, explicit_reconcile
):
    backend = MemPalaceBackend(tmp_path, workspace_id="ws")
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    entered, release = threading.Event(), threading.Event()
    original = backend.prepare_canonical_write

    def delayed(record):
        entered.set()
        assert release.wait(10)
        return original(record)

    monkeypatch.setattr(backend, "prepare_canonical_write", delayed)
    second = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            writer = executor.submit(
                client.remember, "live commit", requested_block_id="stable_project_facts"
            )
            assert entered.wait(5)
            recovery = (
                executor.submit(backend.reconcile, dry_run=False)
                if explicit_reconcile
                else executor.submit(MemPalaceBackend, tmp_path, workspace_id="ws")
            )
            assert not recovery.done()
            release.set()
            saved = writer.result(timeout=10)
            recovered = recovery.result(timeout=10)
            if explicit_reconcile:
                assert recovered.failures == 0
            else:
                second = recovered
        assert saved.saved and client.projection_barrier(saved.memory_id, timeout_seconds=10)["ok"]
    finally:
        release.set()
        if second is not None:
            second.close()
        backend.close()


def test_mempalace_snapshot_excludes_commit_after_read_boundary(tmp_path, monkeypatch):
    from lethyx.memory.backends import mempalace

    backend = MemPalaceBackend(tmp_path / "source", workspace_id="ws")
    client = MemoryClient(tmp_path / "source", workspace_id="ws", backend_instance=backend)
    seed = client.remember("before snapshot", requested_block_id="stable_project_facts")
    assert client.projection_barrier(seed.memory_id, timeout_seconds=10)["ok"]
    original = mempalace._backup_sqlite
    saved = []

    def concurrent_commit(connection, destination):
        with ThreadPoolExecutor(max_workers=1) as executor:
            saved.append(
                executor.submit(
                    client.remember, "after snapshot", requested_block_id="stable_project_facts"
                ).result(timeout=5)
            )
        original(connection, destination)

    monkeypatch.setattr(mempalace, "_backup_sqlite", concurrent_commit)
    restored = None
    try:
        assert backend.create_native_snapshot(tmp_path / "snapshot")["ok"]
        assert saved[0].saved
        restored = MemPalaceBackend.restore_native_snapshot(
            tmp_path / "snapshot", repo_root=tmp_path / "restored", workspace_id="ws"
        )
        assert restored.get_block(seed.memory_id).content == "before snapshot"
        assert restored.get_block(saved[0].memory_id) is None
        assert not backend._index._conn.in_transaction
    finally:
        if restored is not None:
            restored.close()
        backend.close()


@pytest.mark.parametrize("backend_name", ["mempalace", "mnemosyne", "hindsight"])
def test_reconcile_does_not_project_a_new_uncommitted_writer(tmp_path, monkeypatch, backend_name):
    from test_real_backend_requirements import _RecoveryHindsightClient

    from lethyx.memory.backends.hindsight import HindsightBackend
    from lethyx.memory.hindsight.config import HindsightConfig

    if backend_name == "hindsight":
        backend = HindsightBackend(
            repo_root=tmp_path,
            workspace_id="ws",
            config=HindsightConfig(enabled=True),
            client=_RecoveryHindsightClient(),
            reconcile_pending_writes=False,
        )
    else:
        factory = MemPalaceBackend if backend_name == "mempalace" else MnemosyneBackend
        backend = factory(tmp_path, workspace_id="ws", reconcile_pending_writes=False)
    client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=backend)
    start, entered, release = threading.Event(), threading.Event(), threading.Event()
    original_replay = backend._index.replay_pending_canonical_heads
    original_complete = backend.complete_canonical_head

    def replay():
        original_replay()
        start.set()
        assert entered.wait(5)

    def delayed_complete(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_complete(*args, **kwargs)

    def write():
        assert start.wait(5)
        return client.remember("live prepared write", requested_block_id="stable_project_facts")

    monkeypatch.setattr(backend._index, "replay_pending_canonical_heads", replay)
    monkeypatch.setattr(backend, "complete_canonical_head", delayed_complete)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            writing = executor.submit(write)
            try:
                backend.reconcile(dry_run=False)
                # The writer is paused before publication. Its canonical row
                # and outbox are one uncommitted transaction, invisible here.
                assert backend._index.pending_write_outbox() == []
                assert backend._index.pending_canonical_head_reservations() == []
            finally:
                release.set()
            saved = writing.result(timeout=10)
        assert saved.saved and client.projection_barrier(saved.memory_id, timeout_seconds=10)["ok"]
    finally:
        release.set()
        backend.close()


@pytest.mark.parametrize("write_mode", ["standard", "lossless_atomic"])
def test_duplicate_operation_concurrently_returns_original_receipt(
    tmp_path, monkeypatch, write_mode
):
    path = tmp_path / "index.sqlite"
    bootstrap = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    MemoryClient(tmp_path, workspace_id="ws", backend_instance=bootstrap).block_map()
    bootstrap.close()
    start = threading.Barrier(4)

    def write():
        index = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
        try:
            client = MemoryClient(tmp_path, workspace_id="ws", backend_instance=index)
            start.wait(5)
            return client.remember(
                "identical source",
                requested_block_id="stable_project_facts",
                operation_id="same-operation",
                write_mode=write_mode,
                metadata={"source_segments": [{"source_id": "turn-1", "text": "identical source"}]},
            )
        finally:
            index.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: write(), range(4)))
    assert all(result.saved for result in results)
    assert len({result.memory_id for result in results}) == 1
    receipts = [
        json.dumps(
            {key: value for key, value in result.raw.items() if key != "replayed"}, sort_keys=True
        )
        for result in results
    ]
    assert all(receipt == receipts[0] for receipt in receipts)
    receipt_bytes = (tmp_path / ".Lethyx/memory/service_receipts.sqlite").read_bytes()
    assert b"identical source" not in receipt_bytes
    assert b"same-operation" not in receipt_bytes
    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="ws")
    try:
        assert len(reopened.export_blocks()) == 1
        with pytest.raises(ValueError, match="receipt conflict"):
            MemoryClient(tmp_path, workspace_id="ws", backend_instance=reopened).remember(
                "different source",
                requested_block_id="stable_project_facts",
                operation_id="same-operation",
            )
        monkeypatch.setattr(reopened, "get_block", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match="canonical record is unavailable"):
            MemoryClient(tmp_path, workspace_id="ws", backend_instance=reopened).remember(
                "identical source",
                requested_block_id="stable_project_facts",
                operation_id="same-operation",
                write_mode=write_mode,
                metadata={"source_segments": [{"source_id": "turn-1", "text": "identical source"}]},
            )
    finally:
        reopened.close()
