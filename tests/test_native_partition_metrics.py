import sqlite3
from pathlib import Path

from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.memory.backends.contracts import NativePartitionIdentity, NativePartitionMetrics
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


def test_existing_index_adds_native_partition_metric_indexes(tmp_path: Path) -> None:
    path = tmp_path / "existing.sqlite3"
    first = BackendSQLiteIndex(path, backend_id="test", workspace_id="workspace-alpha")
    first.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX idx_native_partition_events_stable_partition")
        connection.execute("DROP INDEX idx_native_partition_events_partition_latency")
    reopened = BackendSQLiteIndex(path, backend_id="test", workspace_id="workspace-alpha")
    reopened.close()
    with sqlite3.connect(path) as connection:
        names = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list('native_partition_events')")
        }
    assert "idx_native_partition_events_stable_partition" in names
    assert "idx_native_partition_events_partition_latency" in names


def _record(record_id: str) -> MemoryBlockRecord:
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
                ref_id=f"metrics:{record_id}",
                produced_by="verifier",
                producer_id="native-partition-metrics-test",
            ),
        ),
        block_membership=RecordBlockMembership(
            primary_block_id="code_resources",
            catalog_revision_id="catalog:revision-a",
        ),
    )


def test_partition_metrics_are_typed_complete_and_aggregate_durable_events(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "metrics.sqlite3",
        backend_id="mempalace",
        workspace_id="workspace-alpha",
    )
    records = (_record("code-1"), _record("code-2"))
    partition = MemPalaceBackend.native_partition_identity(
        workspace_id="workspace-alpha",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:revision-a",
        backend_partition_id=RecordBlockMembership(
            primary_block_id="code_resources", catalog_revision_id="catalog:test"
        ).backend_partition_id,
    )
    try:
        for record in records:
            index.retain_block(record)
        index.record_partition_operation(
            partition,
            operation="write",
            latency_ms=10.0,
            record_bytes=42,
        )
        index.record_partition_operation(
            partition,
            operation="write",
            latency_ms=20.0,
            record_bytes=43,
        )
        index.record_partition_operation(
            partition,
            operation="recall",
            latency_ms=30.0,
            hit_count=2,
            search_mode="scoped",
        )
        index.record_partition_operation(
            partition,
            operation="recall",
            latency_ms=40.0,
            search_mode="search_all",
        )
        index.record_partition_operation(
            partition,
            operation="recall",
            latency_ms=50.0,
            success=False,
            search_mode="scoped",
        )
        index.record_partition_operation(
            partition,
            operation="render",
            rendered_tokens=17,
        )
        for operation in (
            "consolidation",
            "compaction",
            "eviction",
            "expiry",
        ):
            index.record_partition_operation(partition, operation=operation)

        metrics = index.partition_metrics(partition, health="ready")
    finally:
        index.close()

    assert metrics == NativePartitionMetrics(
        identity=partition,
        record_count=2,
        write_count=2,
        recall_count=3,
        hit_count=2,
        zero_hit_count=1,
        failure_count=1,
        record_bytes=85,
        latency_p50_ms=30.0,
        latency_p95_ms=50.0,
        rendered_tokens=17,
        consolidation_count=1,
        compaction_count=1,
        eviction_count=1,
        expiry_count=1,
        scoped_recall_count=2,
        broad_recall_count=1,
        last_accessed_at=metrics.last_accessed_at,
        health="ready",
    )
    assert metrics.last_accessed_at.endswith("Z")
    assert metrics.to_mapping()["partition_id"] == partition.partition_id


def test_partition_metrics_preserve_profile_history_across_catalog_revisions(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "long-history.sqlite3",
        backend_id="mempalace",
        workspace_id="workspace-alpha",
    )
    partition = NativePartitionIdentity(
        backend_id="mempalace",
        workspace_id="workspace-alpha",
        primary_block_id="code_resources",
        catalog_revision_id="catalog:current",
        partition_id="stable-partition",
        isolation_class="physical",
        partition_mode="native_collection_family",
    )
    rows = []
    target_rows = []
    expected_latencies = []
    for event_id in range(20000):
        operation = ("write", "recall", "render", "consolidation")[event_id % 4]
        latency = float((event_id * 37) % 997) / 10
        success = int(event_id % 29 != 0)
        hits = event_id % 5 if operation == "recall" else 0
        record_bytes = event_id % 127 if event_id % 3 == 0 else None
        tokens = event_id % 31
        search_mode = (
            "scoped"
            if operation == "recall" and event_id % 2
            else ("search_all" if operation == "recall" else "")
        )
        row = (
            "workspace-alpha",
            "code_resources",
            f"catalog:{event_id % 7}",
            "stable-partition" if event_id % 20 < 4 else f"decoy-{event_id % 4}",
            "physical",
            "native_collection_family",
            operation,
            latency,
            hits,
            record_bytes,
            tokens,
            success,
            search_mode,
            f"2026-10-23T00:{event_id // 60:02d}:{event_id % 60:02d}Z",
        )
        rows.append(row)
        if event_id % 20 < 4:
            target_rows.append(row)
        if event_id % 20 < 4 and latency > 0:
            expected_latencies.append(latency)
    index._conn.executemany(
        "INSERT INTO native_partition_events(workspace_id, primary_block_id, catalog_revision_id, "
        "partition_id, isolation_class, partition_mode, operation, latency_ms, hit_count, "
        "record_bytes, rendered_tokens, success, search_mode, recorded_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    index._conn.commit()
    query_plan = index._conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM native_partition_events "
        "WHERE workspace_id=? AND primary_block_id=? AND partition_id=? ORDER BY id",
        ("workspace-alpha", "code_resources", "stable-partition"),
    ).fetchall()
    assert any("idx_native_partition_events_stable_partition" in str(row) for row in query_plan)
    percentile_plan = index._conn.execute(
        "EXPLAIN QUERY PLAN SELECT latency_ms FROM native_partition_events "
        "WHERE workspace_id=? AND primary_block_id=? AND partition_id=? AND latency_ms>0 "
        "ORDER BY latency_ms, id LIMIT 1 OFFSET 100",
        ("workspace-alpha", "code_resources", "stable-partition"),
    ).fetchall()
    assert any(
        "idx_native_partition_events_partition_latency" in str(row) for row in percentile_plan
    )
    try:
        metrics = index.partition_metrics(
            partition,
            profile_id="profile-alpha",
            backend_partition_id="stable-partition",
        )
        revision_metrics = index.partition_metrics(partition)
    finally:
        index.close()

    expected_recall = [row for row in target_rows if row[6] == "recall"]
    ordered = sorted(expected_latencies)
    nearest_rank = lambda p: round(ordered[max(1, int(len(ordered) * p + 0.999999999)) - 1], 3)
    assert metrics.write_count == 1000
    assert metrics.recall_count == 1000
    assert metrics.hit_count == sum(row[8] for row in expected_recall if row[11])
    assert metrics.zero_hit_count == sum(row[11] and row[8] == 0 for row in expected_recall)
    assert metrics.failure_count == sum(not row[11] for row in target_rows)
    assert metrics.record_bytes == sum(row[9] for row in target_rows if row[9] is not None)
    assert metrics.rendered_tokens == sum(row[10] for row in target_rows)
    assert metrics.consolidation_count == 1000
    assert metrics.scoped_recall_count == sum(row[12] == "scoped" for row in expected_recall)
    assert metrics.broad_recall_count == sum(row[12] == "search_all" for row in expected_recall)
    assert metrics.latency_p50_ms == nearest_rank(0.50)
    assert metrics.latency_p95_ms == nearest_rank(0.95)
    assert metrics.last_accessed_at == target_rows[-1][-1]
    assert revision_metrics.recall_count == 0
    assert revision_metrics.last_accessed_at == ""
