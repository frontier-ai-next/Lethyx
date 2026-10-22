from pathlib import Path

from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.memory.backends.contracts import NativePartitionMetrics
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


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
