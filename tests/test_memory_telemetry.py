import hashlib

from lethyx.memory.kernel import telemetry

from lethyx.memory.backends.contracts import NativePartitionIdentity
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.cli import main as memory_cli_main
from lethyx.memory.kernel.telemetry import (
    memory_event_report,
    memory_event_trace_path,
    memory_profile_report,
    profile_memory_operation,
    read_memory_events,
    record_memory_event,
)


def test_memory_event_trace_is_structured_content_free_and_reader_backed(tmp_path) -> None:
    secret_identifier = "secret-evidence-id"

    record_memory_event(
        tmp_path,
        event_type="recall_activation",
        workspace_id="workspace-a",
        run_id="run-a",
        backend="mempalace",
        status="ok",
        turn_index=4,
        operation="recall",
        decision="opportunity",
        outcome="available",
        reason="periodic_interval_elapsed",
        candidate_count=3,
        selected_ids=[secret_identifier],
    )
    record_memory_event(
        tmp_path,
        event_type="recall_activation",
        workspace_id="workspace-a",
        run_id="run-a",
        backend="mempalace",
        status="ok",
        turn_index=5,
        operation="recall",
        decision="call",
        outcome="rendered",
        reason="periodic_interval_elapsed",
        candidate_count=3,
        selected_ids=[secret_identifier],
    )

    trace_path = memory_event_trace_path(tmp_path)
    rows = read_memory_events(tmp_path)
    report = memory_event_report(tmp_path)

    assert trace_path.exists()
    assert len(rows) == 2
    assert [row["turn_index"] for row in rows] == [4, 5]
    assert all(row["schema_version"] == 1 for row in rows)
    assert all(row["event_id"] for row in rows)
    assert all(row["event_sha256"] for row in rows)
    assert rows[0]["operation"] == "recall"
    assert rows[0]["decision"] == "opportunity"
    assert rows[1]["decision"] == "call"
    assert rows[1]["outcome"] == "rendered"
    assert rows[0]["selected_id_hashes"] == [
        hashlib.sha256(secret_identifier.encode("utf-8")).hexdigest()[:16]
    ]
    assert secret_identifier not in trace_path.read_text(encoding="utf-8")
    assert report["event_count"] == 2
    assert report["by_operation"] == {"recall": 2}
    assert report["by_decision"] == {"call": 1, "opportunity": 1}


def test_memory_event_reader_preserves_unicode_line_separator(tmp_path) -> None:
    record_memory_event(tmp_path, event_type="runtime_outcome", reason="part\u2028two")
    rows = read_memory_events(tmp_path)
    assert len(rows) == 1
    assert rows[0]["reason"] == "part\u2028two"


def test_memory_event_batches_rotate_without_losing_recent_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "_EVENT_BATCH_LIMIT", 1000)
    monkeypatch.setattr(telemetry, "_EVENT_ROTATE_BYTES", 900)
    for number in range(8):
        record_memory_event(tmp_path, event_type="batch_test", reason=f"first-{number}")
    assert len(read_memory_events(tmp_path)) == 8
    for number in range(8):
        record_memory_event(tmp_path, event_type="batch_test", reason=f"second-{number}")
    rows = read_memory_events(tmp_path)
    assert [row["reason"] for row in rows] == [
        *(f"first-{number}" for number in range(8)),
        *(f"second-{number}" for number in range(8)),
    ]
    path = memory_event_trace_path(tmp_path)
    assert path.with_name(path.name + ".1").exists()


def test_memory_event_reports_bounded_queue_loss(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "_EVENT_MAX_PENDING_BYTES", 1)
    record_memory_event(tmp_path, event_type="batch_test", reason="cannot queue")
    assert memory_event_report(tmp_path)["dropped_event_count"] == 1


def test_memory_event_report_separates_runtime_outcomes(tmp_path) -> None:
    record_memory_event(
        tmp_path,
        event_type="runtime_outcome",
        status="blocked",
        operation="tool_call",
        decision="outcome",
        outcome="backend_mismatch",
        reason="configured_backend_differs",
    )
    record_memory_event(
        tmp_path,
        event_type="runtime_outcome",
        status="failed",
        operation="tool_call",
        decision="outcome",
        outcome="infrastructure_failure",
        reason="subprocess_nonzero",
    )

    report = memory_event_report(tmp_path)

    assert report["by_outcome"] == {
        "backend_mismatch": 1,
        "infrastructure_failure": 1,
    }


def test_memory_profile_records_safe_io_timing_and_nested_parallelism(tmp_path) -> None:
    with profile_memory_operation(
        tmp_path,
        operation="build_context",
        component="memory_client",
        workspace_id="workspace-a",
        profile_id="profile-a",
        user_id="user-a",
        chat_id="chat-a",
        run_id="run-a",
        input_value={"query": "do not persist this query", "max_tokens": 600},
    ) as parent:
        with profile_memory_operation(
            tmp_path,
            operation="backend_recall",
            component="hindsight",
            workspace_id="workspace-a",
            profile_id="profile-a",
            user_id="user-a",
            chat_id="chat-a",
            run_id="run-a",
            input_value={"query": "do not persist this query", "limit": 8},
        ) as child:
            child.set_output({"hit_count": 3, "rendered_tokens": 120})
            child.set_result(outcome="returned", reason="native_recall")
        parent.set_output(
            {
                "items": [{"text": "also do not persist this result"}],
                "diagnostics": {"native_path": "backend.recall"},
            }
        )
        parent.set_result(outcome="returned", reason="context_recall")

    trace_path = memory_event_trace_path(tmp_path)
    rows = read_memory_events(tmp_path)
    profile_rows = [row for row in rows if row["event_type"] == "memory_profile"]
    report = memory_profile_report(tmp_path)

    assert trace_path.exists()
    assert len(profile_rows) == 2
    assert all(row["duration_ms"] >= 0 for row in profile_rows)
    assert all(row["user_id"] == "user-a" for row in profile_rows)
    assert all(row["chat_id"] == "chat-a" for row in profile_rows)
    assert all(row["resources"]["process_cpu_ms"] >= 0 for row in profile_rows)
    assert all(row["resources"]["thread_cpu_ms"] >= 0 for row in profile_rows)
    assert all(row["data"]["input"]["sha256"] for row in profile_rows)
    assert "do not persist" not in trace_path.read_text(encoding="utf-8")
    child = next(row for row in profile_rows if row["operation"] == "backend_recall")
    assert child["parent_span_id"]
    assert child["parallelism"]["max_in_flight"] == 2
    assert report["profiled_operation_count"] == 2
    assert report["by_operation"]["backend_recall"]["count"] == 1
    assert report["by_operation"]["build_context"]["count"] == 1
    assert report["max_in_flight"] == 2


def test_native_partition_metrics_join_the_profile_trace(tmp_path) -> None:
    index = BackendSQLiteIndex(
        tmp_path / "index.sqlite",
        backend_id="mempalace",
        workspace_id="workspace-a",
        repo_root=tmp_path,
    )
    partition = NativePartitionIdentity(
        backend_id="mempalace",
        workspace_id="workspace-a",
        primary_block_id="block-a",
        catalog_revision_id="catalog-a",
        partition_id="partition-a",
        isolation_class="physical",
        partition_mode="native_collection_family",
    )
    try:
        with profile_memory_operation(
            tmp_path,
            operation="backend_recall",
            component="memory_client.backend",
            workspace_id="workspace-a",
            profile_id="profile-a",
            user_id="user-a",
            chat_id="chat-a",
            run_id="run-a",
            backend="mempalace",
            input_value={"query": "do not persist this query"},
        ):
            index.record_partition_operation(
                partition,
                operation="recall",
                latency_ms=12.5,
                hit_count=3,
                success=True,
                search_mode="scoped",
                run_id="run-a",
                profile_id="profile-a",
            )
    finally:
        index.close()

    rows = [
        row
        for row in read_memory_events(tmp_path)
        if row["operation"] == "backend_partition.recall"
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "backend.mempalace.partition"
    assert row["parent_span_id"]
    assert row["duration_ms"] == 12.5
    assert row["user_id"] == "user-a"
    assert row["chat_id"] == "chat-a"
    assert "block-a" not in memory_event_trace_path(tmp_path).read_text(encoding="utf-8")


def test_profile_report_is_available_from_diagnostics_cli(tmp_path, capsys) -> None:
    record_memory_event(
        tmp_path,
        event_type="memory_profile",
        workspace_id="workspace-a",
        operation="backend_recall",
        component="memory_client.backend",
        duration_ms=4.0,
    )

    assert memory_cli_main(["profile-report", "--repo-root", str(tmp_path), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"profiled_operation_count": 1' in output
