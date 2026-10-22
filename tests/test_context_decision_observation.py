"""Context packing exposes an exact journal link and measured nested latency."""

from pathlib import Path

import pytest

from um_mem import MemoryClient, api
from um_mem.memory.blocks.catalog import compile_semantic_catalog
from um_mem.memory.blocks.registry import load_memory_blocks
from um_mem.memory.kernel.telemetry import read_memory_events


def test_context_decision_has_measured_latency_current_catalog_and_exact_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MemoryClient(tmp_path, workspace_id="context-test", backend="mempalace")
    try:
        root = client.block_map()[0]
        leaf = client.create_leaf_block(
            name="Measured facts",
            synopsis="Explicitly selected test facts",
            parent_id=root["block_id"],
            reason="test scope",
        )
        saved = client.remember(
            "The measured input is 1200.",
            title="Input",
            run_id="context-run",
            requested_block_id=leaf["block_id"],
        )
        assert saved.saved
        clock_values = iter((1_000_000_000, 1_037_500_000, 2_000_000_000, 2_011_000_000))
        monkeypatch.setattr(api, "perf_counter_ns", lambda: next(clock_values), raising=False)
        items = client.build_context(
            "measured input",
            logical_block_ids=[leaf["block_id"]],
            record_ids=[saved.memory_id],
            include_candidates=True,
            max_tokens=2048,
            run_id="context-run",
        )
        assert len(items) == 1
        decision_id = items[0]["metadata"]["context_decision_id"]
        trace = client.memory_decision_trace(decision_id)
        assert trace.decision.cost_latency_ms == 37
        assert (
            trace.decision.catalog_revision
            == compile_semantic_catalog(
                load_memory_blocks(tmp_path, "context-test", strict=True),
            ).revision_id
        )
        assert trace.decision.selected_candidate_ids == (saved.memory_id,)
        assert {event.usage_type for event in trace.usage_events} == {
            "candidate",
            "selected",
            "rendered",
        }
        repeated = client.build_context(
            "measured input",
            logical_block_ids=[leaf["block_id"]],
            record_ids=[saved.memory_id],
            include_candidates=True,
            max_tokens=2048,
            run_id="context-run",
        )
        next_id = repeated[0]["metadata"]["context_decision_id"]
        assert next_id != decision_id
        assert client.memory_decision_trace(next_id).decision.cost_latency_ms == 11
        assert client.memory_operator_report(decision_id).entries[0].disposition == "unknown"
        packer_rows = [
            row
            for row in read_memory_events(tmp_path)
            if row["event_type"] == "memory_profile" and row["operation"] == "pack_context"
        ]
        assert len(packer_rows) == 2
        assert all(row["status"] == "ok" and row["run_id"] == "context-run" for row in packer_rows)
    finally:
        client.backend.close()
