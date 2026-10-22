from pathlib import Path

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import EvidenceRef, HostObservation
from um_mem.memory.backends.contracts import EdgeType, MemoryEdgeRecord
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


@pytest.mark.parametrize(
    "event_kind",
    [
        "file_created",
        "file_changed",
        "symbol_observed",
        "test_passed",
        "test_failed",
        "run_receipt",
        "trace_assertion",
        "source_opened",
        "source_snapshot",
        "source_claim",
        "session_episode",
        "handoff",
        "checkpoint",
        "report_created",
        "analysis_report",
        "decision",
        "contract",
        "task_checkpoint",
        "subtask_checkpoint",
        "lesson_candidate",
        "skill_artifact",
    ],
)
def test_typed_host_observation_preserves_only_raw_witness_until_agent_places_it(
    tmp_path: Path,
    event_kind: str,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / f"{event_kind}.sqlite",
        backend_id="host-observation",
        workspace_id="alpha",
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-1",
        task_id="task-1",
        subtask_id="subtask-1",
        backend_instance=backend,
    )
    visible_leaf = client.create_leaf_block(
        name="Agent chosen context",
        synopsis="A visible semantic area chosen by the agent.",
        parent_id=client.block_map()[0]["block_id"],
        reason="test visible topology",
    )
    raw = "full raw evidence\n" + ("x" * 4000)
    try:
        receipt = client.observe(
            HostObservation(
                event_id=f"event-{event_kind}",
                event_kind=event_kind,
                raw_evidence=raw,
                atomic_claim=f"Observed {event_kind} outcome.",
                source_path="artifacts/result.json",
                artifact_hash="sha256:" + ("a" * 64),
                observed_at="2026-07-30T10:00:00Z",
                producer_id="hermes:session-1",
            )
        )

        assert receipt["ok"] is True
        assert receipt["route_outcome"] == "needs_agent_choice"
        assert receipt["requested_block_id"] == ""
        assert receipt["actual_block_id"] == ""
        assert receipt["memory_id"] == ""
        assert receipt["placement_manifest"]["candidate_blocks"] == [
            {
                "block_id": visible_leaf["block_id"],
                "use_when": [],
                "avoid_when": [],
            }
        ]
        witness = backend.get_block(receipt["witness_id"])
        assert witness is not None
        assert witness.content == raw
        assert witness.retention == "ephemeral"
        assert witness.delivery_policy == "hidden"
        assert witness.block_membership.primary_block_id == "transient_logs"
        assert backend.export_blocks() == [witness]
    finally:
        backend.close()


def test_host_provenance_graph_is_connected_and_rejects_cycles_and_dangling_edges(
    tmp_path: Path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "provenance.sqlite",
        backend_id="host-observation",
        workspace_id="alpha",
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-1",
        backend_instance=backend,
    )
    try:
        receipt = client.observe(
            HostObservation(
                event_id="event-provenance",
                event_kind="file_created",
                raw_evidence="created src/feature.py with exact bytes",
                atomic_claim="Created src/feature.py.",
                source_path="src/feature.py",
                artifact_hash="sha256:" + ("b" * 64),
                observed_at="2026-07-30T10:00:00Z",
                producer_id="hermes:session-1",
            )
        )
        visible_leaf = client.create_leaf_block(
            name="Feature work",
            synopsis="Agent-chosen semantic context for feature work.",
            parent_id=client.block_map()[0]["block_id"],
            reason="retain the observed feature fact",
        )
        retained = client.remember(
            "Created src/feature.py.",
            title="Feature created",
            requested_block_id=visible_leaf["block_id"],
            evidence_refs=receipt["evidence_refs"],
        )
        graph = client.export_graph()
        assert graph["audit"]["valid"] is True
        assert graph["audit"]["isolated_durable_semantic_record_ids"] == []
        assert graph["audit"]["dangling_edges"] == []
        assert graph["audit"]["cycle_paths"] == []
        assert (
            graph["provenance_routes"][retained.memory_id][0]["target_id"] == receipt["witness_id"]
        )

        evidence = EvidenceRef(
            ref_type="test_run",
            ref_id="cycle-fixture",
            produced_by="verifier",
            producer_id="verifier:cycle-fixture",
        )
        with pytest.raises(ValueError, match="cycle"):
            client.add_edge(
                MemoryEdgeRecord(
                    src_id=receipt["witness_id"],
                    dst_id=retained.memory_id,
                    edge_type=EdgeType.DERIVED_FROM,
                    reason="illegal reverse provenance",
                    evidence_refs=(evidence,),
                )
            )
        with pytest.raises(ValueError, match="endpoints"):
            backend.add_edge(
                MemoryEdgeRecord(
                    src_id=retained.memory_id,
                    dst_id="missing-record",
                    edge_type=EdgeType.SUPPORTS,
                    workspace_id="alpha",
                    reason="dangling fixture",
                    evidence_refs=(evidence,),
                )
            )
    finally:
        backend.close()


def test_explicit_agent_placement_links_to_ambiguous_raw_witness(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "agent-placement.sqlite",
        backend_id="host-observation",
        workspace_id="alpha",
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-1",
        backend_instance=backend,
    )
    try:
        witness = client.observe(
            HostObservation(
                event_id="agent-retain-1",
                event_kind="ambiguous",
                raw_evidence='{"content":"The parser contract is durable."}',
                observed_at="2026-07-30T10:00:00Z",
                producer_id="hermes:session-1",
            )
        )
        assert witness["route_outcome"] == "needs_agent_choice"

        retained = client.remember(
            "The parser contract is durable.",
            title="Parser contract",
            requested_block_id="decisions_and_contracts",
            evidence_refs=witness["evidence_refs"],
            metadata={"raw_witness_id": witness["witness_id"]},
        )

        assert retained.saved is True
        assert {
            edge.dst_id
            for edge in backend.edges_for(retained.memory_id)
            if edge.src_id == retained.memory_id and edge.edge_type == EdgeType.DERIVED_FROM
        } == {witness["witness_id"]}
        graph = client.export_graph()
        assert graph["audit"]["valid"] is True
        assert graph["audit"]["isolated_durable_semantic_record_ids"] == []
    finally:
        backend.close()


def test_ambiguous_host_observation_is_witnessed_but_not_semantically_routed(
    tmp_path: Path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "ambiguous.sqlite",
        backend_id="host-observation",
        workspace_id="alpha",
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-1",
        backend_instance=backend,
    )
    try:
        receipt = client.observe(
            HostObservation(
                event_id="event-ambiguous",
                event_kind="ambiguous",
                raw_evidence="untyped tool output",
                atomic_claim="This may be useful later.",
                observed_at="2026-07-30T10:00:00Z",
                producer_id="hermes:session-1",
            )
        )

        assert receipt["ok"] is True
        assert receipt["route_outcome"] == "needs_agent_choice"
        assert receipt["memory_id"] == ""
        assert receipt["actual_block_id"] == ""
        assert receipt["placement_manifest"]["requires_one_writable_leaf"] is True
        records = backend.export_blocks()
        assert len(records) == 1
        assert records[0].block_membership.primary_block_id == "transient_logs"
    finally:
        backend.close()


def test_passed_host_test_creates_evidence_without_placing_a_semantic_lesson(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "memory"
    workspace_root = tmp_path / "project"
    workspace_root.mkdir()
    (workspace_root / "result.txt").write_text("passed\n", encoding="utf-8")
    backend = BackendSQLiteIndex(
        tmp_path / "promotion.sqlite",
        backend_id="host-observation",
        workspace_id="alpha",
    )
    client = MemoryClient(
        memory_root,
        workspace_id="alpha",
        session_id="session-1",
        backend_instance=backend,
    )
    try:
        test_receipt = client.observe(
            HostObservation(
                event_id="test-result-1",
                event_kind="test_passed",
                raw_evidence="1 passed",
                atomic_claim="The focused regression test passed.",
                artifact_hash="sha256:" + ("c" * 64),
                observed_at="2026-07-30T10:00:00Z",
                producer_id="hermes:session-1",
                workspace_root=str(workspace_root),
                changed_files=("result.txt",),
            )
        )
        lesson = client.observe(
            HostObservation(
                event_id="lesson-1",
                event_kind="lesson_candidate",
                raw_evidence="The bounded recovery path passed its regression test.",
                atomic_claim="Use the bounded recovery path for this failure mode.",
                observed_at="2026-07-30T10:01:00Z",
                producer_id="hermes:session-1",
            )
        )
        reflection_area = client.create_leaf_block(
            name="Recovery lessons",
            synopsis="Agent-chosen candidate lessons about bounded recovery.",
            parent_id=client.block_map()[0]["block_id"],
            reason="retain a lesson candidate supported by host evidence",
        )
        retained_lesson = client.remember(
            "Use the bounded recovery path for this failure mode.",
            title="Bounded recovery lesson",
            requested_block_id=reflection_area["block_id"],
            evidence_refs=lesson["evidence_refs"],
            metadata={"raw_witness_id": lesson["witness_id"]},
        )

        assert test_receipt["verification_report"]["passed"] is True
        assert test_receipt["memory_id"] == ""
        assert lesson["memory_id"] == ""
        retained = backend.get_block(retained_lesson.memory_id)
        assert retained is not None
        assert retained.block_membership.primary_block_id == reflection_area["block_id"]
    finally:
        backend.close()


def test_conversation_turn_is_raw_only_and_never_hidden_semantic_recall(
    tmp_path: Path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "turn.sqlite",
        backend_id="host-observation",
        workspace_id="alpha",
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-1",
        backend_instance=backend,
    )
    try:
        receipt = client.observe(
            HostObservation(
                event_id="turn-1",
                event_kind="conversation_turn",
                raw_evidence="user secret\nassistant answer",
                observed_at="2026-07-30T10:00:00Z",
                producer_id="hermes:session-1",
            )
        )

        assert receipt["route_outcome"] == "raw_witness_only"
        assert receipt["memory_id"] == ""
        assert (
            client.recall(
                "secret",
                logical_block_ids=["workspace_episodes"],
                include_candidates=True,
            )
            == []
        )
    finally:
        backend.close()
