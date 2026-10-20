from pathlib import Path

import pytest

from lethyx.api import MemoryClient
from lethyx.contracts import HostObservation
from lethyx.contracts.ledger import (
    append_consolidation_semantic_verification_ledger_event,
    append_verification_report_ledger_event,
)
from lethyx.memory.backends.contracts import EdgeType
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex


def _observation(event_id: str, claim: str) -> HostObservation:
    return HostObservation(
        event_id=event_id,
        event_kind="session_episode",
        raw_evidence=f"raw session evidence for {event_id}\n" + ("w" * 2048),
        atomic_claim=claim,
        observed_at="2026-07-30T12:00:00Z",
        producer_id="hermes:session-lifecycle",
    )


def _semantic_report(
    repo_root: Path,
    *,
    source_ids: tuple[str, ...],
    content: str,
    title: str,
    requested_block_id: str,
):
    return append_consolidation_semantic_verification_ledger_event(
        repo_root=repo_root,
        workspace_id="alpha",
        verifier_id="verifier:lifecycle-semantic-fixture",
        verifier_model_version="semantic-fixture-v1",
        proposer_model_version="deterministic-host-proposer-v1",
        source_ids=source_ids,
        content=content,
        title=title,
        requested_block_id=requested_block_id,
        coverage=True,
        preservation=True,
        faithfulness=True,
        reasons=("fixture independently covers and preserves both source observations",),
    )


def _place_observation(
    client: MemoryClient,
    receipt: dict[str, object],
    *,
    claim: str,
    requested_block_id: str,
) -> dict[str, object]:
    retained = client.remember(
        claim,
        title=claim,
        requested_block_id=requested_block_id,
        evidence_refs=receipt["evidence_refs"],
        metadata={"raw_witness_id": receipt["witness_id"]},
    )
    assert retained.saved is True
    return {**receipt, "memory_id": retained.memory_id}


def test_controlled_lifecycle_consolidates_with_tribunal_and_preserves_raw(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    backend = BackendSQLiteIndex(
        tmp_path / "lifecycle.sqlite",
        backend_id="host-lifecycle",
        workspace_id="alpha",
    )
    client = MemoryClient(
        repo_root,
        workspace_id="alpha",
        session_id="session-lifecycle",
        backend_instance=backend,
    )
    try:
        target = client.create_leaf_block(
            name="Bounded episodes",
            synopsis="LLM-selected materialized views.",
            parent_id=client.block_map()[0]["block_id"],
            reason="explicit consolidation target",
        )
        first = _place_observation(
            client,
            client.observe(_observation("episode-1", "Session established the first outcome.")),
            claim="Session established the first outcome.",
            requested_block_id=target["block_id"],
        )
        second = _place_observation(
            client,
            client.observe(_observation("episode-2", "Session established the second outcome.")),
            claim="Session established the second outcome.",
            requested_block_id=target["block_id"],
        )
        report = append_verification_report_ledger_event(
            repo_root=repo_root,
            workspace_id="alpha",
            verifier_id="verifier:lifecycle-fixture",
            report_hash="sha256:lifecycle-fixture",
            passed=True,
        )

        content = "The two host-observed session outcomes form one bounded episode."
        title = "Controlled lifecycle summary"
        semantic_report = _semantic_report(
            repo_root,
            source_ids=(second["memory_id"], first["memory_id"]),
            content=content,
            title=title,
            requested_block_id=target["block_id"],
        )
        receipt = client.consolidate_observations(
            [second["memory_id"], first["memory_id"]],
            title=title,
            content=content,
            reason="controlled session-boundary lifecycle fixture",
            verification_reports=[report],
            semantic_verification_report=semantic_report,
            requested_block_id=target["block_id"],
        )

        assert receipt["ok"] is True
        assert receipt["route_outcome"] == "consolidated"
        assert receipt["actual_block_id"] == target["block_id"]
        assert receipt["tribunal"]["verdict"] == "accept"
        assert receipt["shadow_resolution"]["disposition"] == "committed"
        assert receipt["consolidation_receipt"]["committed_ids"] == (receipt["memory_id"],)
        consolidated = backend.get_block(receipt["memory_id"])
        assert consolidated is not None
        assert consolidated.status == "candidate"
        assert consolidated.authority == "advisory"
        assert consolidated.block_membership.primary_block_id == target["block_id"]
        source_ids = {first["memory_id"], second["memory_id"]}
        assert {
            edge.dst_id
            for edge in backend.edges_for(consolidated.block_id)
            if edge.src_id == consolidated.block_id and edge.edge_type == EdgeType.DERIVED_FROM
        } == source_ids
        for witness_id, event_id in (
            (first["witness_id"], "episode-1"),
            (second["witness_id"], "episode-2"),
        ):
            witness = backend.get_block(witness_id)
            assert witness is not None
            assert witness.content.startswith(f"raw session evidence for {event_id}")
            assert witness.delivery_policy == "hidden"
        assert (
            backend._conn.execute("SELECT COUNT(*) FROM canonical_mutation_receipts").fetchone()[0]
            == 1
        )
        graph = client.export_graph()
        assert graph["audit"]["valid"] is True
        assert graph["audit"]["isolated_durable_semantic_record_ids"] == []

        replay = client.consolidate_observations(
            [first["memory_id"], second["memory_id"]],
            title="Controlled lifecycle summary",
            content="The two host-observed session outcomes form one bounded episode.",
            reason="controlled session-boundary lifecycle fixture",
            verification_reports=[report],
            semantic_verification_report=semantic_report,
            requested_block_id=target["block_id"],
        )
        assert replay["ok"] is True
        assert replay["replayed"] is True
        assert replay["memory_id"] == receipt["memory_id"]
        assert (
            backend._conn.execute("SELECT COUNT(*) FROM canonical_mutation_receipts").fetchone()[0]
            == 1
        )
    finally:
        backend.close()


def test_consolidation_rejects_untyped_and_cross_workspace_sources(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    backend = BackendSQLiteIndex(
        tmp_path / "alpha.sqlite",
        backend_id="host-lifecycle",
        workspace_id="alpha",
    )
    other = BackendSQLiteIndex(
        tmp_path / "beta.sqlite",
        backend_id="host-lifecycle",
        workspace_id="beta",
    )
    client = MemoryClient(
        repo_root,
        workspace_id="alpha",
        session_id="session-lifecycle",
        backend_instance=backend,
    )
    other_client = MemoryClient(
        repo_root,
        workspace_id="beta",
        session_id="session-lifecycle",
        backend_instance=other,
    )
    try:
        target = client.create_leaf_block(
            name="Bounded episodes",
            synopsis="LLM-selected materialized views.",
            parent_id=client.block_map()[0]["block_id"],
            reason="explicit consolidation target",
        )
        first = _place_observation(
            client,
            client.observe(_observation("episode-1", "First same-workspace episode.")),
            claim="First same-workspace episode.",
            requested_block_id=target["block_id"],
        )
        second = _place_observation(
            client,
            client.observe(_observation("episode-2", "Second same-workspace episode.")),
            claim="Second same-workspace episode.",
            requested_block_id=target["block_id"],
        )
        foreign_target = other_client.create_leaf_block(
            name="Foreign episodes",
            synopsis="Foreign workspace materialized views.",
            parent_id=other_client.block_map()[0]["block_id"],
            reason="cross-workspace test fixture",
        )
        foreign = _place_observation(
            other_client,
            other_client.observe(_observation("episode-3", "Foreign workspace episode.")),
            claim="Foreign workspace episode.",
            requested_block_id=foreign_target["block_id"],
        )
        failed_report = append_verification_report_ledger_event(
            repo_root=repo_root,
            workspace_id="alpha",
            verifier_id="verifier:lifecycle-fixture",
            report_hash="sha256:failed-lifecycle-fixture",
            passed=False,
        )
        rejected_content = "This summary has no passed verification."
        rejected_title = "Rejected lifecycle summary"
        rejected_semantic = _semantic_report(
            repo_root,
            source_ids=(first["memory_id"], second["memory_id"]),
            content=rejected_content,
            title=rejected_title,
            requested_block_id=target["block_id"],
        )
        with pytest.raises(ValueError, match="passed verification"):
            client.consolidate_observations(
                [first["memory_id"], second["memory_id"]],
                title=rejected_title,
                content=rejected_content,
                reason="negative fixture",
                verification_reports=[failed_report],
                semantic_verification_report=rejected_semantic,
                requested_block_id=target["block_id"],
            )

        passed_report = append_verification_report_ledger_event(
            repo_root=repo_root,
            workspace_id="alpha",
            verifier_id="verifier:lifecycle-fixture",
            report_hash="sha256:passed-lifecycle-fixture",
            passed=True,
        )
        cross_workspace_content = "This summary attempts to cross workspace scope."
        cross_workspace_title = "Cross-workspace lifecycle summary"
        cross_workspace_semantic = _semantic_report(
            repo_root,
            source_ids=(first["memory_id"], foreign["memory_id"]),
            content=cross_workspace_content,
            title=cross_workspace_title,
            requested_block_id=target["block_id"],
        )
        with pytest.raises(ValueError, match="was not found"):
            client.consolidate_observations(
                [first["memory_id"], foreign["memory_id"]],
                title=cross_workspace_title,
                content=cross_workspace_content,
                reason="negative fixture",
                verification_reports=[passed_report],
                semantic_verification_report=cross_workspace_semantic,
                requested_block_id=target["block_id"],
            )
    finally:
        backend.close()
        other.close()


def test_consolidation_rejects_a_generic_report_as_semantic_attestation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    backend = BackendSQLiteIndex(
        tmp_path / "lifecycle.sqlite",
        backend_id="host-lifecycle",
        workspace_id="alpha",
    )
    client = MemoryClient(
        repo_root,
        workspace_id="alpha",
        session_id="session-lifecycle",
        backend_instance=backend,
    )
    try:
        target = client.create_leaf_block(
            name="Bounded episodes",
            synopsis="LLM-selected materialized views.",
            parent_id=client.block_map()[0]["block_id"],
            reason="explicit consolidation target",
        )
        first = _place_observation(
            client,
            client.observe(_observation("episode-1", "First observed outcome.")),
            claim="First observed outcome.",
            requested_block_id=target["block_id"],
        )
        second = _place_observation(
            client,
            client.observe(_observation("episode-2", "Second observed outcome.")),
            claim="Second observed outcome.",
            requested_block_id=target["block_id"],
        )
        unrelated_report = append_verification_report_ledger_event(
            repo_root=repo_root,
            workspace_id="alpha",
            verifier_id="verifier:unrelated-test-run",
            report_hash="sha256:unrelated-positive-result",
            passed=True,
        )

        with pytest.raises(ValueError, match="dedicated verifier-owned semantic report"):
            client.consolidate_observations(
                [first["memory_id"], second["memory_id"]],
                title="Unsupported summary",
                content="The moon is made of green cheese.",
                reason="must not accept an unrelated positive report",
                verification_reports=[unrelated_report],
                semantic_verification_report=unrelated_report,
                requested_block_id=target["block_id"],
            )
        assert not any(
            record.content == "The moon is made of green cheese."
            for record_id in backend.record_ids_for_logical_blocks(
                (target["block_id"],),
                workspace_id="alpha",
            )
            if (record := backend.get_block(record_id, workspace_id="alpha")) is not None
        )
    finally:
        backend.close()
