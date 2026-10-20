from dataclasses import replace

import pytest

from lethyx import DependencyValidation, MemoryClient
from lethyx.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from lethyx.memory.backends.contracts import EdgeType, MemoryEdgeRecord
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.mutation import MutationDecision, plan_memory_mutation


def _record(record_id: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=record_id,
        canonical_key=f"fact:{record_id}",
        revision_id=f"revision:{record_id}",
        title=record_id,
        content=f"Measured finding {record_id}.",
        workspace_id="alpha",
        kind="fact",
        status="accepted",
        authority="advisory",
        trust_level="workspace_verified",
        verification_status="verified",
        verification_methods=("test_run",),
        promotion_decision="accepted",
        scope="workspace",
        retention="durable",
        valid_from="2026-01-01T00:00:00Z",
        recorded_at="2026-01-01T00:00:00Z",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id="anonymous-trial",
                produced_by="verifier",
                producer_id="independent-fixture",
            ),
        ),
        block_membership=RecordBlockMembership(
            primary_block_id="stable_project_facts", catalog_revision_id="catalog:fixture"
        ),
    )


def _correct(index: BackendSQLiteIndex, source: MemoryBlockRecord) -> str:
    mutation = plan_memory_mutation(
        replace(
            source,
            content="Corrected finding.",
            valid_from="2026-02-01T00:00:00Z",
            recorded_at="2026-02-01T00:00:00Z",
        ),
        existing=source,
        action="correction",
    )
    return _commit(index, mutation)


def _commit(index: BackendSQLiteIndex, mutation: MutationDecision) -> str:
    reservation = index.reserve_canonical_head(
        workspace_id="alpha",
        canonical_key=mutation.record.canonical_key,
        predecessor_id=mutation.superseded_block_id,
        successor_id=mutation.record.block_id,
        successor_status=mutation.record.status,
        successor_authority=mutation.record.authority,
        mutation_action=mutation.action,
        mutation=mutation,
    )
    prepared = index.prepare_canonical_write(mutation.record)
    index.complete_canonical_head(reservation, mutation)
    index.complete_write_outbox(prepared["commit_revision"])
    return mutation.record.block_id


def test_corrected_source_invalidates_transitive_recall_but_preserves_history(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        root = client.block_map()[0]
        leaf = client.create_leaf_block(
            name="Measured findings",
            synopsis="Derived findings under review",
            parent_id=root["block_id"],
            reason="dependency validity fixture",
        )
        source, derived, dependent, cited = (
            replace(
                _record(record_id),
                block_membership=RecordBlockMembership(
                    primary_block_id=leaf["block_id"], catalog_revision_id="catalog:fixture"
                ),
            )
            for record_id in ("source", "derived", "dependent", "cited")
        )
        for record in (source, derived, dependent, cited):
            index.retain_block(record)
        for src, dst, edge_type in (
            (derived, source, EdgeType.DERIVED_FROM),
            (dependent, derived, EdgeType.DERIVED_FROM),
            (cited, source, EdgeType.CITES),
        ):
            index.add_edge(
                MemoryEdgeRecord(
                    src_id=src.block_id,
                    dst_id=dst.block_id,
                    edge_type=edge_type,
                    workspace_id="alpha",
                    created_at="2026-01-01T00:00:00Z",
                )
            )
        client.set_active_mask(
            selected_block_ids=(leaf["block_id"],),
            token_budget=1000,
            activation_reason="current derived findings",
        )
        assert "Measured finding dependent" in client.resolve_active_mask().render()
        prepared = index.prepare_canonical_write(derived)
        index.complete_write_outbox(prepared["commit_revision"])
        query = {"scope_chain": ["workspace:alpha"]}
        assert index.overlay_native_hit(derived, query=query) is not None
        assert client.pin_record("dependent").status == "accepted"
        assert {hit.metadata["block_id"] for hit in client.recall("", search_all=True)} == {
            "source",
            "derived",
            "dependent",
            "cited",
        }

        successor = _correct(index, source)

        assert {hit.metadata["block_id"] for hit in client.recall("", search_all=True)} == {
            successor,
            "cited",
        }
        assert index.get_block("derived") == derived
        assert index.get_block("dependent") == dependent
        assert "Measured finding dependent" not in client.resolve_active_mask().render()
        assert index.overlay_native_hit(derived, query=query) is None
        with pytest.raises(ValueError, match="dependencies are not current"):
            client.pin_record("dependent")
        receipt = client.validate_dependencies("dependent")
        assert isinstance(receipt, DependencyValidation)
        assert receipt.status == "invalidated"
        assert set(receipt.checked_record_ids) == {"source", "derived", "dependent"}
        assert any(issue.record_id == "source" for issue in receipt.issues)
        assert (
            client.validate_dependencies("dependent", as_of="2026-01-15T00:00:00Z").status
            == "valid"
        )
        assert client.validate_dependencies("dependent", max_nodes=1).status == "incomplete"
        assert client.validate_dependencies("dependent", max_edges=1).status == "incomplete"
        with pytest.raises(ValueError, match="max_nodes"):
            client.validate_dependencies("dependent", max_nodes=False)
        assert {
            hit.metadata["block_id"]
            for hit in client.recall(
                "", search_all=True, temporal_mode="historical", as_of="2026-01-15T00:00:00Z"
            )
        } == {"source", "derived", "dependent", "cited"}
    finally:
        index.close()


def test_promotion_carries_source_dependencies_into_the_atomic_accepted_revision(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        source = _record("source")
        candidate = replace(
            _record("derived"),
            status="candidate",
            trust_level="agent_reported",
            verification_status="unverified",
            verification_methods=(),
            promotion_decision="not_requested",
        )
        index.retain_block(source)
        insertion = plan_memory_mutation(candidate, existing=None)
        _commit(index, insertion)
        candidate = insertion.record
        index.add_edge(
            MemoryEdgeRecord(
                src_id=candidate.block_id,
                dst_id="source",
                edge_type=EdgeType.DERIVED_FROM,
                workspace_id="alpha",
            )
        )
        promotion = plan_memory_mutation(
            replace(
                candidate,
                status="accepted",
                trust_level="workspace_verified",
                verification_status="verified",
                verification_methods=("test_run",),
                promotion_decision="accepted",
            ),
            existing=candidate,
            action="promote",
        )
        promoted_id = _commit(index, promotion)
        assert index.validate_dependencies(promoted_id).status == "valid"
        assert any(
            edge.src_id == promoted_id
            and edge.dst_id == "source"
            and edge.edge_type == EdgeType.DERIVED_FROM
            for edge in index.edges_for(promoted_id)
        )

        _correct(index, source)

        assert index.validate_dependencies(promoted_id).status == "invalidated"
        assert index.get_block(promoted_id).status == "accepted"
    finally:
        index.close()


def test_retraction_blocks_dependent_context_without_erasing_accepted_records(tmp_path):
    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        source, derived = _record("source"), _record("derived")
        for record in (source, derived):
            index.retain_block(record)
        index.add_edge(
            MemoryEdgeRecord(
                src_id="derived",
                dst_id="source",
                edge_type=EdgeType.DERIVED_FROM,
                workspace_id="alpha",
            )
        )
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=index)
        assert client.validate_dependencies("derived").status == "valid"

        result = client.forget(
            "source", reason="source was withdrawn", expected_revision=source.revision_id
        )

        assert result["ok"] is True
        assert client.recall("", search_all=True, record_ids=("derived",)) == []
        assert client.validate_dependencies("derived").status == "invalidated"
        assert index.get_block("derived") == derived
        assert client.recall(
            "", search_all=True, temporal_mode="historical", record_ids=("derived",)
        )
    finally:
        index.close()
