"""Recovery and dependency races at the native canonical publication boundary."""

from dataclasses import replace

import pytest
import lethyx.api as api
from lethyx.api import MemoryClient
from lethyx.contracts import MemoryBlockRecord, VerificationReportRef
from lethyx.contracts.ledger import append_candidate_semantic_verification_ledger_event
from lethyx.memory.backends.sqlite_index import BackendSQLiteIndex
from lethyx.memory.backends.contracts import EdgeType, MemoryEdgeRecord
from lethyx.memory.kernel.models import block_record_to_event
from lethyx.memory.kernel.writer import _write_memory_event
from lethyx.memory.mutation import plan_memory_mutation, resolve_canonical_revision_head
from test_memory_authority_and_isolation import _record


def review_report(client: MemoryClient, record: MemoryBlockRecord) -> VerificationReportRef:
    subject = client.verification_subject(
        record.block_id, expected_source_revision_id=record.revision_id
    )
    return append_candidate_semantic_verification_ledger_event(
        repo_root=client.repo_root,
        subject=subject,
        verifier_id="independent-reviewer",
        verifier_model_version="reviewer-fixture",
        proposer_model_version="author-fixture",
        coverage=True,
        preservation=True,
        faithfulness=True,
        instruction_safety=True,
        reasons=("Synthetic review of the exact correction candidate.",),
    )


def correction_setup(tmp_path):
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha"
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    def insert(record):
        mutation = plan_memory_mutation(record, existing=None, action="insert")
        saved = _write_memory_event(
            tmp_path,
            block_record_to_event(mutation.record),
            workspace_id="alpha",
            backend_instance=backend,
            mutation=mutation,
        )
        assert saved.saved
        return backend.get_block(saved.canonical_id)

    target = insert(_record("trusted-target", repo_root=tmp_path, canonical_key="trusted-target"))
    dependency = insert(_record("source", repo_root=tmp_path, canonical_key="source"))
    staged = client.correction(
        target.block_id,
        "Synthetic independently reviewed correction.",
        requested_block_id="stable_project_facts",
    )
    candidate = backend.get_block(staged["block_id"])
    return client, backend, target, dependency, candidate


def publish(client, tmp_path, candidate):
    result = client.promote(
        candidate.block_id,
        expected_source_revision_id=candidate.revision_id,
        operation_id="review-operation",
        evidence_refs=[review_report(client, candidate)],
        reason="Synthetic independent approval.",
    )
    assert result["saved"]
    return client.backend.get_block(result["block_id"])


def head(client, record):
    return resolve_canonical_revision_head(
        record,
        canonical_key=record.canonical_key,
        workspace_id="alpha",
        get_block=client.backend.get_block,
        edges_for=client.backend.edges_for,
    )


def change_record(tmp_path, backend, record):
    mutation = plan_memory_mutation(
        replace(record, content="Synthetic newer accepted revision."),
        existing=record,
        action="correction",
    )
    saved = _write_memory_event(
        tmp_path,
        block_record_to_event(mutation.record),
        workspace_id="alpha",
        backend_instance=backend,
        mutation=mutation,
    )
    assert saved.saved
    return backend.get_block(saved.canonical_id)


def test_correction_publication_keeps_candidate_dependency(tmp_path):
    client, backend, target, dependency, candidate = correction_setup(tmp_path)
    try:
        backend.add_edge(
            MemoryEdgeRecord(
                src_id=candidate.block_id,
                dst_id=dependency.block_id,
                edge_type=EdgeType.DERIVED_FROM,
                workspace_id="alpha",
                metadata={"source_revision_id": dependency.revision_id},
            )
        )
        assert client.validate_dependencies(candidate.block_id).status == "valid"
        published = publish(client, tmp_path, candidate)
        inherited = [
            edge
            for edge in backend.edges_for(published.block_id)
            if edge.src_id == published.block_id and edge.edge_type == EdgeType.DERIVED_FROM
        ]
        change_record(tmp_path, backend, dependency)
        assert client.validate_dependencies(candidate.block_id).status == "invalidated"
        assert len(inherited) == 1
        assert client.validate_dependencies(published.block_id).status == "invalidated"
    finally:
        backend.close()


def test_compensation_after_newer_proposal_keeps_that_proposal_but_restores_owned_target(tmp_path):
    client, backend, target, dependency, candidate = correction_setup(tmp_path)
    try:
        published = publish(client, tmp_path, candidate)
        amended = client.correction(
            candidate.block_id,
            "Synthetic later independent proposal.",
            requested_block_id="stable_project_facts",
        )
        newer = backend.get_block(amended["block_id"])
        assert newer.revision_of == candidate.block_id
        receipt = client.compensate_promotion(
            candidate.block_id,
            expected_revision=candidate.revision_id,
            operation_id="review-operation",
            reason="Synthetic interrupted host publication.",
        )
        assert receipt.outcome == "superseded"
        assert backend.get_block(newer.block_id) == newer
        actual = head(client, target)
        assert actual.content == target.content
        assert actual.block_id != published.block_id
    finally:
        backend.close()


def test_target_compensation_cas_does_not_touch_interleaved_newer_target(tmp_path, monkeypatch):
    client, backend, target, dependency, candidate = correction_setup(tmp_path)
    try:
        published = publish(client, tmp_path, candidate)
        original_write = api._write_memory_event
        observed = {}

        def interleave(*args, **kwargs):
            if kwargs.get("mutation").action == "rollback":
                monkeypatch.setattr(api, "_write_memory_event", original_write)
                observed["newer"] = change_record(tmp_path, backend, published)
            return original_write(*args, **kwargs)

        monkeypatch.setattr(api, "_write_memory_event", interleave)
        with pytest.raises(RuntimeError, match="canonical_head_conflict"):
            client.compensate_promotion(
                candidate.block_id,
                expected_revision=candidate.revision_id,
                operation_id="review-operation",
                reason="Synthetic compensation race.",
            )
        assert head(client, target).block_id == observed["newer"].block_id
        assert head(client, candidate).block_id == candidate.block_id
        receipt = client.compensate_promotion(
            candidate.block_id,
            expected_revision=candidate.revision_id,
            operation_id="review-operation",
            reason="Retry synthetic compensation race.",
        )
        assert receipt.outcome == "superseded"
        assert head(client, target).block_id == observed["newer"].block_id
    finally:
        backend.close()


def test_restore_keeps_prior_invalid_dependency(tmp_path):
    client, backend, target, dependency, candidate = correction_setup(tmp_path)
    try:
        backend.add_edge(
            MemoryEdgeRecord(
                src_id=target.block_id,
                dst_id=dependency.block_id,
                edge_type=EdgeType.DERIVED_FROM,
                workspace_id="alpha",
                metadata={"source_revision_id": dependency.revision_id},
            )
        )
        publish(client, tmp_path, candidate)
        change_record(tmp_path, backend, dependency)
        assert client.validate_dependencies(target.block_id).status == "invalidated"
        result = client.compensate_promotion(
            candidate.block_id,
            expected_revision=candidate.revision_id,
            operation_id="review-operation",
            reason="Synthetic failed host publication.",
        )
        assert result.outcome == "retracted"
        restored = head(client, target)
        assert restored.content == target.content
        assert client.validate_dependencies(restored.block_id).status == "invalidated"
    finally:
        backend.close()


def test_correction_publication_rejects_source_amended_between_planning_and_commit(
    tmp_path, monkeypatch
):
    client, backend, target, dependency, candidate = correction_setup(tmp_path)
    try:
        original_write = api._write_memory_event
        observed = {}

        def interleave(*args, **kwargs):
            mutation = kwargs.get("mutation")
            if mutation.action == "correction" and mutation.record.status == "accepted":
                monkeypatch.setattr(api, "_write_memory_event", original_write)
                changed = client.correction(
                    candidate.block_id,
                    "Synthetic concurrent proposal revision.",
                    requested_block_id="stable_project_facts",
                )
                assert changed["saved"]
                observed["newer"] = backend.get_block(changed["block_id"])
            return original_write(*args, **kwargs)

        monkeypatch.setattr(api, "_write_memory_event", interleave)
        published = client.promote(
            candidate.block_id,
            expected_source_revision_id=candidate.revision_id,
            operation_id="review-operation",
            evidence_refs=[review_report(client, candidate)],
            reason="Synthetic independent approval.",
        )
        assert head(client, candidate).block_id == observed["newer"].block_id
        assert published.get("saved") is not True
        assert published["ok"] is False
        assert published["error"] == "correction_promotion_source_changed"
        assert head(client, target).block_id == target.block_id
    finally:
        backend.close()
