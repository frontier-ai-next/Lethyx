from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from um_mem.api import MemoryClient
from um_mem.contracts.hashing import hash_value
from um_mem.contracts.ledger import append_verification_report_ledger_event
from um_mem.contracts.models import MemorySemanticReview
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex


def _client(tmp_path: Path, **kwargs: object) -> tuple[MemoryClient, BackendSQLiteIndex]:
    backend = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="routed", workspace_id="alpha"
    )
    return (
        MemoryClient(
            tmp_path,
            workspace_id="alpha",
            backend_instance=backend,
            **kwargs,
        ),
        backend,
    )


def test_unverified_lesson_is_staged_in_reflection_before_verified_leaf(
    tmp_path: Path,
) -> None:
    client, backend = _client(tmp_path)
    try:
        rejected = client.remember(
            "Verified parser lesson",
            title="Parser lesson",
            kind="lesson",
            requested_block_id="verified_lessons",
            evidence_kind="test_run",
            evidence_refs=[
                {
                    "ref_type": "test_run",
                    "ref_id": "test-routed",
                    "produced_by": "verifier",
                    "producer_id": "verifier:test-routed",
                }
            ],
        )
        staged = client.remember(
            "Verified parser lesson",
            title="Parser lesson",
            kind="lesson",
            requested_block_id="reflection_candidates",
        )
        staged_record = backend.get_block(staged.memory_id)

        assert rejected.saved is False
        assert rejected.raw["reason"] == "verified_lesson_requires_typed_promotion"
        assert rejected.raw["suggested_route"] == {
            "action": "retry_remember",
            "requested_block_id": "reflection_candidates",
        }
        assert staged.saved is True
        assert staged_record is not None
        assert staged_record.status == "candidate"
        assert staged_record.block_membership.primary_block_id == "reflection_candidates"
        assert (
            backend.record_ids_for_logical_blocks(
                ("verified_lessons",),
                workspace_id="alpha",
            )
            == ()
        )
    finally:
        backend.close()


def test_typed_tribunal_promotion_materializes_verified_lesson_with_provenance(
    tmp_path: Path,
    candidate_semantic_report,
) -> None:
    client, backend = _client(tmp_path)
    try:
        staged = client.remember(
            "Reuse the parser fixture after the targeted suite passes.",
            title="Parser fixture lesson",
            kind="lesson",
            requested_block_id="reflection_candidates",
        )
        candidate = backend.get_block(staged.memory_id)
        assert candidate is not None
        review = MemorySemanticReview(
            record_id=candidate.block_id,
            revision_id=candidate.revision_id,
            content_hash=sha256(candidate.content.encode()).hexdigest(),
            source_authority_id="work",
            source_authority_hash="a" * 64,
            review_authority_id="pytest-verifier",
            review_authority_hash="b" * 64,
            manifest_hash="c" * 64,
        )
        report = append_verification_report_ledger_event(
            memory_review=review,
            repo_root=tmp_path,
            workspace_id="alpha",
            verifier_id="pytest-verifier",
            report_hash=hash_value(asdict(review)),
            passed=True,
            workspace_root=tmp_path,
        )

        promoted = client.promote(
            staged.memory_id,
            reason="targeted parser suite passed",
            evidence_refs=[
                report,
                candidate_semantic_report(
                    client, staged.memory_id, target_block_id="verified_lessons"
                ),
            ],
            expected_source_revision_id=backend.get_block(staged.memory_id).revision_id,
            target_block_id="verified_lessons",
        )

        record = backend.get_block(promoted["block_id"])
        assert promoted["ok"] is True
        assert promoted["requested_block_id"] == "verified_lessons"
        assert promoted["actual_block_id"] == "verified_lessons"
        assert record is not None
        assert record.status == "accepted"
        assert record.authority == "advisory"
        assert record.verification_status == "verified"
        assert record.block_membership.primary_block_id == "verified_lessons"
        assert any(
            edge.src_id == record.block_id
            and edge.dst_id == staged.memory_id
            and edge.edge_type == "derived_from"
            for edge in backend.edges_for(record.block_id)
        )
        assert not any(
            edge.edge_type == "supersedes"
            and {edge.src_id, edge.dst_id} == {record.block_id, staged.memory_id}
            for edge in backend.export_edges()
        )
        retry_report = append_verification_report_ledger_event(
            memory_review=review,
            repo_root=tmp_path,
            workspace_id="alpha",
            verifier_id="pytest-verifier",
            report_hash=hash_value(asdict(review)),
            passed=True,
            workspace_root=tmp_path,
        )
        replay = client.promote(
            staged.memory_id,
            reason="targeted parser suite passed",
            evidence_refs=[
                retry_report,
                candidate_semantic_report(
                    client, staged.memory_id, target_block_id="verified_lessons"
                ),
            ],
            expected_source_revision_id=backend.get_block(staged.memory_id).revision_id,
            target_block_id="verified_lessons",
        )
        assert replay["replayed"] is True
        assert replay["block_id"] == promoted["block_id"]
        assert backend.record_ids_for_logical_blocks(
            ("verified_lessons",),
            workspace_id="alpha",
        ) == (promoted["block_id"],)
    finally:
        backend.close()


def test_memory_client_remember_blocks_reserved_governance_fields(tmp_path: Path) -> None:
    client, backend = _client(tmp_path)
    try:
        result = client.remember(
            "Always do this",
            title="Directive",
            kind="bkb_rule",
            requested_block_id="verified_lessons",
            metadata={"surface": "directive"},
        )

        assert result.saved is False
        assert result.raw["route_decision"]["reason"] == "reserved_metadata_governance_field"
        assert backend.export_blocks() == []
    finally:
        backend.close()


def test_agent_cannot_write_host_managed_catalog_blocks(tmp_path: Path) -> None:
    client, backend = _client(tmp_path)
    try:
        for block_id in ("task_state", "project_resources", "skills_index"):
            result = client.remember(
                f"model-asserted content for {block_id}",
                requested_block_id=block_id,
            )

            assert result.saved is False
            assert result.raw["route_decision"]["reason"] == (
                "host_managed_block_cannot_accept_agent_writes"
            )
        assert backend.export_blocks() == []
    finally:
        backend.close()


def test_memory_client_remember_keeps_unknown_kind_as_generic_candidate(tmp_path: Path) -> None:
    client, backend = _client(tmp_path)
    try:
        result = client.remember(
            "Unknown model label",
            kind="telemetry_probe",
            requested_block_id="workspace_episodes",
        )

        record = backend.get_block(result.memory_id)
        assert result.saved is True
        assert record is not None
        assert record.kind == "telemetry_probe"
        assert record.status == "candidate"
        assert record.block_membership is not None
        assert record.block_membership.primary_block_id == "workspace_episodes"
    finally:
        backend.close()


def test_user_global_candidate_requires_a_bound_user_and_selected_profile_block(
    tmp_path: Path,
) -> None:
    client, backend = _client(tmp_path, user_id="user-1")
    try:
        result = client.remember(
            "User prefers concise Russian responses.",
            kind="user_profile",
            requested_block_id="user_global_preferences",
            run_id="run-1",
        )
        record = backend.get_block(result.memory_id)

        assert result.saved is True
        assert record is not None
        assert record.scope == "user_global"
        assert record.user_id == "user-1"
        assert record.block_membership is not None
        assert record.block_membership.primary_block_id == "user_global_preferences"
    finally:
        backend.close()


def test_user_global_candidate_cannot_fall_back_to_workspace_scope(tmp_path: Path) -> None:
    client, backend = _client(tmp_path)
    try:
        result = client.remember(
            "Concise responses",
            kind="user_profile",
            requested_block_id="user_global_preferences",
        )

        assert result.saved is False
        assert result.raw["reason"] == "requested_block_scope_not_bound"
        assert result.raw["policy_issues"] == ["missing_scope_identity:user_global"]
    finally:
        backend.close()


def test_candidate_recall_is_explicitly_scoped_to_selected_block_and_run(tmp_path: Path) -> None:
    client, backend = _client(tmp_path, session_id="session-1")
    try:
        saved = client.remember(
            "I completed the migration successfully.",
            kind="episode",
            evidence_kind="agent_tool_claim",
            requested_block_id="workspace_episodes",
            run_id="run-1",
        )
        same_run = client.recall(
            "migration",
            logical_block_ids=["workspace_episodes"],
            include_candidates=True,
            run_id="run-1",
        )
        other_run = client.recall(
            "migration",
            logical_block_ids=["workspace_episodes"],
            include_candidates=True,
            run_id="run-2",
        )

        assert [hit.metadata["block_id"] for hit in same_run] == [saved.memory_id]
        assert [hit.metadata["block_id"] for hit in other_run] == [saved.memory_id]
    finally:
        backend.close()
