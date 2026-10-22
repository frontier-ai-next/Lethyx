from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import (
    EvidenceRef,
    MemoryBlockRecord,
    MemoryEvent,
    MemorySemanticReview,
    RecordBlockMembership,
    VerificationReportRef,
)
from um_mem.contracts.hashing import hash_value
from um_mem.contracts.ledger import append_verification_report_ledger_event
from um_mem.memory.backends.base import MemoryHit
from um_mem.memory.backends.contracts import (
    CanonicalHeadReservation,
    EdgeType,
    MemoryEdgeRecord,
    block_record_to_memory_hit,
)
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.catalog import compile_semantic_catalog
from um_mem.memory.blocks.registry import load_memory_blocks
from um_mem.memory.kernel.models import block_record_to_event
from um_mem.memory.kernel.writer import _write_memory_event
from um_mem.memory.mutation import plan_memory_mutation, resolve_canonical_revision_head
from um_mem.memory.recall_plan import RecallPlan

_CATALOG_REVISION = "test-catalog-revision"


def _catalog_revision(repo_root: Path, workspace_id: str = "alpha") -> str:
    return compile_semantic_catalog(
        load_memory_blocks(repo_root, workspace_id=workspace_id, strict=True)
    ).revision_id


def _evidence(*, produced_by: str = "verifier") -> dict[str, str]:
    return {
        "ref_type": "test_run",
        "ref_id": "authority-regression",
        "produced_by": produced_by,
        "producer_id": f"{produced_by}:authority-regression",
    }


def _verification_report(repo_root: Path, *, workspace_id: str = "alpha") -> VerificationReportRef:
    review = MemorySemanticReview(
        record_id="candidate",
        revision_id="rev_candidate",
        content_hash=sha256(b"evidence from candidate").hexdigest(),
        source_authority_id="work",
        source_authority_hash="a" * 64,
        review_authority_id="verifier:authority-regression",
        review_authority_hash="b" * 64,
        manifest_hash="c" * 64,
    )
    return append_verification_report_ledger_event(
        repo_root=repo_root,
        workspace_id=workspace_id,
        verifier_id=review.review_authority_id,
        report_hash=hash_value(asdict(review)),
        passed=True,
        memory_review=review,
    )


def _record(
    block_id: str,
    *,
    workspace_id: str = "alpha",
    repo_root: Path | None = None,
    **overrides: Any,
) -> MemoryBlockRecord:
    payload: dict[str, Any] = {
        "block_id": block_id,
        "revision_id": f"rev_{block_id}",
        "block_membership": RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id=(
                _catalog_revision(repo_root, workspace_id) if repo_root else _CATALOG_REVISION
            ),
        ),
        "content": f"evidence from {block_id}",
        "title": block_id,
        "kind": "observation",
        "status": "accepted",
        "retention": "durable",
        "delivery_policy": "on_demand",
        "temperature": "warm",
        "authority": "advisory",
        "trust_level": "workspace_verified",
        "scope": "workspace",
        "verification_status": "verified",
        "verification_methods": ("test_run",),
        "promotion_decision": "accepted",
        "workspace_id": workspace_id,
        "evidence_refs": (
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"evidence-{block_id}",
                produced_by="verifier",
                producer_id=f"verifier:{block_id}",
            ),
        ),
    }
    payload.update(overrides)
    return MemoryBlockRecord(**payload)


class RecordingBackend:
    def __init__(self, records: list[MemoryBlockRecord] | None = None) -> None:
        self.blocks = {record.block_id: record for record in records or []}
        self.writes: list[MemoryBlockRecord] = []
        self.queries: list[dict[str, Any]] = []
        self.edges: list[Any] = []
        self.status_calls = 0
        self.commit_revision = 0

    def reserve_canonical_head(self, **kwargs: Any) -> CanonicalHeadReservation:
        self.commit_revision += 1
        return CanonicalHeadReservation(
            commit_revision=self.commit_revision,
            workspace_id=str(kwargs["workspace_id"]),
            canonical_key_sha256="test",
            predecessor_id=str(kwargs["predecessor_id"]),
            successor_id=str(kwargs["successor_id"]),
            successor_status=str(kwargs["successor_status"]),
            successor_authority=str(kwargs["successor_authority"]),
        )

    def complete_canonical_head(self, reservation: Any, mutation: Any) -> dict[str, Any]:
        del reservation
        if mutation.edge is not None:
            self.edges.append(mutation.edge)
        return {"ok": True}

    def fail_canonical_head(self, reservation: Any, *, error: str) -> None:
        del reservation, error

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.writes.append(record)
        self.blocks[record.block_id] = record
        return {"ok": True, "block_id": record.block_id, "backend": "recording"}

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self.blocks.get(block_id)

    def recall_evidence(self, query: dict[str, Any]) -> list[MemoryBlockRecord]:
        self.queries.append(dict(query))
        return list(self.blocks.values())

    def update_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        self.status_calls += 1
        raise AssertionError("public promotion must use the kernel writer")

    def add_edge(self, edge: Any) -> dict[str, Any]:
        self.edges.append(edge)
        return {"ok": True}

    def edges_for(self, block_id: str) -> list[Any]:
        return [edge for edge in self.edges if edge.src_id == block_id or edge.dst_id == block_id]


def test_agent_writes_require_a_selected_block_and_trusted_corrections_stay_candidates(
    tmp_path: Path,
    candidate_semantic_report,
) -> None:
    trusted = _record("trusted-fact", repo_root=tmp_path)
    backend = RecordingBackend([trusted])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="requested_block_id"):
        client.remember("Unscoped agent write")

    pending = client.correction(
        trusted.block_id,
        "Corrected but not yet host-approved.",
        requested_block_id="stable_project_facts",
    )

    assert pending["saved"] is True
    assert pending["pending_host_approval"] is True
    assert backend.blocks[trusted.block_id].content == trusted.content
    proposal = backend.blocks[str(pending["block_id"])]
    assert proposal.status == "candidate"
    assert proposal.authority == "advisory"
    assert proposal.trust_level == "agent_reported"
    assert proposal.metadata["correction_target_id"] == trusted.block_id
    assert any(
        edge.src_id == proposal.block_id
        and edge.dst_id == trusted.block_id
        and edge.edge_type == "contradicts"
        for edge in backend.edges
    )

    approved = client.promote(
        proposal.block_id,
        reason="host verified the correction",
        evidence_refs=[candidate_semantic_report(client, proposal.block_id)],
        expected_source_revision_id=proposal.revision_id,
    )
    authoritative = backend.blocks[str(approved["block_id"])]
    assert authoritative.content == proposal.content
    assert authoritative.status == "accepted"
    assert authoritative.revision_of == trusted.block_id
    assert authoritative.supersedes_ids == (trusted.block_id,)
    assert approved["approved_correction_candidate_id"] == proposal.block_id


def test_correction_proposals_bind_the_current_writer_and_isolate_generations(
    tmp_path: Path,
) -> None:
    old_owner = {"owner_id": "previous-host", "generation_id": "previous-attempt"}
    trusted = _record(
        "trusted-fact",
        repo_root=tmp_path,
        canonical_key="trusted-stable-fact",
        session_id="previous-session",
        run_id="previous-session",
        metadata={"working_set_owner": old_owner, "promotion_operation_id": "old-promotion"},
    )
    backend = BackendSQLiteIndex(
        tmp_path / "corrections.sqlite", backend_id="test", workspace_id="alpha"
    )
    backend.retain_block(trusted)
    client = MemoryClient(
        tmp_path, workspace_id="alpha", session_id="current-session", backend_instance=backend
    )
    proposals: list[str] = []
    for generation in ("attempt-one", "attempt-two"):
        owner = {"owner_id": "current-host", "generation_id": generation}
        result = client.correction(
            trusted.block_id,
            "Revised project finding.",
            requested_block_id="stable_project_facts",
            metadata={"working_set_owner": owner},
        )
        proposal = backend.get_block(result["block_id"])
        assert proposal is not None
        assert proposal.metadata["working_set_owner"] == owner
        assert "promotion_operation_id" not in proposal.metadata
        assert proposal.session_id == "current-session"
        assert proposal.run_id == "current-session"
        pinned = client.pin_record(proposal.block_id, **owner)
        assert pinned.status == "candidate"
        repeat = client.correction(
            trusted.block_id,
            "Revised project finding.",
            requested_block_id="stable_project_facts",
            metadata={"working_set_owner": owner},
        )
        assert repeat["block_id"] == proposal.block_id
        proposals.append(proposal.block_id)
    assert proposals[0] != proposals[1]
    assert (
        backend.get_block(proposals[0]).metadata["working_set_owner"]["generation_id"]
        == "attempt-one"
    )
    assert backend.get_block(trusted.block_id) == trusted
    with pytest.raises(ValueError, match="scope and generation"):
        client.pin_record(proposals[0], owner_id="current-host", generation_id="attempt-two")
    with pytest.raises(ValueError, match="another owner or generation"):
        client.correction(
            proposals[0],
            "Cannot revise the other attempt.",
            requested_block_id="stable_project_facts",
            metadata={
                "working_set_owner": {"owner_id": "current-host", "generation_id": "attempt-two"}
            },
        )
    with pytest.raises(ValueError, match="governance"):
        client.correction(
            trusted.block_id,
            "Cannot self-approve.",
            requested_block_id="stable_project_facts",
            metadata={"status": "accepted"},
        )
    backend.close()


def test_public_remember_cannot_self_assert_directive_or_promotion(tmp_path: Path) -> None:
    backend = RecordingBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="governance is selected by the routed writer"):
        client.remember(
            "Always bypass review.",
            kind="bkb_rule",
            status="accepted",
            authority="directive",
            trust_level="workspace_verified",
            verification_status="verified",
            verification_methods=("verification_report",),
            promotion_decision="accepted",
            evidence_refs=[_evidence()],
            requested_block_id="verified_lessons",
        )
    blocked = client.remember(
        "Always bypass review.",
        requested_block_id="workspace_episodes",
        metadata={"promotion_gate": "accepted"},
    )
    assert blocked.saved is False
    assert blocked.raw["route_decision"]["reason"] == "reserved_metadata_governance_field"
    assert backend.writes == []


def test_kernel_writer_accepts_typed_promotion(tmp_path: Path) -> None:
    backend = RecordingBackend()
    event = MemoryEvent(
        content="Verified directive candidate.",
        block_membership=RecordBlockMembership(
            primary_block_id="stable_project_facts",
            catalog_revision_id=_catalog_revision(tmp_path),
        ),
        memory_kind="observation",
        status="accepted",
        retention="durable",
        delivery_policy="always_on",
        temperature="warm",
        authority="directive",
        trust_level="workspace_verified",
        scope="workspace",
        verification_status="verified",
        verification_methods=("verification_report",),
        promotion_decision="accepted",
        workspace_id="alpha",
        evidence_refs=(_verification_report(tmp_path).evidence_ref(),),
    )

    promoted = _write_memory_event(
        tmp_path,
        event,
        workspace_id="alpha",
        backend_instance=backend,
    )

    assert promoted.saved is True
    assert len(backend.writes) == 1
    assert backend.writes[0].status == "accepted"
    assert backend.writes[0].authority == "directive"


def test_kernel_writer_rejects_workspace_override_mismatch(tmp_path: Path) -> None:
    backend = RecordingBackend()
    event = MemoryEvent(
        content="Foreign workspace evidence.",
        workspace_id="beta",
    )

    result = _write_memory_event(
        tmp_path,
        event,
        workspace_id="alpha",
        backend_instance=backend,
    )

    assert result.saved is False
    assert any("workspace_id" in issue for issue in result.policy_issues)
    assert backend.writes == []


def test_memory_event_rejects_hidden_backend_selection_before_backend_write(tmp_path: Path) -> None:
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="reserved governance fields"):
        MemoryEvent(
            content="Retired physical routing must not reach the backend.",
            workspace_id="alpha",
            metadata={"block_id": "retired-policy", "write_policy": "selected"},
        )
    assert backend.writes == []


def test_public_remember_writes_agent_evidence_as_candidate(tmp_path: Path) -> None:
    backend = RecordingBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    result = client.remember(
        "An agent cannot verify its own durable lesson.",
        kind="lesson",
        evidence_refs=[_evidence(produced_by="agent")],
        requested_block_id="reflection_candidates",
    )

    assert result.saved is True
    assert backend.writes[0].status == "candidate"
    assert backend.writes[0].authority == "advisory"
    assert backend.writes[0].trust_level == "agent_reported"
    assert backend.writes[0].verification_status == "unverified"


def test_public_non_directive_remember_remains_usable(tmp_path: Path) -> None:
    backend = RecordingBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    result = client.remember(
        "Ordinary supplemental observation.", requested_block_id="workspace_episodes"
    )

    assert result.saved is True
    assert backend.writes[0].status == "candidate"
    assert backend.writes[0].authority == "advisory"
    assert backend.writes[0].promotion_decision == "not_requested"


def test_promote_runs_tribunal_and_rewrites_through_kernel_writer(
    tmp_path: Path, candidate_semantic_report
) -> None:
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
    )
    backend = RecordingBackend([candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    result = client.promote(
        "candidate",
        reason="verified by the regression suite",
        evidence_refs=[candidate_semantic_report(client, "candidate", authority="directive")],
        expected_source_revision_id=candidate.revision_id,
        authority="directive",
    )

    assert result["ok"] is True
    assert backend.status_calls == 0
    promoted = backend.blocks[result["block_id"]]
    assert backend.blocks["candidate"].status == "candidate"
    assert promoted.status == "accepted"
    assert promoted.authority == "directive"
    assert promoted.verification_status == "verified"
    assert promoted.promotion_decision == "accepted"
    assert result["promotion_receipt_id"] == result["block_id"]


def test_promote_rejects_forged_and_stale_verification_reports(tmp_path: Path) -> None:
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
    )
    backend = RecordingBackend([candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    valid = _verification_report(tmp_path)

    with pytest.raises(ValueError, match="fake_evidence_ref"):
        client.promote(
            "candidate",
            reason="forged report",
            evidence_refs=[replace(valid, report_id="missing-report", ledger_hash=None)],
        )
    with pytest.raises(ValueError, match="proof_stale_rerun_required"):
        client.promote(
            "candidate",
            reason="stale report",
            evidence_refs=[replace(valid, workspace_hash="0" * 64)],
        )

    assert backend.writes == []


def test_promote_accepts_exact_workspace_with_scoped_diff_report(
    tmp_path: Path, candidate_semantic_report
) -> None:
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
    )
    backend = RecordingBackend([candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    changed = tmp_path / "changed.py"
    changed.write_text("answer = 42\n", encoding="utf-8")
    report = append_verification_report_ledger_event(
        repo_root=tmp_path,
        workspace_root=tmp_path,
        workspace_id="alpha",
        verifier_id="pytest",
        passed=True,
        report_hash="sha256:pytest-passed",
        changed_files=("changed.py",),
    )

    result = client.promote(
        "candidate",
        reason="exact full workspace hash proves the scoped diff remains current",
        evidence_refs=[report, candidate_semantic_report(client, "candidate")],
        expected_source_revision_id=candidate.revision_id,
        workspace_root=tmp_path,
    )

    assert result["ok"] is True
    promoted = backend.blocks[result["block_id"]]
    assert promoted.status == "accepted"
    assert promoted.verification_status == "verified"


def test_promote_accepts_unchanged_scoped_files_when_runtime_state_changes(
    tmp_path: Path,
    candidate_semantic_report,
) -> None:
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
    )
    backend = RecordingBackend([candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    source = tmp_path / "source.py"
    source.write_text("answer = 42\n", encoding="utf-8")
    report = append_verification_report_ledger_event(
        repo_root=tmp_path,
        workspace_root=tmp_path,
        workspace_id="alpha",
        verifier_id="pytest",
        passed=True,
        report_hash="sha256:scoped-research-proof",
        changed_files=("source.py",),
    )
    semantic_report = candidate_semantic_report(client, "candidate", changed_files=("source.py",))
    (tmp_path / "runtime-state.bin").write_bytes(b"changed after verification")

    result = client.promote(
        "candidate",
        reason="the exact Research source remained unchanged",
        evidence_refs=[report, semantic_report],
        expected_source_revision_id=candidate.revision_id,
        workspace_root=tmp_path,
        changed_files=("source.py",),
        allow_workspace_hash_mismatch_if_diff_matches=True,
    )

    assert result["ok"] is True


def test_promote_rejects_changed_scoped_file_even_when_full_hash_is_relaxed(
    tmp_path: Path,
) -> None:
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
    )
    backend = RecordingBackend([candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    source = tmp_path / "source.py"
    source.write_text("answer = 42\n", encoding="utf-8")
    report = append_verification_report_ledger_event(
        repo_root=tmp_path,
        workspace_root=tmp_path,
        workspace_id="alpha",
        verifier_id="pytest",
        passed=True,
        report_hash="sha256:stale-scoped-research-proof",
        changed_files=("source.py",),
    )
    source.write_text("answer = 43\n", encoding="utf-8")

    with pytest.raises(ValueError, match="proof_stale_rerun_required"):
        client.promote(
            "candidate",
            reason="stale Research source must not be accepted",
            evidence_refs=[report],
            workspace_root=tmp_path,
            changed_files=("source.py",),
            allow_workspace_hash_mismatch_if_diff_matches=True,
        )


def test_promote_rejects_untyped_evidence_and_cross_workspace_sources(tmp_path: Path) -> None:
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
    )
    foreign = replace(candidate, block_id="foreign", workspace_id="beta")
    backend = RecordingBackend([candidate, foreign])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="passed VerificationReportRef"):
        client.promote(
            "candidate",
            reason="self asserted",
            evidence_refs=[_evidence(produced_by="agent")],
            authority="directive",
        )
    with pytest.raises(ValueError, match="workspace"):
        client.promote(
            "foreign",
            reason="wrong workspace",
            evidence_refs=[_verification_report(tmp_path)],
            authority="directive",
        )

    assert backend.writes == []


def test_promote_rejects_quarantined_memory_but_not_false_index_flag(
    tmp_path: Path, candidate_semantic_report
) -> None:
    quarantined = _record("quarantined", repo_root=tmp_path, status="quarantined")
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
        metadata={"index_only": "false"},
    )
    backend = RecordingBackend([quarantined, candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="status"):
        client.promote(
            "quarantined",
            reason="unsafe restore",
            evidence_refs=[_verification_report(tmp_path)],
            expected_source_revision_id=quarantined.revision_id,
            authority="directive",
        )

    result = client.promote(
        "candidate",
        reason="typed supplemental promotion",
        evidence_refs=[candidate_semantic_report(client, "candidate")],
        expected_source_revision_id=candidate.revision_id,
    )
    assert result["ok"] is True


def test_recall_plan_is_bound_to_client_workspace_and_requested_run(tmp_path: Path) -> None:
    backend = RecordingBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="workspace_id"):
        client.recall(
            "evidence",
            recall_plan=RecallPlan(query="evidence", workspace_id="beta"),
        )
    with pytest.raises(ValueError, match="run_id"):
        client.recall(
            "evidence",
            run_id="run-new",
            recall_plan=RecallPlan(
                query="evidence",
                workspace_id="alpha",
                run_id="run-old",
            ),
        )

    client.recall("evidence", recall_plan=RecallPlan(query="evidence", search_all=True))
    assert backend.queries[-1]["workspace_id"] == "alpha"

    normalized = MemoryClient(
        tmp_path,
        workspace_id="workspaces/alpha",
        backend_instance=backend,
    )
    normalized.recall(
        "evidence",
        recall_plan=RecallPlan(query="evidence", workspace_id="workspaces/alpha", search_all=True),
    )
    assert normalized.workspace_id == "alpha"
    assert backend.queries[-1]["workspace_id"] == "alpha"


def test_explicit_user_global_scope_is_bound_to_trusted_client_lineage(tmp_path: Path) -> None:
    records = [
        _record("current", workspace_id="alpha"),
        _record("global", workspace_id="alpha", scope="user_global", user_id="user-1"),
        _record("other-global", workspace_id="alpha", scope="user_global", user_id="user-2"),
        _record("foreign", workspace_id="beta"),
    ]
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        user_id="user-1",
        backend_instance=RecordingBackend(records),
    )

    hits = client.recall(
        "evidence",
        search_all=True,
        scope_chain=("workspace:alpha", "user_global:user-1"),
        limit=20,
    )

    assert {hit.metadata["block_id"] for hit in hits} == {"current", "global"}


def test_project_scope_is_explicit_and_bound_to_client_lineage(tmp_path: Path) -> None:
    project = _record("project", scope="project", project_id="project-1")
    foreign = _record("foreign-project", scope="project", project_id="project-2")
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        project_id="project-1",
        backend_instance=RecordingBackend([project, foreign]),
    )

    hits = client.recall("evidence", search_all=True, scope_chain=("project:project-1",))

    assert [hit.metadata["block_id"] for hit in hits] == ["project"]


@pytest.mark.parametrize("scope", ["session", "subtask"])
def test_narrow_memory_is_bound_by_trusted_scope_lineage(
    tmp_path: Path,
    scope: str,
) -> None:
    identities = {
        "session_id": "session-old",
        **({"task_id": "task-1", "subtask_id": "subtask-1"} if scope == "subtask" else {}),
    }
    record = _record(
        scope,
        scope=scope,
        delivery_policy="working_set",
        run_id="run-old",
        **identities,
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        **identities,
        backend_instance=RecordingBackend([record]),
    )

    assert [hit.metadata["block_id"] for hit in client.recall("evidence", search_all=True)] == [
        scope
    ]
    assert [
        hit.metadata["block_id"]
        for hit in client.recall("evidence", search_all=True, run_id="other")
    ] == [scope]


def test_identifiable_index_only_rows_never_render_as_evidence(tmp_path: Path) -> None:
    flagged = _record("flagged", metadata={"index_only": True})
    routed = _record(
        "routed",
        block_membership=RecordBlockMembership(
            primary_block_id="parser_index",
            catalog_revision_id="catalog:test",
        ),
    )
    backend = RecordingBackend([flagged, routed])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    plan = RecallPlan(
        query="evidence",
        workspace_id="alpha",
        search_all=True,
    )

    assert [hit.metadata["block_id"] for hit in client.recall("evidence", recall_plan=plan)] == [
        "routed"
    ]


def test_native_hit_without_workspace_provenance_fails_closed(tmp_path: Path) -> None:
    class UnscopedHitBackend:
        def recall_evidence(self, query: dict[str, Any]) -> list[MemoryHit]:
            del query
            return [
                MemoryHit(
                    text="backend omitted workspace provenance",
                    source="unscoped-hit",
                    metadata={"block_id": "unscoped-hit"},
                )
            ]

    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        backend_instance=UnscopedHitBackend(),
    )

    assert client.recall("workspace evidence", search_all=True) == []


def test_record_hit_preserves_run_scope_for_post_backend_isolation(tmp_path: Path) -> None:
    stale = _record(
        "stale-run",
        scope="session",
        run_id="run-old",
        session_id="session-old",
    )

    class HitBackend:
        def recall_evidence(self, query: dict[str, Any]) -> list[MemoryHit]:
            del query
            return [block_record_to_memory_hit(stale)]

    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-new",
        backend_instance=HitBackend(),
    )

    assert client.recall("run evidence", search_all=True, run_id="run-new") == []


def test_dict_hit_canonical_workspace_overrides_conflicting_flexible_metadata(
    tmp_path: Path,
) -> None:
    class DictBackend:
        def recall_evidence(self, query: dict[str, Any]) -> list[dict[str, Any]]:
            del query
            return [
                {
                    "block_id": "foreign",
                    "text": "foreign workspace evidence",
                    "workspace_id": "beta",
                    "scope": "workspace",
                    "metadata": {
                        "block_id": "foreign",
                        "workspace_id": "alpha",
                    },
                }
            ]

    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=DictBackend())

    assert client.recall("workspace evidence", search_all=True) == []


def test_record_rejects_safety_axes_hidden_in_flexible_metadata() -> None:
    with pytest.raises(ValueError, match="reserved governance fields"):
        _record(
            "secret",
            content="TOP_SECRET_VALUE",
            sensitivity="secret",
            content_mode="pointer_only",
            metadata={"sensitivity": "internal"},
        )


def test_public_remember_rejects_typed_sensitive_memory_before_backend_write(
    tmp_path: Path,
) -> None:
    backend = RecordingBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    saved = client.remember(
        "TOP_SECRET_PUBLIC_WRITE",
        title="TOP_SECRET_PUBLIC_TITLE",
        tags=["TOP_SECRET_PUBLIC_TAG"],
        sensitivity="secret",
        requested_block_id="workspace_episodes",
        content_mode="pointer_only",
    )
    assert saved.saved is False
    assert backend.writes == []
    assert (
        "protected or non-injectable memory cannot enter selected backend storage"
        in saved.raw["policy_issues"]
    )


def test_public_remember_rejects_safety_axes_hidden_in_flexible_metadata(tmp_path: Path) -> None:
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        backend_instance=RecordingBackend(),
    )

    blocked = client.remember(
        "secret",
        requested_block_id="workspace_episodes",
        metadata={"sensitivity": "secret"},
    )
    assert blocked.saved is False
    assert blocked.raw["route_decision"]["reason"] == "reserved_metadata_governance_field"


@pytest.mark.parametrize("authority", ["advisory", "directive"])
def test_generic_report_cannot_publish_candidate(tmp_path: Path, authority: str) -> None:
    candidate = _record("candidate", repo_root=tmp_path, status="candidate")
    backend = RecordingBackend([candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    with pytest.raises(ValueError, match="dedicated candidate semantic verification"):
        client.promote(
            candidate.block_id,
            expected_source_revision_id=candidate.revision_id,
            reason="A generic report does not review memory content",
            authority=authority,
            evidence_refs=[_verification_report(tmp_path)],
        )
    assert backend.writes == []


def test_recoverable_correction_requires_semantic_verification(tmp_path: Path) -> None:
    trusted = _record("trusted-target", repo_root=tmp_path, canonical_key="trusted-target")
    backend = BackendSQLiteIndex(
        tmp_path / "correction.sqlite", backend_id="test", workspace_id="alpha"
    )
    backend.retain_block(trusted)
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    proposed = client.correction(
        trusted.block_id,
        "A qualified corrected claim.",
        requested_block_id="stable_project_facts",
        metadata={"working_set_owner": {"owner_id": "work", "generation_id": "attempt"}},
    )
    proposal = backend.get_block(proposed["block_id"])
    assert proposal is not None and proposal.status == "candidate"
    with pytest.raises(ValueError, match="dedicated candidate semantic verification"):
        client.promote(
            proposal.block_id,
            reason="Recoverable publication must preserve the predecessor",
            expected_source_revision_id=proposal.revision_id,
            operation_id="exact-change-set",
            evidence_refs=[_verification_report(tmp_path)],
        )
    assert backend.get_block(proposal.block_id) == proposal
    assert backend.get_block(trusted.block_id) == trusted

    backend.close()


@pytest.mark.parametrize("proof", ["integrity", "other_record", "other_revision", "tampered"])
def test_semantic_enum_cannot_upgrade_unbound_or_wrong_review_evidence(
    tmp_path: Path,
    proof: str,
) -> None:
    candidate = _record(
        "candidate",
        repo_root=tmp_path,
        status="candidate",
        trust_level="agent_reported",
        evidence_refs=(),
        verification_status="unverified",
        verification_methods=(),
        promotion_decision="not_requested",
    )
    backend = RecordingBackend([candidate])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    if proof == "integrity":
        report = append_verification_report_ledger_event(
            repo_root=tmp_path,
            workspace_id="alpha",
            verifier_id="work-integrity",
            report_hash="structural-report",
            passed=True,
        )
    else:
        review = MemorySemanticReview(
            record_id="other" if proof == "other_record" else candidate.block_id,
            revision_id="other-revision" if proof == "other_revision" else candidate.revision_id,
            content_hash=sha256(candidate.content.encode()).hexdigest(),
            source_authority_id="work",
            source_authority_hash="a" * 64,
            review_authority_id="independent-review",
            review_authority_hash="b" * 64,
            manifest_hash="c" * 64,
        )
        report = append_verification_report_ledger_event(
            repo_root=tmp_path,
            workspace_id="alpha",
            verifier_id=review.review_authority_id,
            report_hash=hash_value(asdict(review)),
            passed=True,
            memory_review=review,
        )
        if proof == "tampered":
            report = replace(report, report_hash="self-asserted-review")
    with pytest.raises(
        ValueError, match="dedicated candidate semantic verification|verification evidence rejected"
    ):
        client.promote(
            candidate.block_id,
            reason="Enum is not independent proof",
            expected_source_revision_id=candidate.revision_id,
            evidence_refs=[report],
        )
    assert backend.writes == []


@pytest.mark.parametrize("fresh_client", [False, True])
@pytest.mark.parametrize("advance_at_write", [False, True])
def test_first_trusted_correction_rebases_after_unrelated_topology_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_client: bool,
    advance_at_write: bool,
) -> None:
    from um_mem import api

    trusted = _record(
        "trusted-fact",
        repo_root=tmp_path,
        canonical_key="reviewed-fact",
    )
    backend = BackendSQLiteIndex(
        tmp_path / "correction.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    backend.retain_block(trusted)
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="correction-session",
        backend_instance=backend,
        topology_actor="system",
    )

    def advance_catalog() -> None:
        root = client.block_map()[0]["block_id"]
        client.create_leaf_block(
            name="Unrelated subject",
            synopsis="Independent topology growth.",
            parent_id=root,
            reason="New independent subject.",
        )

    if advance_at_write:
        original_write = api._write_memory_event

        def write_after_growth(*args: Any, **kwargs: Any):
            advance_catalog()
            monkeypatch.setattr(api, "_write_memory_event", original_write)
            return original_write(*args, **kwargs)

        monkeypatch.setattr(api, "_write_memory_event", write_after_growth)
    else:
        advance_catalog()
    if fresh_client:
        backend.close()
        backend = BackendSQLiteIndex(
            tmp_path / "correction.sqlite",
            backend_id="test",
            workspace_id="alpha",
        )
        client = MemoryClient(
            tmp_path,
            workspace_id="alpha",
            session_id="correction-session",
            backend_instance=backend,
            topology_actor="system",
        )
    try:
        result = client.correction(
            trusted.block_id,
            "Revised finding awaiting independent review.",
            requested_block_id="stable_project_facts",
        )
        assert result["saved"] is True
        assert result["pending_host_approval"] is True
        assert backend.get_block(trusted.block_id) == trusted
        proposal = backend.get_block(result["block_id"])
        assert proposal is not None
        assert proposal.status == "candidate"
        assert proposal.authority == "advisory"
        assert proposal.verification_status == "unverified"
        assert proposal.metadata["correction_target_revision_id"] == trusted.revision_id
        assert proposal.block_membership is not None
        assert proposal.block_membership.primary_block_id == "stable_project_facts"
        assert proposal.block_membership.catalog_revision_id == _catalog_revision(tmp_path)
        assert any(
            edge.edge_type == "contradicts" and edge.dst_id == trusted.block_id
            for edge in backend.edges_for(proposal.block_id)
        )
        repeat = client.correction(
            trusted.block_id,
            "Revised finding awaiting independent review.",
            requested_block_id="stable_project_facts",
        )
        assert repeat["block_id"] == proposal.block_id
        assert backend.get_block(trusted.block_id) == trusted
    finally:
        backend.close()


@pytest.mark.parametrize(
    "state",
    [
        "prepared",
        "published",
        "foreign_publication",
        "newer",
        "interrupted_cleanup",
        "retracted_proposal",
    ],
)
@pytest.mark.parametrize("with_dependency", [False, True])
def test_recoverable_correction_restores_only_its_exact_accepted_predecessor(
    tmp_path: Path,
    state: str,
    with_dependency: bool,
    monkeypatch: pytest.MonkeyPatch,
    candidate_semantic_report,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "correction.sqlite3", backend_id="test", workspace_id="alpha"
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    original = _record("trusted-correction", repo_root=tmp_path, canonical_key="trusted-correction")
    insertion = plan_memory_mutation(original, existing=None, action="insert")
    retained = _write_memory_event(
        tmp_path,
        block_record_to_event(insertion.record),
        workspace_id="alpha",
        backend_instance=backend,
        mutation=insertion,
    )
    assert retained.saved
    original = backend.get_block(retained.canonical_id)
    assert original is not None
    dependency = None
    if with_dependency:
        dependency = _record(
            "exact-dependency", repo_root=tmp_path, canonical_key="exact-dependency"
        )
        insertion = plan_memory_mutation(dependency, existing=None, action="insert")
        saved = _write_memory_event(
            tmp_path,
            block_record_to_event(insertion.record),
            workspace_id="alpha",
            backend_instance=backend,
            mutation=insertion,
        )
        assert saved.saved
        dependency = backend.get_block(saved.canonical_id)
        assert dependency is not None
        backend.add_edge(
            MemoryEdgeRecord(
                src_id=original.block_id,
                dst_id=dependency.block_id,
                edge_type=EdgeType.DERIVED_FROM,
                workspace_id="alpha",
                metadata={"source_revision_id": dependency.revision_id},
            )
        )
    proposed = client.correction(
        original.block_id,
        "The independently reviewed replacement.",
        requested_block_id="stable_project_facts",
    )
    source = backend.get_block(proposed["block_id"])
    assert source is not None and source.status == "candidate"

    published_id = ""
    try:
        if state != "prepared":
            published = client.promote(
                source.block_id,
                expected_source_revision_id=source.revision_id,
                operation_id="foreign-operation"
                if state == "foreign_publication"
                else "correction-operation",
                evidence_refs=[candidate_semantic_report(client, source.block_id)],
                reason="Exact independent approval of the correction candidate.",
            )
            assert published["ok"] is True
            published_id = published["block_id"]
            accepted = backend.get_block(published_id)
            assert accepted is not None and accepted.status == "accepted"
            assert accepted.revision_of == original.block_id
            assert accepted.metadata["approved_correction_candidate_id"] == source.block_id
            if state == "newer":
                later = plan_memory_mutation(
                    replace(accepted, content="A later independently accepted target revision."),
                    existing=accepted,
                    action="correction",
                )
                saved = _write_memory_event(
                    tmp_path,
                    block_record_to_event(later.record),
                    workspace_id="alpha",
                    backend_instance=backend,
                    mutation=later,
                )
                assert saved.saved
                published_id = saved.canonical_id
        if state == "retracted_proposal":
            assert client.forget(
                source.block_id,
                expected_revision=source.revision_id,
                reason="Late old candidate cleanup.",
            )["saved"]
        if state == "interrupted_cleanup":
            with monkeypatch.context() as interrupted:

                def fail_retraction(*args: object, **kwargs: object) -> None:
                    raise RuntimeError("interrupted after predecessor restoration")

                interrupted.setattr(client, "_retract_current_revision", fail_retraction)
                with pytest.raises(RuntimeError, match="interrupted after predecessor restoration"):
                    client.compensate_promotion(
                        source.block_id,
                        expected_revision=source.revision_id,
                        operation_id="correction-operation",
                        reason="Recover the interrupted host operation.",
                    )
    finally:
        backend.close()

    reopened_backend = BackendSQLiteIndex(
        tmp_path / "correction.sqlite3", backend_id="test", workspace_id="alpha"
    )
    reopened = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=reopened_backend)
    try:
        with pytest.raises(ValueError, match="revision"):
            reopened.compensate_promotion(
                source.block_id,
                expected_revision="wrong-revision",
                operation_id="correction-operation",
                reason="Recover the interrupted host operation.",
            )
        before = reopened_backend.get_block(published_id) if published_id else None
        receipt = reopened.compensate_promotion(
            source.block_id,
            expected_revision=source.revision_id,
            operation_id="correction-operation",
            reason="Recover the interrupted host operation.",
        )
        if state in {"foreign_publication", "newer"}:
            assert receipt.outcome == "superseded"
            assert reopened_backend.get_block(published_id) == before
        else:
            assert receipt.outcome == (
                "already_retracted" if state == "retracted_proposal" else "retracted"
            )
            assert reopened_backend.get_block(receipt.record_id).status == "retracted"
            history = reopened_backend.canonical_history(
                canonical_key=original.canonical_key,
                workspace_id="alpha",
            )
            accepted_heads = [record for record in history if record.status == "accepted"]
            assert any(record.content == original.content for record in accepted_heads)
            root = reopened_backend.get_block(original.block_id)
            head = resolve_canonical_revision_head(
                root,
                canonical_key=original.canonical_key,
                workspace_id="alpha",
                get_block=reopened_backend.get_block,
                edges_for=reopened_backend.edges_for,
            )
            assert head.content == original.content and head.status == "accepted"
            assert head.trust_level == original.trust_level
            assert head.verification_status == original.verification_status
            replay = reopened.compensate_promotion(
                source.block_id,
                expected_revision=source.revision_id,
                operation_id="correction-operation",
                reason="Recover the interrupted host operation.",
            )
            assert replay.outcome == "already_retracted"
            if dependency is not None:
                inherited = [
                    edge
                    for edge in reopened_backend.edges_for(head.block_id)
                    if edge.src_id == head.block_id and edge.edge_type == EdgeType.DERIVED_FROM
                ]
                assert len(inherited) == 1 and inherited[0].dst_id == dependency.block_id
                assert inherited[0].metadata["source_revision_id"] == dependency.revision_id
                assert reopened.validate_dependencies(head.block_id).status == "valid"
                changed = plan_memory_mutation(
                    replace(dependency, content="A newer source invalidates the original claim."),
                    existing=dependency,
                    action="correction",
                )
                assert _write_memory_event(
                    tmp_path,
                    block_record_to_event(changed.record),
                    workspace_id="alpha",
                    backend_instance=reopened_backend,
                    mutation=changed,
                ).saved
                assert reopened.validate_dependencies(head.block_id).status == "invalidated"
        retained = reopened_backend.get_block(original.block_id)
        assert (
            retained.content == original.content
            and retained.evidence_refs == original.evidence_refs
        )
    finally:
        reopened_backend.close()
