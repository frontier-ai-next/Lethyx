import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from um_mem.contracts import (
    ContinuationLiteralSet,
    ContinuationPacket,
    EvidenceRef,
    FoldedBranch,
    MemoryBlockRecord,
    RecordBlockMembership,
    WorkingStateCheckpoint,
)
from um_mem.contracts.memory import CURRENT_SCHEMA_VERSION
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.blocks.registry import create_workspace_block, load_workspace_topology
from um_mem.memory.working_continuity import (
    ContinuityTrial,
    assess_continuation_sufficiency,
    commit_host_compaction_projection,
    commit_working_state_checkpoint,
    compare_continuity_strategies,
    compile_continuation_packet,
    continuation_source_hash,
    evaluate_folded_branch_reopen,
    fold_completed_branch,
    load_working_state_history,
    page_back_continuation,
    recover_continuation_after_failure,
    verify_forced_restart_continuation,
    working_state_canonical_key,
)

_WORKSPACE = "alpha"
_SESSION = "session-1"
_TASK = "task-1"
_RAW_REFS = (
    "event-1",
    "event-2",
    "task-anchor",
    "system-anchor",
    "tail-1",
    "page-1",
    "page-2",
)


def _evidence() -> tuple[EvidenceRef, ...]:
    return (
        EvidenceRef(
            ref_type="artifact",
            ref_id="artifact:working-state",
            produced_by="harness",
            producer_id="harness:working-continuity",
        ),
    )


def _raw_record(
    block_id: str,
    *,
    workspace_id: str = _WORKSPACE,
    session_id: str = _SESSION,
    task_id: str = _TASK,
) -> MemoryBlockRecord:
    revision = hashlib.sha256(block_id.encode("utf-8")).hexdigest()
    return MemoryBlockRecord(
        block_id=block_id,
        canonical_key=f"{workspace_id}:host_event:{block_id}",
        revision_id=f"sha256:{revision}",
        recorded_at="2026-07-14T09:00:00Z",
        content=f"immutable host event {block_id}",
        block_membership=RecordBlockMembership(
            primary_block_id="transient_logs",
            catalog_revision_id="test-catalog",
        ),
        kind="observation",
        status="candidate",
        retention="durable",
        delivery_policy="on_demand",
        authority="advisory",
        trust_level="source_observed",
        scope="task",
        workspace_id=workspace_id,
        session_id=session_id,
        task_id=task_id,
        producer="host_observer",
        evidence_refs=_evidence(),
        metadata={"record_kind": "host_event", "host_event_id": block_id},
    )


def _source_event_sha256(refs: tuple[str, ...]) -> str:
    manifest = [
        {
            "block_id": record.block_id,
            "canonical_key": record.canonical_key,
            "revision_id": record.revision_id,
            "content_sha256": hashlib.sha256(record.content.encode("utf-8")).hexdigest(),
        }
        for ref in refs
        for record in (_raw_record(ref),)
    ]
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _literal_manifest(values: dict[str, object]) -> tuple[ContinuationLiteralSet, ...]:
    rejected = values["rejected_approaches"]
    assert isinstance(rejected, tuple)
    pages = values["evidence_page_refs"]
    assert isinstance(pages, tuple)
    return (
        ContinuationLiteralSet(
            dimension="goals_constraints",
            literals=(
                str(values["working_view"]),
                str(values["current_goal"]),
                str(values["current_subgoal"]),
                *values["plan"],
                *values["unresolved_constraints"],
            ),
        ),
        ContinuationLiteralSet(
            dimension="paths_symbols",
            literals=(*values["touched_files"], *values["touched_symbols"]),
        ),
        ContinuationLiteralSet(dimension="diff", literals=(str(values["current_diff"]),)),
        ContinuationLiteralSet(dimension="error_literals", literals=values["errors"]),
        ContinuationLiteralSet(dimension="tests", literals=values["tests"]),
        ContinuationLiteralSet(
            dimension="open_hypotheses",
            literals=values["open_hypotheses"],
        ),
        ContinuationLiteralSet(
            dimension="rejected_reasons",
            literals=tuple(item.reason for item in rejected),
        ),
        ContinuationLiteralSet(dimension="safety_state", literals=values["safety_state"]),
        ContinuationLiteralSet(
            dimension="raw_page_back_coverage",
            literals=(
                *values["source_event_refs"],
                *values["task_anchor_refs"],
                *values["system_anchor_refs"],
                *values["exact_recent_tail_refs"],
                *(ref for page in pages for ref in page),
                str(values["rollback_or_page_back_ref"]),
            ),
        ),
    )


def _checkpoint(**overrides: object) -> WorkingStateCheckpoint:
    values: dict[str, object] = {
        "workspace_id": _WORKSPACE,
        "profile_id": "default",
        "owner_generation": "owner-1",
        "requested_block_id": "task_state",
        "working_view": "Continue the task.",
        "host_event_id": "checkpoint-event-1",
        "session_id": _SESSION,
        "parent_session_id": "",
        "task_id": _TASK,
        "source_event_refs": ("event-1", "event-2"),
        "source_event_sha256": "",
        "task_anchor_refs": ("task-anchor",),
        "system_anchor_refs": ("system-anchor",),
        "exact_recent_tail_refs": ("tail-1",),
        "evidence_page_refs": (("page-1",), ("page-2",)),
        "rollback_or_page_back_ref": "page-1",
        "artifact_id": "artifact:working-state",
        "artifact_revision": "sha256:checkpoint-1",
        "valid_from": "2026-07-14T09:00:00Z",
        "valid_to": "2026-07-14T10:00:00Z",
        "current_goal": "finish the current task",
        "plan": ("preserve exact state",),
        "current_subgoal": "commit a task-local projection",
        "unresolved_constraints": ("memory remains advisory",),
        "project_id": "project:alpha",
        "repository_id": "repository:alpha",
        "checkout": "C:/repo/UM-MEM",
        "worktree": "C:/repo/UM-MEM",
        "branch": "main",
        "revision": "abc123",
        "dirty_state": "sha256:dirty",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "hard_policy_version": 3,
        "semantic_catalog_version": "catalog/experimental",
        "host_version": "codex/2026-07-14",
        "model_version": "sol",
        "prompt_version": "master-plan/phase-3.5",
        "touched_files": ("src/um_mem/memory/working_continuity.py",),
        "touched_symbols": ("commit_working_state_checkpoint",),
        "current_diff": "sha256:working-diff",
        "tests": ("tests/test_working_continuity.py",),
        "errors": (),
        "open_hypotheses": ("folding must preserve exact literals",),
        "verified_hypotheses": ("canonical history is append-only",),
        "rejected_approaches": (),
        "safety_state": ("working memory remains advisory",),
        "open_branch_ids": ("branch-open",),
        "completed_branch_ids": ("branch-1",),
        "isolated_branch_ids": ("branch-1",),
        "pending_prospective_obligations": (),
        "remaining_work": (),
        "continuation_sufficiency": "sufficient",
        "evidence_refs": _evidence(),
    }
    values.update(overrides)
    if "source_event_sha256" not in overrides:
        values["source_event_sha256"] = _source_event_sha256(values["source_event_refs"])
    if "exact_literal_manifest" not in overrides:
        values["exact_literal_manifest"] = _literal_manifest(values)
    refs = {
        *values["source_event_refs"],
        *values["task_anchor_refs"],
        *values["system_anchor_refs"],
        *values["exact_recent_tail_refs"],
        *(ref for page in values["evidence_page_refs"] for ref in page),
        *values.get("transitive_raw_refs", ()),
    }
    values["source_ref_sha256"] = tuple(
        (ref, continuation_source_hash(_raw_record(ref))) for ref in sorted(refs)
    )
    return WorkingStateCheckpoint(**values)


def _seed(index: BackendSQLiteIndex, refs: tuple[str, ...] = _RAW_REFS) -> str:
    for ref in refs:
        result = index.retain_block(_raw_record(ref))
        assert result["ok"] is True

    root = next(
        state.block.block_id
        for state in load_workspace_topology(index.path.parent, _WORKSPACE)
        if not state.block.parent_id
    )
    return create_workspace_block(
        index.path.parent,
        _WORKSPACE,
        name="Interview working view",
        synopsis="Model-selected current task context",
        parent_id=root,
        reason="model choice in fixture",
        scope="task",
    )["block_id"]


def test_continuity_contracts_require_versioned_task_local_raw_refs() -> None:
    checkpoint = _checkpoint()

    with pytest.raises(ValueError, match="source_event_refs"):
        replace(checkpoint, source_event_refs=())
    with pytest.raises(ValueError, match="evidence_page_refs"):
        replace(checkpoint, evidence_page_refs=((),))

    with pytest.raises(ValueError, match="checkpoint scope"):
        ContinuationPacket(
            profile_id="default",
            owner_generation="owner-1",
            workspace_id=_WORKSPACE,
            session_id=_SESSION,
            task_id="foreign-task",
            checkpoint=checkpoint,
            source_event_range=("event-1", "event-2"),
            source_event_sha256="sha256:" + "2" * 64,
            task_anchor_refs=checkpoint.task_anchor_refs,
            system_anchor_refs=checkpoint.system_anchor_refs,
            exact_recent_tail_refs=checkpoint.exact_recent_tail_refs,
            evidence_page_refs=checkpoint.evidence_page_refs,
            schema_version=CURRENT_SCHEMA_VERSION,
            hard_policy_version=checkpoint.hard_policy_version,
            semantic_catalog_version=checkpoint.semantic_catalog_version,
            host_version=checkpoint.host_version,
            model_version=checkpoint.model_version,
            prompt_version=checkpoint.prompt_version,
            exact_literal_manifest=checkpoint.exact_literal_manifest,
            continuation_sufficiency=checkpoint.continuation_sufficiency,
            expires_at=checkpoint.valid_to,
            rollback_or_page_back_ref=checkpoint.rollback_or_page_back_ref,
        )


def test_compile_packet_preserves_exact_anchors_tail_pages_and_source_range(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        checkpoint = _checkpoint(requested_block_id=leaf)
        packet = compile_continuation_packet(checkpoint, backend=index)
    finally:
        index.close()

    assert packet.checkpoint == checkpoint
    assert packet.source_event_range == ("event-1", "event-2")
    assert packet.source_event_sha256.startswith("sha256:")
    assert len(packet.source_event_sha256) == 71
    assert packet.task_anchor_refs == ("task-anchor",)
    assert packet.system_anchor_refs == ("system-anchor",)
    assert packet.exact_recent_tail_refs == ("tail-1",)
    assert packet.evidence_page_refs == (("page-1",), ("page-2",))
    assert packet.host_version == checkpoint.host_version
    assert packet.hard_policy_version == checkpoint.hard_policy_version
    assert packet.model_version == checkpoint.model_version
    assert packet.prompt_version == checkpoint.prompt_version
    assert packet.exact_literal_manifest == checkpoint.exact_literal_manifest
    assert packet.expires_at == checkpoint.valid_to
    assert packet.rollback_or_page_back_ref == "page-1"


def test_checkpoint_commit_is_candidate_scoped_append_only_and_restart_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "continuity.sqlite"
    index = BackendSQLiteIndex(path, backend_id="mempalace")
    leaf = _seed(index)
    first_checkpoint = _checkpoint(requested_block_id=leaf)

    first = commit_working_state_checkpoint(tmp_path, first_checkpoint, backend=index)
    replay = commit_working_state_checkpoint(tmp_path, first_checkpoint, backend=index)
    first_history = load_working_state_history(
        index,
        profile_id="default",
        owner_generation="owner-1",
        workspace_id=_WORKSPACE,
        session_id=_SESSION,
        task_id=_TASK,
    )
    assert first.saved and replay.saved
    assert len(first_history) == 1

    index.retain_block(_raw_record("event-3"))
    second_checkpoint = _checkpoint(
        requested_block_id=leaf,
        host_event_id="checkpoint-event-2",
        artifact_revision="sha256:checkpoint-2",
        current_goal="finish the revised task",
        source_event_refs=("event-2", "event-3"),
        exact_recent_tail_refs=("event-3",),
    )
    second = commit_working_state_checkpoint(tmp_path, second_checkpoint, backend=index)
    history = load_working_state_history(
        index,
        profile_id="default",
        owner_generation="owner-1",
        workspace_id=_WORKSPACE,
        session_id=_SESSION,
        task_id=_TASK,
    )
    index.close()

    assert second.saved
    assert len(history) == 2
    previous, current = history
    assert previous.block_id != current.block_id
    assert (
        previous.canonical_key
        == current.canonical_key
        == working_state_canonical_key(
            profile_id="default",
            owner_generation="owner-1",
            workspace_id=_WORKSPACE,
            session_id=_SESSION,
            task_id=_TASK,
        )
    )
    assert previous.revision_id != current.revision_id
    assert current.kind == "task_state"
    assert current.status == "candidate"
    assert current.authority == "advisory"
    assert current.scope == "task"
    assert current.workspace_id == _WORKSPACE
    assert current.session_id == _SESSION
    assert current.task_id == _TASK
    assert current.promotion_decision == "not_requested"
    payload = json.loads(current.content)
    assert payload["source_event_range"] == ["event-2", "event-3"]
    assert payload["exact_recent_tail_refs"] == ["event-3"]
    assert payload["evidence_page_refs"] == [["page-1"], ["page-2"]]
    assert payload["checkpoint"]["open_branch_ids"] == ["branch-open"]

    reopened = BackendSQLiteIndex(path, backend_id="mempalace")
    try:
        restored = load_working_state_history(
            reopened,
            profile_id="default",
            owner_generation="owner-1",
            workspace_id=_WORKSPACE,
            session_id=_SESSION,
            task_id=_TASK,
        )
    finally:
        reopened.close()
    assert [record.revision_id for record in restored] == [record.revision_id for record in history]


def test_checkpoint_rejects_same_scope_anchor_drift_without_appending_revision(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        first = commit_working_state_checkpoint(
            tmp_path, _checkpoint(requested_block_id=leaf), backend=index
        )
        index.retain_block(_raw_record("task-anchor-2"))
        index.retain_block(_raw_record("system-anchor-2"))

        drifted = _checkpoint(
            requested_block_id=leaf,
            host_event_id="checkpoint-event-2",
            artifact_revision="sha256:checkpoint-2",
            task_anchor_refs=("task-anchor-2",),
            system_anchor_refs=("system-anchor-2",),
        )
        with pytest.raises(ValueError, match="anchor drift"):
            commit_working_state_checkpoint(tmp_path, drifted, backend=index)

        history = load_working_state_history(
            index,
            profile_id="default",
            owner_generation="owner-1",
            workspace_id=_WORKSPACE,
            session_id=_SESSION,
            task_id=_TASK,
        )
    finally:
        index.close()

    assert first.saved
    assert len(history) == 1
    payload = json.loads(history[0].content)
    assert payload["task_anchor_refs"] == ["task-anchor"]
    assert payload["system_anchor_refs"] == ["system-anchor"]


def test_missing_or_foreign_raw_ref_fails_before_projection_write(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index, tuple(ref for ref in _RAW_REFS if ref != "page-2"))
        with pytest.raises(ValueError, match="page-2.*missing"):
            commit_working_state_checkpoint(
                tmp_path, _checkpoint(requested_block_id=leaf), backend=index
            )

        index.retain_block(_raw_record("page-2", task_id="foreign-task"))
        with pytest.raises(ValueError, match="outside the task"):
            commit_working_state_checkpoint(
                tmp_path, _checkpoint(requested_block_id=leaf), backend=index
            )

        assert (
            load_working_state_history(
                index,
                profile_id="default",
                owner_generation="owner-1",
                workspace_id=_WORKSPACE,
                session_id=_SESSION,
                task_id=_TASK,
            )
            == ()
        )
    finally:
        index.close()


def test_open_branch_cannot_fold_and_remains_explicit_working_state(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        checkpoint = _checkpoint(requested_block_id=leaf)
        committed = commit_working_state_checkpoint(tmp_path, checkpoint, backend=index)
        assessment = assess_continuation_sufficiency(
            checkpoint,
            branch_id="branch-open",
            backend=index,
        )

        assert committed.saved
        assert assessment.sufficient is False
        with pytest.raises(ValueError, match="completed and isolated"):
            fold_completed_branch(
                checkpoint,
                branch_id="branch-open",
                backend=index,
                result="must not fold",
            )

        [stored] = load_working_state_history(
            index,
            profile_id="default",
            owner_generation="owner-1",
            workspace_id=_WORKSPACE,
            session_id=_SESSION,
            task_id=_TASK,
        )
    finally:
        index.close()

    payload = json.loads(stored.content)
    assert payload["checkpoint"]["open_branch_ids"] == ["branch-open"]


def test_completed_isolated_branch_requires_full_sufficiency_and_carries_versions(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        checkpoint = _checkpoint(requested_block_id=leaf)
        folded = fold_completed_branch(
            checkpoint,
            branch_id="branch-1",
            backend=index,
            result="branch verified complete",
            remaining_risks=("page back if verifier fails",),
        )
    finally:
        index.close()

    assert isinstance(folded, FoldedBranch)
    assert folded.sufficiency_assessment.sufficient is True
    assert folded.sufficiency_assessment.covered_dimensions == tuple(
        item.dimension for item in checkpoint.exact_literal_manifest
    )
    assert folded.source_event_sha256 == checkpoint.source_event_sha256
    assert folded.host_version == checkpoint.host_version
    assert folded.hard_policy_version == checkpoint.hard_policy_version
    assert folded.model_version == checkpoint.model_version
    assert folded.prompt_version == checkpoint.prompt_version
    assert folded.exact_literal_manifest == checkpoint.exact_literal_manifest
    assert folded.expires_at == checkpoint.valid_to
    assert folded.rollback_or_page_back_ref == checkpoint.rollback_or_page_back_ref
    assert folded.continuation_sufficiency == "sufficient"


def test_checkpoint_rejects_incomplete_exact_literal_manifest() -> None:
    checkpoint = _checkpoint()

    with pytest.raises(ValueError, match="exact_literal_manifest"):
        replace(checkpoint, exact_literal_manifest=checkpoint.exact_literal_manifest[:-1])


def test_source_event_hash_mismatch_fails_before_continuation_projection(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        checkpoint = replace(
            _checkpoint(requested_block_id=leaf), source_event_sha256="sha256:" + "0" * 64
        )
        with pytest.raises(ValueError, match="source_event_sha256 mismatch"):
            compile_continuation_packet(checkpoint, backend=index)
    finally:
        index.close()


@pytest.mark.parametrize(
    "trigger",
    (
        "uncertainty",
        "failure",
        "plan_transition",
        "source_transition",
        "state_transition",
        "missing_literal",
        "verifier_coverage_gap",
    ),
)
def test_page_back_triggers_return_exact_task_local_raw_view(
    tmp_path: Path,
    trigger: str,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        checkpoint = _checkpoint(requested_block_id=leaf)
        view = page_back_continuation(checkpoint, trigger=trigger, backend=index)
    finally:
        index.close()

    assert view.trigger == trigger
    assert view.raw_refs == _RAW_REFS
    assert tuple(record.block_id for record in view.records) == _RAW_REFS
    assert all(record.workspace_id == _WORKSPACE for record in view.records)
    assert all(record.session_id == _SESSION for record in view.records)
    assert all(record.task_id == _TASK for record in view.records)


def test_page_back_rejects_non_triggering_query(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        with pytest.raises(ValueError, match="page-back trigger"):
            page_back_continuation(
                _checkpoint(requested_block_id=leaf), trigger="routine_query", backend=index
            )
    finally:
        index.close()


def test_host_compaction_commits_only_candidate_working_projection(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        initial = _checkpoint(requested_block_id=leaf)
        commit_working_state_checkpoint(tmp_path, initial, backend=index)
        [previous] = load_working_state_history(
            index,
            profile_id="default",
            owner_generation="owner-1",
            workspace_id=_WORKSPACE,
            session_id=_SESSION,
            task_id=_TASK,
        )
        compacted = _checkpoint(
            requested_block_id=leaf,
            host_event_id="checkpoint-event-compacted",
            artifact_revision="sha256:checkpoint-compacted",
            current_goal="continue after host compaction",
            summary_source_refs=(previous.block_id,),
            transitive_raw_refs=initial.source_event_refs,
            preservation_verified=True,
        )

        result = commit_host_compaction_projection(
            tmp_path,
            compacted,
            backend=index,
        )
        history = load_working_state_history(
            index,
            profile_id="default",
            owner_generation="owner-1",
            workspace_id=_WORKSPACE,
            session_id=_SESSION,
            task_id=_TASK,
        )
    finally:
        index.close()

    assert result.saved
    assert len(history) == 2
    stored = history[-1]
    assert stored.kind == "task_state"
    assert stored.status == "candidate"
    assert stored.authority == "advisory"
    assert stored.promotion_decision == "not_requested"
    payload = json.loads(stored.content)["checkpoint"]
    assert payload["summary_source_refs"] == [previous.block_id]
    assert payload["transitive_raw_refs"] == ["event-1", "event-2"]
    assert payload["preservation_verified"] is True


def test_recursive_summary_requires_transitive_raw_refs_and_reverification() -> None:
    with pytest.raises(ValueError, match="transitive_raw_refs"):
        _checkpoint(
            summary_source_refs=("projection-1",),
            preservation_verified=True,
        )
    with pytest.raises(ValueError, match="preservation_verified"):
        _checkpoint(
            summary_source_refs=("projection-1",),
            transitive_raw_refs=("event-1",),
        )
    with pytest.raises(ValueError, match="referenced immutable raw events"):
        _checkpoint(
            summary_source_refs=("projection-1",),
            transitive_raw_refs=("foreign-event",),
            preservation_verified=True,
        )


def test_working_projection_key_is_isolated_by_workspace_session_and_task() -> None:
    base = working_state_canonical_key(
        profile_id="default",
        owner_generation="owner-1",
        workspace_id=_WORKSPACE,
        session_id=_SESSION,
        task_id=_TASK,
    )
    assert base != working_state_canonical_key(
        profile_id="default",
        owner_generation="owner-1",
        workspace_id="beta",
        session_id=_SESSION,
        task_id=_TASK,
    )
    assert base != working_state_canonical_key(
        profile_id="default",
        owner_generation="owner-1",
        workspace_id=_WORKSPACE,
        session_id="session-2",
        task_id=_TASK,
    )
    assert base != working_state_canonical_key(
        profile_id="default",
        owner_generation="owner-1",
        workspace_id=_WORKSPACE,
        session_id=_SESSION,
        task_id="task-2",
    )


def _continuity_trial(
    strategy: str,
    *,
    action_succeeded: bool,
    tokens_used: int,
    repeated_work_turns: int,
    reacquisition_turns: int,
    page_back_count: int,
) -> ContinuityTrial:
    return ContinuityTrial(
        strategy=strategy,
        task_id="task-comparison-1",
        budget_tokens=10_000,
        tool_turn_count=200,
        action_succeeded=action_succeeded,
        repeated_work_turns=repeated_work_turns,
        reacquisition_turns=reacquisition_turns,
        expected_exact_literals=10,
        survived_exact_literals=10,
        tokens_used=tokens_used,
        latency_samples_ms=(10.0, 20.0, 30.0, 40.0, 50.0),
        prefix_cache_eligible_tokens=5_000,
        prefix_cache_reused_tokens=4_000,
        page_back_count=page_back_count,
        folded_branch_count=1,
        folded_branches_with_raw_refs=1,
        working_projection_advisory=True,
        semantic_truth_created=False,
    )


def test_continuity_comparison_reports_required_metrics_and_adaptive_pareto() -> None:
    comparison = compare_continuity_strategies(
        (
            _continuity_trial(
                "no_compaction",
                action_succeeded=False,
                tokens_used=10_000,
                repeated_work_turns=20,
                reacquisition_turns=20,
                page_back_count=0,
            ),
            _continuity_trial(
                "threshold_summary",
                action_succeeded=True,
                tokens_used=8_000,
                repeated_work_turns=16,
                reacquisition_turns=12,
                page_back_count=2,
            ),
            _continuity_trial(
                "structured_checkpoint",
                action_succeeded=True,
                tokens_used=7_000,
                repeated_work_turns=8,
                reacquisition_turns=6,
                page_back_count=2,
            ),
            _continuity_trial(
                "branch_fold",
                action_succeeded=True,
                tokens_used=6_500,
                repeated_work_turns=6,
                reacquisition_turns=4,
                page_back_count=1,
            ),
            _continuity_trial(
                "adaptive_manager",
                action_succeeded=True,
                tokens_used=6_000,
                repeated_work_turns=4,
                reacquisition_turns=2,
                page_back_count=1,
            ),
        )
    )

    assert comparison.task_ids == ("task-comparison-1",)
    assert comparison.budget_tokens == (10_000,)
    assert comparison.adaptive_pareto_improves_threshold is True
    assert comparison.safety_and_literal_gates_pass is True
    adaptive = comparison.metric_for("adaptive_manager")
    assert adaptive.continuation_success_rate == 1.0
    assert adaptive.repeated_work_rate == 0.02
    assert adaptive.reacquisition_rate == 0.01
    assert adaptive.exact_literal_survival_rate == 1.0
    assert adaptive.tokens_used == 6_000
    assert adaptive.latency_p50_ms == 30.0
    assert adaptive.latency_p95_ms == 50.0
    assert adaptive.prefix_cache_reuse_rate == 0.8
    assert adaptive.page_back_frequency == 0.005


def test_continuity_comparison_rejects_missing_strategy_or_unequal_budget() -> None:
    complete = [
        _continuity_trial(
            strategy,
            action_succeeded=True,
            tokens_used=6_000,
            repeated_work_turns=4,
            reacquisition_turns=2,
            page_back_count=1,
        )
        for strategy in (
            "no_compaction",
            "threshold_summary",
            "structured_checkpoint",
            "branch_fold",
            "adaptive_manager",
        )
    ]
    with pytest.raises(ValueError, match="all five continuity strategies"):
        compare_continuity_strategies(tuple(complete[:-1]))
    with pytest.raises(ValueError, match="same token budget"):
        compare_continuity_strategies((*complete[:-1], replace(complete[-1], budget_tokens=9_000)))


@pytest.mark.parametrize("tool_turn_count", (100, 250, 500))
def test_forced_restart_is_transcript_free_and_preserves_required_literals(
    tmp_path: Path,
    tool_turn_count: int,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        checkpoint = _checkpoint(requested_block_id=leaf)
        proof = verify_forced_restart_continuation(
            checkpoint,
            backend=index,
            tool_turn_count=tool_turn_count,
            required_literals=("finish the current task", "memory remains advisory"),
            required_evidence_ref_ids=("artifact:working-state",),
        )
    finally:
        index.close()

    assert proof.tool_turn_count == tool_turn_count
    assert proof.transcript_required is False
    assert proof.constraints_preserved is True
    assert proof.evidence_preserved is True
    assert proof.packet.checkpoint == checkpoint


def test_forced_restart_requires_declared_constraints_and_evidence(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        with pytest.raises(ValueError, match="required_literals"):
            verify_forced_restart_continuation(
                _checkpoint(requested_block_id=leaf),
                backend=index,
                tool_turn_count=100,
                required_literals=(),
                required_evidence_ref_ids=("artifact:working-state",),
            )
        with pytest.raises(ValueError, match="required_evidence_ref_ids"):
            verify_forced_restart_continuation(
                _checkpoint(requested_block_id=leaf),
                backend=index,
                tool_turn_count=100,
                required_literals=("finish the current task",),
                required_evidence_ref_ids=(),
            )
    finally:
        index.close()


def test_completed_branch_reopens_only_for_traced_evidence_or_state_change(
    tmp_path: Path,
) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        folded = fold_completed_branch(
            _checkpoint(requested_block_id=leaf),
            branch_id="branch-1",
            backend=index,
            result="complete",
        )
    finally:
        index.close()

    unchanged = evaluate_folded_branch_reopen(
        folded,
        observed_source_event_sha256=folded.source_event_sha256,
        previous_state_sha256="sha256:" + "1" * 64,
        observed_state_sha256="sha256:" + "1" * 64,
    )
    changed = evaluate_folded_branch_reopen(
        folded,
        observed_source_event_sha256="sha256:" + "2" * 64,
        previous_state_sha256="sha256:" + "1" * 64,
        observed_state_sha256="sha256:" + "1" * 64,
    )
    state_changed = evaluate_folded_branch_reopen(
        folded,
        observed_source_event_sha256=folded.source_event_sha256,
        previous_state_sha256="sha256:" + "1" * 64,
        observed_state_sha256="sha256:" + "3" * 64,
    )

    assert unchanged.reopen is False
    assert unchanged.reason == "unchanged"
    assert changed.reopen is True
    assert changed.reason == "new_evidence"
    assert state_changed.reopen is True
    assert state_changed.reason == "state_change"


def test_failure_recovery_selects_last_known_good_or_exact_raw_view(tmp_path: Path) -> None:
    index = BackendSQLiteIndex(tmp_path / "continuity.sqlite", backend_id="mempalace")
    try:
        leaf = _seed(index)
        checkpoint = _checkpoint(requested_block_id=leaf)
        last_known_good = recover_continuation_after_failure(
            checkpoint,
            backend=index,
            last_known_good=checkpoint,
        )
        exact_raw = recover_continuation_after_failure(
            checkpoint,
            backend=index,
            last_known_good=None,
        )
    finally:
        index.close()

    assert last_known_good.mode == "last_known_good"
    assert last_known_good.packet is not None
    assert last_known_good.raw_view is None
    assert exact_raw.mode == "exact_recent_raw"
    assert exact_raw.packet is None
    assert exact_raw.raw_view is not None
    assert exact_raw.raw_view.trigger == "failure"


def test_checkpoint_accepts_non_repository_work_and_pins_owner() -> None:
    state = _checkpoint()
    from dataclasses import asdict, fields

    optional = {
        "repository_id",
        "checkout",
        "worktree",
        "branch",
        "revision",
        "dirty_state",
        "touched_files",
        "touched_symbols",
        "current_diff",
        "tests",
        "errors",
    }
    values = {
        field.name: getattr(state, field.name)
        for field in fields(state)
        if field.name not in optional | {"exact_literal_manifest"}
    }
    values.update(
        profile_id="profile-a",
        owner_generation="owner-7",
        requested_block_id="model-chosen-leaf",
        working_view="Prepare the interview; preserve the participant's consent.",
    )
    generic = WorkingStateCheckpoint(**values)
    assert generic.repository_id == ""
    assert generic.profile_id == "profile-a"
    assert generic.owner_generation == "owner-7"
    assert generic.requested_block_id == "model-chosen-leaf"
    assert asdict(generic)["working_view"].startswith("Prepare the interview")


def test_continuation_rejects_changed_tail_and_foreign_profile() -> None:
    checkpoint = _checkpoint()

    class Source:
        def get_block(self, ref):
            record = _raw_record(ref)
            return replace(record, content="changed tail") if ref == "tail-1" else record

    with pytest.raises(ValueError, match="exact continuation source hash"):
        compile_continuation_packet(checkpoint, backend=Source())
    with pytest.raises(ValueError, match="workspace"):
        compile_continuation_packet(replace(checkpoint, profile_id="foreign"), backend=Source())


def test_host_verified_checkpoint_stores_view_without_host_transcript(tmp_path) -> None:
    from um_mem.contracts.ledger import append_candidate_semantic_verification_ledger_event
    from um_mem.memory.working_continuity import checkpoint_verification_subject

    source_index = BackendSQLiteIndex(tmp_path / "host-history.sqlite", backend_id="test")
    target_index = BackendSQLiteIndex(
        tmp_path / ".UM-MEM" / "memory.sqlite", backend_id="test", workspace_id=_WORKSPACE
    )
    try:
        leaf = _seed(source_index)
        checkpoint = _checkpoint(requested_block_id=leaf)
        compile_continuation_packet(checkpoint, backend=source_index)
        report = append_candidate_semantic_verification_ledger_event(
            repo_root=tmp_path,
            subject=checkpoint_verification_subject(checkpoint),
            verifier_id="host-reviewer",
            verifier_model_version="reviewer",
            proposer_model_version=checkpoint.model_version,
            coverage=True,
            preservation=True,
            faithfulness=True,
            instruction_safety=True,
            reasons=("Compared exact native sources.",),
        )
        committed = commit_working_state_checkpoint(
            tmp_path, checkpoint, backend=target_index, verification_report=report
        )
        assert committed.saved
        assert len(target_index.export_blocks()) == 1
        assert "immutable host event" not in target_index.export_blocks()[0].content
        with pytest.raises(ValueError, match="verification.*bound"):
            commit_working_state_checkpoint(
                tmp_path,
                replace(checkpoint, working_view="tampered", exact_literal_manifest=()),
                backend=target_index,
                verification_report=report,
            )
    finally:
        source_index.close()
        target_index.close()
