import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import EvidenceRef, MemoryBlockRecord, MemoryEvent, RecordBlockMembership
from um_mem.contracts.ledger import (
    append_supervisor_ledger_event,
    append_verification_report_ledger_event,
)
from um_mem.memory import mcp_facade
from um_mem.memory.backends import mempalace as mempalace_backend
from um_mem.memory.backends.contracts import (
    CanonicalHeadReservation,
    MemoryEdgeRecord,
    MemoryUsageRecord,
)
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.blocks.catalog import compile_semantic_catalog
from um_mem.memory.blocks.registry import load_memory_blocks
from um_mem.memory.kernel.models import block_record_to_event
from um_mem.memory.kernel.telemetry import read_memory_events
from um_mem.memory.kernel.writer import _write_memory_event
from um_mem.memory.mutation import plan_memory_mutation


class _MutationBackend:
    def __init__(self, record: MemoryBlockRecord) -> None:
        self.backend_id = "capture"
        self.record = record
        self.transitions: list[dict[str, Any]] = []
        self.expire_calls = 0

    def get_block(self, block_id: str) -> MemoryBlockRecord | None:
        return self.record if block_id == self.record.block_id else None

    def reserve_canonical_head(
        self,
        **kwargs: Any,
    ) -> CanonicalHeadReservation:
        return CanonicalHeadReservation(
            commit_revision=1,
            workspace_id=str(kwargs["workspace_id"]),
            canonical_key_sha256="capture",
            predecessor_id=str(kwargs["predecessor_id"]),
            successor_id=str(kwargs["successor_id"]),
            successor_status=str(kwargs["successor_status"]),
            successor_authority=str(kwargs["successor_authority"]),
        )

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        return {
            "ok": True,
            "block_id": record.block_id,
            "backend": "capture",
        }

    def complete_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        mutation: Any,
    ) -> dict[str, Any]:
        self.transitions.append(
            {
                "block_id": reservation.successor_id,
                "status": mutation.record.status,
                "reason": mutation.record.metadata.get("status_reason", ""),
                "evidence_refs": list(mutation.record.evidence_refs),
                "retention": mutation.record.retention,
            }
        )
        return {"ok": True, "block_id": reservation.successor_id, "backend": "capture"}

    def expire_scope(self, *args: Any, **kwargs: Any) -> int:
        del args, kwargs
        self.expire_calls += 1
        raise AssertionError("governed erase must not call expire_scope")


def _record(
    workspace_id: str,
    evidence_refs: tuple[EvidenceRef, ...] = (),
    *,
    repo_root: Path | None = None,
) -> MemoryBlockRecord:
    if not evidence_refs:
        evidence_refs = (
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"record:{workspace_id}",
                produced_by="verifier",
                producer_id=f"verifier:{workspace_id}",
            ),
        )
    catalog = compile_semantic_catalog(
        load_memory_blocks(repo_root or Path.cwd(), workspace_id=workspace_id, strict=True)
    )
    record = MemoryBlockRecord(
        block_id="memory-1",
        content="SENSITIVE_MEMORY_CONTENT_MUST_NOT_REACH_JOURNAL",
        block_membership=RecordBlockMembership(
            primary_block_id="project_ideas",
            catalog_revision_id=catalog.revision_id,
        ),
        title="candidate memory",
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
        workspace_id=workspace_id,
        run_id="run-1",
        stage_id="execute",
        evidence_refs=evidence_refs,
    )
    planned = plan_memory_mutation(record, existing=None).record
    return replace(planned, block_id="memory-1")


def test_forget_rejects_foreign_workspace_before_backend_mutation(tmp_path) -> None:
    backend = _MutationBackend(_record("beta", repo_root=tmp_path))
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match="workspace"):
        client.forget("memory-1", reason="foreign workspace")

    assert backend.transitions == []


def test_forget_records_content_free_lifecycle_journal(tmp_path) -> None:
    backend = _MutationBackend(_record("alpha", repo_root=tmp_path))
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    result = client.forget("memory-1", reason="operator retraction")

    [event] = [
        row
        for row in read_memory_events(tmp_path)
        if row["event_type"] == "memory_status_transition"
    ]
    assert result["ok"] is True
    assert backend.transitions[0]["status"] == "retracted"
    assert event["workspace_id"] == "alpha"
    assert event["data"]["from_status"] == "accepted"
    assert event["data"]["to_status"] == "retracted"
    assert event["selected_id_hashes"]
    assert "SENSITIVE_MEMORY_CONTENT_MUST_NOT_REACH_JOURNAL" not in json.dumps(event)


def test_exact_forget_preserves_a_newer_native_revision_after_reopen(tmp_path) -> None:
    backend = MemPalaceBackend(tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
        saved = client.remember(
            "Initial working result",
            title="Versioned result",
            requested_block_id="workspace_episodes",
        )
        assert saved.saved is True
        initial = backend.get_block(saved.memory_id)
        assert initial is not None
        changed = client.amend(
            initial.block_id,
            expected_revision=initial.revision_id,
            content="Newer working result",
            reason="another completed operation",
        )
        assert changed["ok"] is True
        current_id = changed["block_id"]
    finally:
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id="alpha")
    try:
        client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=reopened)
        current = reopened.get_block(current_id)
        assert current is not None
        for record_id, revision in (
            (initial.block_id, initial.revision_id),
            (initial.block_id, current.revision_id),
            (current.block_id, initial.revision_id),
        ):
            result = client.forget(record_id, expected_revision=revision, reason="stale cleanup")
            assert result["ok"] is False
            assert result["error"] == "expected_revision_conflict"
        assert reopened.get_block(current_id) == current
        result = client.forget(
            current_id,
            expected_revision=current.revision_id,
            reason="exact cleanup",
        )
        assert result["saved"] is True
        retracted = reopened.get_block(result["block_id"])
        assert retracted is not None and retracted.status == "retracted"
        assert retracted.revision_of == current_id
    finally:
        reopened.close()


def test_forget_rebinds_unchanged_membership_after_catalog_growth(tmp_path) -> None:
    backend = _MutationBackend(_record("alpha", repo_root=tmp_path))
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    [root] = client.block_map()
    client.create_leaf_block(
        name="Later topology",
        synopsis="An unrelated block created after the record was written.",
        parent_id=root["block_id"],
        reason="advance the catalog revision",
    )

    result = client.forget("memory-1", reason="operator retraction")

    assert result["saved"] is True
    assert backend.transitions[0]["status"] == "retracted"


@pytest.mark.parametrize("state", ["prepared", "published", "foreign_publication", "newer"])
def test_promotion_compensation_recovers_only_its_exact_native_write(
    tmp_path, state, candidate_semantic_report
) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha", backend="mempalace")
    try:
        saved = client.remember(
            "A reviewed observation",
            title="Recoverable publication",
            requested_block_id="stable_project_facts",
        )
        assert saved.saved is True
        source = client.backend.get_block(saved.memory_id)
        assert source is not None
        current_id = source.block_id
        if state != "prepared":
            report = append_verification_report_ledger_event(
                repo_root=tmp_path,
                workspace_id="alpha",
                verifier_id="reviewer",
                report_hash="review-one",
                passed=True,
            )
            published = client.promote(
                source.block_id,
                expected_source_revision_id=source.revision_id,
                operation_id="foreign" if state == "foreign_publication" else "operation-one",
                evidence_refs=(report, candidate_semantic_report(client, source.block_id)),
                reason="reviewed publication",
            )
            assert published["ok"] is True
            current_id = published["block_id"]
        if state == "newer":
            later_report = append_verification_report_ledger_event(
                repo_root=tmp_path,
                workspace_id="alpha",
                verifier_id="later-reviewer",
                report_hash="review-two",
                passed=True,
            )
            later = client.promote(
                current_id,
                evidence_refs=(later_report, candidate_semantic_report(client, current_id)),
                expected_source_revision_id=client.backend.get_block(current_id).revision_id,
                reason="later independent review",
            )
            assert later["ok"] is True
            assert later["block_id"] != current_id
            current_id = later["block_id"]
    finally:
        client.backend.close()

    reopened = MemoryClient(tmp_path, workspace_id="alpha", backend="mempalace")
    try:
        current = reopened.backend.get_block(current_id)
        with pytest.raises(ValueError, match="revision"):
            reopened.compensate_promotion(
                source.block_id,
                expected_revision="wrong-revision",
                operation_id="operation-one",
                reason="recover interrupted host",
            )
        result = reopened.compensate_promotion(
            source.block_id,
            expected_revision=source.revision_id,
            operation_id="operation-one",
            reason="recover interrupted host",
        )
        if state in {"prepared", "published"}:
            assert result.outcome == "retracted"
            hidden = reopened.backend.get_block(result.record_id)
            assert hidden is not None and hidden.status == "retracted"
            replay = reopened.compensate_promotion(
                source.block_id,
                expected_revision=source.revision_id,
                operation_id="operation-one",
                reason="recover interrupted host",
            )
            assert replay.outcome == "already_retracted"
        else:
            assert result.outcome == "superseded"
            assert reopened.backend.get_block(current_id) == current
            assert [
                hit.text
                for hit in reopened.recall(
                    "reviewed observation",
                    logical_block_ids=["stable_project_facts"],
                )
            ] == ["A reviewed observation"]
    finally:
        reopened.backend.close()


def test_forget_preserves_existing_and_new_evidence(tmp_path) -> None:
    existing = EvidenceRef(
        ref_type="test_run",
        ref_id="verify-1",
        produced_by="verifier",
        producer_id="verifier:verify-1",
    )
    requested = EvidenceRef(
        ref_type="session_event",
        ref_id="um-mem-forget:operator",
        produced_by="harness",
        producer_id="harness:forget-request",
    )
    backend = _MutationBackend(_record("alpha", (existing,), repo_root=tmp_path))
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    client.forget("memory-1", reason="operator retraction", evidence_refs=(requested,))

    assert backend.transitions[0]["evidence_refs"] == [existing, requested]


def test_working_retraction_requires_exact_ownership_and_follows_its_amendment(tmp_path) -> None:
    client = MemoryClient(tmp_path, workspace_id="alpha", backend="mempalace")
    try:
        saved = client.remember(
            "A working observation",
            title="Owned working state",
            requested_block_id="workspace_episodes",
            metadata={"working_set_owner": {"owner_id": "owner-one", "generation_id": "gen-one"}},
        )
        source = client.backend.get_block(saved.memory_id)
        assert saved.saved and source is not None
        amended = client.amend(
            source.block_id,
            expected_revision=source.revision_id,
            content="A corrected working observation",
            reason="same owner rework",
        )
        assert amended["ok"] is True
        for owner, generation in (("owner-two", "gen-one"), ("owner-one", "gen-two")):
            with pytest.raises(ValueError, match="another owner or generation"):
                client.retract_working_record(
                    source.block_id,
                    owner_id=owner,
                    generation_id=generation,
                    reason="wrong owner",
                )
        result = client.retract_working_record(
            source.block_id,
            owner_id="owner-one",
            generation_id="gen-one",
            reason="stop work",
        )
        assert result.outcome == "retracted"
        record = client.backend.get_block(result.record_id)
        assert record is not None and record.revision_of == amended["block_id"]
        assert record.status == "retracted"
        assert (
            client.retract_working_record(
                source.block_id,
                owner_id="owner-one",
                generation_id="gen-one",
                reason="stop work",
            ).outcome
            == "already_retracted"
        )
    finally:
        client.backend.close()


def test_mempalace_does_not_publish_before_atomic_canonical_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    workspace_id = "alpha"
    content = "A native projection without canonical commit must stay invisible."
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)

    def sidecar_commit_failure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise OSError("test sidecar canonical commit failure")

    monkeypatch.setattr(backend._index, "complete_canonical_head", sidecar_commit_failure)
    try:
        result = MemoryClient(
            tmp_path,
            workspace_id=workspace_id,
            backend="mempalace",
            backend_instance=backend,
        ).remember(
            content,
            title="canonical normal write failure",
            requested_block_id="workspace_episodes",
        )
        block_id = result.memory_id
        recalled = backend.recall(
            {
                "query": "native projection canonical",
                "workspace_id": workspace_id,
                "scope_chain": [f"workspace:{workspace_id}"],
                "include_candidates": True,
                "search_all": True,
                "limit": 8,
            }
        )
    finally:
        backend.close()

    assert result.saved is False
    assert all(record.block_id != block_id for record in recalled)


def test_mempalace_rolls_back_normal_write_after_sidecar_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    workspace_id = "alpha"
    content = "An uncommitted selected-adapter write must roll back before restart."
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    original_commit = backend._index.complete_canonical_head
    attempts = 0

    def fail_first_sidecar_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("test sidecar commit failure")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(backend._index, "complete_canonical_head", fail_first_sidecar_commit)
    try:
        first = MemoryClient(
            tmp_path,
            workspace_id=workspace_id,
            backend="mempalace",
            backend_instance=backend,
        ).remember(
            content,
            title="pending normal write recovery",
            requested_block_id="workspace_episodes",
        )
        block_id = first.memory_id
        assert all(
            record.block_id != block_id
            for record in backend.recall(
                {
                    "query": "pending selected-adapter",
                    "workspace_id": workspace_id,
                    "scope_chain": [f"workspace:{workspace_id}"],
                    "include_candidates": True,
                    "search_all": True,
                    "limit": 8,
                }
            )
        )
    finally:
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        assert reopened.get_block(block_id) is None
        assert reopened._index.write_outbox_status(block_id) is None
        retried = MemoryClient(
            tmp_path, workspace_id=workspace_id, backend_instance=reopened
        ).remember(
            content,
            title="pending normal write recovery",
            requested_block_id="workspace_episodes",
        )
        assert retried.saved is True
        assert MemoryClient(
            tmp_path, workspace_id=workspace_id, backend_instance=reopened
        ).projection_barrier(retried.memory_id, timeout_seconds=5)["ok"]
        recovered = reopened.get_block(retried.memory_id)
        recalled = reopened.recall(
            {
                "query": "pending selected-adapter",
                "workspace_id": workspace_id,
                "include_candidates": True,
                "source_horizon": "durable",
                "logical_block_ids": ["workspace_episodes"],
                "record_ids": [retried.memory_id],
                "scope_chain": [f"workspace:{workspace_id}"],
                "limit": 8,
            }
        )
        with sqlite3.connect(reopened._index.path) as connection:
            outbox_rows = connection.execute(
                "SELECT status, error, commit_revision FROM write_outbox WHERE block_id=?",
                (retried.memory_id,),
            ).fetchall()
    finally:
        reopened.close()

    replayed = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        replayed_record = replayed.get_block(retried.memory_id)
        assert replayed_record is not None
        native = replayed._native.collection(
            "palace.idea",
            partition=mempalace_backend._record_partition_identity(replayed_record),
        ).get(ids=[retried.memory_id])
    finally:
        replayed.close()

    assert first.saved is False
    assert recovered is not None
    assert recovered.content == content
    assert "commit_revision" not in recovered.metadata
    assert [record.block_id for record in recalled] == [retried.memory_id]
    assert len(outbox_rows) == 1
    assert outbox_rows[0][:2] == ("committed", "")
    assert isinstance(outbox_rows[0][2], int)
    assert replayed_record == recovered
    assert native.ids == [retried.memory_id]


def test_mempalace_retries_transient_native_visibility_before_failing_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = MemPalaceBackend(
        tmp_path,
        workspace_id="alpha",
        reconcile_pending_writes=False,
    )
    record = _record("alpha", repo_root=tmp_path)
    entry = backend._index.open_write_outbox(record=record, store="palace.idea")
    backend._index.commit_write_outbox_record(entry)
    backend._native.add_record(record, commit_revision=entry.commit_revision)
    observed: list[MemoryBlockRecord | None] = [None, None, record]
    monkeypatch.setattr(
        backend,
        "_native_record_for_write_outbox",
        lambda _entry: observed.pop(0),
    )
    monkeypatch.setattr(mempalace_backend.time, "sleep", lambda _seconds: None)

    try:
        native_id = backend._project_write_outbox(entry, record)
        status = backend._index.write_outbox_status(record.block_id)
    finally:
        backend.close()

    assert native_id == record.block_id
    assert status == "committed"
    assert observed == []


def test_mempalace_recovers_pending_native_projection_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    workspace_id = "alpha"
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)

    def reject_native_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("test native projection failure")

    monkeypatch.setattr(backend._native, "add_record", reject_native_write)
    try:
        result = MemoryClient(
            tmp_path,
            workspace_id=workspace_id,
            backend="mempalace",
            backend_instance=backend,
        ).remember(
            "A failed native projection must not become recoverable memory.",
            title="native normal write failure",
            requested_block_id="workspace_episodes",
        )
        block_id = result.memory_id
    finally:
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        assert MemoryClient(
            tmp_path, workspace_id=workspace_id, backend_instance=reopened
        ).projection_barrier(block_id, timeout_seconds=5)["ok"]
        with sqlite3.connect(reopened._index.path) as connection:
            outbox_rows = connection.execute(
                "SELECT status, error FROM write_outbox WHERE block_id=?",
                (block_id,),
            ).fetchall()
        recovered = reopened.get_block(block_id)
        recalled = reopened.recall(
            {
                "query": "failed native projection",
                "workspace_id": workspace_id,
                "search_all": True,
                "limit": 8,
            }
        )
    finally:
        reopened.close()

    assert result.saved is True
    assert result.raw["projection_pending"] is True
    assert result.raw["native_written"] is False
    assert outbox_rows == [("committed", "")]
    assert recovered is not None
    assert recalled == []


def test_mempalace_recalls_a_single_record_partition_after_restart(tmp_path) -> None:
    workspace_id = "alpha"
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        stored = _record(workspace_id, repo_root=tmp_path)
        backend.retain_block(stored)
        assert stored.block_membership is not None
        logical_block_id = stored.block_membership.primary_block_id
    finally:
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        recalled = reopened.recall(
            {
                "query": "current GIL state",
                "workspace_id": workspace_id,
                "run_id": "run-1",
                "scope_chain": [f"workspace:{workspace_id}"],
                "logical_block_ids": [logical_block_id],
                "limit": 4,
            }
        )
    finally:
        reopened.close()

    assert [record.block_id for record in recalled] == [stored.block_id]


def test_mempalace_rejects_stale_successor_before_native_projection(tmp_path) -> None:
    workspace_id = "alpha"
    stale_id = "stale-mempalace-successor"
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        client = MemoryClient(
            tmp_path,
            workspace_id=workspace_id,
            backend="mempalace",
            backend_instance=backend,
        )
        root = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        committed = client.remember(
            "The package uses FastAPI.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        committed_block = backend.get_block(committed.memory_id)
        assert committed_block is not None

        stale = _write_memory_event(
            tmp_path,
            MemoryEvent(
                content="The package uses Django.",
                block_membership=committed_block.block_membership,
                title="Package framework",
                memory_kind="fact",
                status="candidate",
                trust_level="agent_reported",
                scope="workspace",
                delivery_policy="on_demand",
                workspace_id=workspace_id,
                revision_of=root.memory_id,
                supersedes_ids=(root.memory_id,),
                canonical_key=committed_block.canonical_key,
                revision_id="rev_stale_mempalace_successor",
                recorded_at="2026-07-14T00:00:00Z",
                block_id=stale_id,
            ),
            workspace_id=workspace_id,
            backend_instance=backend,
        )
        native = backend._native.collection(
            "palace.idea",
            partition=mempalace_backend._record_partition_identity(committed_block),
        ).get(ids=[stale_id])
        recalled = backend.recall(
            {
                "query": "package framework",
                "workspace_id": workspace_id,
                "include_candidates": True,
                "source_horizon": "durable",
                "logical_block_ids": ["stable_project_facts"],
                "record_ids": [committed.memory_id],
                "scope_chain": [f"workspace:{workspace_id}"],
                "limit": 8,
            }
        )
    finally:
        backend.close()

    assert stale.saved is False
    assert stale.error == "canonical_head_conflict"
    assert native.ids == []
    assert [record.block_id for record in recalled] == [committed.memory_id]


def test_mempalace_batch_reconciles_native_row_after_sidecar_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    workspace_id = "alpha"
    first = replace(
        _record(workspace_id, repo_root=tmp_path),
        block_id="batch-outbox-first",
        content="The first batch record remains committed.",
        title="first batch record",
    )
    second = replace(
        _record(workspace_id, repo_root=tmp_path),
        block_id="batch-outbox-second",
        content="The second batch record must recover from a sidecar failure.",
        title="second batch record",
    )
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    original_retain = backend._index.retain_block

    def fail_second_sidecar(record: MemoryBlockRecord) -> dict[str, Any]:
        if record.block_id == second.block_id:
            raise OSError("test batch sidecar failure")
        return original_retain(record)

    monkeypatch.setattr(backend._index, "retain_block", fail_second_sidecar)
    try:
        with pytest.raises(OSError, match="test batch sidecar failure"):
            backend.retain_blocks([first, second])
        first_before_restart = backend.get_block(first.block_id)
        second_before_restart = backend.get_block(second.block_id)
    finally:
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        deadline = time.monotonic() + 5
        while reopened.write_outbox_status(second.block_id) != "committed":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        first_after_restart = reopened.get_block(first.block_id)
        second_after_restart = reopened.get_block(second.block_id)
        native = reopened._native.collection(
            "palace.idea",
            partition=mempalace_backend._record_partition_identity(first),
        ).get(ids=[first.block_id, second.block_id])
        with sqlite3.connect(reopened._index.path) as connection:
            outbox_rows = connection.execute(
                """
                SELECT block_id, status, error, commit_revision
                FROM write_outbox
                WHERE block_id IN (?, ?)
                ORDER BY block_id
                """,
                (first.block_id, second.block_id),
            ).fetchall()
    finally:
        reopened.close()

    assert first_before_restart is not None
    assert second_before_restart is None
    assert first_after_restart is not None
    assert second_after_restart is not None
    assert set(native.ids) == {first.block_id, second.block_id}
    assert [(block_id, status, error) for block_id, status, error, _ in outbox_rows] == [
        (first.block_id, "committed", ""),
        (second.block_id, "committed", ""),
    ]
    assert len({revision for _, _, _, revision in outbox_rows}) == 2


def test_governed_erase_requires_supervisor_ledger_authorization_before_mutation(tmp_path) -> None:
    backend = _MutationBackend(_record("alpha", repo_root=tmp_path))
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    reason = "ERASE_REASON_MUST_NOT_REACH_JOURNAL"

    result = client.erase(
        "memory-1",
        request_id="erase-request-1",
        reason=reason,
        authorization=EvidenceRef(
            ref_type="ledger_event",
            ref_id="missing-supervisor-approval",
            produced_by="supervisor",
            producer_id="supervisor:missing-supervisor-approval",
        ),
    )

    [event] = [
        row for row in read_memory_events(tmp_path) if row["event_type"] == "memory_erase_rejected"
    ]
    assert result == {"ok": False, "error": "invalid_erase_authorization"}
    assert backend.transitions == []
    assert backend.expire_calls == 0
    assert event["error"] == "invalid_erase_authorization"
    assert event["selected_id_hashes"]
    serialized = json.dumps(event)
    assert "SENSITIVE_MEMORY_CONTENT_MUST_NOT_REACH_JOURNAL" not in serialized
    assert reason not in serialized
    assert "memory-1" not in serialized


def test_governed_erase_without_verified_native_purge_fails_closed(tmp_path) -> None:
    backend = _MutationBackend(_record("alpha", repo_root=tmp_path))
    backend.backend_id = "mempalace"
    backend.purge_exact_native_artifacts = lambda *_args, **_kwargs: {
        "ok": True,
        "native_exact_id_absent": True,
        "native_edges_absent": True,
    }
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        backend="mempalace",
        backend_instance=backend,
    )
    request_id = "erase-request-2"
    reason = "ERASE_REASON_MUST_NOT_REACH_UNAVAILABLE_EVENT"
    approval = append_supervisor_ledger_event(
        repo_root=tmp_path,
        workspace_id="alpha",
        actor="supervisor",
        stage_id="privacy",
        tool="authorize_memory_erase",
        result={
            "action": "erase_memory",
            "approved": True,
            "request_id": request_id,
            "memory_id_sha256": hashlib.sha256(b"memory-1").hexdigest(),
            "reason_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
        },
    )

    result = client.erase(
        "memory-1",
        request_id=request_id,
        reason=reason,
        authorization=EvidenceRef(
            ref_type="ledger_event",
            ref_id=approval.event_id,
            hash=approval.event_hash,
            produced_by="supervisor",
            producer_id=f"supervisor:{approval.event_id}",
        ),
    )

    [event] = [
        row for row in read_memory_events(tmp_path) if row["event_type"] == "memory_erase_rejected"
    ]
    assert result["ok"] is False
    assert result["error"] == "native_purge_unsupported"
    assert result["erase_request"]["status"] == "unavailable"
    assert "certificate" not in result
    assert backend.transitions == []
    assert backend.expire_calls == 0
    assert event["status"] == "unavailable"
    assert event["error"] == "native_purge_unsupported"
    serialized = json.dumps(event)
    assert "SENSITIVE_MEMORY_CONTENT_MUST_NOT_REACH_JOURNAL" not in serialized
    assert reason not in serialized
    assert "memory-1" not in serialized


def test_governed_erase_unavailable_request_is_restart_idempotent_and_rejects_conflict(
    tmp_path,
) -> None:
    backend = _MutationBackend(_record("alpha", repo_root=tmp_path))
    request_id = "erase-request-idempotent"
    reason = "ERASE_REASON_MUST_NOT_REACH_REQUEST_STATE"
    approval = append_supervisor_ledger_event(
        repo_root=tmp_path,
        workspace_id="alpha",
        actor="supervisor",
        stage_id="privacy",
        tool="authorize_memory_erase",
        result={
            "action": "erase_memory",
            "approved": True,
            "request_id": request_id,
            "memory_id_sha256": hashlib.sha256(b"memory-1").hexdigest(),
            "reason_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
        },
    )
    authorization = EvidenceRef(
        ref_type="ledger_event",
        ref_id=approval.event_id,
        hash=approval.event_hash,
        produced_by="supervisor",
        producer_id=f"supervisor:{approval.event_id}",
    )

    first = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend).erase(
        "memory-1",
        request_id=request_id,
        reason=reason,
        authorization=authorization,
    )
    replay = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend).erase(
        "memory-1",
        request_id=request_id,
        reason=reason,
        authorization=authorization,
    )

    assert first["ok"] is False
    assert first["error"] == "native_purge_unsupported"
    assert replay == first
    assert first["erase_request"]["status"] == "unavailable"
    assert (
        first["erase_request"]["request_id_sha256"]
        == hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    )
    assert "certificate" not in first
    serialized = json.dumps(first["erase_request"])
    for value in (request_id, reason, "memory-1", approval.event_id, approval.event_hash):
        assert value not in serialized

    changed_reason = "ERASE_REASON_WITH_DIFFERENT_FINGERPRINT"
    changed_approval = append_supervisor_ledger_event(
        repo_root=tmp_path,
        workspace_id="alpha",
        actor="supervisor",
        stage_id="privacy",
        tool="authorize_memory_erase",
        result={
            "action": "erase_memory",
            "approved": True,
            "request_id": request_id,
            "memory_id_sha256": hashlib.sha256(b"memory-1").hexdigest(),
            "reason_sha256": hashlib.sha256(changed_reason.encode("utf-8")).hexdigest(),
        },
    )
    conflict = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend).erase(
        "memory-1",
        request_id=request_id,
        reason=changed_reason,
        authorization=EvidenceRef(
            ref_type="ledger_event",
            ref_id=changed_approval.event_id,
            hash=changed_approval.event_hash,
            produced_by="supervisor",
            producer_id=f"supervisor:{changed_approval.event_id}",
        ),
    )

    assert conflict == {"ok": False, "error": "erase_request_conflict"}
    assert backend.transitions == []
    assert backend.expire_calls == 0


def test_governed_erase_real_mempalace_closes_exact_native_artifacts_after_reopen(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_id = "alpha"
    request_id = "erase-mempalace-request"
    reason = "ERASE_REASON_MUST_NOT_REACH_COMPLETION_CERTIFICATE"
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    queued, release_projection = threading.Event(), threading.Event()
    mutation = plan_memory_mutation(_record(workspace_id, repo_root=tmp_path), existing=None)
    record = mutation.record
    neighbor = replace(
        record,
        block_id="memory-neighbor",
        content="neighbor native evidence must remain available",
        title="neighbor memory",
    )

    def _forbidden_expire(*_args: Any, **_kwargs: Any) -> int:
        raise AssertionError("governed erase must not call expire_scope")

    try:
        seeded = _write_memory_event(
            tmp_path,
            block_record_to_event(record),
            workspace_id=workspace_id,
            backend_instance=backend,
            mutation=mutation,
        )
        assert seeded.saved is True
        assert MemoryClient(
            tmp_path, workspace_id=workspace_id, backend_instance=backend
        ).projection_barrier(record.block_id, timeout_seconds=10)["ok"]
        backend.retain_block(neighbor)
        backend.add_edge(
            MemoryEdgeRecord(
                src_id=record.block_id,
                dst_id=neighbor.block_id,
                edge_type="supports",
                workspace_id=workspace_id,
            )
        )
        backend.record_usage(
            MemoryUsageRecord(
                block_id=record.block_id,
                workspace_id=workspace_id,
                outcome="neutral",
            )
        )
        monkeypatch.setattr(backend, "expire_scope", _forbidden_expire)
        monkeypatch.setattr(backend._index, "expire_scope", _forbidden_expire)
        approval = append_supervisor_ledger_event(
            repo_root=tmp_path,
            workspace_id=workspace_id,
            actor="supervisor",
            stage_id="privacy",
            tool="authorize_memory_erase",
            result={
                "action": "erase_memory",
                "approved": True,
                "request_id": request_id,
                "memory_id_sha256": hashlib.sha256(record.block_id.encode("utf-8")).hexdigest(),
                "reason_sha256": hashlib.sha256(reason.encode("utf-8")).hexdigest(),
            },
        )
        authorization = EvidenceRef(
            ref_type="ledger_event",
            ref_id=approval.event_id,
            hash=approval.event_hash,
            produced_by="supervisor",
            producer_id=f"supervisor:{approval.event_id}",
        )

        original_project = backend._index._project_partition
        original_enqueue = backend.enqueue_native_projection

        def delayed_project(key, first):
            queued.set()
            assert release_projection.wait(10)
            return original_project(key, first)

        def enqueue_retraction(record):
            result = original_enqueue(record)
            assert queued.wait(5)
            return result

        monkeypatch.setattr(backend._index, "_project_partition", delayed_project)
        monkeypatch.setattr(backend, "enqueue_native_projection", enqueue_retraction)

        result = MemoryClient(
            tmp_path,
            workspace_id=workspace_id,
            backend="mempalace",
            backend_instance=backend,
        ).erase(
            record.block_id,
            request_id=request_id,
            reason=reason,
            authorization=authorization,
        )

        assert result["ok"] is True
        assert not backend._index.pending_write_outbox()
        release_projection.set()
        backend._index.stop_native_projection()
        retracted_id = str(result["block_id"])
        assert retracted_id != record.block_id
        assert result["erase_request"]["status"] == "completed"
        assert result["certificate"]["backend"] == "mempalace"
        assert result["certificate"]["closure"] == {
            "native_exact_id_absent": True,
            "native_edges_absent": True,
            "raw_evidence_retained": True,
            "sidecar_status": "retracted",
        }
        serialized = json.dumps(result)
        [completion_event] = [
            event
            for event in read_memory_events(tmp_path)
            if event["event_type"] == "memory_erase_completed"
        ]
        for value in (
            request_id,
            reason,
            record.block_id,
            record.content,
            approval.event_id,
            approval.event_hash,
        ):
            assert value not in serialized
            assert value not in json.dumps(completion_event)
        original = backend.get_block(record.block_id)
        retained = backend.get_block(retracted_id)
        assert original is not None
        assert original.content == record.content
        assert original.status == "accepted"
        assert retained is not None
        assert retained.content == record.content
        assert retained.status == "retracted"
        assert retained.retention == "archive"
        assert retained.delivery_policy == "hidden"
        assert backend.edges_for(record.block_id)
        assert {
            row.block_id
            for row in backend.canonical_history(
                canonical_key=record.canonical_key,
                workspace_id=workspace_id,
            )
        } == {record.block_id, retracted_id}
        assert backend.value_stats(record.block_id)["uses_count"] == 1
        native = backend._native.collection(
            "palace.idea",
            partition=mempalace_backend._record_partition_identity(record),
        ).get(ids=[record.block_id, retracted_id, neighbor.block_id])
        assert record.block_id not in native.ids
        assert retracted_id not in native.ids
        assert neighbor.block_id in native.ids
    finally:
        release_projection.set()
        backend.close()

    reopened = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    try:
        original = reopened.get_block(record.block_id)
        retained = reopened.get_block(retracted_id)
        assert original is not None
        assert original.content == record.content
        assert original.status == "accepted"
        assert retained is not None
        assert retained.content == record.content
        assert retained.status == "retracted"
        assert retained.retention == "archive"
        assert retained.delivery_policy == "hidden"
        assert reopened.edges_for(record.block_id)
        assert {
            row.block_id
            for row in reopened.canonical_history(
                canonical_key=record.canonical_key,
                workspace_id=workspace_id,
            )
        } == {record.block_id, retracted_id}
        assert reopened.value_stats(record.block_id)["uses_count"] == 1
        native = reopened._native.collection(
            "palace.idea",
            partition=mempalace_backend._record_partition_identity(record),
        ).get(ids=[record.block_id, retracted_id, neighbor.block_id])
        assert record.block_id not in native.ids
        assert retracted_id not in native.ids
        assert neighbor.block_id in native.ids
        replay = MemoryClient(
            tmp_path,
            workspace_id=workspace_id,
            backend="mempalace",
            backend_instance=reopened,
        ).erase(
            record.block_id,
            request_id=request_id,
            reason=reason,
            authorization=authorization,
        )
    finally:
        reopened.close()

    assert replay == result


def test_mempalace_recovery_preflight_requires_native_and_sidecar_components(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_id = "alpha"
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    record = _record(workspace_id, repo_root=tmp_path)
    neighbor = replace(
        record,
        block_id="memory-neighbor",
        content="neighbor native evidence must remain available",
        title="neighbor memory",
    )
    try:
        backend.retain_block(record)
        backend.retain_block(neighbor)
        backend.add_edge(
            MemoryEdgeRecord(
                src_id=record.block_id,
                dst_id=neighbor.block_id,
                edge_type="supports",
                workspace_id=workspace_id,
            )
        )
        backend.record_usage(
            MemoryUsageRecord(
                block_id=record.block_id,
                workspace_id=workspace_id,
                outcome="neutral",
            )
        )
        partition = mempalace_backend._record_partition_identity(record)
        native_before = backend._native.collection(
            "palace.idea",
            partition=partition,
        ).get(ids=[record.block_id])
        sidecar_before = backend.get_block(record.block_id)
        edges_before = backend.edges_for(record.block_id)

        manifest = backend.recovery_preflight()

        assert manifest["ok"] is True
        assert manifest["manifest_complete"] is True
        assert manifest["preflight_only"] is True
        assert manifest["backup_restore_performed"] is False
        assert manifest["snapshot_available"] is False
        assert manifest["restore_available"] is False
        assert manifest["backend"] == "mempalace"
        assert set(manifest["components"]) == {
            "native_collections",
            "sidecar_index",
        }
        assert all(component["ok"] for component in manifest["components"].values())
        assert set(manifest["components"]["native_collections"]["available"])
        assert (
            backend._native.collection(
                "palace.idea",
                partition=partition,
            )
            .get(ids=[record.block_id])
            .ids
            == native_before.ids
        )
        assert backend.get_block(record.block_id) == sidecar_before
        assert backend.edges_for(record.block_id) == edges_before
        serialized = json.dumps(manifest)
        for value in (str(tmp_path), record.block_id, record.content):
            assert value not in serialized

        monkeypatch.setattr(
            backend._native,
            "health",
            lambda _partitions=(): {"ok": False},
        )
        unavailable = backend.recovery_preflight()

        assert unavailable["ok"] is False
        assert unavailable["manifest_complete"] is False
        assert unavailable["snapshot_available"] is False
        assert unavailable["restore_available"] is False
        assert unavailable["components"]["native_collections"]["ok"] is False
        assert unavailable["components"]["sidecar_index"]["ok"] is True
    finally:
        backend.close()


def test_mempalace_snapshot_gate_restores_a_real_native_cut(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_id = "alpha"
    backend = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    record = _record(workspace_id, repo_root=tmp_path)
    neighbor = replace(
        record,
        block_id="memory-neighbor",
        content="neighbor native evidence must remain available",
        title="neighbor memory",
    )
    later = replace(
        record,
        block_id="memory-after-snapshot",
        content="must not appear in the snapshot",
        title="later memory",
    )
    snapshot_path = tmp_path / "snapshot"
    restored: MemPalaceBackend | None = None
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    snapshot_result: dict[str, object] = {}
    snapshot_errors: list[Exception] = []
    write_done = threading.Event()

    try:
        forbidden_target = (
            tmp_path / "workspaces" / workspace_id / ".memory" / "palace" / "snapshot"
        )
        assert backend.create_native_snapshot(forbidden_target) == {
            "ok": False,
            "error": "snapshot_target_not_authorized",
        }
        backend.retain_block(record)
        backend.retain_block(neighbor)
        backend.add_edge(
            MemoryEdgeRecord(
                src_id=record.block_id,
                dst_id=neighbor.block_id,
                edge_type="supports",
                workspace_id=workspace_id,
            )
        )
        backend.record_usage(
            MemoryUsageRecord(
                block_id=record.block_id,
                workspace_id=workspace_id,
                outcome="neutral",
            )
        )
        original_capture = backend._write_native_snapshot

        def _paused_capture(stage_path):
            snapshot_started.set()
            assert release_snapshot.wait(timeout=5)
            return original_capture(stage_path)

        monkeypatch.setattr(backend, "_write_native_snapshot", _paused_capture)

        def _snapshot() -> None:
            try:
                snapshot_result.update(backend.create_native_snapshot(snapshot_path))
            except Exception as exc:
                snapshot_errors.append(exc)

        def _write_after_snapshot() -> None:
            backend.retain_block(later)
            write_done.set()

        snapshot_thread = threading.Thread(target=_snapshot)
        snapshot_thread.start()
        assert snapshot_started.wait(timeout=5)

        writer_thread = threading.Thread(target=_write_after_snapshot)
        writer_thread.start()
        assert not write_done.wait(timeout=0.2)

        release_snapshot.set()
        snapshot_thread.join(timeout=10)
        writer_thread.join(timeout=10)

        assert snapshot_errors == []
        assert snapshot_result["ok"] is True
        assert write_done.is_set()
        assert backend.get_block(later.block_id) is not None
        assert not any(path.name == "chroma.sqlite3" for path in snapshot_path.rglob("*"))

        manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["backend"] == "mempalace"
        assert manifest["workspace_id"] == workspace_id
        assert manifest["logical_native_export"] is True
        assert str(tmp_path) not in json.dumps(manifest)

        restored = MemPalaceBackend.restore_native_snapshot(
            snapshot_path,
            repo_root=tmp_path / "restored",
            workspace_id=workspace_id,
        )

        restored_record = restored.get_block(record.block_id)
        assert restored_record is not None
        assert restored_record.content == record.content
        assert restored.get_block(later.block_id) is None
        assert restored.edges_for(record.block_id)
        assert restored.value_stats(record.block_id)["uses_count"] == 1
        native = restored._native.collection(
            "palace.idea",
            partition=mempalace_backend._record_partition_identity(record),
        ).get(ids=[record.block_id, later.block_id])
        assert record.block_id in native.ids
        assert later.block_id not in native.ids
        assert restored.edges_for(record.block_id)
        with pytest.raises(ValueError, match="native_snapshot_restore_target_not_fresh"):
            MemPalaceBackend.restore_native_snapshot(
                snapshot_path,
                repo_root=tmp_path / "restored",
                workspace_id=workspace_id,
            )
    finally:
        release_snapshot.set()
        if restored is not None:
            restored.close()
        backend.close()


def test_mempalace_restore_rejects_malformed_logical_export_without_target_residue(
    tmp_path,
) -> None:
    workspace_id = "alpha"
    source = MemPalaceBackend(tmp_path, workspace_id=workspace_id)
    record = _record(workspace_id, repo_root=tmp_path)
    snapshot_path = tmp_path / "snapshot"
    restored_root = tmp_path / "restored"
    restored: MemPalaceBackend | None = None

    try:
        source.retain_block(record)
        assert source.create_native_snapshot(snapshot_path)["ok"] is True

        collections_path = snapshot_path / "native_collections.jsonl"
        original_collections = collections_path.read_text(encoding="utf-8")
        collections_path.write_text("{\n", encoding="utf-8")
        manifest_path = snapshot_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for component in manifest["components"]:
            if component["file"] == "native_collections.jsonl":
                component["sha256"] = hashlib.sha256(collections_path.read_bytes()).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="native_snapshot_collection_row_invalid"):
            MemPalaceBackend.restore_native_snapshot(
                snapshot_path,
                repo_root=restored_root,
                workspace_id=workspace_id,
            )

        assert not mempalace_backend.palace_path_for(restored_root, workspace_id).exists()
        restored_index = mempalace_backend._index_path(restored_root, workspace_id)
        assert not restored_index.exists()
        assert not restored_index.with_name(f"{restored_index.name}-wal").exists()
        assert not restored_index.with_name(f"{restored_index.name}-shm").exists()

        collections_path.write_text(original_collections, encoding="utf-8")
        for component in manifest["components"]:
            if component["file"] == "native_collections.jsonl":
                component["sha256"] = hashlib.sha256(collections_path.read_bytes()).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        restored = MemPalaceBackend.restore_native_snapshot(
            snapshot_path,
            repo_root=restored_root,
            workspace_id=workspace_id,
        )
        assert restored.get_block(record.block_id) is not None
    finally:
        if restored is not None:
            restored.close()
        source.close()


def test_mcp_forget_delegates_to_memory_client(monkeypatch) -> None:
    received: dict[str, Any] = {}

    class _Client:
        workspace_id = "alpha"

        def forget(self, memory_id: str, **kwargs: Any) -> dict[str, Any]:
            received["memory_id"] = memory_id
            received.update(kwargs)
            return {"ok": True, "block_id": memory_id, "backend": "capture"}

    client = _Client()
    monkeypatch.setattr(mcp_facade, "_client", lambda **_kwargs: client)
    monkeypatch.setattr(mcp_facade, "_close_client", lambda _client: None)

    payload = mcp_facade.um_mem_forget(
        memory_id="memory-1",
        reason="operator retraction",
        workspace_id="alpha",
    )

    assert payload["ok"] is True
    assert received["memory_id"] == "memory-1"
    evidence = received["evidence_refs"][0]
    assert (evidence.ref_type, evidence.ref_id, evidence.produced_by, evidence.producer_id) == (
        "session_event",
        "um-mem-forget:operator retraction",
        "harness",
        "mcp:um_mem_forget",
    )
