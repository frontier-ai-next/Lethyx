import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from um_mem import MemoryClient, api
from um_mem.contracts import EvidenceRef, MemoryBlockRecord, RecordBlockMembership
from um_mem.memory import cli
from um_mem.memory.backends.base import ReconcileReport
from um_mem.memory.backends.contracts import (
    block_from_json,
    block_record_to_memory_hit,
    block_to_json,
    memory_event_to_block_record,
)
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.backends.mnemosyne import MnemosyneBackend
from um_mem.memory.backends.query_policy import NATIVE_PROJECTION_REBUILD_REQUIRED
from um_mem.memory.hindsight.config import HindsightConfig
from um_mem.memory.kernel.models import block_record_to_event, normalize_memory_event
from um_mem.memory.mcp_facade import _dedup_records
from um_mem.memory.mutation import plan_lifecycle_mutation, plan_memory_mutation
from um_mem.memory.recall_plan import RecallPlan


def test_recall_keeps_independent_and_opposing_records() -> None:
    positive = plan_memory_mutation(
        replace(_record("positive"), content="delta = +5"), existing=None
    ).record
    negative = plan_memory_mutation(
        replace(_record("negative"), content="delta = -5"), existing=None
    ).record
    independent = plan_memory_mutation(
        replace(_record("independent"), content="delta = +5"), existing=None
    ).record
    assert positive.duplicate_cluster_id != negative.duplicate_cluster_id
    assert _dedup_records([positive, negative, independent, positive]) == [
        positive,
        negative,
        independent,
    ]


def _record(block_id: str, *, expires_at: str = "") -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        revision_id=f"revision:{block_id}",
        content=f"content for {block_id}",
        block_membership=RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
        workspace_id="workspace",
        run_id="run",
        scope="workspace",
        expires_at=expires_at,
    )


def _accepted_record(block_id: str) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=block_id,
        revision_id=f"revision:{block_id}",
        title=block_id,
        content="canonical live root",
        block_membership=RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
        canonical_key="workspace:observation:order17",
        kind="observation",
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
        workspace_id="workspace",
        run_id="run",
        evidence_refs=(
            EvidenceRef(
                ref_type="test_run",
                ref_id=f"order17:{block_id}",
                produced_by="verifier",
                producer_id="order17-test",
            ),
        ),
    )


def _canonical_native_metadata(record: MemoryBlockRecord, **extra: Any) -> dict[str, Any]:
    return {
        "block_id": record.block_id,
        "workspace_id": record.workspace_id,
        "canonical_record_json": json.dumps(
            block_to_json(record), ensure_ascii=False, sort_keys=True
        ),
        **extra,
    }


class _StaleNative:
    def __init__(self, record: MemoryBlockRecord) -> None:
        self.calls = 0
        self._row = {
            "id": record.block_id,
            "content": record.content,
            "canonical_record_json": _canonical_native_metadata(record)["canonical_record_json"],
        }

    def rows(self, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        return [dict(self._row)]

    def release(self) -> None:
        return None

    def close(self) -> None:
        return None


class _StaleHindsight:
    def __init__(self, record: MemoryBlockRecord) -> None:
        self.calls = 0
        self._row = {
            "id": record.block_id,
            "text": record.content,
            "score": 0.9,
            "metadata": _canonical_native_metadata(record),
        }

    def recall(self, **_kwargs: Any) -> list[dict[str, Any]]:
        self.calls += 1
        return [dict(self._row)]


class _StaleMnemosyne:
    def __init__(self, record: MemoryBlockRecord, commit_revision: int) -> None:
        self.calls = 0
        self._row = {
            "id": record.block_id,
            "content": MnemosyneBackend._native_expected_content(record, commit_revision),
            "source": f"um-mem:{record.workspace_id}:{record.block_id}",
            "metadata": _canonical_native_metadata(
                record,
                commit_revision=str(commit_revision),
                source_identity=MnemosyneBackend._native_source_identity(record),
            ),
        }

    def remember(self, *_args: Any, **_kwargs: Any) -> str:
        return "unused"

    def get_all_memories(self) -> list[dict[str, Any]]:
        return [dict(self._row)]

    def recall(self, _query: str, *, top_k: int, source: str = "") -> list[dict[str, Any]]:
        self.calls += 1
        if source and self._row["source"] != source:
            return []
        return [{"id": self._row["id"], "score": 0.9}][:top_k]

    def get(self, native_id: str) -> dict[str, Any] | None:
        return dict(self._row) if native_id == self._row["id"] else None


def _t12_backend(name: str, tmp_path: Path, stale_root: MemoryBlockRecord) -> tuple[Any, Any]:
    if name == "mempalace":
        backend = MemPalaceBackend(
            repo_root=tmp_path,
            workspace_id="workspace",
            reconcile_pending_writes=False,
        )
        native = _StaleNative(stale_root)
        backend._native = native
        return backend, native
    if name == "hindsight":
        native = _StaleHindsight(stale_root)
        return (
            HindsightBackend(
                repo_root=tmp_path,
                workspace_id="workspace",
                config=HindsightConfig(enabled=True),
                client=native,
                reconcile_pending_writes=False,
            ),
            native,
        )
    backend = MnemosyneBackend(
        repo_root=tmp_path,
        workspace_id="workspace",
        enabled=True,
        reconcile_pending_writes=False,
    )
    backend.available = True
    native = _StaleMnemosyne(stale_root, 0)
    backend._native_memory = native
    return backend, native


def _stage_committed(backend: Any, record: MemoryBlockRecord, store: str) -> None:
    entry = backend._index.open_write_outbox(record=record, store=store)
    backend._index.retain_block(record)
    backend._index.set_write_outbox_native_ref(entry.commit_revision, record.block_id)
    backend._index.complete_write_outbox(entry.commit_revision)


def _backend_factories() -> tuple[tuple[str, object], ...]:
    return (
        (
            "mempalace",
            lambda root: MemPalaceBackend(
                repo_root=root,
                workspace_id="workspace",
                reconcile_pending_writes=False,
            ),
        ),
        (
            "hindsight",
            lambda root: HindsightBackend(
                repo_root=root,
                workspace_id="workspace",
                config=HindsightConfig(enabled=False),
                reconcile_pending_writes=False,
            ),
        ),
        (
            "mnemosyne",
            lambda root: MnemosyneBackend(
                repo_root=root,
                workspace_id="workspace",
                enabled=False,
                reconcile_pending_writes=False,
            ),
        ),
    )


@pytest.mark.parametrize("backend_name", ("mempalace", "hindsight", "mnemosyne"))
def test_t12_cap_scope_ttl_preserve_current_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
) -> None:
    root = _accepted_record("root")
    backend, native = _t12_backend(backend_name, tmp_path, root)
    try:
        _stage_committed(backend, root, backend_name)
        expired_decoy = _record(
            "expired-decoy",
            expires_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        )
        _stage_committed(backend, expired_decoy, backend_name)
        if backend_name == "mnemosyne":
            root_revision = backend._index._conn.execute(
                "SELECT commit_revision FROM write_outbox WHERE block_id=?",
                (root.block_id,),
            ).fetchone()[0]
            native = _StaleMnemosyne(root, int(root_revision))
            backend._native_memory = native
        mutation = plan_lifecycle_mutation(
            root,
            action="retract",
            status="retracted",
            reason="order17 tombstone",
            retention="archive",
            delivery_policy="hidden",
        )
        tombstone = mutation.record
        entry = backend._index.open_write_outbox(record=tombstone, store=backend_name)
        reservation = backend._index.reserve_canonical_head(
            workspace_id="workspace",
            canonical_key=tombstone.canonical_key,
            predecessor_id=root.block_id,
            successor_id=tombstone.block_id,
            successor_status=tombstone.status,
            successor_authority=tombstone.authority,
            mutation_action=mutation.action,
            mutation=mutation,
        )
        backend._index.retain_block(tombstone)
        backend._index.complete_canonical_head(reservation, mutation)
        backend._index.complete_write_outbox(entry.commit_revision)
        monkeypatch.setenv("UM_MEM_BACKEND_INDEX_MAX_ROWS", "1")

        assert backend._index._evict_over_cap() == 0
        assert backend._index.expire_scope("workspace", "workspace", run_id="run") == 0
        assert backend._index.get_block("root", workspace_id="workspace") is not None
        assert backend._index.get_block(tombstone.block_id, workspace_id="workspace") is not None
        assert backend._index.get_block("expired-decoy", workspace_id="workspace") is not None
        assert (
            backend._index._conn.execute(
                "SELECT block_id FROM canonical_heads WHERE workspace_id=?",
                ("workspace",),
            ).fetchone()[0]
            == tombstone.block_id
        )
        query = {
            "query": "stale native fact",
            "workspace_id": "workspace",
            "run_id": "run",
            "scope_chain": ["workspace:workspace"],
            "logical_block_ids": ["workspace_episodes"],
        }
        if backend_name in {"mempalace", "mnemosyne"}:
            assert backend.recall(query) == []
            assert (
                MemoryClient(
                    tmp_path,
                    workspace_id="workspace",
                    backend_instance=backend,
                ).recall("stale native fact", search_all=True, run_id="run")
                == []
            )
            if backend_name == "mempalace":
                # The canonical tombstone rules out the stale partition before
                # native access; retained history must not cause repair work.
                assert native.calls == 0
            else:
                assert native.calls >= 1
        else:
            with pytest.raises(
                (RuntimeError, ValueError), match=NATIVE_PROJECTION_REBUILD_REQUIRED
            ):
                backend.recall(query)
            with pytest.raises(
                (RuntimeError, ValueError), match=NATIVE_PROJECTION_REBUILD_REQUIRED
            ):
                MemoryClient(
                    tmp_path,
                    workspace_id="workspace",
                    backend_instance=backend,
                ).recall("stale native fact", search_all=True, run_id="run")
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name, factory", _backend_factories())
def test_t13_reconcile_dry_run_does_not_replay_pending_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
    factory: object,
) -> None:
    backend = factory(tmp_path)  # type: ignore[operator]
    try:
        entry = backend._index.open_write_outbox(record=_record("pending"), store=backend_name)
        assert entry.status == "pending"
    finally:
        backend.close()

    reopened = factory(tmp_path)  # type: ignore[operator]
    try:
        native_calls: list[str] = []
        monkeypatch.setattr(
            reopened,
            "_project_write_outbox",
            lambda entry, record: native_calls.append(record.block_id),
        )
        before = reopened._index._conn.execute(
            "SELECT status, error FROM write_outbox WHERE block_id=?",
            ("pending",),
        ).fetchone()
        report = reopened.reconcile(dry_run=True)
        after = reopened._index._conn.execute(
            "SELECT status, error FROM write_outbox WHERE block_id=?",
            ("pending",),
        ).fetchone()
        assert report.dry_run is True
        assert report.pending_write_outbox == 1
        assert report.repaired == 0
        assert report.reason_code == "pending_work"
        assert len(reopened._index.pending_write_outbox()) == 1
        assert before == after == ("pending", "")
        assert native_calls == []
    finally:
        reopened.close()


@pytest.mark.parametrize("backend_name, factory", _backend_factories())
def test_t13_reconcile_apply_is_idempotent_when_clean(
    tmp_path: Path,
    backend_name: str,
    factory: object,
) -> None:
    del backend_name
    backend = factory(tmp_path)  # type: ignore[operator]
    try:
        first = backend.reconcile(dry_run=False)
        second = backend.reconcile(dry_run=False)
        assert first.result_code == second.result_code == "clean"
        assert first.failures == second.failures == 0
        assert first.remaining == second.remaining == 0
        assert first.repaired == second.repaired == 0
    finally:
        backend.close()


def test_empty_backend_open_does_not_acquire_global_operation_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_gate(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("empty startup must not acquire the global operation gate")

    monkeypatch.setattr(MemPalaceBackend, "operation_gate", unexpected_gate)
    backend = MemPalaceBackend(
        repo_root=tmp_path,
        workspace_id="workspace",
        reconcile_pending_writes=True,
    )
    monkeypatch.undo()
    backend.close()


def test_t13_apply_reports_remaining_pending_rows_as_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = MemPalaceBackend(
        repo_root=tmp_path,
        workspace_id="workspace",
        reconcile_pending_writes=False,
    )
    try:
        backend._index.open_write_outbox(record=_record("pending"), store="mempalace")
        monkeypatch.setattr(backend, "_reconcile_pending_write_outbox_locked", lambda: None)
        report = backend.reconcile(dry_run=False)
        assert report.remaining == 1
        assert report.failures == 1
        assert report.result_code == "failed"
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", ("mempalace", "hindsight", "mnemosyne"))
def test_t13_reconcile_apply_dispatches_pending_write_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_name: str
) -> None:
    record = _accepted_record(f"pending-{backend_name}")
    backend, _native = _t12_backend(backend_name, tmp_path, record)
    native_calls: list[int] = []
    try:
        entry = backend._index.open_write_outbox(record=record, store=backend_name)

        def project(replay_entry: Any, projected: MemoryBlockRecord, **_kwargs: Any) -> str:
            native_calls.append(replay_entry.commit_revision)
            backend._index.complete_write_outbox(replay_entry.commit_revision)
            return projected.block_id

        monkeypatch.setattr(backend, "_project_write_outbox", project)
        first = backend.reconcile(dry_run=False)
        second = backend.reconcile(dry_run=False)
        assert first.repaired == 1
        assert first.remaining == first.failures == 0
        assert second.repaired == second.remaining == second.failures == 0
        assert native_calls == [entry.commit_revision]
    finally:
        backend.close()


def test_t13_reconcile_marks_conflicting_head_failed_and_not_repaired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = MemPalaceBackend(
        repo_root=tmp_path,
        workspace_id="workspace",
        reconcile_pending_writes=False,
    )
    try:
        root = _accepted_record("root")
        _stage_committed(backend, root, "mempalace")
        mutation = plan_lifecycle_mutation(
            root,
            action="retract",
            status="retracted",
            reason="conflict regression",
            retention="archive",
            delivery_policy="hidden",
        )
        entry = backend._index.open_write_outbox(record=mutation.record, store="mempalace")
        backend._index.reserve_canonical_head(
            workspace_id="workspace",
            canonical_key=mutation.record.canonical_key,
            predecessor_id=root.block_id,
            successor_id=mutation.record.block_id,
            successor_status=mutation.record.status,
            successor_authority=mutation.record.authority,
            mutation_action=mutation.action,
            mutation=mutation,
        )
        backend._index.retain_block(mutation.record)
        backend._index._conn.execute(
            "UPDATE canonical_heads SET block_id='foreign-head' WHERE workspace_id=?",
            ("workspace",),
        )
        backend._index._conn.commit()
        monkeypatch.setattr(backend, "_reconcile_pending_write_outbox_locked", lambda: None)
        report = backend.reconcile(dry_run=False)
        assert entry.status == "pending"
        assert report.repaired == 0
        assert report.remaining == report.failures == 2
        assert report.result_code == "failed"
        assert backend._index.canonical_head_status(mutation.record.block_id) == "failed"
        audit = backend.reconcile(dry_run=True)
        assert audit.result_code == "failed"
        assert audit.pending_canonical_heads == 0
        assert audit.pending_write_outbox == 0
        assert audit.failures == 2
        assert audit.remaining == 2
    finally:
        backend.close()


def test_t13_cli_defaults_to_dry_run_and_writes_content_free_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("UM_MEM_BACKEND", raising=False)
    monkeypatch.delenv("UM_MEM_MEMORY_BACKEND", raising=False)
    assert (
        cli.main(
            [
                "reconcile",
                "--backend",
                "mempalace",
                "--repo-root",
                str(tmp_path),
                "--workspace",
                "workspace",
            ]
        )
        == 0
    )
    report_path = (
        tmp_path
        / "workspaces"
        / "workspace"
        / ".memory"
        / "drive"
        / "reports"
        / "memory_reconcile_report_latest.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert set(payload) == {
        "backend",
        "dry_run",
        "pending_canonical_heads",
        "pending_write_outbox",
        "repaired",
        "remaining",
        "failures",
        "result_code",
        "reason_code",
        "timestamp",
    }


def test_t13_cli_apply_returns_nonzero_when_failures_remain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailedBackend:
        def reconcile(self, *, dry_run: bool = True) -> ReconcileReport:
            assert dry_run is False
            return ReconcileReport(
                backend="mempalace",
                dry_run=False,
                pending_canonical_heads=0,
                pending_write_outbox=1,
                repaired=0,
                remaining=1,
                failures=1,
                result_code="failed",
                reason_code="reconcile_failures",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_durable_backend", lambda *args, **kwargs: FailedBackend())
    assert (
        cli.main(
            [
                "reconcile",
                "--apply",
                "--backend",
                "mempalace",
                "--repo-root",
                str(tmp_path),
                "--workspace",
                "workspace",
            ]
        )
        == 1
    )


def test_t14_clusters_preserve_exact_content_and_workspace_identity(tmp_path: Path) -> None:
    first = replace(
        _accepted_record("first"),
        content="Same normalized semantic content.",
        canonical_key="workspace:observation:first",
    )
    second = replace(
        _accepted_record("second"),
        content="same   normalized semantic content",
        canonical_key="workspace:observation:second",
    )
    other_workspace = replace(first, workspace_id="Alpha")
    other_case_workspace = replace(first, workspace_id="alpha")
    first_decision = plan_memory_mutation(first, existing=None)
    second_decision = plan_memory_mutation(second, existing=None)
    other_decision = plan_memory_mutation(other_workspace, existing=None)
    other_case_decision = plan_memory_mutation(other_case_workspace, existing=None)
    assert first_decision.record.duplicate_cluster_id != second_decision.record.duplicate_cluster_id
    assert first_decision.record.canonical_key != second_decision.record.canonical_key
    assert first_decision.record.block_id != second_decision.record.block_id
    assert (
        other_decision.record.duplicate_cluster_id
        != other_case_decision.record.duplicate_cluster_id
    )
    backend = MemPalaceBackend(
        tmp_path,
        workspace_id="workspace",
        reconcile_pending_writes=False,
    )
    try:
        for decision in (first_decision, second_decision):
            entry = backend._index.open_write_outbox(
                record=decision.record,
                store="mempalace",
            )
            reservation = backend._index.reserve_canonical_head(
                workspace_id="workspace",
                canonical_key=decision.record.canonical_key,
                predecessor_id="",
                successor_id=decision.record.block_id,
                successor_status=decision.record.status,
                successor_authority=decision.record.authority,
                mutation_action=decision.action,
                mutation=decision,
            )
            backend._index.retain_block(decision.record)
            backend._index.complete_canonical_head(reservation, decision)
            backend._index.complete_write_outbox(entry.commit_revision)
        heads = {
            str(row[0])
            for row in backend._index._conn.execute(
                "SELECT block_id FROM canonical_heads WHERE workspace_id=?",
                ("workspace",),
            ).fetchall()
        }
        assert heads == {first_decision.record.block_id, second_decision.record.block_id}
    finally:
        backend.close()


def test_t14_cluster_is_observational_and_correction_rederives(tmp_path: Path) -> None:
    index = MemPalaceBackend(
        tmp_path,
        workspace_id="workspace",
        reconcile_pending_writes=False,
    )
    try:
        client = MemoryClient(tmp_path, workspace_id="workspace", backend_instance=index)
        first = client.remember(
            "The package uses Flask.",
            title="Package framework",
            kind="fact",
            metadata={"source_path": "README.md"},
            requested_block_id="stable_project_facts",
        )
        root = index.get_block(first.memory_id)
        assert root is not None
        explicit_cluster = replace(root, duplicate_cluster_id="dup_explicit")
        retried = plan_memory_mutation(explicit_cluster, existing=root)
        assert retried.action == "retry"
        assert retried.record.revision_id == root.revision_id
        assert retried.record.block_id == root.block_id
        corrected = client.correction(
            first.memory_id,
            "The package uses FastAPI.",
            requested_block_id="stable_project_facts",
        )
        successor = index.get_block(str(corrected["block_id"]))
        assert successor is not None
        assert index._index._conn.execute(
            "SELECT mutation_action FROM canonical_mutations "
            "WHERE successor_id=? ORDER BY commit_revision DESC LIMIT 1",
            (successor.block_id,),
        ).fetchone() == ("correction",)
        assert successor.duplicate_cluster_id != root.duplicate_cluster_id
    finally:
        index.close()


def test_t14_corrections_and_opposing_content_keep_distinct_clusters() -> None:
    root = replace(
        _accepted_record("root"),
        content="The deployment is healthy.",
        canonical_key="workspace:observation:health",
    )
    root = plan_memory_mutation(root, existing=None).record
    correction = plan_memory_mutation(
        replace(root, content="The deployment is unhealthy", duplicate_cluster_id=""),
        existing=root,
        action="correction",
    )
    opposing = plan_memory_mutation(
        replace(root, content="The deployment is unhealthy", duplicate_cluster_id=""),
        existing=None,
    )
    assert correction.action == "correction"
    assert correction.edge is not None
    assert correction.edge.edge_type == "supersedes"
    assert correction.record.duplicate_cluster_id != root.duplicate_cluster_id
    assert opposing.record.duplicate_cluster_id == correction.record.duplicate_cluster_id


def test_t14_cluster_round_trips_event_json_block_hit_and_reopen() -> None:
    planned = plan_memory_mutation(_record("roundtrip"), existing=None).record
    event = block_record_to_event(planned)
    normalized = normalize_memory_event(event.__dict__)
    block = memory_event_to_block_record(normalized)
    reopened = block_from_json(block_to_json(block))
    hit = block_record_to_memory_hit(reopened)
    from_hit = api._record_from_hit(hit, RecallPlan(query="roundtrip"))
    assert normalized.duplicate_cluster_id == planned.duplicate_cluster_id
    assert reopened.duplicate_cluster_id == planned.duplicate_cluster_id
    assert from_hit.duplicate_cluster_id == planned.duplicate_cluster_id


def test_t14_metadata_cannot_forge_typed_cluster() -> None:
    with pytest.raises(ValueError, match="reserved governance fields"):
        MemoryBlockRecord(
            block_id="metadata-cluster",
            content="typed cluster",
            metadata={"duplicate_cluster_id": "forged"},
        )
