from pathlib import Path

import pytest

from um_mem import MemoryClient
from um_mem.contracts import MemoryBlockRecord, RecordBlockMembership
from um_mem.memory.backends.query_policy import ensure_record_is_storable
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.recall_plan import RecallPlan
from um_mem.memory.scopes import ScopeLineage, bind_scope_chain


class _RecordingBackend:
    def __init__(self) -> None:
        self.queries: list[dict[str, object]] = []

    def recall(self, query: dict[str, object]) -> list[object]:
        self.queries.append(dict(query))
        return []


def test_workspace_only_lineage_is_the_simple_local_default() -> None:
    lineage = ScopeLineage(workspace_id="alpha")

    assert bind_scope_chain(lineage, ()) == ("workspace:alpha",)
    assert "user_global" not in " ".join(bind_scope_chain(lineage, ()))


def test_requested_scope_chain_is_an_exact_authorized_subset() -> None:
    lineage = ScopeLineage(
        workspace_id="alpha",
        project_id="project-a",
        session_id="session-a",
        task_id="task-a",
        subtask_id="subtask-a",
        user_id="user-a",
    )

    assert bind_scope_chain(lineage, ()) == (
        "subtask:subtask-a",
        "task:task-a",
        "session:session-a",
        "workspace:alpha",
        "project:project-a",
    )
    assert bind_scope_chain(lineage, ("task:task-a", "workspace:alpha")) == (
        "task:task-a",
        "workspace:alpha",
    )
    assert bind_scope_chain(lineage, ("user_global:user-a",)) == ("user_global:user-a",)


@pytest.mark.parametrize(
    "requested",
    (
        ("workspace:foreign",),
        ("workspace:*",),
        ("workspace:alpha", "workspace:alpha"),
        ("workspace:alpha", "session:session-a"),
        ("user_global:foreign",),
    ),
)
def test_scope_binding_rejects_untrusted_or_malformed_claims(
    requested: tuple[str, ...],
) -> None:
    lineage = ScopeLineage(
        workspace_id="alpha",
        session_id="session-a",
        user_id="user-a",
    )

    with pytest.raises(ValueError):
        bind_scope_chain(lineage, requested)


def test_recall_plan_does_not_authorize_its_own_workspace_claim() -> None:
    plan = RecallPlan(query="parser", workspace_id="alpha")

    assert plan.scope_chain == ()


def test_memory_client_binds_default_and_explicit_scope_to_constructor_lineage(
    tmp_path: Path,
) -> None:
    backend = _RecordingBackend()
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        project_id="project-a",
        session_id="session-a",
        task_id="task-a",
        backend_instance=backend,
    )

    client.recall("parser", search_all=True)
    assert backend.queries[-1]["scope_chain"] == [
        "task:task-a",
        "session:session-a",
        "workspace:alpha",
        "project:project-a",
    ]

    client.recall(
        "parser",
        search_all=True,
        scope_chain=("session:session-a", "workspace:alpha"),
    )
    assert backend.queries[-1]["scope_chain"] == [
        "session:session-a",
        "workspace:alpha",
    ]

    with pytest.raises(ValueError, match="scope"):
        client.recall("parser", search_all=True, scope_chain=("workspace:foreign",))
    assert len(backend.queries) == 2


@pytest.mark.parametrize(
    ("scope", "fields", "message"),
    (
        ("workspace", {}, "workspace_id"),
        ("session", {"workspace_id": "alpha"}, "session_id"),
        (
            "task",
            {"workspace_id": "alpha", "session_id": "session-a"},
            "task_id",
        ),
        (
            "subtask",
            {
                "workspace_id": "alpha",
                "session_id": "session-a",
                "task_id": "task-a",
            },
            "subtask_id",
        ),
        ("project", {}, "project_id"),
        ("user_global", {}, "user_id"),
    ),
)
def test_records_cannot_be_stored_with_invisible_scope_identity(
    scope: str,
    fields: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MemoryBlockRecord(block_id="record-a", content="evidence", scope=scope, **fields)


def test_backend_admission_rechecks_scope_identity() -> None:
    valid = MemoryBlockRecord(
        block_id="record-a",
        revision_id="revision:record-a",
        content="evidence",
        block_membership=RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
        scope="workspace",
        workspace_id="alpha",
    )
    invalid = object.__new__(MemoryBlockRecord)
    for field_name in valid.__dataclass_fields__:
        object.__setattr__(invalid, field_name, getattr(valid, field_name))
    object.__setattr__(invalid, "workspace_id", "")

    with pytest.raises(ValueError, match="workspace_id"):
        ensure_record_is_storable(invalid)


def test_identity_fields_are_not_accepted_as_free_metadata() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        MemoryBlockRecord(
            block_id="record-a",
            content="evidence",
            workspace_id="alpha",
            metadata={"workspace_id": "foreign"},
        )


def test_scope_lineage_rejects_broken_parent_chain() -> None:
    with pytest.raises(ValueError, match="session_id"):
        ScopeLineage(workspace_id="alpha", task_id="task-a")
    with pytest.raises(ValueError, match="task_id"):
        ScopeLineage(workspace_id="alpha", session_id="session-a", subtask_id="subtask-a")


def test_explicit_narrow_write_uses_bound_scope_without_rerouting(tmp_path: Path) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        session_id="session-a",
        task_id="task-a",
        subtask_id="subtask-a",
        backend_instance=backend,
    )
    try:
        saved = client.remember(
            "Observed a parser implementation detail.",
            kind="research_finding",
            requested_block_id="execution_working_state",
            evidence_refs=[
                {
                    "ref_type": "tool_result",
                    "ref_id": "tool:parser-read",
                    "produced_by": "verifier",
                    "producer_id": "host:parser-read",
                }
            ],
        )
        record = backend.get_block(saved.memory_id, workspace_id="alpha")

        assert saved.saved is True
        assert record is not None
        assert record.scope == "subtask"
        assert record.block_membership is not None
        assert record.block_membership.primary_block_id == "execution_working_state"
    finally:
        backend.close()


def test_explicit_unbound_narrow_route_and_per_call_identity_claim_fail_before_write(
    tmp_path: Path,
) -> None:
    backend = BackendSQLiteIndex(
        tmp_path / "memory.sqlite",
        backend_id="test",
        workspace_id="alpha",
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        result = client.remember(
            "Unbound execution state.",
            kind="research_finding",
            requested_block_id="execution_working_state",
        )
        assert result.saved is False
        assert result.raw["reason"] == "requested_block_scope_not_bound"
        assert result.raw["suggested_route"] == {
            "action": "retry_remember",
            "requested_block_id": "decisions_and_contracts",
        }

        with pytest.raises(ValueError, match="session_id"):
            client.remember(
                "Untrusted session claim.",
                session_id="foreign-session",
                requested_block_id="workspace_episodes",
            )
        assert backend._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0
    finally:
        backend.close()
