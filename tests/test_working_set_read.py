from dataclasses import asdict, replace
from pathlib import Path

import pytest

from lethyx import MemoryClient, WorkingSnapshotBudgetError
from lethyx.contracts import MemoryBlockRecord


class RecordBackend:
    def __init__(self, records: list[MemoryBlockRecord]) -> None:
        self.records = {record.block_id: record for record in records}

    def get_block(self, record_id: str) -> MemoryBlockRecord | None:
        return self.records.get(record_id)


def test_public_working_snapshot_is_complete_pinned_and_never_promotes(tmp_path: Path) -> None:
    import hashlib

    content = "  Exact candidate: α\nRetain this final sentence.  "
    record = MemoryBlockRecord(
        block_id="candidate",
        content=content,
        title="Measurement",
        workspace_id="alpha",
        revision_id="revision-1",
        metadata={
            "working_set_owner": {"owner_id": "node-a", "generation_id": "generation-1"},
            "private_transport": "must-not-project",
        },
    )
    backend = RecordBackend([record])
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    request = {
        "owner_id": "node-a",
        "generation_id": "generation-1",
        "expected_revision_id": "revision-1",
        "expected_content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    snapshot = client.snapshot_working_record("candidate", **request)
    assert snapshot.content == content
    assert snapshot.reference.status == "candidate"
    assert snapshot.reference.content_hash == request["expected_content_hash"]
    assert "metadata" not in asdict(snapshot)
    assert record.status == "candidate"
    with pytest.raises(WorkingSnapshotBudgetError, match="content budget"):
        client.snapshot_working_record("candidate", **request, max_bytes=16)
    with pytest.raises(WorkingSnapshotBudgetError, match="content budget"):
        client.snapshot_working_record("candidate", **request, max_tokens=1)
    with pytest.raises(ValueError, match="unavailable"):
        client.snapshot_working_record("candidate", **{**request, "generation_id": "other"})
    for changed in (
        replace(record, revision_id="revision-2"),
        replace(record, content="Amended after pinning"),
        replace(record, status="retracted", retention="archive", delivery_policy="hidden"),
        replace(record, workspace_id="foreign"),
        replace(record, scope="session", session_id="foreign-session"),
        replace(record, sensitivity="secret", content_mode="non_injectable"),
        replace(record, content_mode="pointer_only"),
    ):
        backend.records[record.block_id] = changed
        with pytest.raises(ValueError, match="unavailable"):
            client.snapshot_working_record("candidate", **request)


def test_working_read_is_exact_generation_scoped_and_bounded(tmp_path: Path) -> None:
    record = MemoryBlockRecord(
        block_id="own",
        content="Current unfinished experiment " * 1000,
        workspace_id="alpha",
        profile_id="default",
        revision_id="revision-own",
        metadata={"working_set_owner": {"owner_id": "node-a", "generation_id": "generation-1"}},
    )
    foreign = replace(
        record,
        block_id="foreign",
        metadata={"working_set_owner": {"owner_id": "node-b", "generation_id": "generation-1"}},
    )
    old = replace(
        record,
        block_id="old",
        metadata={"working_set_owner": {"owner_id": "node-a", "generation_id": "generation-0"}},
    )
    retracted = replace(
        record,
        block_id="retracted",
        status="retracted",
        retention="archive",
        delivery_policy="hidden",
    )
    cross_profile = replace(record, block_id="cross-profile", profile_id="other")
    cross_workspace = replace(record, block_id="cross-workspace", workspace_id="beta")
    cross_session = replace(
        record, block_id="cross-session", scope="session", session_id="other-session"
    )
    protected = replace(
        record, block_id="protected", sensitivity="secret", content_mode="non_injectable"
    )
    backend = RecordBackend(
        [
            record,
            foreign,
            old,
            retracted,
            cross_profile,
            cross_workspace,
            cross_session,
            protected,
        ]
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    result = client.read_working_set(
        owner_id="node-a",
        generation_id="generation-1",
        record_ids=(
            "own",
            "foreign",
            "old",
            "retracted",
            "cross-profile",
            "cross-workspace",
            "cross-session",
            "protected",
            "missing",
        ),
        max_tokens=256,
        max_bytes=512,
    )

    assert [ref.record_id for ref in result.records] == ["own"]
    assert result.records[0].revision_id == record.revision_id
    assert result.records[0].content_hash
    assert result.estimated_tokens <= 256
    assert result.rendered_bytes <= 512
    assert result.truncated is True
    assert "Current unfinished experiment" in result.rendered_context
    assert set(result.unavailable_record_ids) == {
        "foreign",
        "old",
        "retracted",
        "cross-profile",
        "cross-workspace",
        "cross-session",
        "protected",
        "missing",
    }
    assert record.status == "candidate"

    tiny = client.read_working_set(
        owner_id="node-a",
        generation_id="generation-1",
        record_ids=("own",),
        max_tokens=1,
        max_bytes=1,
    )
    assert tiny.rendered_context == ""
    assert tiny.estimated_tokens == 0
    assert tiny.rendered_bytes == 0
    assert tiny.omitted_record_ids == ("own",)


def test_public_record_pin_preserves_exact_bytes_and_rejects_foreign_generation(
    tmp_path: Path,
) -> None:
    import hashlib

    content = "  Exact source text\n"
    record = MemoryBlockRecord(
        block_id="own",
        content=content,
        workspace_id="alpha",
        profile_id="default",
        revision_id="revision-1",
        metadata={"working_set_owner": {"owner_id": "node-a", "generation_id": "generation-1"}},
    )
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=RecordBackend([record]))
    ref = client.pin_record("own", owner_id="node-a", generation_id="generation-1")
    assert ref.record_id == "own"
    assert ref.revision_id == "revision-1"
    assert ref.content_hash == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert ref.status == "candidate"
    assert not hasattr(ref, "content")
    with pytest.raises(ValueError, match="unavailable"):
        client.pin_record("own", owner_id="node-a", generation_id="generation-2")
    with pytest.raises(ValueError, match="unavailable"):
        client.pin_record("own")
    for excluded in (
        replace(record, workspace_id="beta"),
        replace(record, profile_id="another-profile"),
        replace(record, scope="session", session_id="foreign-session"),
        replace(record, status="retracted", retention="archive", delivery_policy="hidden"),
        replace(record, sensitivity="secret", content_mode="non_injectable"),
    ):
        isolated = MemoryClient(
            tmp_path, workspace_id="alpha", backend_instance=RecordBackend([excluded])
        )
        with pytest.raises(ValueError, match="unavailable"):
            isolated.pin_record("own", owner_id="node-a", generation_id="generation-1")

    preference = replace(record, scope="user_global", user_id="user-a")
    scoped = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        user_id="user-a",
        backend_instance=RecordBackend([preference]),
    )
    owner = {"owner_id": "node-a", "generation_id": "generation-1"}
    assert scoped.pin_record("own", **owner, scope_chain=("user_global:user-a",)).record_id == "own"
    working = scoped.read_working_set(
        **owner,
        record_ids=("own",),
        scope_chain=("user_global:user-a",),
    )
    assert working.records[0].record_id == "own"
    with pytest.raises(ValueError, match="not bound"):
        scoped.pin_record("own", **owner, scope_chain=("user_global:user-b",))
