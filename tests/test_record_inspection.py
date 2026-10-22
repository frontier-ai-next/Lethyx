from dataclasses import replace
from pathlib import Path

import pytest

from um_mem import BackendCapabilities, MemoryClient
from um_mem.contracts import MemoryBlockRecord, RecordBlockMembership
from um_mem.contracts.memory import MemoryStatus
from um_mem.contracts.models import EvidenceRef


def record() -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id="record-1",
        revision_id="revision-1",
        workspace_id="alpha",
        profile_id="operator",
        content="Exact historical content.",
        block_membership=RecordBlockMembership(
            primary_block_id="findings",
            catalog_revision_id="catalog-1",
        ),
    )


class Backend:
    def __init__(self, value: MemoryBlockRecord) -> None:
        self.value = value

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("mempalace", exact_record_read=True)

    def get_block(self, memory_id: str) -> MemoryBlockRecord:
        assert memory_id == "record-1"
        return self.value


@pytest.mark.parametrize("status", ["candidate", "accepted", "retracted"])
def test_inspection_reads_exact_scoped_revision_without_changing_its_status(
    tmp_path: Path,
    status: MemoryStatus,
) -> None:
    expected = record()
    if status == "accepted":
        expected = replace(
            expected,
            status=status,
            verification_status="verified",
            verification_methods=("test_run",),
            trust_level="workspace_verified",
            promotion_decision="accepted",
            evidence_refs=(EvidenceRef("test_run", "test:fixture", producer_id="test-host"),),
        )
    elif status == "retracted":
        expected = replace(expected, status=status, retention="archive", delivery_policy="hidden")
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        profile_id="operator",
        backend_instance=Backend(expected),
    )
    inspected = client.inspect_record(
        "record-1",
        logical_block_id="findings",
        expected_revision_id="revision-1",
    )
    assert inspected == expected
    assert inspected.status == status


@pytest.mark.parametrize(
    "field,value",
    [
        ("workspace_id", "foreign"),
        ("profile_id", "foreign"),
        ("block_id", "another-record"),
        ("revision_id", "another-revision"),
        (
            "block_membership",
            RecordBlockMembership(primary_block_id="other", catalog_revision_id="c"),
        ),
    ],
)
def test_inspection_rejects_foreign_or_replaced_record(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    client = MemoryClient(
        tmp_path,
        workspace_id="alpha",
        profile_id="operator",
        backend_instance=Backend(replace(record(), **{field: value})),
    )
    with pytest.raises(ValueError):
        client.inspect_record(
            "record-1",
            logical_block_id="findings",
            expected_revision_id="revision-1",
        )
