import json
from pathlib import Path

import pytest

from um_mem.api import MemoryClient
from um_mem.contracts import MemoryBlockRecord, RecordBlockMembership
from um_mem.memory.backends.sqlite_index import BackendSQLiteIndex
from um_mem.memory.kernel.telemetry import read_memory_events
from um_mem.memory.obsidian_export import export_records_to_obsidian

_PROTECTED_CASES = (
    ("confidential", "full", "low"),
    ("secret", "pointer_only", "low"),
    ("internal", "non_injectable", "low"),
)


class _CaptureBackend:
    def __init__(self) -> None:
        self.records = []

    def retain_block(self, record):
        self.records.append(record)
        return {"ok": True, "block_id": record.block_id}


def _record(
    marker: str,
    *,
    sensitivity: str,
    content_mode: str,
    operational_risk: str,
) -> MemoryBlockRecord:
    return MemoryBlockRecord(
        block_id=marker,
        revision_id=f"revision:{marker}",
        title=marker,
        content=marker,
        block_membership=RecordBlockMembership(
            primary_block_id="workspace_episodes",
            catalog_revision_id="test-catalog",
        ),
        kind="observation",
        scope="workspace",
        workspace_id="alpha",
        sensitivity=sensitivity,
        content_mode=content_mode,
        operational_risk=operational_risk,
    )


@pytest.mark.parametrize(("sensitivity", "content_mode", "operational_risk"), _PROTECTED_CASES)
def test_protected_canonical_axes_never_reach_backend_prompt_or_export(
    tmp_path: Path,
    sensitivity: str,
    content_mode: str,
    operational_risk: str,
) -> None:
    marker = f"SENSITIVE_CANONICAL_{sensitivity}_{content_mode}_{operational_risk}"
    backend = _CaptureBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    result = client.remember(
        marker,
        title=marker,
        requested_block_id="workspace_episodes",
        sensitivity=sensitivity,
        content_mode=content_mode,
        operational_risk=operational_risk,
    )
    assert result.saved is False
    assert backend.records == []

    index = BackendSQLiteIndex(tmp_path / "memory.sqlite", backend_id="test", workspace_id="alpha")
    try:
        with pytest.raises(ValueError, match="cannot enter selected backend storage"):
            index.retain_block(
                _record(
                    marker,
                    sensitivity=sensitivity,
                    content_mode=content_mode,
                    operational_risk=operational_risk,
                )
            )
        assert index.render_prompt_items(workspace_id="alpha") == []
    finally:
        index.close()

    exported = export_records_to_obsidian(
        blocks=[
            _record(
                marker,
                sensitivity=sensitivity,
                content_mode=content_mode,
                operational_risk=operational_risk,
            )
        ],
        edges=[],
        vault_dir=tmp_path / "vault",
        workspace_id="alpha",
    )
    assert exported.block_paths == {}
    assert marker not in exported.memory_overview_path.read_text(encoding="utf-8")
    assert marker not in json.dumps(read_memory_events(tmp_path), ensure_ascii=False)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sensitivity", "secret_pointer_only"),
        ("content_mode", "secret-pointer-only"),
        ("operational_risk", "secret"),
        ("sensitivity", None),
    ),
)
def test_safety_axes_reject_closed_vocabulary_and_keep_marker_out_of_telemetry(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    marker = "SENSITIVE_TELEMETRY_MARKER"
    backend = _CaptureBackend()
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)

    with pytest.raises(ValueError, match=field):
        client.remember(
            marker,
            title=marker,
            requested_block_id="workspace_episodes",
            **{field: value},
        )
    assert backend.records == []
    assert marker not in json.dumps(read_memory_events(tmp_path), ensure_ascii=False)
