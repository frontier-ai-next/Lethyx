"""Capability admission must precede topology or native mutations."""

from dataclasses import asdict
from pathlib import Path

import pytest

from lethyx import BackendCapabilities, MemoryClient
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mempalace import MemPalaceBackend
from lethyx.memory.backends.mnemosyne import MnemosyneBackend
from lethyx.memory.catalog_store import BlockEvolutionTarget, BlockMergeSource


@pytest.mark.parametrize("backend_type", [MemPalaceBackend, HindsightBackend, MnemosyneBackend])
def test_native_capabilities_distinguish_revision_recovery_from_batch_atomicity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_type: type
) -> None:
    backend = backend_type(repo_root=tmp_path, workspace_id="alpha", reconcile_pending_writes=False)
    monkeypatch.setattr(backend, "health", lambda: {"native_ok": True, "effective_mode": "native"})
    if isinstance(backend, HindsightBackend):

        class NativeClient:
            def retain(self):
                raise AssertionError("capability discovery must not retain records")

            def recall(self):
                raise AssertionError("capability discovery must not recall records")

        monkeypatch.setattr(backend, "_client", lambda: NativeClient())
    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=backend)
    try:
        capabilities = client.capabilities()
        assert capabilities.exact_record_read
        assert capabilities.scoped_search
        assert capabilities.revision_history
        assert capabilities.revision_compare_and_swap
        assert capabilities.recoverable_writes
        assert not capabilities.atomic_record_reassignment
        assert capabilities.staged_topology_evolution
        assert not capabilities.atomic_batch_publication
        assert not capabilities.batch_read
        assert not capabilities.cancellation

        monkeypatch.setattr(backend, "health", lambda: {"native_ok": False})
        unavailable = asdict(client.capabilities())
        assert not any(value for value in unavailable.values() if isinstance(value, bool))
    finally:
        backend.close()


def test_undeclared_topology_mutations_fail_before_catalog_changes(tmp_path: Path) -> None:
    class Backend:
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities("mempalace", persistent=True)

        def reassign_logical_block_records(self, *_args, **_kwargs):
            raise AssertionError("method presence must not grant reassignment")

    client = MemoryClient(tmp_path, workspace_id="alpha", backend_instance=Backend())
    root = client.block_map()[0]
    revision = client.catalog_revision()
    target = BlockEvolutionTarget(name="Target", synopsis="Explicit target", record_ids=())
    with pytest.raises(ValueError, match="staged_topology_evolution"):
        client.split_block(
            root["block_id"], expected_revision=root["revision"], targets=(target,), reason="test"
        )
    with pytest.raises(ValueError, match="staged_topology_evolution"):
        client.merge_blocks(
            (BlockMergeSource("a", 1), BlockMergeSource("b", 1)),
            target=target,
            parent_id=root["block_id"],
            reason="test",
        )
    assert client.catalog_revision() == revision


def test_public_capabilities_reject_missing_untyped_or_foreign_declarations(tmp_path: Path) -> None:
    class Backend:
        def __init__(self, result: object) -> None:
            self.result = result

        def capabilities(self) -> object:
            return self.result

    for backend in (
        object(),
        Backend({"exact_record_read": True}),
        Backend(BackendCapabilities("hindsight")),
    ):
        client = MemoryClient(tmp_path, backend="mempalace", backend_instance=backend)
        with pytest.raises(ValueError, match="capabilit"):
            client.capabilities()


def test_effective_capabilities_reject_unknown_features_and_non_boolean_claims() -> None:
    with pytest.raises(ValueError, match="unknown"):
        BackendCapabilities.effective(
            "mempalace", native_ok=True, implemented=frozenset({"atomic_maybe"})
        )
    with pytest.raises(ValueError, match="boolean"):
        BackendCapabilities("mempalace", exact_record_read="false")
