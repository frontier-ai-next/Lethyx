import base64
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError

import pytest

from um_mem import MemoryClient, recall, remember
from um_mem.contracts import MemoryBlockRecord, memory_scope_key
from um_mem.memory.backends.base import MemoryQuery
from um_mem.memory.backends.contracts import CanonicalHeadReservation, block_record_to_memory_hit
from um_mem.memory.backends.hindsight import HindsightBackend
from um_mem.memory.backends.mempalace import MemPalaceBackend
from um_mem.memory.hindsight.config import HindsightConfig


class FakeBackend:
    def __init__(self) -> None:
        self.blocks: dict[str, MemoryBlockRecord] = {}
        self._commit_revision = 0

    def reserve_canonical_head(
        self,
        *,
        workspace_id: str,
        canonical_key: str,
        predecessor_id: str,
        successor_id: str,
        successor_status: str,
        successor_authority: str,
        **_kwargs: Any,
    ) -> CanonicalHeadReservation:
        self._commit_revision += 1
        return CanonicalHeadReservation(
            commit_revision=self._commit_revision,
            workspace_id=workspace_id,
            canonical_key_sha256=hashlib.sha256(canonical_key.encode("utf-8")).hexdigest(),
            predecessor_id=predecessor_id,
            successor_id=successor_id,
            successor_status=successor_status,
            successor_authority=successor_authority,
        )

    def complete_canonical_head(
        self,
        reservation: CanonicalHeadReservation,
        _mutation: Any,
    ) -> dict[str, Any]:
        return {"ok": True, "commit_revision": reservation.commit_revision}

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.blocks[record.block_id] = record
        return {"ok": True, "block_id": record.block_id, "backend": "fake"}

    def recall_evidence(self, query: Any) -> list[Any]:
        q = str(query.get("query", "") if isinstance(query, dict) else "").lower()
        hits = []
        for block in self.blocks.values():
            haystack = f"{block.title}\n{block.content}".lower()
            if not q or any(part in haystack for part in q.split()):
                hits.append(block_record_to_memory_hit(block))
        return hits

    def update_lifecycle(self, block_id: str, lifecycle: str, **kwargs: Any) -> dict[str, Any]:
        current = self.blocks[block_id]
        self.blocks[block_id] = replace(
            current,
            lifecycle=lifecycle,
            surface=kwargs.get("surface") or current.surface,
        )
        return {"ok": True, "block_id": block_id, "lifecycle": lifecycle}

    def export_blocks(self) -> list[Any]:
        return list(self.blocks.values())

    def get_block(self, block_id: str, *, workspace_id: str = "") -> MemoryBlockRecord | None:
        if (
            workspace_id
            and block_id in self.blocks
            and self.blocks[block_id].workspace_id != workspace_id
        ):
            return None
        return self.blocks.get(block_id)

    def exact_reference_block_ids(self, reference: str, *, workspace_id: str) -> tuple[str, ...]:
        return tuple(
            record.block_id
            for record in self.blocks.values()
            if record.workspace_id == workspace_id
            and (
                any(reference in {ref.ref_id, ref.hash} for ref in record.evidence_refs)
                or any(
                    reference
                    in {str(segment.get("source_id") or ""), str(segment.get("sha256") or "")}
                    for segment in record.metadata.get("source_segments", ())
                    if isinstance(segment, dict)
                )
            )
        )

    def scoped_record_ids(
        self,
        *,
        workspace_id: str,
        profile_id: str,
        scope_key: str,
        logical_block_id: str = "",
    ) -> tuple[str, ...]:
        return tuple(
            record.block_id
            for record in self.blocks.values()
            if record.workspace_id == workspace_id
            and record.profile_id == profile_id
            and memory_scope_key(record) == scope_key
            and (
                not logical_block_id
                or (
                    record.block_membership is not None
                    and record.block_membership.primary_block_id == logical_block_id
                )
            )
        )

    def edges_for(self, block_id: str) -> list[Any]:
        return []

    def export_edges(self) -> list[Any]:
        return []


def test_projection_barrier_waits_for_named_revision_without_store_reconcile(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PendingBackend:
        def __init__(self) -> None:
            self.reads = 0

        def write_outbox_status(self, _record_id: str) -> str:
            self.reads += 1
            return "pending"

        def latest_write_outbox_entry(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(commit_revision=7)

        def reconcile(self, *, dry_run: bool) -> Any:
            pytest.fail("barrier must not reconcile unrelated work")

    backend = PendingBackend()
    client = MemoryClient(tmp_path, workspace_id="test", backend_instance=backend)
    monkeypatch.setattr(client, "_require_profile_record", lambda _id: None)
    result = client.projection_barrier("record-1", commit_revision=7, timeout_seconds=0.12)

    assert backend.reads >= 2
    assert result["reason_code"] == "projection_deadline_exceeded"
    assert result["attention_required"] is False
    assert result["retryable"] is True
    assert result["commit_revision"] == 7
    assert result["projection_revision"] == 0


def test_projection_barrier_observes_named_revision_commit(tmp_path, monkeypatch) -> None:
    class RecoveringBackend:
        def __init__(self) -> None:
            self.reads = 0

        def write_outbox_status(self, _record_id: str) -> str:
            self.reads += 1
            return "committed" if self.reads == 3 else "pending"

        def latest_write_outbox_entry(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(commit_revision=9)

        def reconcile(self, *, dry_run: bool) -> Any:
            pytest.fail("barrier must not start full reconciliation")

    backend = RecoveringBackend()
    client = MemoryClient(tmp_path, workspace_id="test", backend_instance=backend)
    monkeypatch.setattr(client, "_require_profile_record", lambda _id: None)
    result = client.projection_barrier("record-1", commit_revision=9, timeout_seconds=1)

    assert backend.reads == 3
    assert result["ok"] is True
    assert result["projection_revision"] == 9


def test_projection_barrier_deadline_does_not_start_recovery(tmp_path, monkeypatch) -> None:
    class PendingBackend:
        def write_outbox_status(self, _record_id: str) -> str:
            return "pending"

        def reconcile(self, *, dry_run: bool) -> Any:
            pytest.fail("recovery started after deadline")

    client = MemoryClient(tmp_path, workspace_id="test", backend_instance=PendingBackend())
    monkeypatch.setattr(client, "_require_profile_record", lambda _id: None)
    result = client.projection_barrier("record-1", timeout_seconds=0)

    assert result["reason_code"] == "projection_deadline_exceeded"
    assert result["retryable"] is True


def test_projection_barrier_requires_outbox_entry_for_success(tmp_path, monkeypatch) -> None:
    class MissingEntryBackend:
        def write_outbox_status(self, _record_id: str) -> str:
            return "committed"

        def latest_write_outbox_entry(self, **_kwargs: Any) -> None:
            return None

    client = MemoryClient(tmp_path, workspace_id="test", backend_instance=MissingEntryBackend())
    monkeypatch.setattr(client, "_require_profile_record", lambda _id: None)

    result = client.projection_barrier("record-1", timeout_seconds=0)

    assert result["ok"] is False
    assert result["projection_status"] == "pending"
    assert result["reason_code"] == "projection_deadline_exceeded"


def test_projection_barrier_rejects_different_revision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DifferentRevisionBackend:
        def write_outbox_status(self, _record_id: str) -> str:
            return "committed"

        def latest_write_outbox_entry(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(commit_revision=5)

        def reconcile(self, *, dry_run: bool) -> Any:
            pytest.fail("barrier must not start full reconciliation")

    client = MemoryClient(
        tmp_path, workspace_id="test", backend_instance=DifferentRevisionBackend()
    )
    monkeypatch.setattr(client, "_require_profile_record", lambda _id: None)
    result = client.projection_barrier("record-1", commit_revision=4, timeout_seconds=10)

    assert result["reason_code"] == "projection_revision_mismatch"
    assert result["projection_revision"] == 5
    assert result["retryable"] is False
    assert result["attention_required"] is True


def test_projection_barrier_ignores_unrelated_pending_revision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ScopedBackend:
        def write_outbox_status(self, record_id: str) -> str:
            assert record_id == "record-1"
            return "committed"

        def latest_write_outbox_entry(self, *, block_id: str, workspace_id: str) -> Any:
            assert (block_id, workspace_id) == ("record-1", "test")
            return SimpleNamespace(commit_revision=11)

        def reconcile(self, *, dry_run: bool) -> Any:
            pytest.fail("unrelated pending work must not be reconciled")

    client = MemoryClient(tmp_path, workspace_id="test", backend_instance=ScopedBackend())
    monkeypatch.setattr(client, "_require_profile_record", lambda _id: None)

    result = client.projection_barrier("record-1", commit_revision=11, timeout_seconds=1)

    assert result["ok"] is True
    assert result["projection_revision"] == 11


def test_public_client_remember_and_recall_are_framework_neutral(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = MemoryClient(tmp_path, backend_instance=backend, workspace_id="demo")

    saved = client.remember(
        "Use evidence-gated promotion for agent memory.",
        tags=["policy"],
        requested_block_id="workspace_episodes",
    )
    hits = client.recall(
        "promotion",
        logical_block_ids=["workspace_episodes"],
        include_candidates=True,
    )

    assert saved.saved is True
    assert saved.backend == "fake"
    assert hits[0].metadata["workspace_id"] == "demo"
    assert hits[0].metadata["authority"] == "advisory"


def test_public_client_reads_exact_records_in_requested_order(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = MemoryClient(tmp_path, backend_instance=backend, workspace_id="demo")
    first = client.remember("first fact", requested_block_id="workspace_episodes")
    second = client.remember("second fact", requested_block_id="workspace_episodes")

    records = client.read_exact_records([second.memory_id, first.memory_id])

    assert [record.content for record in records] == ["second fact", "first fact"]
    with pytest.raises(ValueError, match="distinct record IDs"):
        client.read_exact_records([first.memory_id, first.memory_id])
    with pytest.raises(ValueError, match="not found"):
        client.read_exact_records([first.memory_id, "absent"])


def test_convenience_functions_accept_injected_backend(tmp_path: Path) -> None:
    backend = FakeBackend()

    saved = remember(
        "Host nodes can request non-directive recall.",
        repo_root=tmp_path,
        backend_instance=backend,
        requested_block_id="workspace_episodes",
    )
    hits = recall(
        "Host",
        repo_root=tmp_path,
        backend_instance=backend,
        logical_block_ids=["workspace_episodes"],
        include_candidates=True,
    )

    assert saved.saved is True
    assert [hit.text for hit in hits] == ["Host nodes can request non-directive recall."]


def test_list_records_uses_bounded_scope_snapshot_across_new_writes(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = MemoryClient(tmp_path, backend_instance=backend, workspace_id="demo")
    first = client.remember("first", requested_block_id="workspace_episodes")
    second = client.remember("second", requested_block_id="workspace_episodes")
    hidden = client.remember("hidden", requested_block_id="workspace_episodes")
    backend.blocks[hidden.memory_id] = replace(
        backend.blocks[hidden.memory_id], delivery_policy="hidden"
    )
    other_profile = MemoryClient(
        tmp_path, backend_instance=backend, workspace_id="demo", profile_id="beta"
    )
    other_profile.remember("other profile", requested_block_id="workspace_episodes")

    page_one = client.list_records(
        scope="workspace", scope_id="demo", logical_block_id="workspace_episodes", limit=1
    )
    client.remember("later", requested_block_id="workspace_episodes")
    page_two = client.list_records(
        scope="workspace",
        scope_id="demo",
        logical_block_id="workspace_episodes",
        limit=1,
        cursor=page_one["next_cursor"],
    )

    listed = [row["record_id"] for row in page_one["records"] + page_two["records"]]
    assert set(listed) == {first.memory_id, second.memory_id}
    assert page_two["next_cursor"] == ""
    with pytest.raises(ValueError, match="cursor scope"):
        client.list_records(
            scope="workspace",
            scope_id="other",
            logical_block_id="workspace_episodes",
            cursor=page_one["next_cursor"],
        )
    decoded = json.loads(base64.urlsafe_b64decode(page_one["next_cursor"] + "=="))
    decoded["unexpected"] = True
    forged = base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode()
    with pytest.raises(ValueError, match="cursor is invalid"):
        client.list_records(scope="workspace", scope_id="demo", cursor=forged)


def test_record_cursor_is_small_reopenable_and_does_not_rescan(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = MemoryClient(tmp_path, backend_instance=backend, workspace_id="demo")
    first = client.remember("first", requested_block_id="workspace_episodes")
    template = backend.blocks[first.memory_id]
    for i in range(369):
        record = replace(template, block_id=f"record-{i}", revision_id=f"revision-{i}")
        backend.blocks[record.block_id] = record
    page = client.list_records(scope="workspace", scope_id="demo", limit=128)
    assert len(page["next_cursor"]) < 1024
    reopened = MemoryClient(tmp_path, backend_instance=backend, workspace_id="demo")
    backend.scoped_record_ids = lambda **kwargs: pytest.fail("Continuation rescanned scope")
    second = reopened.list_records(
        scope="workspace", scope_id="demo", limit=128, cursor=page["next_cursor"]
    )
    assert len(second["records"]) == 128
    assert not {r["record_id"] for r in page["records"]} & {
        r["record_id"] for r in second["records"]
    }
    with pytest.raises(ValueError, match="cursor scope"):
        MemoryClient(
            tmp_path, backend_instance=backend, workspace_id="demo", profile_id="other"
        ).list_records(scope="workspace", scope_id="demo", cursor=page["next_cursor"])


def test_recall_and_context_apply_same_typed_speaker_filter(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = MemoryClient(tmp_path, backend_instance=backend, workspace_id="demo")
    for speaker in ("Ada", "Bob"):
        client.remember(
            f"{speaker} reported the date.",
            requested_block_id="workspace_episodes",
            write_mode="lossless_atomic",
            metadata={
                "source_segments": [
                    {
                        "source_id": f"turn-{speaker}",
                        "text": f"{speaker} reported the date.",
                        "speaker": speaker,
                    }
                ]
            },
        )

    hits = client.recall(
        "reported",
        logical_block_ids=["workspace_episodes"],
        include_candidates=True,
        source_speakers=["Ada"],
    )
    context = client.build_context(
        "reported",
        logical_block_ids=["workspace_episodes"],
        include_candidates=True,
        source_speakers=["Ada"],
    )

    assert len(hits) == 1 and hits[0].text == "Ada reported the date."
    assert len(context) == 1 and context[0]["text"] == "Ada reported the date."
    with pytest.raises(ValueError, match="source_speakers"):
        client.recall("reported", logical_block_ids=["workspace_episodes"], source_speakers=[])


def test_mempalace_backend_accepts_memory_query_payload() -> None:
    payload = MemPalaceBackend._query_payload(
        MemoryQuery(
            query="smoke tests",
            workspace_id="workspace-1",
            run_id="run-1",
            tags=["testing"],
            max_tokens=512,
        )
    )

    assert payload["query"] == "smoke tests"
    assert payload["workspace_id"] == "workspace-1"
    assert payload["tags"] == ["testing"]


def test_hindsight_health_does_not_report_native_without_probe_api(tmp_path) -> None:
    class UnsupportedBanks:
        pass

    class ClientWithoutProbe:
        banks = UnsupportedBanks()

    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="demo",
        config=HindsightConfig(enabled=True),
        client=ClientWithoutProbe(),
    )

    health = backend.health()

    assert health["ok"] is False
    assert health["native_ok"] is False
    assert health["strict_ok"] is False
    assert health["effective_mode"] == "unavailable"
    assert health["reason"] == "native_health_probe_unavailable"


def test_hindsight_health_starts_and_accepts_the_embedded_client_api(tmp_path) -> None:
    class EmbeddedClient:
        def __init__(self) -> None:
            self.started = False

        @property
        def client(self) -> object:
            self.started = True
            return object()

        @property
        def is_running(self) -> bool:
            return self.started

    client = EmbeddedClient()
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="demo",
        config=HindsightConfig(enabled=True, embedded=True, api_key="test-key"),
        client=client,
    )

    health = backend.health()

    assert client.started is True
    assert health["ok"] is True
    assert health["native_ok"] is True
    assert health["effective_mode"] == "native"


def test_hindsight_health_fails_fast_without_embedded_remote_provider_key(tmp_path) -> None:
    calls = 0

    def client_factory():
        nonlocal calls
        calls += 1
        raise AssertionError("embedded client must not start without required credentials")

    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="workspace-a",
        config=HindsightConfig(
            embedded=True,
            llm_provider="openai",
            api_key="",
        ),
    )
    backend._client = client_factory  # type: ignore[method-assign]

    health = backend.health()

    assert calls == 0
    assert health["native_ok"] is False
    assert health["effective_mode"] == "unavailable"
    assert health["reason"] == "embedded_llm_credentials_missing"


def test_hindsight_preflight_checks_retain_llm_and_cleans_its_ephemeral_bank(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HindsightBackend(
        repo_root=tmp_path,
        workspace_id="demo",
        config=HindsightConfig(enabled=True, base_url="http://hindsight.test"),
        reconcile_pending_writes=False,
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        backend,
        "_health",
        lambda *, verify_native_partitions: {
            "ok": True,
            "native_ok": True,
            "strict_ok": True,
            "effective_mode": "native",
            "backend": "hindsight",
        },
    )

    def external(
        path: str,
        *,
        method: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del payload
        calls.append((method, path))
        if method == "POST":
            raise HTTPError(
                "http://hindsight.test",
                500,
                "Internal Server Error",
                {},
                io.BytesIO(b'{"detail":"AuthenticationError: invalid proxy server token"}'),
            )
        return {"ok": True}

    monkeypatch.setattr(backend, "_external_json", external)
    try:
        health = backend.preflight_health()
    finally:
        backend.close()

    assert health["ok"] is False
    assert health["llm_write_ready"] is False
    assert health["reason"] == "native_llm_authentication_failed"
    assert [method for method, _path in calls] == ["PUT", "POST", "DELETE"]
    assert calls[0][1] == calls[-1][1]


def test_record_cursor_rejects_unreturnable_identity(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = MemoryClient(
        tmp_path, backend_instance=backend, workspace_id="demo", profile_id="x" * 3000
    )
    client.remember("one", requested_block_id="workspace_episodes")
    client.remember("two", requested_block_id="workspace_episodes")
    with pytest.raises(ValueError, match="cursor budget"):
        client.list_records(scope="workspace", scope_id="demo", limit=1)
