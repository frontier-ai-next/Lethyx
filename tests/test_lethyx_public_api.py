from dataclasses import replace
from typing import Any

from lethyx import MemoryClient, recall, remember
from lethyx.adapters import LangGraphMemoryAdapter
from lethyx.memory.backends.base import MemoryQuery
from lethyx.memory.backends.contracts import MemoryBlockRecord, block_record_to_memory_hit
from lethyx.memory.backends.hindsight import HindsightBackend
from lethyx.memory.backends.mempalace import MemPalaceBackend
from lethyx.memory.hindsight.config import HindsightConfig


class FakeBackend:
    def __init__(self) -> None:
        self.blocks: dict[str, MemoryBlockRecord] = {}

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.blocks[record.block_id] = record
        return {"ok": True, "block_id": record.block_id, "backend": "fake"}

    def recall_evidence(self, query: Any) -> list[Any]:
        q = str(getattr(query, "query", "") or "").lower()
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

    def edges_for(self, block_id: str) -> list[Any]:
        return []


def test_public_client_remember_and_recall_are_framework_neutral() -> None:
    backend = FakeBackend()
    client = MemoryClient(backend_instance=backend, workspace_id="demo")

    saved = client.remember("Use evidence-gated promotion for agent memory.", tags=["policy"])
    hits = client.recall("promotion")

    assert saved.saved is True
    assert saved.backend == "fake"
    assert hits[0].metadata["workspace_id"] == "demo"
    assert hits[0].metadata["surface"] == "supplemental_evidence"


def test_convenience_functions_accept_injected_backend() -> None:
    backend = FakeBackend()

    saved = remember("LangGraph nodes can request non-directive recall.", backend_instance=backend)
    hits = recall("LangGraph", backend_instance=backend)

    assert saved.saved is True
    assert [hit.text for hit in hits] == ["LangGraph nodes can request non-directive recall."]


def test_langgraph_adapter_returns_state_patch_and_saves_when_requested() -> None:
    backend = FakeBackend()
    adapter = LangGraphMemoryAdapter(MemoryClient(backend_instance=backend), context_key="mem")

    adapter.client.remember("Debugger node should recall flaky test lessons.")
    patch = adapter.before_node({"task": "debug flaky tests"})
    saved = adapter.after_node(
        {"lethyx_save": True},
        {"summary": "The verify node found a regression test.", "save_to_memory": True},
        node_name="verify",
    )

    assert patch["mem"]
    assert saved is not None and saved.saved is True
    assert any("regression test" in block.content for block in backend.blocks.values())


def test_mempalace_backend_accepts_memory_query_payload() -> None:
    payload = MemPalaceBackend._query_payload(
        MemoryQuery(
            query="smoke tests",
            workspace_id="workspace-1",
            run_id="run-1",
            phase_id="verify",
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
