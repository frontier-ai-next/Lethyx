from dataclasses import replace
from typing import Any

from lethyx import MemoryClient
from lethyx.adapters import LangGraphMemoryAdapter
from lethyx.evals.langgraph_experiment import run_langgraph_adapter_experiment
from lethyx.evals.langgraph_llm_experiment import _chat_json
from lethyx.memory.backends.contracts import MemoryBlockRecord, block_record_to_memory_hit


class FakeBackend:
    def __init__(self) -> None:
        self.blocks: dict[str, MemoryBlockRecord] = {}

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.blocks[record.block_id] = record
        return {"ok": True, "block_id": record.block_id, "backend": "fake"}

    def recall_evidence(self, query: Any) -> list[Any]:
        workspace_id = str(getattr(query, "workspace_id", "") or "")
        terms = [
            term for term in str(getattr(query, "query", "") or "").lower().split() if len(term) > 2
        ]
        hits = []
        for block in self.blocks.values():
            if workspace_id and block.workspace_id != workspace_id:
                continue
            haystack = f"{block.title}\n{block.content}\n{' '.join(block.tags)}".lower()
            if not terms or any(term in haystack for term in terms):
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


class RuntimeLike:
    def __init__(self) -> None:
        self.config = {"configurable": {"thread_id": "runtime-thread"}}
        self.context = {"user_id": "runtime-user"}


def test_langgraph_before_node_uses_messages_and_config_metadata() -> None:
    backend = FakeBackend()
    client = MemoryClient(backend_instance=backend, workspace_id="demo")
    adapter = LangGraphMemoryAdapter(client)
    client.remember("Use pytest smoke tests before refactors.", tags=["testing"])

    patch = adapter.before_node(
        {"messages": [{"role": "user", "content": "How should I test refactors?"}]},
        config={"configurable": {"thread_id": "thread-1", "user_id": "user-1"}},
    )

    assert patch["memory_context"]
    assert "non-directive" in patch["memory_context_text"]
    assert patch["lethyx_metadata"]["thread_id"] == "thread-1"
    assert patch["lethyx_metadata"]["user_id"] == "user-1"


def test_langgraph_metadata_accepts_runtime_like_object() -> None:
    adapter = LangGraphMemoryAdapter(MemoryClient(backend_instance=FakeBackend()))

    metadata = adapter.metadata_from_state({}, config=RuntimeLike())

    assert metadata.thread_id == "runtime-thread"
    assert metadata.user_id == "runtime-user"


def test_langgraph_wrap_node_injects_context_and_saves_output() -> None:
    backend = FakeBackend()
    client = MemoryClient(backend_instance=backend, workspace_id="demo")
    adapter = LangGraphMemoryAdapter(client)
    client.remember("Prefer direct unit tests for parser changes.", tags=["testing"])

    def node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        assert config["configurable"]["thread_id"] == "thread-2"
        assert "direct unit tests" in state["memory_context_text"]
        return {
            "answer": "Use direct unit tests.",
            "summary": "Parser plan: use direct unit tests.",
            "save_to_memory": True,
        }

    wrapped = adapter.wrap_node(node, node_name="planner")
    result = wrapped(
        {"messages": [{"role": "user", "content": "parser change test plan"}]},
        {"configurable": {"thread_id": "thread-2", "user_id": "user-2"}},
    )

    assert result["answer"] == "Use direct unit tests."
    assert result["lethyx_last_save"]["saved"] is True
    saved_blocks = list(backend.blocks.values())
    assert any(block.metadata.get("thread_id") == "thread-2" for block in saved_blocks)


def test_langgraph_adapter_experiment_cross_thread_recall() -> None:
    result = run_langgraph_adapter_experiment()

    assert result.ok, result.failures
    assert "pytest smoke tests" in result.first_answer
    assert "non-directive" in result.second_context_text
    assert result.saved_count >= 2


def test_langgraph_workspace_isolation() -> None:
    shared_backend = FakeBackend()
    a = LangGraphMemoryAdapter(
        MemoryClient(backend_instance=shared_backend, workspace_id="workspace-a")
    )
    b = LangGraphMemoryAdapter(
        MemoryClient(backend_instance=shared_backend, workspace_id="workspace-b")
    )

    a.client.remember("Workspace A uses API A.", tags=["workspace"])
    b.client.remember("Workspace B uses API B.", tags=["workspace"])

    patch_a = a.before_node({"query": "Which API does this workspace use?"})
    patch_b = b.before_node({"query": "Which API does this workspace use?"})

    assert "API A" in patch_a["memory_context_text"]
    assert "API B" not in patch_a["memory_context_text"]
    assert "API B" in patch_b["memory_context_text"]
    assert "API A" not in patch_b["memory_context_text"]


def test_langgraph_llm_parser_preserves_raw_fallback() -> None:
    class Message:
        content = '{"answer": "", "summary": ""}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs: Any) -> Response:
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    parsed = _chat_json(Client(), model="model", system="system", user="user")

    assert parsed["answer"] == ""
    assert parsed["_raw"] == '{"answer": "", "summary": ""}'
