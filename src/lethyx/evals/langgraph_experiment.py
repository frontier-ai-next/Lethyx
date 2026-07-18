"""Self-contained LangGraph adapter experiment.

The experiment intentionally avoids importing LangGraph so it can run in a
minimal Lethyx development environment. It exercises the same callable shape a
LangGraph ``StateGraph`` node uses: ``node(state, config) -> state_patch``.
"""

from dataclasses import dataclass, field, replace
from typing import Any

from lethyx import MemoryClient
from lethyx.adapters import LangGraphMemoryAdapter
from lethyx.memory.backends.contracts import MemoryBlockRecord, block_record_to_memory_hit


class ExperimentBackend:
    """Tiny in-memory backend with Lethyx contract semantics."""

    def __init__(self) -> None:
        self.blocks: dict[str, MemoryBlockRecord] = {}

    def retain_block(self, record: MemoryBlockRecord) -> dict[str, Any]:
        self.blocks[record.block_id] = record
        return {"ok": True, "block_id": record.block_id, "backend": "experiment"}

    def recall_evidence(self, query: Any) -> list[Any]:
        workspace_id = str(getattr(query, "workspace_id", "") or "")
        q = str(getattr(query, "query", "") or "").lower()
        terms = [part for part in q.split() if len(part) > 2]
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
        del block_id
        return []


@dataclass(frozen=True)
class LangGraphExperimentResult:
    ok: bool
    first_answer: str
    second_context_text: str
    saved_count: int
    failures: list[str] = field(default_factory=list)


def run_langgraph_adapter_experiment() -> LangGraphExperimentResult:
    """Run a two-thread memory experiment against the LangGraph adapter."""

    backend = ExperimentBackend()
    client = MemoryClient(
        backend_instance=backend,
        workspace_id="langgraph-demo",
    )
    adapter = LangGraphMemoryAdapter(client, default_limit=4)
    client.remember(
        "The user prefers pytest smoke tests before risky refactors.",
        title="testing preference",
        tags=["preference", "testing"],
    )

    def planner_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        del config
        memory_text = str(state.get("memory_context_text") or "")
        if "pytest smoke" in memory_text:
            answer = "Plan: run pytest smoke tests before the refactor."
        else:
            answer = "Plan: inspect the task first."
        return {
            "answer": answer,
            "summary": answer,
            "save_to_memory": True,
        }

    wrapped = adapter.wrap_node(planner_node, node_name="planner")
    first = wrapped(
        {"messages": [{"role": "user", "content": "How should I test this refactor?"}]},
        {"configurable": {"thread_id": "thread-a", "user_id": "user-1"}},
    )
    second = adapter.before_node(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What did the planner decide about refactor testing?",
                }
            ]
        },
        config={"configurable": {"thread_id": "thread-b", "user_id": "user-1"}},
    )
    context_text = str(second.get("memory_context_text") or "")
    failures: list[str] = []
    if "pytest smoke tests" not in str(first.get("answer") or ""):
        failures.append("first answer did not use recalled preference")
    if "thread-a" not in {
        str(block.metadata.get("thread_id") or "") for block in backend.blocks.values()
    }:
        failures.append("saved memory did not include LangGraph thread metadata")
    if "non-directive" not in context_text:
        failures.append("context text is missing non-directive label")
    if "refactor" not in context_text.lower():
        failures.append("second thread did not recall prior planner output")
    return LangGraphExperimentResult(
        ok=not failures,
        first_answer=str(first.get("answer") or ""),
        second_context_text=context_text,
        saved_count=len(backend.blocks),
        failures=failures,
    )


if __name__ == "__main__":
    result = run_langgraph_adapter_experiment()
    print(result)
    raise SystemExit(0 if result.ok else 1)
