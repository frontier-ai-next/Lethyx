"""Minimal LangGraph + Lethyx example.

Install optional deps first:

    python -m pip install -e .[langgraph]

Then run:

    python examples/langgraph/minimal_agent.py
"""

from typing import Any, TypedDict

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - optional example dependency.
    raise SystemExit(
        "Install optional dependency with: python -m pip install -e .[langgraph]"
    ) from exc

from lethyx import MemoryClient
from lethyx.adapters import LangGraphMemoryAdapter


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    memory_context: list[dict[str, Any]]
    memory_context_text: str
    answer: str
    summary: str
    save_to_memory: bool


def build_graph():
    memory = MemoryClient(repo_root=".", workspace_id="langgraph-example", backend="mempalace")
    adapter = LangGraphMemoryAdapter(memory)
    memory.remember(
        "The user prefers quick smoke tests before broad refactors.",
        tags=["preference", "testing"],
    )

    def agent_node(state: AgentState, config: dict[str, Any]) -> AgentState:
        del config
        question = state["messages"][-1]["content"]
        memory_text = state.get("memory_context_text", "")
        answer = (
            f"Use this supplemental memory while deciding:\n{memory_text}\n\nQuestion: {question}"
        )
        return {
            "answer": answer,
            "summary": f"Answered with Lethyx context for: {question}",
            "save_to_memory": True,
        }

    builder = StateGraph(AgentState)
    builder.add_node("agent", adapter.wrap_node(agent_node, node_name="agent"))
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "How should I test a risky refactor?"}]},
        {"configurable": {"thread_id": "demo-thread", "user_id": "demo-user"}},
    )
    print(result["answer"])
