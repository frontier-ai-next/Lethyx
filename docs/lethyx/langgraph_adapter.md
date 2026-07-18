# Lethyx LangGraph Adapter

Lethyx should be used as long-term, advisory memory beside LangGraph's own
short-term checkpointing. LangGraph keeps graph state and thread checkpoints;
Lethyx stores reusable cross-thread evidence, preferences, lessons, and
observations.

## Contract

- Recalled memory is supplemental evidence, not instructions.
- `thread_id` and `user_id` are read from `config["configurable"]` when present.
- Manual hook usage can also pass a runtime-like object with `.config` and `.context`.
- The adapter writes bounded context into state keys:
  - `memory_context`: structured recalled items.
  - `memory_context_text`: prompt-safe text headed as non-directive evidence.
- Node outputs are saved only when:
  - state contains `lethyx_save=True`;
  - output contains `save_to_memory=True`; or
  - `wrap_node(..., save_output=True)` is used.

## Minimal Pattern

```python
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from lethyx import MemoryClient
from lethyx.adapters import LangGraphMemoryAdapter


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    memory_context: list[dict[str, Any]]
    memory_context_text: str
    answer: str
    summary: str
    save_to_memory: bool


memory = MemoryClient(repo_root=".", workspace_id="my-agent", backend="mempalace")
adapter = LangGraphMemoryAdapter(memory)


def agent_node(state: AgentState, config: dict[str, Any]) -> AgentState:
    memory_text = state.get("memory_context_text", "")
    question = state["messages"][-1]["content"]
    answer = f"{memory_text}\n\nAnswer the user question: {question}"
    return {
        "answer": answer,
        "summary": f"Answered: {question}",
        "save_to_memory": True,
    }


builder = StateGraph(AgentState)
builder.add_node("agent", adapter.wrap_node(agent_node, node_name="agent"))
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile(checkpointer=InMemorySaver())
result = graph.invoke(
    {"messages": [{"role": "user", "content": "How should I test this refactor?"}]},
    {"configurable": {"thread_id": "thread-1", "user_id": "user-1"}},
)
```

## Local Experiment

Run the dependency-free adapter experiment:

```powershell
$env:PYTHONPATH = "src"
python -m lethyx.evals.langgraph_experiment
```

The experiment verifies that:

- a wrapped node receives recalled Lethyx context;
- node output is saved with LangGraph thread metadata;
- a second thread can recall useful prior memory;
- context is explicitly labelled non-directive.

## Real LLM Smoke

When `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` are configured, run:

```powershell
$env:PYTHONPATH = "src;benchmarks"
python -m lethyx.evals.langgraph_llm_experiment
```

This executes a real `StateGraph` with two LLM-backed nodes:

- `planner`: recalls Lethyx context and proposes the release gate.
- `reviewer`: recalls Lethyx context and checks the planner output.

The smoke passes only when the agents use the recalled validation memory,
preserve the non-directive memory authority rule, save their summaries, and a
second thread can recall the run.
