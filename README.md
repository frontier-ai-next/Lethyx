# Lethyx

Lethyx is a framework-neutral memory layer for agent systems. It packages typed
agent memory blocks, durable backend adapters, lifecycle policy, graph links,
promotion gates, and evaluation harnesses behind a small public API.

The goal is simple: let an agent framework such as LangGraph, an MCP server, a
REST service, or a plain Python agent add useful long-term memory without making
memory a second control plane.

## Install

```bash
python -m pip install lethyx
```

For development from a clone:

```bash
python -m pip install -e .[dev,mcp]
```

## Quick Start

```python
from lethyx import MemoryClient
from lethyx.adapters import LangGraphMemoryAdapter

memory = MemoryClient(repo_root=".", workspace_id="my-agent", backend="mempalace")

memory.remember(
    "When fixing auth regressions, check refresh-token rotation tests first.",
    tags=["auth", "regression-tests"],
)

context = memory.build_context("debug auth refresh failure")
adapter = LangGraphMemoryAdapter(memory)
```

For a complete LangGraph wrapper pattern and real LLM smoke command, see
`docs/lethyx/langgraph_adapter.md` and `examples/langgraph/minimal_agent.py`.

## Public API Target

- `remember(content, metadata, evidence=None)`
- `recall(query, scope=None, limit=10, filters=None)`
- `forget(memory_id, reason)`
- `start_session(session_id, actor=None, project=None)`
- `end_session(session_id)`
- `promote(candidate, evidence)`
- `build_context(query, budget)`
- `export_graph(scope=None)`

The first public client lives in `lethyx.MemoryClient`. It is intentionally thin:
host frameworks own routing, task control, permissions, and final decisions;
Lethyx owns retention, recall, evidence, lifecycle, and graph metadata.

## What Is Inside

- `src/lethyx/memory/`: core memory implementation.
- `src/lethyx/contracts/`: typed evidence and memory-contract support.
- `src/lethyx/evals/memory_scenarios/`: scenario harness for memory invariants.
- `src/lethyx/adapters/`: optional framework adapters, starting with LangGraph-style hooks.
- `examples/langgraph/`: minimal optional LangGraph integration example.
- `benchmarks/memory_bench/`: provider-agnostic memory benchmark suite.
- `docs/lethyx/`: architecture, methodology, lifecycle, graph, durable backend, and eval docs.


## Backends

Lethyx supports one selected durable backend per run or workspace:

- `mempalace`
- `hindsight`
- `mnemosyne`

Set `LETHYX_BACKEND=mempalace|hindsight|mnemosyne|composite`, or pass
`backend=` to `MemoryClient`. Legacy extraction env names were removed from the
new package surface; use `LETHYX_*` for new integrations.

## MCP + Hermes Integration

Lethyx can run as a stdio MCP server for Hermes or any MCP-compatible host:

```bash
python -m pip install "lethyx[mcp]"
```

```bash
lethyx-mcp
# or:
python -m lethyx.memory.mcp_server
```

The server exposes scoped tools for `health`, `remember`, `recall`,
`build_context`, `promote`, `forget`, `add_edge`, and `export_obsidian`.
Hermes prefixes them with the configured server name, e.g.
`mcp_lethyx_recall` when the server is named `lethyx`.

Recommended Hermes MCP config keeps memory project-scoped:

```bash
hermes mcp add lethyx \
  --command lethyx-mcp \
  --env LETHYX_REPO_ROOT="$HOME/.lethyx" \
        LETHYX_WORKSPACE="my_project" \
        LETHYX_BACKEND="mempalace"
```

Use a different `LETHYX_WORKSPACE` for each project profile. Global memory is
not included unless a tool call explicitly passes `include_global=true`.

Hermes can also use Lethyx as a native memory provider after installing the
package:

```bash
python -m lethyx.memory.cli hermes-provider-install --json
hermes config set memory.provider lethyx
hermes config set memory.memory_enabled true
```

The provider stores runtime data under the Hermes home by default and does not
require repository-local paths.

## Development Status

This repository has completed the first independence pass:

1. The releasable package namespace is `lethyx`.
2. Packaging entrypoints point to `lethyx`.
3. A public `MemoryClient` and LangGraph-compatible adapter exist, including
   node wrappers and a dependency-free adapter experiment.
4. The benchmark registry exposes `lethyx`, `lethyx-mempalace`,
   `lethyx-hindsight`, `lethyx-mnemosyne`, and `lethyx-composite`.
5. The old extraction namespace has been removed from the release tree.

See `RELEASE_PLAN.md` for the remaining release work.
