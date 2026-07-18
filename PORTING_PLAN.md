# Lethyx Porting Plan

## North Star

Turn Lethyx into a framework-neutral memory package that plugs into LangGraph,
MCP clients, REST clients, and plain Python agents without importing the
original orchestration system.

## Boundary To Keep

Memory remains advisory. Routing, proof validity, permissions, and task
completion belong to the host framework or control plane.

## Current Porting State

Completed in the first independence pass:

- `src/lethyx/` is now the primary package namespace.
- `pyproject.toml` package discovery includes only `lethyx*`.
- Console scripts point to `lethyx.memory.cli` and `lethyx.evals.memory_scenarios.cli`.
- `lethyx.MemoryClient` exposes a framework-neutral public API.
- `lethyx.adapters.LangGraphMemoryAdapter` provides version-light LangGraph-style hooks.
- `benchmarks/memory_bench` registers `lethyx*` provider ids.
- `src/lethyx/evals/memory_scenarios` runs without host orchestration imports.
- `docs/lethyx/` is the active documentation path.

Removed from the release tree:

- The legacy extraction namespace.
- Old orchestration integration snapshots.
- Compatibility bridge code that was not behind a deliberate adapter boundary.

## Public API Target

- `remember(content, metadata, evidence=None)`
- `recall(query, scope=None, limit=10, filters=None)`
- `forget(memory_id, reason)`
- `start_session(session_id, actor=None, project=None)`
- `end_session(session_id)`
- `promote(candidate, evidence)`
- `build_context(query, budget)`
- `export_graph(scope=None)`

## Adapter Shape

Framework adapters should be thin wrappers over `MemoryClient`:

- store observations from node or tool outputs;
- recall context before selected graph nodes;
- expose memory context as explicit state fields;
- bridge checkpoint/session ids into Lethyx sessions;
- keep promotion evidence-gated;
- never make retrieved memory directive by default.

## Remaining Refactor Sequence

1. Harden the `lethyx.contracts` facade and reduce copied contract surface.
2. Move host-specific phase/runner hooks into optional adapter packages.
3. Keep third-party memory bridges outside the core release package unless they are explicit optional adapters.
4. Add production MCP and REST server adapters over `MemoryClient`.
5. Harden LangGraph integration with examples and version-specific smoke tests.
6. Port tests in layers: public API, backend contracts, graph, lifecycle, proactive/BKB, integrations.
7. Keep compatibility cleanup guarded by tests so the legacy namespace cannot reappear.
