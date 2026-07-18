# Lethyx Release Plan

## Release Goal

Ship Lethyx as an installable agent-memory package with a stable Python API,
optional LangGraph/MCP/REST adapters, selectable durable backends, and a release
gate that proves memory stays advisory, scoped, evidence-aware, and useful.

## Phase 1: Package Boundary

Status: complete.

- Keep `lethyx` as the only packaged namespace.
- Remove runtime dependency on the old compatibility namespace.
- Keep deleted compatibility sources out of the release tree.
- Rename or quarantine any future orchestration-specific snapshots under explicit adapter folders.
- Define public import guarantees in `lethyx.__all__`, `lethyx.api`, and `lethyx.adapters`.

Exit criteria:

- `pip install -e .` exposes `lethyx` only.
- `python -m lethyx.memory.cli --help` works.
- Static import scan returns no legacy runtime imports under `src/lethyx`.
- Deterministic scenario harness passes `run --all` with local fallback fixtures.

## Phase 2: Core API Hardening

Status: started.

- Finish `MemoryClient` methods for `forget`, `promote`, `build_context`, and `export_graph`.
- Add typed request/response models for public API calls.
- Decide stable defaults for observation lifecycle, workspace scope, trust, and token budgets.
- Add clear exceptions for policy failures and backend unavailability.
- Provide sync API first; add async wrappers only after sync contracts settle.

Exit criteria:

- Public API tests use fake backends and all three real backends behind optional markers.
- API docs include copy-paste examples for plain Python and agent frameworks.

## Phase 3: Backend Contracts

Status: copied contracts exist, needs migration.

- Port backend contract tests from copied fixtures to `lethyx` imports.
- Standardize `LETHYX_*` env vars across MemPalace, Hindsight, and Mnemosyne adapters.
- Remove legacy backend modes from docs and errors.
- Confirm exactly one selected durable backend per run/workspace unless composite block mode is explicitly requested.
- Add backend health matrix output for CLI and REST.

Exit criteria:

- `mempalace`, `hindsight`, and `mnemosyne` pass retain/recall/lifecycle/graph/privacy contract tests.
- Backend health failures are visible and fail closed when configured.

## Phase 4: LangGraph Adapter

Status: initial version exists.

- Add examples for wrapping graph nodes with `before_node` and `after_node`.
- Add optional integration with LangGraph store/checkpointer concepts without requiring LangGraph in core installs.
- Preserve state field names such as `memory_context` and never inject recalled text as instructions by default.
- Add one minimal LangGraph smoke app under `examples/langgraph/`.

Exit criteria:

- Example agent recalls useful context across sessions.
- Recalled memory is labelled as supplemental evidence in state.
- No LangGraph dependency is required for `pip install lethyx`.

## Phase 5: MCP And REST Adapters

Status: planned.

- Build `lethyx-mcp` server over `MemoryClient` with tools for remember, recall, forget, promote, graph export, and health.
- Build optional FastAPI REST bridge with the same contracts.
- Keep auth and remote persistence opt-in.
- Add JSON schemas for tool inputs and outputs.

Exit criteria:

- MCP inspector lists tools and can round-trip remember/recall.
- REST OpenAPI docs match the Python API models.

## Phase 6: Evaluation And Benchmarks

Status: benchmark suite and deterministic scenario harness are migrated to `lethyx`.

- Run static ablations: no memory, retrieval-only, proactive directive, supplemental evidence, Hindsight candidates, triggers/utility, backend variants.
- Keep metrics: first-action alignment, directive precision/recall, contamination rate, retrieval leakage, evidence coverage, token ROI, false promotion rate, candidate yield.
- Add benchmark configs for LoCoMo, LongMemEval, BEAM, MemoryArena, and EvoMemBench where datasets are available.

Exit criteria:

- Release report is generated under `.memory/drive/reports/`.
- Benchmark CLI can run `lethyx:mempalace`, `lethyx:hindsight`, and `lethyx:mnemosyne`.

## Phase 7: Safety And Privacy

Status: policies exist, needs release gating.

- Prove directive memory requires active lifecycle, verified trust, accepted promotion gate, and scope fit.
- Prove retrieval cannot become directive without promotion.
- Prove workspace isolation and run-state expiry.
- Add redaction and secret-pointer tests for public API, MCP, REST, and reports.

Exit criteria:

- Contamination and leakage tests are release blockers.
- Public docs explain what Lethyx will not store by default.

## Phase 8: Documentation And Release

Status: started.

- Write `docs/lethyx/quickstart.md`.
- Write adapter docs for Python, LangGraph, MCP, and REST.
- Add migration notes for users coming from the extraction snapshot.
- Add package metadata, classifiers, license, and changelog.
- Build wheel/sdist and test install in a clean virtual environment.

Exit criteria:

- New user can install, remember, recall, and inspect health in under 10 minutes.
- Release tag includes test report, benchmark report, and API stability note.
