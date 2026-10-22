# UM-MEM Agent Guide

UM-MEM is a framework-neutral agent memory package. Treat `src/um_mem` as the only product namespace.

## Read First

- `README.md`, `PORTING_PLAN.md`, and `RELEASE_PLAN.md` for product status and release work.
- `docs/um-mem/memory_architecture.md` for current authority and backend rules.
- `docs/um-mem/memory_methodology.md` for backend-neutral memory invariants.
- `docs/um-mem/memory_eval_protocol.md` before claiming a memory quality result.
- `src/um_mem/memory/README.md` for older structured-memory context that may still explain copied code.

## Project Invariants

- Memory is advisory. Host control planes own routing, proof validity, permissions, and task completion.
- Durable writes must go through the backend-neutral writer and exactly one selected backend per run/workspace: `mempalace`, `hindsight`, or `mnemosyne`.
- Retrieval, graph recall, and backend-native recall stay supplemental unless promoted through evidence and tribunal gates.
- Only active, curated, scoped, evidence-backed memory may become directive.
- Candidate, quarantined, superseded, deprecated, and retracted memory must not render as directive.
- Keep orchestration-specific phase/runner hooks in optional adapters, not core `um-mem`.
- Do not add cloud memory providers, remote persistence, or API-key requirements without explicit user approval.

## Development Workflow

- Prefer narrow changes that preserve the porting path: core API first, adapters second, compatibility cleanup last.
- Use existing copied tests and memory scenario harnesses as contract material while porting them to `um-mem`.
- When public behavior changes, test authority boundaries: directive contamination, retrieval leakage, workspace isolation, evidence coverage, promotion gates, lifecycle state, and backend selection.
- Keep third-party reference material outside the release tree unless it is turned into an explicit adapter or documented benchmark fixture.
- Keep docs honest about implemented behavior versus planned optional live smoke or provider integrations.

## Local Memory Setup

Keep local tool/MCP experiments out of the release tree. Use UM-MEM's packaged
MCP server or Hermes provider integration for project memory, and do not commit
machine-local wrappers, runtime stores, credentials, logs, or generated profiles.
