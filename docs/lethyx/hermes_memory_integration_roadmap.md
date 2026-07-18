# Lethyx → Hermes Memory Integration Roadmap

**Status:** active roadmap  
**Created:** 2026-06-27T21:16:52Z  
**Principle:** one project uses one selected Lethyx memory surface by default; cross-project/global recall is opt-in, explicit, and auditable.

## Goal

Make Lethyx the project-scoped memory facade for Hermes agents before wiring it into the admin/default profile. The system must improve recall quality without turning memory into a second control plane or leaking one project's assumptions into another.

## Non-negotiable invariants

1. **No cross-project auto-mixing.** A profile/project reads its own workspace memory by default. Global/manager memory is included only when the caller explicitly asks for it.
2. **Exactly one durable backend per run/workspace.** Use `mempalace`, `hindsight`, `mnemosyne`, or a declared Lethyx composite block plan. Do not silently mirror to multiple unrelated providers.
3. **Memory stays advisory unless promoted.** Retrieved backend content renders as `supplemental_evidence` unless it is active, scoped, verified, evidence-backed, and promotion-gated.
4. **Hindsight Docker is a Lethyx benchmark/dev resource unless explicitly repurposed.** Hermes profile memory must not assume it can use the benchmark container.
5. **Human knowledge surfaces are exports, not authority.** Obsidian and NotebookLM can summarize and visualize memory, but their outputs re-enter Lethyx only as candidate/supplemental evidence until promoted.
6. **Secrets do not leave local control.** `secret_pointer`, `non-injectable`, and equivalent privacy classes are never exported to NotebookLM bundles.

## Target architecture

```text
Hermes profile / project bot
        │
        ▼
Lethyx facade (workspace_id + backend selection + authority gate)
        │
        ├── selected durable backend: mempalace | hindsight | mnemosyne | composite
        │
        ├── scoped recall/build_context
        │     ├── current project only by default
        │     └── manager/global only with include_global=true
        │
        ├── Obsidian export
        │     ├── project notes
        │     ├── graph wikilinks
        │     ├── active/candidate/retracted indexes
        │     └── NotebookLM-safe source bundles
        │
        └── Hermes integration
              ├── MCP first, provider later
              ├── profile-specific backend/workspace mapping
              └── verification through isolation + quality evals
```

## Profile/backend policy proposal

| Hermes profile | Workspace | Default memory stance | Candidate Lethyx backend |
|---|---|---|---|
| `default` / admin | `admin` | Minimal built-in memory + explicit scoped Lethyx calls only | none initially, then Lethyx tools-only |
| project-control-plane / Albert7 | control_plane | Project-scoped engineering memory | `mempalace` first, Hindsight only if explicitly selected |
| `lethyx` / Mim | `lethyx` | Dogfood strongest Lethyx behavior | `hindsight` or declared `composite` after benchmark proof |
| `research` / Andrew | `research` | Literature/citation memory + Obsidian/NotebookLM exports | `mnemosyne` or dedicated research backend |
| `ops` / CC | `ops` | Short-lived operational memory, avoid stale automation | lightweight Lethyx or built-in/session search only |

## Phase 0 — Baseline and guardrails

- [ ] Record existing memory providers per Hermes profile without changing them.
- [ ] Record current Lethyx backend health for `mempalace`, `hindsight`, and `mnemosyne`.
- [ ] Add a small repository note that Hindsight Docker belongs to Lethyx benchmark/dev unless explicitly repurposed.
- [ ] Keep current uncommitted benchmark changes isolated from memory-integration changes.

**Verification:** `git diff --name-only`, backend health commands, and no edits outside planned files.

## Phase 1 — Obsidian/NotebookLM export from Lethyx

Build the first concrete feature in Lethyx: export typed memory blocks and graph edges to Obsidian-compatible Markdown.

### Acceptance criteria

- Export accepts a list of `MemoryBlockRecord` objects and optional `MemoryEdgeRecord` edges.
- Export writes per-project notes under `Lethyx/Projects/<workspace_id>/`.
- Each note has YAML frontmatter with Lethyx authority metadata.
- Graph edges render as Obsidian wikilinks.
- Project index groups notes by lifecycle/surface/trust.
- NotebookLM bundle is generated as a safe source file.
- Non-injectable/secret privacy classes are excluded from NotebookLM bundles by default.
- `include_global=false` excludes manager/global blocks by default.
- Tests prove workspace isolation, wikilinks, and privacy filtering.

### Planned files

- `src/lethyx/memory/obsidian_export.py`
- `tests/test_obsidian_export.py`
- `src/lethyx/memory/cli.py`
- `docs/lethyx/hermes_memory_integration_roadmap.md`

**Verification:** targeted pytest for exporter tests plus a manual export to a temporary vault.

## Phase 2 — Lethyx MCP facade

Expose project-scoped Lethyx operations as MCP tools before touching Hermes provider internals.

### Tools

```text
lethyx_health(workspace_id, backend)
lethyx_remember(workspace_id, backend, content, scope, surface, lifecycle, evidence_refs, tags)
lethyx_recall(workspace_id, backend, query, limit, include_global=false)
lethyx_build_context(workspace_id, backend, query, budget, include_global=false)
lethyx_promote(workspace_id, backend, memory_id, evidence_refs, target_surface)
lethyx_forget(workspace_id, backend, memory_id, reason)
lethyx_export_obsidian(workspace_id, backend, vault_dir, include_global=false)
```

### Rules

- `workspace_id` is required for all project operations.
- `include_global` defaults false.
- Tool outputs label all recalled items as non-directive unless promoted.
- The server must not require cloud credentials to start.
- Secret-like and non-injectable fields are redacted in tool output.

**Verification:** MCP smoke listing tools, direct tool call against temp workspace, and isolation tests.

## Phase 3 — Hermes profile integration, tools-only

Wire Lethyx MCP into selected Hermes profiles as an explicit toolset first.

### Initial mapping

- `lethyx` profile: enable Lethyx MCP against `workspace_id=lethyx`.
- control-plane profile: enable Lethyx MCP against its own control-plane workspace only after Phase 2 tests pass.
- `default` profile: keep Lethyx MCP available but do not auto-recall.

### Do not do yet

- Do not replace Hermes' built-in memory.
- Do not enable cross-project automatic recall.
- Do not point Hermes at the Lethyx benchmark Hindsight Docker unless explicitly approved.

**Verification:** start a fresh Hermes profile session, ask for a scoped recall, prove unrelated workspace memories do not appear.

## Phase 4 — Optional Hermes memory provider plugin

Only after MCP behavior is proven, consider a Hermes `lethyx` memory provider.

### Provider modes

- `tools`: expose tools only, no auto-inject.
- `context`: auto-prefetch only from current `workspace_id`.
- `hybrid`: prefetch + tools, still project scoped.

### Required config

```yaml
memory:
  provider: lethyx
lethyx:
  workspace_id: control_plane
  backend: mempalace
  memory_mode: tools
  include_global_default: false
```

**Verification:** plugin status, prompt block shows workspace/backend, recall isolation tests, and no prompt-cache-breaking mid-session config mutation.

## Phase 5 — Obsidian + NotebookLM operational loop

Use Obsidian/NotebookLM as a polished human-facing research layer.

### Pipeline

```text
Lethyx export → Obsidian vault → Google Drive sync → NotebookLM project notebook
NotebookLM artifacts → Obsidian artifacts folder → optional curated Lethyx candidate memory
```

### Rules

- One NotebookLM notebook per project/workspace.
- NotebookLM source bundle excludes secrets and non-injectable memory.
- NotebookLM summaries are imported back only as `reflection_candidate` or `supplemental_evidence`.
- Promotion to directive requires evidence and explicit gate.

**Verification:** generate vault, import source bundle manually or via NotebookLM MCP, save artifact link back into Obsidian, do not auto-promote.

## Phase 6 — Quality proof

Prove it is actually better.

### Tests

- Unit: exporter and MCP contracts.
- Integration: scoped recall through selected backend.
- Isolation: cross-project negative tests.
- Prompt quality: retrieved metadata appears in reader context.
- Memory benchmark smoke: `agent_quality` for selected backend.
- Human UX: Obsidian graph and NotebookLM bundle are usable.

### Success metrics

- No cross-project leakage in negative tests.
- Relevant recall contains source/session/time/role/tags metadata.
- Lethyx selected backend quality is not worse than raw backend for targeted smoke cases.
- Obsidian vault has navigable indexes and graph links.
- NotebookLM bundle can answer project questions with citations to exported source.

## Execution order

1. Implement Phase 1 exporter with tests.
2. Add CLI command for manual export.
3. Use exporter on a temporary vault and inspect files.
4. Design/implement MCP facade.
5. Configure `lethyx` profile only.
6. Verify isolation and recall quality.
7. Expand to the control-plane profile.
8. Only then consider default/admin integration.
