# Hermes Memory Profile Matrix

Release-gate benchmark for comparing Hermes memory profiles and Lethyx backend configurations on clean profiles.

## What it measures

- Clean profile setup availability.
- Memory-only answer latency.
- Simple keyword correctness for seeded project facts.
- Obvious cross-repo leakage on negative probes.
- Backend-specific Lethyx behavior across Hindsight, MemPalace, Mnemosyne, and Composite.

## Providers

Supported provider IDs:

- `builtin` — built-in Hermes `MEMORY.md` only.
- `hindsight` — direct Hermes Hindsight provider against local external service `http://127.0.0.1:8888`.
- `lethyx-hindsight` — Hermes Lethyx provider with `backend=hindsight`.
- `lethyx-mempalace` — Hermes Lethyx provider with `backend=mempalace`.
- `lethyx-mnemosyne` — Hermes Lethyx provider with `backend=mnemosyne`.
- `lethyx-composite` — Hermes Lethyx provider with `backend=composite`, active blocks for MemPalace, Hindsight, and Mnemosyne.

Honcho should be added only when credentials or a self-hosted config are available; missing Honcho setup is an availability result, not a quality failure.

## Scenarios/questions

The current expanded matrix has 6 scenarios × 4 questions = 24 questions per provider:

1. `synthetic_fastapi_tasks`
2. `synthetic_typer_reports`
3. `synthetic_vite_dashboard`
4. `lymanny/FastAPI-CRUD-Todo`
5. `alisterburt/minimal-rich-typer-cli`
6. `Daaviddev/vite-dashboard-starter`

Question types include architecture recall, path/command recall, testing pitfall recall, and negative leakage probes.

## Run

Full matrix across all built-in/direct/Lethyx configs:

```bash
PYTHONPATH=src python benchmarks/hermes_memory_profiles/run_profile_matrix.py \
  --providers builtin,hindsight,lethyx-hindsight,lethyx-mempalace,lethyx-mnemosyne,lethyx-composite
```

Lethyx configs only:

```bash
PYTHONPATH=src python benchmarks/hermes_memory_profiles/run_profile_matrix.py \
  --providers lethyx-hindsight,lethyx-mempalace,lethyx-mnemosyne,lethyx-composite
```

Fast smoke across all configs:

```bash
PYTHONPATH=src python benchmarks/hermes_memory_profiles/run_profile_matrix.py \
  --providers builtin,hindsight,lethyx-hindsight,lethyx-mempalace,lethyx-mnemosyne,lethyx-composite \
  --max-questions 4
```

Outputs:

- `runs/hermes_memory_profiles/<timestamp>/results.jsonl`
- `runs/hermes_memory_profiles/<timestamp>/summary.json`
- `runs/hermes_memory_profiles/<timestamp>/report.md`

## Public repo facts used

- `lymanny/FastAPI-CRUD-Todo`: README confirms FastAPI + SQLite + SQLAlchemy + Pydantic; run command `uvicorn main:app --reload`; docs at `http://127.0.0.1:8000/docs`; structure includes `database.py`, `models.py`, `schemas.py`, `routers/todo.py`, `main.py`.
- `alisterburt/minimal-rich-typer-cli`: README confirms a minimal pip-installable Typer/Rich CLI with `pip install -e .` and two entry programs: `single_command` and `multi_command`.
- `Daaviddev/vite-dashboard-starter`: README confirms React + TypeScript + Vite + Tailwind CSS + ShadCn + Zustand; install via `pnpm install`, dev via `pnpm dev`, build via `pnpm build`; state store at `src/store/globalStore.ts`, routes at `src/AppRoutes.tsx`, utility `cn` at `src/lib/utils.ts`.

## Interpretation rules

- Separate setup availability from answer quality.
- Do not treat raw Hermes `state.db` as active memory.
- For Hindsight, allow an indexing delay between retain and recall.
- For Lethyx, inspect diagnostics to distinguish native backend hits from metadata/index fallback.
- Composite must be interpreted as fanout/merge behavior, not a replacement for single-backend quality.
