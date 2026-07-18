# Memory Scenario Harness / Memory Audit Lab

Offline declarative tests for Lethyx memory injection, BKB filtering, MemPalace, and host-agent dedup.

## CLI

```bash
uv run memory-scenarios list
uv run memory-scenarios list-llm
uv run memory-scenarios run --all --verbose
uv run memory-scenarios run --scenario 01_bkb_filtering
uv run memory-scenarios run --phase verify
uv run memory-scenarios run-llm --all
uv run memory-scenarios inspect --scenario 01_bkb_filtering --open-report
```

Deterministic `run --all` excludes `scenarios/llm/` (use `run-llm` for behavioral LLM checks).

Reports: `.mrt/memory_scenarios/<scenario_id>/` (`report.md`, `result.json`, per-step prompts and injection reports).

## Pytest

```bash
uv run pytest -q tests/test_memory_scenarios.py
uv run pytest -q -m "memory_scenario or memory_contract"
uv run pytest -q tests/test_workspaces_test_memory_live.py
```

## Opt-in real LLM (behavioral)

```bash
LETHYX_MEMORY_LLM_REAL_TESTS=1 uv run pytest -q tests/test_memory_scenarios_llm_optional.py -m memory_llm_real
```

Requires `LLM_API_KEY` or `OPENAI_API_KEY` (see `Lethyx/env.py`).

## Web UI

Open **Memory Lab** at `/memory-lab` when running `uv run bridge` with the built web app.

API: `GET /api/memory/scenarios`, `POST /api/memory/scenarios/run`, `GET /api/memory/scenarios/<id>/report`, `GET /api/memory/scenarios/latest`.

## Fixtures

Committed workspace seed: `workspaces/test/`. Memory state overlays: `tests/fixtures/memory_scenarios/`. Scenarios: `lethyx/evals/memory_scenarios/scenarios/*.yaml`.
