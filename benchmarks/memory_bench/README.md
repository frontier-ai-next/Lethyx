# memory_bench — Any Memory × Any Benchmark

Provider-agnostic evaluation framework for memory systems.  lethyx's three
durable backends (`mempalace`, `hindsight`, `mnemosyne`) are the first-class
providers; baseline controls (NoMemory, FullContext, BM25) are included for
reference.

## Benchmarks

| Benchmark | Kind | Metric | Categories |
|---|---|---|---|
| AgentQuality | recall | F1 | cross_session_fact, procedural_memory, conflict_resolution, preference_update, abstention |
| [LoCoMo](https://github.com/snap-research/locomo) | recall | F1 | single_hop, multi_hop, temporal, open_domain, adversarial |
| [LongMemEval](https://github.com/xiaowu0162/long-mem-eval) | recall | LLM-judge | 7 question types → 5 memory abilities |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | recall | nugget | 10 categories × 4 context-length tiers (100K–10M) |
| [MemoryArena](https://memoryarena.github.io) | agentic | task_success | web_shopping, travel_planning, progressive_search, formal_reasoning |
| [EvoMemBench](https://arxiv.org/abs/2605.18421) | agentic | task_success | in/cross-episode × knowledge/execution |

## Quick start

### 0. Fast quality smoke (no dataset downloads)

Use this first when you want a simple agent-memory quality matrix. It runs the
built-in `agent_quality` benchmark across Lethyx's three selected backends,
direct native MemPalace/Mnemosyne baselines, and pure-Python controls:

```bash
# From repo root; the script fixes PYTHONPATH for the child process.
python benchmarks/memory_bench/scripts/quality_bench.py \
  --max-episodes 1 \
  --max-probes-per-episode 2
```

Default providers:
- `lethyx-mempalace`, `lethyx-hindsight`, `lethyx-mnemosyne`
- `raw-mempalace`, `raw-mnemosyne`
- `baseline-no-memory`, `baseline-full-context`, `baseline-bm25`

Optional flags:
- `--include-native-lethyx` also adds native Lethyx MemPalace/Mnemosyne rows.
- `--include-raw-hindsight` adds direct Hindsight; it requires a configured
  Hindsight service/client and is excluded by default to keep the smoke simple.
- `--external` appends LoCoMo/LongMemEval/BEAM/MemoryArena/EvoMemBench when
  their datasets are installed locally.

Outputs use the same report format as full runs:
`runs/memory_bench_quality/<timestamp>/{report.json,report.md,report.csv,diagnostics.jsonl}`.

### 1. Install datasets

Datasets are **not** included in the repository.  Download each to
`data/memory_bench/<name>/`:

```bash
# LoCoMo
git clone https://github.com/snap-research/locomo /tmp/locomo
cp /tmp/locomo/data/locomo10.json data/memory_bench/locomo/

# LongMemEval (cleaned release)
git clone https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned /tmp/lme
cp /tmp/lme/longmemeval_s_cleaned.json data/memory_bench/longmemeval/longmemeval_s.json

# BEAM
# Either clone mohammadtavakoli78/BEAM and expose its upstream chats/ directory:
git clone https://github.com/mohammadtavakoli78/BEAM /tmp/beam
cp -r /tmp/beam/chats data/memory_bench/beam/
# or symlink data/memory_bench/beam -> a local BEAM checkout.

# MemoryArena
# Clone/download the HF dataset and place JSONL domains under tasks/:
git clone https://huggingface.co/datasets/ZexueHe/memoryarena /tmp/memoryarena
mkdir -p data/memory_bench/memoryarena/tasks
cp -r /tmp/memoryarena/* data/memory_bench/memoryarena/tasks/

# EvoMemBench — public repo currently says "Coming soon"; see https://arxiv.org/abs/2605.18421
# When released, place files under data/memory_bench/evomembench/
```

Check availability:

```bash
python -m memory_bench datasets check
# `python -m memory_bench datasets` is equivalent.
```

Current public sources used by the local loader:
- LoCoMo: `snap-research/locomo`, copy `data/locomo10.json` to `data/memory_bench/locomo/locomo10.json`.
- LongMemEval: Hugging Face `xiaowu0162/longmemeval-cleaned`; copy `longmemeval_s_cleaned.json` as `longmemeval_s.json` (optionally `longmemeval_m_cleaned.json` as `longmemeval_m.json`).
- BEAM: `mohammadtavakoli78/BEAM`; the loader supports the upstream `chats/<tier>/<id>/chat.json` + `probing_questions/probing_questions.json` layout.
- MemoryArena: Hugging Face `ZexueHe/memoryarena`; the loader supports its `*.jsonl` task files under `data/memory_bench/memoryarena/tasks/<domain>/`.
- EvoMemBench: the public `dsail-memory/evomembench` repository currently says “Coming soon”; without released data the loader emits explicit stub episodes.

### 2. Configure `.env`

```
LLM_API_KEY=sk-...
LLM_BASE_URL=https://gateway.frontierai.ru/v1
LLM_MODEL=gemma-4-31B-it
MEMORY_BENCH_JUDGE_MODEL=gemma-4-31B-it
MEMORY_BENCH_WORKERS=20
```

### 3. Run

```bash
# Full suite (all providers × all benchmarks in the checked-in suite)
python -m memory_bench run --suite memory_bench/suite.toml

# One provider, one benchmark, one episode (smoke test)
python -m memory_bench run --provider lethyx:mempalace --benchmark locomo --max-episodes 1

# BM25 baseline on BEAM 100K tier
python -m memory_bench run --provider bm25 --benchmark beam --max-episodes 3

# List registered components
python -m memory_bench list

# Re-render report from existing results
python -m memory_bench report runs/memory_bench/2026…

# Fast full matrix on this workstation (20 workers, all providers, all benchmarks)
MEMORY_BENCH_REASONING_EFFORT=low \
python -m memory_bench.scripts.full_bench \
  --workers 20 \
  --reader-model gemma-4-31B-it \
  --judge-model gemma-4-31B-it \
  --reader-max-tokens 2048
```

### Reasoning-model settings

For answer/judge isolation benchmarks, reasoning is not part of the memory system under test; it mostly burns completion budget before the final answer.  The OpenAI-compatible backends used here do not expose a reliable universal “off” switch:

- Current local publication-style runs use `gemma-4-31B-it` for both reader and judge. Do not leave `gpt-4o`, `gpt-oss`, or other fallback models in generated suites.
- If the configured Gemma model is unavailable, the run should fail closed. Do not substitute another reader/judge model and then compare the rows as the same experiment.

Do not compare providers with different reader/judge models or reasoning settings.  Keep reader and judge fixed across all memory providers; only the memory backend should vary.

### 4. Output

```
runs/memory_bench/<timestamp>/
  manifest.json            # full reproducibility manifest
  report.json              # schema memory_bench/report/v1
  report.md                # markdown summary tables
  report.csv               # flat CSV of all category rollups
  diagnostics.jsonl         # Lethyx cases below best peer, with why/context traces
  results.jsonl            # all CaseResult rows
  <provider-id>/
    <benchmark>.jsonl      # per-provider per-benchmark rows (resumable)
```

## Evaluation mode: memory isolation

The **reader LLM** (`LLMResponder`) and **judge LLM** are fixed constants across
all providers.  Only the memory backend varies.  This isolates memory quality
as the single experimental variable.

## Running tests

```bash
# From repo root
PYTHONPATH=benchmarks python -m pytest benchmarks/memory_bench/tests -v
```

Tests use in-memory fixtures — no datasets or network required.

## Citation

If you publish results generated with this framework, please cite the upstream
benchmark papers and include the `manifest.json` from your run (contains dataset
checksums, model version, seed, and git SHA).
