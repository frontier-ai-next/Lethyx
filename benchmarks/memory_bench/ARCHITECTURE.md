# memory_bench Architecture

## Layer overview

```
suite.toml / CLI
     │
core/runner.py       ← resolves registry, runs matrix, writes JSONL
     ├── MemoryProvider.namespace(ns) → MemoryStore   ← per episode isolation
     │        └── write_many(records)                 ← ingest
     │        └── retrieve(query, k)                  ← recall
     ├── Responder.answer(question, context)           ← fixed reader LLM
     ├── Judge.judge(probe, answer)                    ← metric
     └── CaseResult → JSONL → report.py
```

## Extension points

### Add a new memory provider

1. Create a file in `adapters/` (or a separate pip package).
2. Implement two classes: `YourStore` (with `write`, `write_many`, `retrieve`,
   `update`, `stats`, `close`) and `YourProvider` (with `capabilities()` and
   `namespace(ns) -> YourStore`).
3. Register it:

```python
from memory_bench.core.registry import register_provider

@register_provider("my_provider")
class MyProvider:
    name = "my_provider"
    ...
```

Or for an external package, declare the entry-point in `pyproject.toml`:

```toml
[project.entry-points."memory_bench.providers"]
my_provider = "my_package.adapters:MyProvider"
```

### Add a new benchmark

1. Create a file in `benchmarks/` implementing the `Benchmark` protocol.
2. Choose `kind = "recall"` or `kind = "agentic"`.
3. Register:

```python
from memory_bench.core.registry import register_benchmark

@register_benchmark("my_bench")
class MyBenchmark:
    name = "my_bench"
    kind = "recall"
    default_judge = "f1"

    def load(self, spec):
        # yield (Episode, [Probe]) pairs
        ...
```

Or entry-point:

```toml
[project.entry-points."memory_bench.benchmarks"]
my_bench = "my_package.benchmarks:MyBenchmark"
```

### Add a new judge

```python
from memory_bench.core.registry import register_judge
from memory_bench.core.types import Judgment

@register_judge("my_judge")
class MyJudge:
    name = "my_judge"
    def judge(self, probe, answer) -> Judgment:
        ...
```

Or entry-point group `memory_bench.judges`.

## Protocol contracts

### MemoryStore

| Method | Required | Notes |
|---|---|---|
| `write(record)` | yes | persist one MemoryRecord |
| `write_many(records)` | no | default loops over write() |
| `retrieve(query, k, max_tokens, filters)` | yes | return ≤k RetrievedItem |
| `update(record_id, **changes)` | no | only when capabilities.supports_update |
| `stats()` | yes | diagnostic dict |
| `close()` | yes | flush and free resources |

### MemoryProvider

| Method | Required | Notes |
|---|---|---|
| `capabilities()` | yes | returns ProviderCapabilities |
| `namespace(ns)` | yes | fresh isolated MemoryStore |

### ProviderCapabilities flags

| Flag | Benchmark use |
|---|---|
| `supports_update` | knowledge-update probes (LongMemEval, EvoMemBench) |
| `supports_temporal_filter` | temporal-reasoning probes |
| `supports_graph` | graph-structure probes |
| `memory_as_tool` | agent decides when to call memory |

## Data model

```
Episode
  └── Session[]
        └── MemoryRecord[]  (id, content, ts, role, session_id, evidence_id)
Probe               (question, gold, category, evidence_ids, abstain, tier)
RetrievedItem       (text, score, source_id, rank)
Answer              (text, abstained, usage)
Judgment            (correct, score, metric, detail)
CaseResult          (provider, benchmark, episode_id, probe_id, category,
                     status, judgment, retrieval, usage, latency_ms)
```

## Reproducibility

Every run writes `manifest.json` containing:
- Provider capability declarations and lethyx version.
- Dataset path and sha256 checksums.
- Reader model, endpoint, and temperature.
- Seed, max_parallel settings.
- Git SHA of the repository.

LLM responses (reader and judge) are disk-cached by
`sha256(model, messages, temperature)` so re-runs are deterministic and free.

## Future extensibility (deferred)

- Deep lethyx integration: test `compile_active_overlay` / injection reports.
- Lethyx-enabled agent heads as `Responder` / `AgentPolicy`.
- Leaderboard schema from versioned `memory_bench/report/v1` JSON.
- Automated dataset download with integrity verification.
- Vector RAG provider using local embeddings (no external API).
