# Developer profiling

Lethyx has an append-only, fail-open developer trace for understanding memory
latency without changing the host UI or the memory protocol. The default local
path is:

```text
<repo_root>/.Lethyx/memory/telemetry/memory_events.jsonl
```

When a runtime supplies a separate drive root, the same file name is written to
`<drive_root>/logs/memory_events.jsonl`.
Diagnostic rows are buffered in bounded process-local batches, flushed within
250 ms or after 32 rows, and rotated at 8 MiB with three retained segments.
Reports read UTF-8 JSONL by physical newline across those segments and expose
`dropped_event_count` if the bounded queue overflows. This trace is best-effort;
the canonical record and projection journals remain separate durable stores.

## What is recorded

Profile rows use `event_type=memory_profile` and carry the correlation fields
`workspace_id`, `profile_id`, `user_id`, `chat_id`, `run_id`, and `stage_id` when
available. The main timing fields are:

- `operation`: the logical operation, such as `recall`, `backend_recall`,
  `pack_context`, or `backend_partition.recall`;
- `component`: the owning layer, such as `hermes_provider.transport`,
  `local_service.core`, `memory_client.backend`, `memory_client.packer`, or a
  concrete native backend partition;
- `duration_ms`, `status`, `outcome`, and `reason`;
- `resources`: process/thread CPU milliseconds and, when `tracemalloc` is
  already enabled by the host, current/peak Python allocation bytes;
- `parent_span_id`: the nesting relationship between Hermes, the local service,
  the MemoryClient, and backend work;
- `parallelism.max_in_flight`: the maximum observed overlap for the span's
  workspace/backend concurrency key.

`data.input` and `data.output` are bounded, content-free summaries. They expose
types, field names, counts, character/byte sizes, estimated tokens, and SHA-256
digests. They do not persist query text, prompts, memory content, tool results,
credentials, or private transcripts. This makes the trace safe to keep beside
the runtime while still showing where payload size and token growth come from.

The currently instrumented path is:

```text
Hermes transport
  └─ local service core
       └─ MemoryClient backend recall
            ├─ native/backend partition timings
            └─ context packing
```

Hermes batch calls also receive one transport row per item in addition to the
batch timing already maintained by Hermes.

## Reading the report

```python
from lethyx.memory.kernel.telemetry import memory_event_report, memory_profile_report

report = memory_profile_report(repo_root)
# or: report = memory_event_report(repo_root)["profile"]
```

There is also a diagnostics command for a quick JSON snapshot:

```console
python -m lethyx.memory.cli profile-report --repo-root <repo_root> --json
```

Use `--drive-root <drive_root>` when the runtime wrote the trace to its
separate drive log location.

The report contains aggregate totals and p50/p95/max latency by operation,
component counts, failed counts, input/output token totals, observed concurrency, and
CPU totals, and the slowest recent rows. Use `parent_span_id` plus `run_id`/`chat_id` to join the
layers for one request. A large transport row with a small local-service row
points to transport or process overhead; a large backend row points to the
selected backend; a large packer row points to evidence shaping or token
budgeting.

For parallelization work, first group rows by the same parent span and compare
independent partition operations. `max_in_flight` is an observation, not a
recommendation: only operations with no ordering, mutation, or shared-resource
dependency should be moved to concurrent execution.

The profiler is diagnostic only. If writing the trace fails, the memory request
continues; the trace must never become a new dependency of the memory path.
