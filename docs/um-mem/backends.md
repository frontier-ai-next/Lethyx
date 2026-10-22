# Backends

UM-MEM selects one durable backend for a configured workspace runtime. The
semantic tree and contracts stay the same; each backend provides a different
native partition primitive and operating model.
Hindsight and Mnemosyne batch writes may use temporary workers for distinct leaves;
their worker-owned SQLite connections close when each worker finishes.
Backend close waits for active partition projection calls before closing native
SDK resources and SQLite connections. Work not yet started remains durable in
the outbox for reopening; closing is not a claim that the entire outbox drained.

## Comparison

| Backend | Good fit | Native leaf partition | External service |
| --- | --- | --- | --- |
| MemPalace | Local development and self-contained profiles | Chroma collection family | No |
| Hindsight | Service-backed retain/recall and Hindsight deployments | Hindsight bank | Usually |
| Mnemosyne | Local Mnemosyne/BEAM storage | Separate database and native session per leaf | No |
| Zvec | Embedded dense and full-text search | Separate Zvec collection per leaf | No |

Mnemosyne's native session identifier belongs to a semantic leaf, rather than
to a Hermes session. Its database also isolates that leaf before native top-k
retrieval.

## MemPalace

MemPalace is installed with the base package and is the default for the Python
and MCP interfaces:

```console
set UM_MEM_BACKEND=mempalace
um-memory backend-health --repo-root ./agent-memory --workspace alpha --backend mempalace --json
```

Data is stored under the workspace's `.memory/mempalace` area. Each logical
leaf has a native collection-family identity. A new leaf is materialized on its
first eligible write.

## Hindsight

Install the client extra:

```console
python -m pip install "um-mem[hindsight]"
```

Configure the runtime:

| Variable | Default | Meaning |
| --- | --- | --- |
| `UM_MEM_HINDSIGHT_BASE_URL` | `http://localhost:8888` | Service URL |
| `UM_MEM_HINDSIGHT_API_KEY` | empty | Service credential |
| `UM_MEM_HINDSIGHT_TIMEOUT_SECONDS` | `30` | Client timeout |
| `UM_MEM_HINDSIGHT_EMBEDDED` | `0` | Use embedded mode when supported |
| `UM_MEM_HINDSIGHT_PROFILE` | `um-mem-dev` | Hindsight profile |
| `UM_MEM_HINDSIGHT_RETAIN_ASYNC` | `0` | Asynchronous retention |
| `UM_MEM_HINDSIGHT_FAIL_CLOSED` | `1` | Refuse native-unhealthy operation |

Each UM-MEM leaf maps to its own Hindsight bank. Health is native only when the
configured service and retain/recall surface are available. The sidecar index
does not masquerade as a healthy Hindsight recall path.
Semantic leaf banks use the native extraction defaults, without imposing a
technical workspace or BKB mission on model-authored topics.

## Mnemosyne

Mnemosyne support is installed with the base package. Select it with:

```console
set UM_MEM_BACKEND=mnemosyne
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `UM_MEM_MNEMOSYNE_ENABLED` | `1` | Enable native Mnemosyne access |
| `UM_MEM_MNEMOSYNE_DATA_DIR` | workspace `.memory/mnemosyne` | Native data directory |

Each leaf receives a stable native session-partition identifier derived from
its workspace, block ID, and stable backend partition key. A separate native
database lives under `partitions/<partition-identity-sha256>/mnemosyne.db`.
Native FTS and dense retrieval therefore rank only that partition's records.
UM-MEM also maintains its typed sidecar index for contracts and audit operations;
semantic recall remains a native Mnemosyne operation. Discovery of existing
leaf partitions uses indexed canonical memberships instead of loading every
record in the store.

All four native backends support staged split/merge projection into new leaf
partitions. Each staged copy preserves the exact canonical record and is verified
before the catalog switches membership. A failed native stage leaves the source
leaf readable and the generation pending for a bounded retry; the sidecar is
never used to claim successful native recall. Hindsight requires a healthy
retain/get/recall service for this operation.

An empty `retain_blocks([])` batch returns zero writes without opening a native
partition or starting batch workers.

An older shared `mnemosyne.db` directly in the native data directory reports
`native_projection_rebuild_required` and refuses native access. Its history is
preserved. Use a clean store for a fresh run; existing deployments require an
explicit reconstruction of native projections from their retained canonical
history. UM-MEM does not silently discard or reinterpret the shared database.

## Zvec

Install the optional embedded backend with `python -m pip install "um-mem[zvec]"`
and select `UM_MEM_BACKEND=zvec`. Zvec 0.7 provides a separate collection for
each leaf. Canonical records, receipts, and the projection outbox remain in
UM-MEM's SQLite store. Native indexing uses overlapping source windows with
exact offsets; neither the windows nor their embeddings replace the full
canonical record. Projection fetches, embeds, writes and verifies at most 256
windows per batch, below Zvec's 1024-document write limit. A retry verifies
already written windows and continues with missing ones; it does not create new
records or truncate long sources. The outbox commits only after all batches
have been verified. Windows never become independent semantic facts. The full
source remains available by exact record ID. Dense and native full-text
retrieval operate only on the selected leaf collections. A pending or failed
native projection is reported as such, rather than silently treated as an
empty memory. Zvec's embedded index has no external server to start.

## Semantic tier (optional)

The backend decides how records are stored and searched. On top of it UM-MEM
can run one local retrieval tier that scores the newest records with
multilingual sentence embeddings, so a question written in one script reaches
a record written in another script. Install it with
`python -m pip install "um-mem[semantic]"`; the default install never
downloads a model.

| Variable | Meaning |
| --- | --- |
| `UM_MEM_SEMANTIC` | `0` switches the tier off. Default is on when the extra is installed. |
| `UM_MEM_SEMANTIC_MODEL` | Model id. Default `intfloat/multilingual-e5-large`. |
| `UM_MEM_SEMANTIC_CACHE_DIR` | Where the model files live. Default `<um-mem home>/cache/fastembed`. |
| `UM_MEM_SEMANTIC_FLOOR` | Minimum similarity that counts as a hit. Default is chosen per model. |
| `UM_MEM_SEMANTIC_WARMUP` | `0` defers the model load to the first question instead of warming it in the background at service start. |

The tier runs in the local service, so the model is loaded once per service
process and stays warm for the MCP tools and the pre-tool helper alike. It
only reorders candidates: the stored record text remains the evidence, and a
question with no real match returns nothing instead of a plausible-looking
record. Prefetch the model with `um-memory semantic-model` instead of waiting
for the first question.

The model is a few gigabytes of resident memory, and it is only useful while
a question is being scored. An unused model is released after
`UM_MEM_SEMANTIC_IDLE_SECONDS` (default 120 seconds) and loaded again on the
next question. The native backend stays keyword-only, so no second model is
ever loaded or downloaded; ask for native dense recall explicitly with
`MNEMOSYNE_EMBEDDING_MODEL` if you want it.

The index covers the whole workspace: every write joins it in the background and
`um-memory semantic-index` backfills what is missing, so a question in any script
reaches old records and not only the newest window. Stored procedures are
records of kind `procedure`; `um-memory map` renders the workspace as one
self-contained HTML page for a human look.

## Switching backends

Changing `UM_MEM_BACKEND` does not move existing records. Start a separate
clean store for the new backend; this release has no old-store conversion
command. Do not point a new backend at old projections and rely on fallback
filtering.

Always check health before starting a long session:

```console
um-memory backend-health --repo-root ./agent-memory --workspace alpha --backend mempalace --json
```
