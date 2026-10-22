# Lethyx

Lethyx is a framework-neutral local memory layer for agent systems. It can be
used as an embedded Python package, connected through MCP, or installed through
a host adapter. The host remains responsible for routing, permissions, proof
validity, and task completion; memory supplies scoped advisory context.

## Release 0.6.0

This release adds lossless atomic records with durable-first native projection,
recoverable split and merge generations, bounded evidence reads, event-time and
knowledge-time recall, and the Zvec backend. Candidate creation has independent
identity and optional durable operation replay. Publication requires semantic
verification bound to an exact revision, source evidence, profile and target.
Hermes can opt into independently verified task working views at its native
compression boundary. This mode defaults off pending paired quality, latency
and total model-cost evaluation. See [Developer profiling](docs/lethyx/developer_profiling.md)
for diagnostics.

## Install

Lethyx requires Python 3.12 or newer.

```console
git clone https://github.com/frontier-ai-next/Lethyx.git
cd Lethyx
python -m pip install .
```

For development from a clone:

```console
python -m pip install -e ".[dev,mcp]"
```

### Optional: multilingual semantic tier

The default install stays small, runs on any machine, and never downloads a
model. A local semantic tier is available as an extra:

```console
python -m pip install "lethyx[semantic]"
```

It downloads one multilingual sentence-embedding model on first use (about
2.1 GB, CPU only, no API key) and lets a question in one script reach a record
written in another script. `LETHYX_SEMANTIC=0` switches it off;
`LETHYX_SEMANTIC_MODEL` and `LETHYX_SEMANTIC_CACHE_DIR` choose the model and
where its files live. See [Quickstart](docs/lethyx/quickstart.md) for details.

## Embedded Python

`MemoryClient.agent_instructions()` returns the memory-read guidance for your host's
agent instruction channel. The MCP initialization handshake and Hermes provider expose
the same guidance. It explains navigation, scoped recall, query refinement, explicit
broad search and exact-scope active memory. Your model makes the semantic choices;
Lethyx does not choose a branch from the prompt or block the primary task on recall.

Host operator surfaces can call `MemoryClient.inspect_record(record_id,
logical_block_id=..., expected_revision_id=...)` to inspect one exact revision.
The public API checks backend capability, workspace, profile, record identity and
primary block membership. It does not follow a newer revision or make unpublished
or historical content eligible for recall. The host remains responsible for
authorizing its operator and bounding any content returned by its HTTP/UI layer.

MemPalace ordinary recall opens native partitions only when they contain current
records eligible for the requested scope and lifecycle policy. Historical revisions
and unpublished candidates remain retained and exactly addressable; their partitions
are not opened or repaired as a side effect of unrelated current recall. Relevance
ranking still runs through the selected native backend.

```python
from lethyx import MemoryClient

memory = MemoryClient(
    repo_root=".",
    workspace_id="project-alpha",
    session_id="session-42",
    task_id="task-7",
    backend="mempalace",
)

roots = memory.block_map()
root_id = roots[0]["block_id"]

research = memory.create_leaf_block(
    name="Parser research",
    synopsis="Stable parser experiments and findings.",
    parent_id=root_id,
    reason="this subject evolves independently",
)

saved = memory.remember(
    "The parser regression is reproduced by tests/test_parser.py.",
    kind="project_fact",
    requested_block_id=research["block_id"],
    evidence_refs=[
        {
            "ref_type": "test_run",
            "ref_id": "pytest:tests/test_parser.py",
            "produced_by": "verifier",
            "producer_id": "pytest",
        }
    ],
    run_id="session-42",
)

amended = memory.amend(
    saved.memory_id,
    expected_revision=saved.raw["revision_id"],
    content="The parser regression is fixed and covered by tests/test_parser.py.",
    reason="verified implementation replaced the reproduced failure",
    requested_block_id=research["block_id"],
)

mask = memory.set_active_mask(
    selected_block_ids=(research["block_id"],),
    pinned_record_ids=(amended.memory_id,),
    token_budget=600,
    activation_reason="keep the current parser decision between rounds",
)
active_context = memory.resolve_active_mask()

context = memory.build_context(
    "How was the parser regression reproduced?",
    max_tokens=1200,
    run_id="session-42",
    include_candidates=True,
    logical_block_ids=[research["block_id"]],
)

memory.deactivate_active_mask(
    mask.mask_id,
    expected_revision=mask.revision,
    reason="the parser context is no longer active",
)
memory.archive_block(
    research["block_id"],
    expected_revision=research["revision"],
    reason="research phase complete",
)
```

Agent-authored writes are candidates by default. Durable writes use exactly one
selected backend per workspace: `mempalace`, `hindsight`, `mnemosyne`, or `zvec`.
Different leaves are native partitions of that selected backend, not independently
selected providers. Cross-backend federation and score calibration are not implemented.
Canonical and sidecar schemas are version-exact and require a clean-store
cutover when their format changes.

Hosts compensating an earlier write can call `forget(record_id,
expected_revision=revision_id, reason=...)` to retract only that exact current
revision. A stale ID or revision returns `ok: false` with
`error: expected_revision_conflict` and leaves the newer canonical head intact.
The existing canonical compare-and-swap also guards concurrent updates during
the write. Exact retraction remains available after the owning leaf is archived:
it appends a hidden successor without reviving the leaf or changing retained authored
bytes. Repeating the same source ID/revision returns already_retracted and resumes a
pending native projection through the existing write journal. A newer unrelated head
is still a conflict. Omitting the revision retains the deliberate whole-chain retraction
used for user-requested forgetting; compensation should always pin a revision.

`MemoryClient.capabilities()` returns the public, immutable `BackendCapabilities`
contract for the selected backend. It separates exact record reads, scoped search,
revision history, revision compare-and-swap and recoverable writes from atomic
record reassignment, atomic batch publication, batch reads and cancellation.
`enabled_features` is a deterministic tuple suitable for host admission reports.
These are effective capabilities of the backend and its owned indexes; they are
false when its native readiness check fails. They do not grant user authority or
promise that a later operation cannot fail. Missing or untyped declarations fail
explicitly; method presence does not enable split/merge. The current native backends
do not claim atomic reassignment, atomic batch publication, batch reads or cancellation.

MemPalace semantic recall merges all authorized partition results by their shared
native cosine similarity before the caller's global limit. A later-created leaf
and a strong second result in one leaf therefore do not lose solely to block
iteration order. Exact record reads retain caller order. Retrieval metadata records
the native rank, score and score mode, while prompt packing keeps its existing
token and byte limits. Empty-query traversal and backends without a qualified shared
score space retain partition-rank selection; cross-backend calibration is not implied.

For recoverable candidate publication, pass a stable `operation_id` and
`expected_source_revision_id` to `promote`. Persist that source identity before
calling the method. After interruption, `compensate_promotion(source_id,
expected_revision=source_revision, operation_id=operation_id, reason=...)` can
retract the unchanged candidate or the direct publication bearing that operation
identity, even if the host lost the publication receipt. Its typed receipt
distinguishes retraction, an already retracted chain, and a newer or differently
owned head that was preserved. A concurrent write conflict remains an error for
explicit recovery. This path covers ordinary candidate publication in the same
block; correction approval and publication into another block require their own
governed lifecycle. It does not provide atomic visibility for a multi-record batch.

Candidate cleanup uses `retract_working_record(record_id, owner_id=...,
generation_id=..., reason=...)`. It follows amendments only while the current
revision remains a candidate owned by that exact working generation. An accepted
or differently owned successor is preserved. Both recovery operations return a
`MemoryRetractionReceipt`; they do not grant publication authority.

## MCP

### Unpublished working state

Accepted recall remains separate from a task's unfinished memory. A host can bind
`metadata.working_set_owner` on candidate writes to an opaque `owner_id` and
`generation_id`, then call `MemoryClient.read_working_set` (or MCP
`read_working_set`) with those identifiers and 1–32 exact record IDs. The host
must supply these identities from its authorization context, not model claims.
The read checks workspace, profile, caller scope lineage, owner, generation,
candidate status and safety. It cannot promote records or broaden normal recall.

The result labels its context `working_unverified` and reports exact record,
revision and content hashes. `max_tokens` uses the existing Lethyx estimator;
`max_bytes` is a hard UTF-8 limit on `rendered_context`. Metadata references are
reported separately. Callers must inspect truncation, omitted IDs and unavailable
IDs before treating returned context as complete. A budget too small for a card
returns empty context rather than a fragment of a heading.

Host integrations can call `MemoryClient.pin_record` for a content-free
`PinnedRecordRef` containing the exact record ID, revision, source-byte SHA-256,
logical block and lifecycle status. Accepted records must be currently eligible;
candidates additionally require the exact working owner and generation. Pins
cannot revive retracted, expired, hidden or protected records. They establish
identity, not semantic correctness. Both host read APIs accept an explicit
`scope_chain` validated against the client's bound lineage; user-global scope
requires an explicit selection and the same bound user. Scope identity is not
exposed in the MCP tool schema.

For independent content review, a host can pass that pin's revision and hash to
`MemoryClient.snapshot_working_record`, together with the bound owner,
generation and scope. The returned `WorkingRecordSnapshot` contains the complete
candidate text, title, kind, scope and declared evidence references at that exact
revision. Private metadata is excluded. Byte and estimated-token budgets apply
to the complete serialized snapshot; a record that does not fit raises the
public `WorkingSnapshotBudgetError` (`ValueError`), never silently excerpted.
Hosts can distinguish this content-free limit from other snapshot failures
without parsing exception messages. Changed, foreign, protected or non-candidate records
are unavailable. This read neither promotes the record nor validates the
semantic claims or declared evidence. The reviewing host must issue its own
decision and bind promotion to the reviewed identity.

### Server

Catalog snapshots assemble hierarchy paths, descriptor lookup, native partition
rows and metrics once per snapshot. Their assembly cost follows the number of
blocks, edges and emitted path lengths, instead of scanning the whole catalog
for each block. Deep full paths still cost space proportional to their output;
native record/index access and search have their own costs. Snapshot content
and ordering remain authoritative and are not served from a stale cache.

Install the MCP extra and start the packaged stdio server:

```console
python -m pip install "lethyx[mcp]"
lethyx mcp-server
```

The portable MCP surface provides scoped health, free-topology and typed-edge
CRUD, active-mask lifecycle and resolution, immutable record amend, explicit
recall, bounded consolidation, durable retraction, graph, and export tools.

The local listener uses the calling Python environment. Its absolute launcher
path is preserved, including POSIX virtual-environment symlinks, so listener
dependencies come from that environment instead of the base interpreter.
If a newly launched listener exits before publishing readiness, the call fails
promptly with its exit code instead of waiting for the startup deadline. A ready
registered listener takes precedence; process arguments and stderr stay private.
Listener discovery, bootstrap credentials and startup locks are local to the host
execution boundary. On Linux this includes the kernel boot and PID/network
namespaces, so containers sharing memory storage cannot reuse, remove or shut
down each other's loopback listener. The workspace, profile and persistent
backend identity remain unchanged; backend operation locks still coordinate
shared storage. Processes in the same execution boundary reuse one listener.
An exited Linux listener is unavailable even while its parent has not collected
its exit status. Its stale discovery record is retired so the next client call
can start the selected listener again. An unreachable process that is still alive
is never replaced merely because a health probe failed.
Queue, scheduler, operating-ledger and mutation-journal transactions close their
SQLite connections when each operation exits, including rollback and early return.
Listener availability does not depend on garbage collection releasing file handles.

Consolidation is a host-owned authority boundary. The ordinary Hermes provider
keeps its parent scope and returns `host_authority_required` for a consolidation
request; it never creates a supervisor identity from an agent tool call. Hosts
can submit exact source IDs and verification reports through the supervisor or
operator API, where the existing semantic and provenance gates still apply.

Configure repository root,
workspace, selected backend, caller identity, capabilities, and scope lineage
in the trusted host configuration rather than in model-generated arguments.

## Host adapters

Optional adapters connect Lethyx to agent lifecycle events such as session
start/end and observed tool outcomes. MCP remains the tools-only integration;
an adapter adds automatic observation capture without making memory a control
plane.

At a real turn-completion boundary, an adapter may request one agent-authored
`MemoryReviewDecision` and bind it to exact observed memory activity in a
`MemoryReviewReceipt`. This records `changed`, `unchanged`, `unavailable`, or
`forbidden` without retaining the transcript or inventing semantic placement.
The review is advisory and cannot turn successful host work into a failure.

## Documentation

- [Quickstart](docs/lethyx/quickstart.md)
- [Operator guide](docs/lethyx/operator_guide.md)
- [Architecture](docs/lethyx/memory_architecture.md)
- [Hermes integration](docs/lethyx/hermes_runtime_integration_user_guide.md)

## Development

```console
uv sync --locked --extra dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest -q
```

The package type check targets the Python 3.12 baseline and is a required CI
gate. Optional host and backend imports keep narrow import-only annotations;
their absence never substitutes another runtime or grants host authority.

Runtime stores, generated profiles, credentials, logs, and local wrappers must
stay outside the release tree.

Trusted Python hosts may construct MemoryClient with topology_actor="system" for code-owned catalog provisioning and cleanup. The default "llm" remains appropriate for agent-authored topology. This label affects the mutation journal only; CAS, protected blocks, scope and operation authority remain enforced. MCP clients cannot change it through a tool argument.


Provider cost observations
--------------------------
MemoryClient.observe_runtime_cost accepts actual input/total/cache-read token counters and a stable
observation ID. It records partial OperationalSloLedger costs and request-level PrefixCacheSample
observations; unavailable measurements remain null. Request cache counts do not attribute reuse to
memory or prove savings. Replayed IDs are idempotent; conflicting values are rejected. Existing
cost/prefix tables are preserved. Recall/packing latency is measured as a nested operation and its
packing token estimate stays separate from provider billing. No background job count implies an
LLM call or a saving.
