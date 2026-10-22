# Data and isolation

## Retrieval mode

`RecallPlan.retrieval_mode` is a closed `semantic | hybrid` value; `semantic`
remains the default. The client, MCP and the local HTTP `/call` dispatcher carry
the same field. `hybrid` requires MemPalace or Zvec. Unsupported backends fail
explicitly. Exact record IDs and historical reads use their existing path and
reject an explicit hybrid request.

Hybrid retrieval reads up to 64 dense and 64 native full-text candidates within
each selected leaf partition. Scope, canonical revision, lifecycle and dependency
checks precede fusion. Dense ranks use the shared qualified cosine space;
lexical lists are interleaved by local rank and stable logical-block ID. Equal
weight RRF uses constant 60 and applies the final limit after fusion. Only the
same record and revision are deduplicated. The fused score is an RRF score,
not cosine similarity. Native full-text errors are visible, without a fallback
scan or alternative successful search claim. Chroma's trigram index cannot
match terms shorter than three characters; no lexical hits for such a query
does not mean the index failed.

Agent-authored `remember` creates an independent record. Titles, source paths,
tags and content are provenance, not update keys. An explicit `amend` or
`correction` identifies the record to change. `remember(operation_id=...)`
replays the original committed creation across restarts; changing its arguments
is a receipt conflict. Without an operation ID, another call creates another
record. Recall deduplicates repeated hits for the same exact record revision,
not records with equal or normalized text.
Concurrent calls with the same operation ID are serialized only against each
other, so they return the same creation receipt without duplicating a record.

Publication requires `expected_source_revision_id` and a verifier-owned
`candidate_semantic_verification` report. Hosts obtain the exact typed subject
with `MemoryClient.verification_subject`, then persist an independent review
with `append_candidate_semantic_verification_ledger_event`. The subject binds
the full candidate payload, profile, revision, source evidence and requested
publication target. Generic test receipts remain supporting evidence. Candidate
creation and editing stay unverified and do not require a review.

This page names the durable objects and the boundaries between them. It is the
reference to use when writing a host adapter or reviewing a storage change.

## Identity hierarchy

| Field | Purpose | Persistence |
| --- | --- | --- |
| `repo_root` | Physical root for UM-MEM runtime data | Operator configured |
| `workspace_id` | Separates bodies of work and backend data | Durable |
| `profile_id` | Separates agents or memory personalities inside a workspace | Durable |
| `session_id` | Identifies one host session | Record provenance and active scope |
| `task_id` | Narrows current working focus | Active scope and provenance |
| `subtask_id` | Records child work without inventing topology | Provenance |
| `user_id` / `project_id` | Optional trusted scope lineage | Provenance and access filtering |

The semantic catalog is owned by `(workspace_id, profile_id)`. Two profiles can
share a storage root without sharing roots, masks, or visible records. Sessions
in the same workspace/profile can reuse the passive tree; their exact-scope
active masks remain separate.

## Durable objects

### Topology block

A topology block has a stable UUID, one parent, a name, a short synopsis,
tags, typed metadata, lifecycle state, token budget, and integer revision. An
index block navigates to children. A leaf can own records and maps to a native
backend partition.

Operational health is attached to the leaf: record and write counts, scoped
and broad recall counts, hits and zero-hits, failures, latency observations,
rendered tokens, lifecycle counters, and partition health. These numbers inform
the agent; they do not trigger topology changes by themselves.

### Record

A trusted host can call `MemoryClient.register_source` with an exact source ID,
UTF-8 text, producer, and optional speaker/time fields. The source is retained
as a hidden raw witness in the bound workspace/profile/session. Repeating the
same event returns its existing witness ID; changing its bytes or scope is a
conflict. Registration does not create a model-authored semantic record or
select a leaf.

A record is an immutable revision of one advisory memory item. It carries its
workspace/profile scope, primary leaf membership, session and task provenance,
lifecycle, trust and verification state, evidence references, and backend
projection identity. A correction creates a successor; it does not edit the
old body in place.

The versioned membership binds the record to the leaf's stable
`backend_partition_id`. Its `catalog_revision_id` records the catalog observed
when the membership was written. Unrelated catalog changes and leaf renames
leave the native partition unchanged. The membership relation in the SQLite
index carries the same partition ID for scoped lookup.

For split/merge, `topology_evolutions` retains the agent's exact source revisions,
target blocks, record mapping, request hash, and state (`prepared`,
`projections_ready`, `published`, or `aborted`). The native target projections are
prepared first. Publication changes topology and `published_record_memberships`
in one catalog transaction. The latter is the current effective placement pointer;
it does not alter the original canonical record payload. Repeating an operation ID
with different arguments is a conflict. `pending_topology_evolutions` and
`resume_topology_evolution` expose crash recovery without guessing a new mapping.
An explicit `abort_topology_evolution` records the reason and releases a pending
source claim without publishing any target.

In `lossless_atomic` write mode the record metadata also contains the complete
`source_segments` payload. Each segment preserves its source ID, original text,
optional speaker and timestamp, and a SHA-256 over the exact UTF-8 text.
Registered bindings additionally preserve the full source hash, exact half-open
UTF-8 byte offsets and witness ID. The service checks the selected byte slice
before storage. The write receipt separates exact segment fidelity from
registered source byte coverage. Inline unregistered segments have unknown
whole-source coverage; a host supplies expected source IDs for a history-level
coverage calculation, which unions overlapping ranges.
Bound and unbound segments cannot be mixed in one record, even when they name
the same source ID.
`summary_count` remains zero: summaries are optional derived indexes and never
replace the canonical source. `source_coverage` in the write receipt reports
the verified fraction of registered source bytes selected by that record, or
zero with `source_coverage_status=unregistered` for inline segments.
MemPalace indexes a long canonical source using overlapping derived search
windows of at most 200 MiniLM tokens. Every window carries exact UTF-8 byte
offsets into its canonical record; it is a search pointer, never a second
semantic fact or a substitute for the full source.

One durable transaction makes the canonical payload, head, exact links and
pending outbox visible together before native projection. A precommit failure
rolls them all back. Normal writes acknowledge that commit without waiting
for the native retain call. A bounded worker projects different leaf partitions
in parallel and serializes writes within one partition. A pending projection is
not evidence of a failed canonical write. Receipts expose
`operation_id`, `commit_revision`, `durable_status`, `projection_status`,
`projection_revision`, `retryable`, and `source_coverage`. A projection barrier
is the explicit read-after-write synchronization point.
An operation-ID creation also stores a content-free immutable canonical receipt
in that transaction. The caller-facing response ledger is separate; after a
process interruption the same operation ID resolves against the canonical record.
It reports `reason_code` and `attention_required` for unresolved projection.
The backend index uses schema version 16 for indexed exact evidence references,
profile/scope enumeration, stable leaf partition identity and trusted knowledge
time. Older index stores
are rejected without an implicit migration; create a fresh store for this schema.
The barrier reads only the named record's outbox state until the deadline. It
does not start store-wide reconciliation or native work. A different commit
revision reports `projection_revision_mismatch`; a pending projection at the
deadline reports `projection_deadline_exceeded`. Recovery is a separate action.
Python callers can inspect a creation by its original `operation_id` with
`MemoryClient.operation_status`, which returns the immutable creation receipt
and a separate current projection status. If the separate caller response was
lost, status verifies the canonical creation receipt and exact profile before
reporting `receipt_recovery=canonical_creation_receipt`; it cannot reproduce the
lost response body. `commit_revision` names the durable outbox entry, while
`projection_revision` is zero until that entry is actually projected.
`read_exact_records` reads a bounded
batch of canonical record IDs in the requested order.
MCP, local-service, and Hermes tools expose these exact reads, operation status,
and the projection barrier. Local-service status reads the original receipt in
the authenticated caller role; callers cannot select another role. The MCP
`remember` tool accepts `write_mode`, exact `source_segments`, and `operation_id`
as explicit arguments. Hermes uses the native tool-call identity when the model
does not supply an `operation_id`. MCP generates and returns one when omitted;
the returned ID can be passed to `operation_status` under the same caller role.
`list_records` takes an explicit semantic scope and scope ID, with an optional
logical block ID and a 1–128 item page limit. Its opaque cursor freezes up to
2048 visible record IDs and revisions, so later writes do not enter the page
sequence. A changed or no longer visible record invalidates the snapshot;
callers then start a new listing. Hidden raw witnesses are not enumerated.

`saved=True` confirms the canonical commit, not native index visibility. An
acknowledged record remains exactly readable even after an amendment while its
projection is pending. `projection_barrier` polls the scoped receipt without
performing blocking native work; its timeout cannot be extended by a slow provider.
Backend `health()["projection"]` exposes pending/failed entries, exhausted retries,
active partitions, the worker limit, worker errors and oldest pending age. Automatic
native retry is capped at three attempts with persisted backoff. Exhausted entries
remain durable and require explicit `reconcile(dry_run=False)` after the cause is
fixed. Reopening or polling never resets that budget. Native projection events
record operation and commit IDs, queue/lock wait, native duration, lag and retries;
native duration is not durable acknowledgement latency.

Concurrent calls with the same scoped `operation_id` share one receipt lease.
They return the original response, including its projection status at acceptance;
the barrier supplies the current projection status. A different payload under the
same ID is a conflict. Different operation IDs do not share this receipt lease.
The service receipt does not duplicate source text or the original operation ID.
Replay restores source segments from the canonical record; an unavailable record
is an explicit error, never a successful replay with silently missing evidence.

### Tree edge and graph edge

The ownership parent is part of the tree. Typed graph edges capture cross-branch
relationships. Keeping them separate prevents a dense graph from making
ownership or physical partitioning ambiguous.

### Active mask and activation lease

The active mask is mutable through explicit revisioned operations. The
activation lease is a per-round immutable result. The lease trace records the
mask, catalog, and backend revisions; selected and rendered record IDs;
lifecycle exclusions; scope validation; token use; and refresh reason.

### Host observation

A host observation is a raw, authenticated witness. It can later support a
semantic write or verification result, but it is not itself a model-visible
topology decision. Session boundaries can be retained as observations; they do
not become session blocks automatically.

## Where data lives

With `repo_root=/memory` and `workspace_id=alpha`, the backend data lives below
`/memory/workspaces/alpha/.memory/`. The semantic catalog and service state are
local SQLite data; the selected backend stores its native partitions beside
that workspace data or at its explicitly configured service/data location.

Generated Hermes provider/observer wrappers, manifests, and profile config live
inside the selected Hermes profile, under `<HERMES_HOME>/plugins/` and
`<HERMES_HOME>/um-mem/`. Legacy cutover backups, when present from older
installations, remain there for explicit restore.
They are not source files and must not be committed to this repository.

## Version boundary

Canonical and sidecar schemas are version-exact. UM-MEM does not silently read
an old partition format or mix projections from different schema lines. This
release supports fresh stores on the current schema and recovery of their own
pending journals; it does not include an older-schema conversion tool.
`native_projection_rebuild_required` does not authorize a hidden compatibility
read. Keep any older store separate rather than pointing this release at it.

## Publication assurance

`MemoryClient.promote` requires a passed dedicated candidate semantic verification
receipt for the exact source revision, payload, source evidence, destination leaf,
authority and scope. A generic test report or a host-declared assurance label does
not establish semantic review. The receipt identifies independent proposer and
verifier models and records coverage, preservation, faithfulness and instruction
safety. Deterministic code validates that binding; it does not judge meaning.

Publication writes verified records to the selected native leaf partition.
Context packing checks the retained receipt, exact source and destination binding,
and matching verifier evidence. A malformed or unbound receipt is projected as
unverified advisory memory without rewriting immutable history. Ordinary committed
candidates remain available as advisory memory without authoritative promotion.

A candidate explicitly authored with `kind="procedure"` also carries
`procedure_applicability`, `source_task_signature`, and exact
`source_episode_ids` in metadata. Its episodes must still be current semantic
records. Publication requires two distinct applied decisions and action traces
in different independence domains, each with a passed
`memory_application_verification` event from the supervisor ledger bound to
the exact decision, action trace, memory ID and applied usage ID. A generic
passed test report or claimed success without that ledger evidence does not
count; a ledger-verified harmful application blocks publication. The verified
credit IDs and domains remain in the published record's provenance. These
credits establish repeatability, not a causal quality gain; cross-signature
reuse still needs an accepted `TransferEnvelope`.

### Contract versions for 0.4.1b2

Record schema is 9, canonical SQLite schema is 16, and local-service API and
schema are 2 (transport protocol remains 1). Serialized checkpoint boundaries
reject unknown fields. WorkingStateCheckpoint requires profile_id,
owner_generation, requested_block_id, working_view, and a hash for every exact
source/anchor/tail/page reference. Repository, diff, error, and test fields are
optional. PublicationCommitReceipt records exact source and result revisions
and destination scope in the existing canonical mutation receipt transaction.
It is an acknowledgement of history, not a live authority grant.
AmendmentCommitReceipt binds the full typed request and scope to the exact
predecessor and correction revisions in the same canonical transaction. An
operation retry cannot be recovered from caller-editable witness metadata.
SQLite schema 16 records trusted `committed_at` separately from the source's
`recorded_at` in immutable block and mutation rows. `known_as_of` uses this
commit timestamp and a committed canonical revision; old stores fail closed
and require a fresh store rather than implicit migration.

### Record-list pagination snapshots

`list_records` returns a short version-2 cursor referencing an ID/revision snapshot
in a derived SQLite cache, isolated by workspace and profile. It survives client
reopen and does not rescan the native store on continuation. Appends cannot change
its membership; each returned record is revalidated for scope, visibility, logical
leaf and revision. Changed records invalidate affected pages.

The cache retains at most 64 snapshots per workspace/profile and expires entries
after 24 hours without use. Evicted or expired cursors fail explicitly and require
new enumeration. A cursor cannot exceed 3072 decoded bytes. Old inline snapshot
cursors are rejected; this does not migrate canonical memory or catalog formats.
