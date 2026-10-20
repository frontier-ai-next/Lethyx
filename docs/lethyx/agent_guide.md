# Guide for agents and contributors

Read this page before changing memory behavior. The goal is to preserve the
central design while keeping the implementation small and reviewable.

## Start with the actual contract

Before editing code:

1. read [Mental model](memory_architecture.md) and
   [Data and isolation](data_model.md);
2. trace the public API or host adapter that owns the behavior;
3. inspect existing tests for identity, lifecycle, and revision invariants;
4. distinguish a host fact, a semantic model decision, and a deterministic
   safety check;
5. make the narrowest change that satisfies the typed contract.

Do not add a parallel memory engine, compatibility layer, or convenience path
when the existing client, catalog, backend, or local service already owns the
operation.

## Non-negotiable design rules

### Facts within blocks

A meaningful leaf can contain several independently maintained records. Write one
independently changeable fact, or a small connected group, per record. Preserve
participants, events, event dates distinct from reporting dates, exact numbers,
units, constraints and source references. With MiniLM, aim for 80–200 tokenizer
tokens. Write receipts diagnose its 256-token embedding window; they do not
truncate stored content. The agent splits an oversized record by meaning and
chooses its placement. Record length does not require a new leaf.

Correct facts through existing CAS revisions. Equal text does not establish
entity identity. Keep source witnesses recoverable; existing stores are not
automatically rewritten. A migration requires explicit model proposals.

For a repeatable procedure, write an advisory candidate with `kind="procedure"`,
explicit applicability, its source task signature, and exact source episode
record IDs. Do not report a successful application merely because the advice
was retrieved. Publication waits for two independent applied uses with passed
host application-verification events bound to the exact usage and the separate
semantic review; a verified harmful use blocks
it. A different task signature requires an accepted transfer envelope.

When answering, choose current, previous or historical values according to the
question. Combine the necessary records and report missing information. A more
recent report is not always the answer to a question about an earlier event.
Use `as_of` for the date when an event was true and `known_as_of` for the
timezone-aware cutoff on what this memory had durably learned. The latter
returns an advisory historical snapshot; an unknown event date remains unknown.
`recorded_at` is provenance, never a substitute for `valid_from`. An `as_of`
query retains otherwise eligible records with unknown event dates and labels
them `event_time_status=unknown`; this does not prove they were true on that date.
Use the source to decide whether it supports the requested time. Explicit event
intervals and expiration still apply. No stored source is rewritten by this read.

Before answering, verify the exact claim against the source: subject, action,
object, time and certainty. The question's premise is not evidence. Adjacent
messages and shared topics do not establish a missing relationship; positive
sentiment does not establish a ranking or superlative. Check attribution when
uncertain, and report the requested fact as unknown if the link remains absent.
Keep this reasoning within the existing read budget. The agent decides whether
the evidence supports an answer; the host does not grade prose or force a retry.
An expanded retrieval query must not weaken the predicate of the original
question. When the host provides an answer-evidence field, record a brief
assessment with exact quotations and references before the final answer in the
same round. Name any missing relation; a sole related example does not prove an
unstated comparison. This is model-declared support, not verified evidence use
or a host verdict, and must not be credited as such in telemetry.

### The topology is free-form

There is one protected root per workspace/profile. Do not require a project
level or standard branches such as `code`, `logs`, `skills`, or `reflection`.
Projects, experiments, conversations, and reusable knowledge are examples of
meaning the model may choose, not schema categories.

The model may create, amend, move, split, merge, archive, revive, delete, and
link blocks through explicit typed operations. Deterministic code validates the
operation; it does not decide the semantics.

Before placing new material, compare its meaning with the leaf synopsis; a
writable leaf is not automatically a fitting leaf. Keep block tags topical and
descriptions representative of the current contents, rather than the first
session or one source date. Record tags describe that record, not an automatic
copy of its block's tags.

At a meaningful work boundary, review the touched leaves. Amend stale synopses
with CAS, split subjects that need independent retrieval, and merge redundant
boundaries when their meanings fit. A deliberate unchanged decision is valid;
block counts do not establish semantic quality. Shared participants or interleaved
dialogue alone do not establish a useful retrieval boundary. Preserve their source
provenance and use explicit graph links for cross-cutting context rather than
keeping unrelated retrieval subjects in one conversation bucket. Enumerate all pages using
`list_records` with workspace scope and the selected `logical_block_id`, then
inspect exact records to construct the full split/merge mapping. Navigation
previews and top-k recall are not exhaustive membership lists. Pagination cursors
reference a persisted, scope-checked ID/revision snapshot rather than embedding
the full membership in the prompt. The derived cache retains up to 64 snapshots
per workspace/profile for 24 hours of inactivity; an expired cursor requires a
new enumeration. Appends do not alter an existing snapshot, and changed revisions
or memberships invalidate affected pages. Old cursor formats are rejected. The public MCP
and Hermes schemas expose the exact target and source fields for these calls.
Hosts should budget this review separately from acquisition and answering;
never force a mutation or rewrite lossless sources merely to complete a review.

For a full topology review, separate the semantic decision from execution. First
compare the existing boundaries with split and merge alternatives using actual
records and current leaf descriptions. Save a short plan naming source leaves,
destination meanings, representative evidence IDs and the expected retrieval
benefit relative to projection and navigation cost. Compare leaves with each
other for redundant meanings; reviewing each description independently misses
merge opportunities. Shared entities alone justify neither keeping a mixed leaf
nor merging distinct subjects. A justified unchanged plan remains valid.

Then execute the agent's plan with complete exact mappings and current CAS
revisions. Representative evidence in the plan is not a complete mapping. Verify
published receipts and the resulting tree, and distinguish completed changes from
deferred decisions. A broader description does not complete a chosen split. If
new evidence changes the decision, explain it; report an actual missing input or
typed tool error rather than claiming that supplied records or tools are absent.
The host may persist the plan and resume execution, but must not grade semantic
reasons, require a particular operation or infer mappings from the plan's prose.

Large exact mappings can be authored in bounded batches after the agent chooses
the destination meanings. Present explicit `record_id` separately from the
current leaf ID. The agent assigns one destination to every record; the host
checks complete coverage, persists each choice, and assembles the exact native
request. Do not default unassigned records to a remainder leaf. A merge chosen
over explicit source leaves means their full union, not a content-based filter.
No partial mapping is published. Reuse the existing native mutation journal for
CAS and recovery rather than adding another publication engine. Small batches
bound model work, not semantic partition size or the number of destinations.

Independent facts do not each require a leaf. Complementary facts about one
stable subject or procedure can share a leaf; a boundary needs a navigation or
isolation benefit. Conversely, common people and interleaved messages do not
make unrelated subjects one useful retrieval unit.

### Active memory is a mask

Active memory is not “recent recall” and not a second cache. It is a durable
exact-scope pointer set over passive memory. A native host resolves it before
each round into a bounded lease. Recall remains temporary.

Never silently add recalled records to a mask. Never infer a mask from the
current prompt. Never reuse a session/task mask in a different exact scope.

### Observations do not mutate meaning

Tool events, tests, files, session boundaries, and other host observations are
typed witnesses. They may support later model-authored records or verifier
receipts. They do not choose a leaf, create a visible record, restructure the
tree, or change active focus.

Do not infer semantic placement, success, failure, relevance, or lifecycle from
keywords, regular expressions, filenames, command strings, prompts, or
tool-output prose. Parsing a declared wire format is different from inferring
meaning from free text; keep that boundary visible in code and tests.

### Memory is advisory

Lethyx can fail closed on an invalid memory mutation, scope violation, stale
revision, or corrupt projection. It must not fail the host's primary task merely
because recall or active delivery is unavailable. Report degraded operation and
let the host continue without memory for that round.

Do not add machine content contracts that grade or block an agent answer. Answer
quality belongs to the host, benchmark, or LLM judge. Lethyx checks structural
and evidence integrity only.

### History is immutable

Corrections append successors and use compare-and-swap revisions. Split, merge,
and lifecycle changes preserve exact mappings, provenance, receipts, and
recoverability. Do not update canonical record bodies in place or hide a
partial topology mutation behind best-effort cleanup.

Before split or merge, choose the destination meanings and map every current
source record ID exactly once. Relocate any active masks and semantic graph
references that still point at source leaves; an unresolved reference rejects
publication. Outgoing `derived_from` links remain as provenance and do not
prevent another split of a previously evolved leaf.
Keep the returned `operation_id`. If a native projection or process fails, read
`pending_topology_evolutions` and call `resume_topology_evolution` with that ID.
If projection cannot be recovered, explicitly call `abort_topology_evolution`
with a reason; this preserves the failed journal and frees the source leaf.
Reuse the same arguments and ID after an unknown outcome. A changed request
under the same ID is rejected; a source CAS conflict requires rereading the
current tree and making a new explicit decision. The old tree remains readable
until the new generation is published.

## Sessions and projects

`workspace_id` and `profile_id` own the persistent semantic catalog. Session,
task, and subtask IDs are scope and provenance. They do not require visible
blocks. Several sessions can extend the same tree; several projects can coexist
in one tree when the agent chooses that organization; one project can also use
separate profiles when isolation is required.

When adding an adapter, bind stable host identity before exposing tools. Never
let model-supplied arguments switch the runtime root, workspace, profile, or
backend.

## Code discipline

- Target Python 3.12+ and use its native typing syntax. Do not add compatibility
  imports such as `from __future__ import annotations`.
- Prefer existing typed objects and direct control flow over generic helpers,
  reflection, broad exception handling, or stringly typed dispatch.
- Keep imports at module scope unless an optional dependency or host-only
  integration requires a local import.
- Reject unknown fields at trust boundaries. Do not silently coerce malformed
  identity, revisions, or lifecycle values.
- Do not read credentials from repository files or expose record bodies in
  navigation and health payloads.
- Keep runtime data, generated wrappers, profiles, logs, benchmarks, and
  credentials outside the repository.
- Update user documentation when a public command, configuration key, tool
  name, or lifecycle behavior changes.

## Review questions

Before accepting a change, ask:

- Who made the semantic decision: the model or a hidden heuristic?
- Can a different project, experiment, or conversation use this without a
  forced taxonomy?
- Does the active/passive/recall distinction still hold?
- Are workspace, profile, session, and task scopes explicit?
- Is a failed memory operation visible but non-blocking to the host task?
- Are mutations revisioned, auditable, and recoverable?
- Does backend recall remain native to the selected leaf partition?
- Does the documentation describe what the code actually does?

Run focused tests first, then the full suite and strict documentation build:

```console
python -m pytest -q
mkdocs build --strict
```

## Creation and independent verification

Use a fresh creation for independent evidence, even if its text equals another
record. Supply an operation ID only to retry the same intended creation; changing
its arguments is a conflict. Correct an existing record through the explicit
CAS amendment or trusted correction route. Duplicate-cluster membership never
proves truth and never hides an independent record.
CAS amendments also accept an operation ID: retries acknowledge the original
immutable correction, even after later revisions. The full typed amendment
request and scope must match its atomically committed receipt.

## Lossless atomic writes and projection barriers

The trusted host can register exact source text with
`MemoryClient.register_source(source_id=..., text=..., producer_id=...)`.
Registration retains a hidden raw witness with source hash, event identity,
session scope and optional speaker/time provenance. It does not choose a
semantic leaf; the model still makes that placement explicitly.
To bind a semantic segment to that source, provide its `source_id`, full
`source_sha256`, and half-open UTF-8 `byte_start` / `byte_end` offsets together
with the exact segment `text`. The service verifies those bytes and the source
scope. `source_coverage(expected_source_ids)` unions overlapping ranges across
records and reports byte and source-ID coverage. The host must supply the
expected IDs; unknown source IDs leave whole-history byte coverage unknown.

When exact source recovery matters, call `remember(..., write_mode="lossless_atomic")`.
The Hermes remember tool exposes the same mode and a closed `source_segments`
array. `content` must equal the exact concatenation of segment texts, with no
added separators, whitespace normalization, or paraphrase. A mismatch is rejected
before storage. Inline segments without a registered source prove only segment
fidelity; they report `source_coverage_status=unregistered` and no registered
source byte coverage.
The write keeps the complete source text in the immutable canonical record and
stores `metadata.source_segments` with a source ID, speaker/time fields and a
SHA-256 of every segment. `summary_count` is zero in this mode; a native
embedding window may limit only its derived index. Read the canonical record by
`MemoryClient.read_exact_record(record_id=...)` or by an exact evidence/source
reference before constructing an answer.
For deliberate enumeration, `list_records` requires a scope and scope ID and
returns content-free record IDs in pages. Its cursor freezes the initial ID set;
new writes require a new listing. Use `read_exact_record` or `read_exact_records`
to inspect selected IDs. After a write, keep its `operation_id` and query
`operation_status` for the original durable receipt and current projection
state; use `projection_barrier` only when native indexing must have caught up.
In Hermes, pass `record_ids` and the matching `logical_block_ids` to scoped recall
to expand a truncated card. Keep the date and speaker attribution with the source.
The compact recall receipt also lists `unrendered_record_refs` when ranked
candidates could not fit in the context budget. These are references, not
evidence already read: choose relevant IDs and fetch their exact records before
using them. If the selected leaf lacks the needed event, the agent chooses a
different leaf or explicitly calls search-all; repeating a query against the
same unchanged scope does not establish absence. For totals, verify each
distinct event against the speaker's report, rather than an assistant suggestion.
For speaker-specific questions, the agent may pass `source_speakers=["user"]`
to `recall`, `build_context`, `recall_blocks`, or `search_all` on the
corresponding surface. The scoped native recall first retrieves a
bounded wider candidate set, then filters by the typed source speaker and
packs the requested limit. The agent still chooses the query and leaf; a filtered
miss is not proof that no matching source exists.
For an exhaustive personal total, request enough candidates within the configured
limit to cover the distinct source events. Verify any omitted exact IDs before
answering; a small top-k of user-authored records can still miss one event.
Packing preserves backend order and complete records while space permits, using
compact headers before truncating facts. It does not rerank by prompt keywords.
Failed native reads carry `reason_code` and `retryable`; they are not empty results
and do not establish absence of knowledge.

Every successful write has a durable commit followed by a native projection.
Use `MemoryClient.projection_barrier` when a dependent recall must observe the
projection revision. The barrier polls only that record's projection state
until its deadline; it does not start reconciliation. A pending projection
returns `projection_deadline_exceeded` at the deadline, while a failed
projection returns `native_projection_attention`. A different commit revision
returns `projection_revision_mismatch`. These states leave the durable record
intact and do not establish absence of knowledge. Request recovery separately
when needed.
Exact lookup by evidence reference rejects a reference shared by multiple
canonical records with `ambiguous_evidence_ref`; read those records by ID.

Keep the returned record ID and commit revision. Exact canonical reads are ready
after durable acknowledgement; scoped native recall may return the typed advisory
`native_projection_pending` until its barrier succeeds. Do not infer absence from
that response. If backend projection health reports `attention`, an operator must
repair the native cause and explicitly reconcile; repeated remember/barrier calls
do not clear the retry budget. Closing a backend drains its current native calls;
pending outbox work survives termination and is recovered on reopening. These
mechanisms enforce identity and ordering only: agents still choose the leaves,
topology and amendments by meaning.

To publish, name the exact source revision and destination leaf. The host
verifier compares the full revision with resolved evidence and records coverage,
exact preservation, faithfulness and instruction safety. Passed tests alone
cannot certify the lesson. Unknown source evidence or unknown author-model
identity prevents independent publication. Memory remains advisory.
