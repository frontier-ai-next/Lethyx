# Memory architecture

Hybrid retrieval changes candidate discovery inside explicitly selected leaves;
it does not author topology, choose active focus, rewrite records or grant
authority. One leaf can group multiple independently revised facts. MiniLM
embedding-window diagnostics leave stored evidence intact. Existing MemPalace
projection checks and exclusive snapshot/repair boundaries remain in force.

Workspace is the physical isolation boundary, including for `project:` scope:
`project_id` is trusted lineage within a workspace, not a group of stores.
Only `user_global` blocks may declare `isolation_policy.cross_workspace: true`.
The store-level manager identity (`um-mem-manager`) maps its backend, catalog
and semantic index to `.UM-MEM/memory/`. A workspace client reads that store
only with explicit `include_manager=True` in the Python recall/context API.
The cross-workspace view requires a bound matching `user_id`, a user-global
scope and a manager block whose isolation policy opts in; manager-local
workspace and project records never cross. These hits follow workspace hits
within the recall limit and remain advisory, including when the source record
had stronger authority. Default workspace reads do not query the manager
store. Existing user-global records in per-workspace stores are not migrated
or replicated. MCP `recall` and `build_context` expose the same explicit
`include_manager` read flag, bound to the configured server principal.

Canonical writes use a short interprocess lane and one SQLite transaction for
the canonical payload, head, source/index links and pending native outbox. A
failure before commit leaves none of those rows visible. Startup recovery uses
the same lane to handle reservations left by earlier releases. SQLite WAL readers use independent connections; native
or model calls never run in this lane. The existing durable outbox schedules up to
four native projection workers per backend instance, ordered within each physical
partition and concurrent across partitions. A partition file lease prevents two
processes from projecting the same pending entry simultaneously. Provider IDs stay
deterministic: retry after a crash may repeat a provider call, but must not create
a second logical record. If the caller response receipt is lost after canonical
commit, `operation_status` verifies the immutable creation receipt, scoped record
and outbox before reporting the recovered durable commit. This status read does
not advance native projection; a same-ID retry remains idempotent.
`commit_revision` names the durable outbox entry. `projection_revision` remains
zero until that exact native entry is committed; a pending or failed barrier
does not present its target commit as a completed projection.

MemPalace retains one cross-process native owner because Chroma caches local
state. Within that process different partitions compute embeddings concurrently;
the adapter passes precomputed embeddings into a short serialized native commit.
MemPalace's own palace-wide mining lock remains enabled. Collection creation uses
the same short gate. Native reads that may repair HNSW, snapshot/restore and repair are exclusive. Exact
canonical reads and durable commits do not wait for this native gate. A snapshot
pins a WAL read transaction while native projection is fenced, so concurrent
canonical commits fall outside its consistent cut. Governed erasure closes pending
projection before purging and certifying absence.

UM-MEM uses one lifecycle:

`capture -> block placement -> block-scoped recall -> pack -> use -> outcome -> consolidation`

Hermes sees the content-free block map, expands a branch, and chooses the primary leaf that
owns each independently changeable record. At recall time Hermes selects one or more blocks;
the native backend searches physically inside each selected primary block. UM-MEM then applies
scope, status, trust, deduplication, and budget checks. The selected backend's declared
score mode determines how results from its partitions are merged. There is no alias
router, substring coverage selector, or budget-filling set-cover stage.

Raw host evidence remains a hidden witness. Atomic records retain origin references, and
materialized block descriptors never replace canonical records. Secondary block memberships
are navigation links rather than additional physical search partitions.

Native physical isolation follows each provider's actual partition primitive. A materialized leaf
is keyed by its stable `backend_partition_id` within the workspace: MemPalace uses a distinct
Chroma collection family, Hindsight uses a distinct native bank, Mnemosyne uses a distinct
native session partition, and Zvec uses a distinct native collection. Catalog revision remains provenance and does not rotate a partition
when an unrelated block changes or the leaf is renamed. Partitions are created lazily on the
first eligible write. Metadata still carries the typed identity for validation, but scoped recall never scans a shared native
store and then filters by `logical_block_id`.

Multi-block recall searches only the selected partitions. MemPalace semantic queries merge
their shared native cosine scores before applying the global result limit. Empty queries
and backends without a qualified shared score space use partition rank rounds
(`A1, B1, A2, B2`). One workspace selects one backend; this is not federation or score
calibration across different providers. Explicit `search_all` enumerates materialized partitions and
is counted separately from scoped recall. Projections that predate the current provider marker or
whose bank/session/collection identity does not match fail closed with
`native_projection_rebuild_required`; start a fresh store rather than performing a hidden dual read.

Health exports content-free identity and metrics per materialized leaf: records, writes, scoped
and broad recalls, hits and zero-hits, failures, stored record bytes when known, p50/p95 observed
latency, rendered tokens, consolidation/compaction/eviction/expiry counters, last access, and
partition health. `expand_block` exposes this operational summary together with retention,
lifecycle, retrieval mode, token budget, writable reason, promotion requirements, and allowed
operations. The branch navigation envelope never includes record bodies, titles, or record IDs.

Workspace navigation is paginated. `expand_block(block_id, limit=50, cursor="")`
returns at most one page of direct children, `child_count`, `next_cursor`, and the
catalog revision. Cursors bind workspace, profile, parent and revision; a topology
change invalidates the cursor instead of silently skipping or duplicating children.
`block_map(child_limit=50)` bounds child previews, and a deep workspace view is
limited to 201 materialized node descriptions. Child summaries retain the full
count and branch/writability semantics even when their child-ID preview is empty.
Follow their cursor with `expand_block` to inspect the omitted children. Protected
policy-tree cursors additionally bind the compiled policy-catalog revision.
Graph neighbors have a separate bounded page: `related_blocks_page(limit=50, cursor="")`
returns `related_count` and `related_next_cursor`. `navigation_envelope` accepts
`related_cursor` independently of its child cursor. Graph cursors bind the
workspace, profile, block and catalog revision; a changed graph requires a fresh
page walk. Expansion previews the selected block's first edge page and at most
16 displayed children's first eight edges; its completeness flags distinguish
that preview from a full neighborhood.

Expansion uses `block_health` for only the selected node and visible child page.
It reads persisted native-partition identities and observations without probing a
provider or enumerating unrelated partitions. `diagnostics_status=not_probed` and
`metrics_scope=native_partition` distinguish these observations from current
backend readiness; explicit `health`/preflight operations retain that role. A
custom backend without this operation reports `unsupported`. Partition counters
describe the physical partition, not causal usefulness or per-user outcome credit.
Scoped current-record counts use both the profile and stable physical key,
including published membership overrides after split/merge. Historical physical
partitions may coexist within a leaf: their observations remain visible, while
superseded records do not inflate current counts. Event observations span catalog
revisions of the same physical identity. Identity/key pairs are validated using
the owning backend's constructor before deduplication or profile admission.
The internal materialized leaf descriptor keeps exact current-record counts from indexed SQLite
membership rows and at most 64 record/revision IDs as a diagnostic preview. Its
`record_ids_truncated` flag marks a larger leaf; exact record lookup and paginated
listing use the canonical membership index rather than that preview. Descriptor
refresh and catalog navigation do not deserialize every record in the leaf.
Derived descriptor refresh has its own short leaf-scoped interprocess gate,
outside the canonical writer lane; it never runs model or native calls.
Lossless source-coverage checks resolve registered source IDs through the exact
reference index and verify the returned canonical segments byte for byte; they
do not enumerate the whole store.

Exact record IDs and host governance constraints remain deterministic. The model owns semantic
placement and block choice but cannot claim trust, verification, authority, or promotion.
Historical recall evaluates retained records at the explicit `as_of` instant, including
their validity interval and expiration. An unrestricted historical query may inspect
retained expired versions; it does not restore deleted bytes or make them current.
`as_of` denotes event validity. Optional `known_as_of` is a separate, timezone-aware
cutoff on the trusted SQLite commit time: recall first selects the canonical head
committed by that time, then applies `as_of` to its event interval. A source's
`recorded_at` is provenance, not the time the memory learned it. If the source has
no event start, the result carries `event_time_status=unknown`; no date is inferred.
Known-time reads use the retained SQLite history rather than a current native index.
Historical dependency validity at the old knowledge cutoff is not yet reconstructed,
so these results remain advisory and should be checked against source evidence.
Source-speaker filtering in the public MCP context path currently runs after a
bounded native candidate pool; its diagnostics report `source_speaker_filter_complete=false`
and the candidate limit rather than implying that excluded speakers were absent.
Historical material is never eligible for directive delivery. Invalid query timestamps
are rejected at admission, before a backend compares or selects records.
Canonical hit conversion retains expiration alongside the validity interval. Public
historical hits carry the requested `as_of` and advisory authority. Final evidence
packing retains those temporal fields and labels historical cards explicitly in both
full and compact headers, within the same total context budget. Compact current cards
may omit the start date from the header to retain the fact; the date remains in item
metadata. End dates, expiration and historical query dates remain explicit. A retained historical
fact must not acquire a current-status label merely because it is still accepted.

Canonical `derived_from` edges are exact dependencies on retained source revisions.
Current recall, native hit admission, accepted-record pinning and active-mask resolution
check these dependencies transitively against canonical heads, lifecycle and validity.
A correction, retraction or expired source makes the dependent record unavailable as
current context; it does not rewrite or erase its accepted bytes. Promotion preserves
the candidate's exact dependencies in the same canonical commit. A correction is a
new authored revision and must declare its own changed derivation links.

Hosts can call `MemoryClient.validate_dependencies(record_id, as_of=..., max_nodes=...,
max_edges=...)` for a content-free `DependencyValidation` receipt: `valid`, `invalidated`
or `incomplete`, checked record IDs and exact issue codes. The defaults bound each
traversal to 256 nodes and 1024 edges; incomplete traversal grants no current-context
eligibility. An explicit historical `as_of` checks source validity at that instant;
unrestricted historical inspection retains the original record history. Ordinary
references and `cites` edges are not dependencies. Index-only witnesses and external
evidence have no canonical record lifecycle and remain the verifier's responsibility.

Existing exact, temporal, usage, and outcome stores form one system; there is no parallel
memory engine.

Agent-authored amendments are immutable canonical successors. The caller supplies the expected
revision, and the writer commits one `supersedes` edge or rejects a stale compare-and-swap.
Corrections to accepted records create separate, unverified proposals. The first proposal creates a
new canonical candidate with a correction mutation identity and no predecessor; subsequent changes
revise that candidate. The kernel writer rebases its exact membership against a newer compatible
catalog before validating the target blocks. Unrelated topology growth does not rewrite the accepted
record or grant promotion. Host-issued
`working_set_owner` metadata identifies the current author and generation; the proposal
uses the current client session lineage, without inheriting the accepted record's owner
or publication identity. Independent generations have separate proposal identities, while
an exact repeat within one generation is idempotent. The MCP correction path preserves
this metadata. Promotion remains a separate evidence-authorized action.
Lifecycle successors preserve the record's exact primary and secondary block IDs while binding
their membership to the current catalog snapshot. Unrelated later topology growth therefore does
not prevent an otherwise valid retraction, while missing or invalid target blocks still fail closed.

Each `(workspace, profile)` owns one protected root and an otherwise free semantic tree. Block IDs
are service-generated UUIDs and remain stable across rename and move. No project node, branch name,
or subject taxonomy is required. The governance catalog remains an internal policy vocabulary; it
does not dictate the model-visible topology. Tree ownership is separate from the eight typed graph
relations: `related_to`, `supports`, `contradicts`, `derived_from`, `supersedes`, `applies_to`,
`learned_from`, and `validated_by`.

Topology, edges, catalog revisions, active masks, maintenance observations, mutation receipts,
rejected mutations, and tombstones share the canonical local SQLite catalog. Structural operations
use integer compare-and-swap revisions and explicit reasons. Split and merge require exact record
coverage. Archive fails closed around active children, incoming links and active mask
references; it preserves record bytes and history. Hard deletion additionally requires an
archived block with no children, graph edges or current records and leaves a topology
tombstone. Neither operation is a whole-workspace data purge. There is no dual read of
the retired YAML dynamic-block store.

Split and merge stage exact record copies in each proposed leaf's native partition before
publishing a new topology generation. One catalog transaction archives the source leaves,
publishes the new leaves and provenance edges, and switches each record's effective membership.
The immutable record payload is never rewritten. Until publication, readers retain the old
tree and native partitions. A prepared or projections-ready generation remains in the journal
after a process failure and can be resumed by its operation ID; a source CAS conflict aborts
that generation without changing the visible tree. Active masks and graph references must be
explicitly moved by the agent before evolving a referenced leaf. A new catalog format fails
closed against older stores and requires a clean store; there is no automatic migration.

Topology expiry is an explicit maintenance operation over stored
`lifecycle_policy.expires_at` values. A block without that field has no implicit
deadline: host metadata, session scope, age and emptiness never assign one.
Maintenance preserves occupied or referenced blocks and rechecks the current
expiry and revision before archival. It does not delete record history. Hosts
own retention choices and the decision to invoke maintenance.

An evidence-authorized mutation refreshes the record membership against the current catalog before
commit, while the kernel writer still validates every referenced block. A later phase may therefore
extend the shared topology without invalidating an earlier staged candidate whose block placement
has not changed.

`ActiveMemoryMask` is the durable exact-scope pointer set for context that must survive between
rounds. It stores selected blocks, pinned records or selectors, priorities, budgets, lifecycle, and
catalog revision. `ActivationLease` is its immutable per-round materialization and trace. Active
materialization never promotes authority, excludes suppressed lifecycle states, and refreshes
canonical heads. Recall remains a separate temporary operation over passive memory.

## Passive memory, active memory, and recall

These are deliberately different mechanisms:

- **Passive memory** is the durable tree, graph, and canonical record history. It survives turns,
  compaction, session boundaries, and later work in the same workspace/profile. Passive records are
  not injected into every model call.
- **The catalog** is the content-free navigation layer over passive memory. Descriptions, paths,
  links, lifecycle facts, and health metrics let the model decide where to look without loading all
  record bodies.
- **Active memory** is a durable mask over selected parts of passive memory. The mask stores typed
  pointers and selectors, not copied prose. Before each round, UM-MEM resolves it into one bounded
  `ActivationLease` and injects that context automatically. The model changes or suspends the mask
  explicitly when its working focus changes. A typical model workflow is: navigate, recall or write,
  decide that part of the context must remain in focus, then set or amend the mask. When focus ends,
  the model suspends or deactivates it.
  A host may pass an exact `scope_chain` to `resolve_active_mask`; records outside
  it are excluded before rendering. Codex and OpenCode derive a project identity
  from the current Git root for automatic context, so older records in a shared
  workspace remain available through explicit search without entering that context.
- **Recall** is a temporary, model-requested read from explicitly selected passive leaves. A recall
  result does not become sticky and does not amend the active mask unless the model performs a
  separate typed mask mutation.

This distinction is what permits long sessions without treating repeated context compaction as the
memory system. The transcript may compact, while durable records, corrections, topology, and graph
links remain available. Only the currently useful active slice is automatically rematerialized;
everything else stays navigable and recallable.

## Host observations and failure semantics

A host observation is evidence, not a semantic memory write. `file_created`, `source_opened`,
`test_passed`, `decision`, and every other host event retain one hidden raw witness and return its
evidence reference. They never select a semantic block, create a model-visible record, mutate the
tree, or alter the active mask. If the fact is worth retaining semantically, the model chooses an
existing visible leaf or creates one and performs a separate explicit write linked to the witness.
There is no event-kind-to-block table and no second hidden semantic tree.

Memory is advisory to the host. Listener failures, missing observations, empty recall, unavailable
active-mask materialization, and analysis warnings are recorded explicitly, but they do not change
the success status of the agent's task or benchmark run. The host continues without injected memory
for that round and may retry later. This is degraded operation, not a silent successful recall.
Machine content checks may score benchmark answers, but provider telemetry and memory-maintenance
diagnostics are never completion gates.

An explicit-reference context load and a semantic recall are different operations. Loading declared
block or record references uses deterministic scoped retrieval and does not manufacture a search
query merely to rank them through a vector index. Semantic recall remains query-driven. MemPalace
serializes native reads and writes across processes so one phase cannot observe a partially updated
native projection while another phase is promoting memory.

## Illustrative evolution across sessions

There is no required `project` level and the names below are examples, not a schema. A model could
organize one workspace/profile like this:

```text
Memory (protected root)
├── Host integration evaluation work
│   ├── current harness decisions
│   └── verified run evidence
├── provider comparison experiment
│   ├── open questions
│   └── rejected approaches
├── design conversation with Alice
│   └── agreed constraints
└── reusable lessons
```

In session A, the model may create the first branch, record two independently changeable facts in
separate leaves, and activate only the current decision plus the most relevant evidence. On a later
turn in the same scope, that active slice is supplied without recall.

In session B, the same workspace/profile catalog and passive records still exist. The model may add
the comparison-experiment branch and connect one reusable lesson to both branches with typed graph
edges instead of copying it. It then explicitly selects a new mask for the new focus; an unrelated
session-scoped mask is never guessed or silently reused.

In session C, the work may be a conversation rather than a code project. The model can add a
conversation-shaped branch directly below the root, reuse an existing constraint through a graph
link, or decide that no new block is warranted. A different profile receives a different protected
root and cannot navigate or recall these blocks or records.

As a branch evolves, the model may rename or move it without changing its stable ID, split a
genuinely overloaded leaf, merge compatible small leaves, archive obsolete material, and revive it
later. Metrics and maintenance observations are evidence for those choices; they do not perform the
choices.

`navigation_envelope` joins the tree path, children, typed related edges, content-free backend health,
advisory maintenance observations, and explicit proposal options. Numeric thresholds may create an
observation, never a block, edge, lifecycle transition, consolidation, or mask mutation.

Outcome reporting preserves incomplete observation: selected, rendered or cited records without
an applied event are unknown, never automatically unnecessary. A host may issue the exact
not_applied usage event only after completing its action-use observation, with supporting evidence.
Contradictory applied/not_applied events for one decision and record are rejected. Explicit nonuse
does not claim causal benefit or harm. Applied records without verified causal credit remain neutral
with the missing linkage stated; helped/hurt still require the existing outcome-credit proof.
Negative outcome credit requires typed decision, applied-record, action-trace, and verifier linkage;
tool-result prose cannot create `hurt`. Consolidation accepts an explicit source-ID set and explicit
target leaf, preserves raw witnesses and `derived_from` provenance, stages one bounded candidate,
and commits only an accepted independent tribunal result. A rejection leaves the active head intact
and records a digest-bound receipt.

Memory remains advisory. Only typed host evidence and verified outcomes may change policy or
authority.

## Agent-authored memory review at work boundaries

A host with a real turn-completion boundary may ask the same working agent for one final memory
review before ending the managed session. The review is not transcript retention and does not let
the host infer semantic placement. The agent first performs any useful recall, record correction,
candidate write, or active selection through its ordinary scoped tools, then emits a
`MemoryReviewDecision`. The host binds that decision to the exact content-free tool observations in
a `MemoryReviewReceipt`.

The closed outcomes are `changed`, `unchanged`, `unavailable`, and `forbidden`. A `changed` decision
must name the exact pending candidate or selected block IDs observed by the host. An `unchanged`
decision cannot hide a retained candidate or selection. `unavailable` and `forbidden` carry no
false activity. Receipts are idempotent and seal further memory activity for that work boundary.

This review is advisory: a missing or unavailable review never changes a successful task result.
It exists to distinguish a deliberate no-change decision from an accidentally skipped memory
step, not to force filler into the tree. Promotion remains a separate host-governed transition
after independent acceptance.

Execution state remains separate from semantic memory. `WorkingStateCheckpoint` is available for
hosts that can provide immutable raw events and have negotiated the required lifecycle callbacks.
An MCP-tools-only adapter must not pretend to provide live-state capture or transcript replacement;
it keeps working continuity disabled and may still use the boundary review above. This preserves
the useful SKILL.state principle—explicit validated state where the host can prove it—without
turning UM-MEM into a hidden agent loop or requiring a fixed schema for every domain.

## Non-negotiable anti-drift boundary

The model owns semantic topology: block meaning, placement, cross-links, active focus, and proposed
consolidation. Those decisions must arrive as explicit typed mutations with actor, reason, scope,
and expected revision where applicable.

Deterministic code may enforce identity, workspace/profile isolation, exact scope, schema shape,
compare-and-swap revisions, lifecycle legality, tree acyclicity, edge integrity, record coverage,
budgets, cryptographic binding, and typed verifier receipts. It may report numeric health facts and
retain authenticated raw witnesses.

Deterministic code must not infer semantic placement, relevance, lifecycle, topology, active focus,
negative experience, or consolidation from prompt prose, tool-result prose, filenames, branch
names, keywords, regular expressions, substring matches, or similar heuristics. It must not create a
mandatory project/code/log/reflection taxonomy, silently mutate the tree from a metric threshold, or
turn a model-authored claim into machine verification. Failure to materialize active memory is an
explicit degraded diagnostic; it is never reported as a successful recall and never blocks the host
task.

Composite topology operations such as split and merge must be recoverable as one logical mutation:
either catalog changes, record reassignment, provenance links, and receipts all commit, or the
operation remains explicitly incomplete and recoverable. A best-effort in-process rollback is not a
substitute for a persisted transaction or compensating-saga contract.

Context packing records elapsed monotonic milliseconds for recall and packing, before journal writes. This nested span is not additional end-to-end latency. The decision binds the current catalog revision; returned item metadata carries its exact context_decision_id for host callbacks. cost_tokens remains a packing estimate, not provider usage, and excludes this returned diagnostic metadata. Rendering records delivery preparation only; application and causal benefit require separate observations.


## Recoverable correction publication

Correcting an accepted record stages an advisory candidate in its own canonical chain. Its
`correction_target_id` and `correction_target_revision_id` pin the accepted predecessor; creating
or amending that proposal does not change the accepted record. A recoverable host promotion
requires exact independent semantic review of the candidate ID, revision and content hash.

The native canonical commit checks both the target CAS and the exact candidate head, profile,
workspace, revision and content in the same transaction. The accepted successor records its
operation and source identity. Its dependency edges come from the reviewed candidate, while
rollback restores the exact predecessor's dependency edges atomically. Neither operation
silently drops a claim's source constraints.

`compensate_promotion` starts from the journal's pinned source even if the host crashed before
appending a publication receipt. It restores only the accepted correction owned by that operation
and source revision, using the committed canonical mutation identity and target CAS. An exact
rollback successor is recognized on retry. Foreign or newer target revisions remain unchanged.
A newer proposal is also preserved, but does not prevent restoration of the operation-owned
target. Recovery inspects the target even when late cleanup has already retracted the proposal.

Retraction receipts describe the proposal: `retracted` and `already_retracted` follow target
restoration where needed; `superseded` preserves a newer proposal or target. They do not assert
that unrelated memory was removed. The restored accepted record is a new historical revision,
with the predecessor's content, assurance, evidence and dependencies; the original audit trail
remains intact. A CAS conflict remains a recoverable failure rather than a successful rollback.
## Candidate identity and publication

An independent creation has an opaque identity. Equal content is only a duplicate
cluster diagnostic; recall preserves independent records and deduplicates only
exact record/revision references. A caller-issued operation ID replays the
original creation acknowledgement, including after later amendments. Amendments
use explicit record identity and a current CAS revision.

Publication requires a verifier-owned semantic assessment of the exact full
candidate revision, source evidence, profile, destination leaf and scope. The
assessment records the actual author and independent verifier model identities,
versioned verifier instructions, four typed outcomes and reasons. A generic
passed-test receipt supplies evidence but cannot verify candidate meaning.
Exact extracts have a narrower deterministic route: when a host has registered
source bytes, `lossless_atomic` checks every UTF-8 segment against its source
hash and byte range during writing. The matching record is accepted as advisory
`evidence_backed`, with an exact-source ledger receipt. This certifies only
faithful copying; it cannot certify an inferred claim or grant directive authority.
For semantic publication, quoted instruction text is evaluated by the semantic
verifier; deterministic code enforces structure, scope, revisions and typed
outcomes. Publication replay
acknowledges the original immutable published revision without republishing a
newer head.

## Host working views

A working checkpoint stores a model-authored advisory view and exact source
references. It is owned by workspace, profile, session, task and owner generation.
Coding fields are optional; semantic placement is one model-selected task leaf.
The host owns raw history and recovery pages. Exact anchors and the recent tail
remain outside the replaceable compression window. Independent semantic
verification precedes checkpoint commit. A failed memory path leaves standard
host compression available.
