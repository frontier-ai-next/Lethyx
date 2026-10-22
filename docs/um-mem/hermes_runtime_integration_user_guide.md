# Hermes runtime integration

## Experimental reading profile

The profile accepts `retrieval_mode: "hybrid"` with `backend: "mempalace"`.
Its default remains `semantic`. Explicit record-ID reads retain their exact
semantic path even when the profile selects hybrid for search.

Use the blocks the working agent already chose while planning or calling a
tool. If none have been chosen, the agent inspects the navigation envelope.
There is no additional selector-model call. Search-all remains a separate
explicit action; temporary recall does not change an active mask.
When the final context budget hides ranked candidates, the compact recall
receipt includes their record and leaf IDs in `unrendered_record_refs`.
The agent may use a scoped `record_ids` recall to read a relevant full source;
the references themselves do not contain evidence. No automatic semantic
rerouting or shared-store scan is performed.

For an experimental host profile set `recall_total_tokens: 4096` and
`recall_max_calls: 2`. Both default to `null`. The first native host turn opens
the allowance; repeated start notifications for that same turn do not reset it.
Active/granted prefetch, one ordinary recall and one agent-requested refinement
share the allowance, including navigation/JSON headers and error responses.
Prefetch uses at most half the allowance and reserves 256 bytes for the native
Hermes memory fence and system note. The first recall reserves room for the
refinement. Late prefetch from an earlier turn is discarded, and transport does
not hold the budget lock. The current strict ceiling counts UTF-8 bytes as a conservative
token upper bound for byte-tokenized host models; it can deliver fewer than
4096 actual tokens. This differs from the existing per-call four-characters
estimate. At exhaustion the provider returns a bounded refusal, then no bytes
when even that refusal cannot fit. Recall failure remains advisory.

`working_context_enabled` remains opt-in. Independent candidate verification is
a publication/write operation, not a verifier-model call on every read.
Procedure candidates additionally need two independent applied outcomes bound
to passed supervisor-ledger verifier reports before publication. The provider
does not count an agent's unverified success statement as such an outcome.

The native provider imports UM-MEM inside Hermes' Python process. UM-MEM
requires Python 3.12+, so the installer checks Hermes' actual interpreter and
refuses to stage the native provider on older runtimes. For those Hermes
versions, use Hermes' MCP client, which starts UM-MEM in the Python environment
where the MCP extra is installed:

```console
python -m pip install "um-mem[mcp]"
um-memory hermes-mcp-install --workspace default --profile default --backend mempalace --json
um-memory hermes-mcp-status --json
```

This edits only the `mcp_servers.um-mem` entry in `$HERMES_HOME/config.yaml`,
preserving other YAML settings and comments. It exposes UM-MEM tools, but does
not select UM-MEM as Hermes' native memory provider and does not deliver an
active mask before every model round.

On Hermes with Python 3.12 or newer, stage and inspect the native provider with:

```console
um-memory hermes-provider-install --hermes-home "$HERMES_HOME" --code-root /path/to/UM-MEM --workspace default --profile default --backend mempalace --json
um-memory hermes-provider-status --json
```

The installer writes a user provider wrapper under
`$HERMES_HOME/plugins/um-mem`, a manifest with the selected backend's Python
dependencies, and an identity-bound UM-MEM profile config under
`$HERMES_HOME/um-mem/`. It does not activate the provider, edit Hermes
`config.yaml`, disable built-in memory, or remove `MEMORY.md`/`USER.md`. Finish
setup through Hermes so its package manager prepares dependencies in Hermes'
own environment and selects the provider:

```console
hermes memory setup
hermes memory status
```

Restart Hermes after dependency preparation. The optional `um_mem_observer`
plugin can be enabled from `hermes plugins`; it adds passive typed lifecycle
capture and block-first tools. Do not treat a staged wrapper or the observer's
presence on disk as proof that a live callback ran. UM-MEM status separates
`staged`, `active`, and importability in the selected Hermes Python;
`live_callback_proven` stays false until a runtime receipt exists. The obsolete
`--replacement` hard-cutover option now fails without modifying the profile.

`hermes-provider-restore` remains only for restoring profile backups created by
older UM-MEM installers. It is not the uninstall command for the new staged
provider.

Standalone Hermes receives a content-free topology map and exact-scope active context.
The required `prefetch` hook is query-independent: only explicit active selectors can invoke
backend recall; user text does not select a memory branch or trigger an additional search.

Managed `prefetch_only` profiles receive one packed advisory context from their exact
`read_block_ids` and optional `read_record_ids`. An active mask requires its pinned
`active_mask_revision`; it may prioritize already granted records but cannot expand that
grant. Active and passive selections are deduplicated before the final packer applies one
`recall_max_tokens` ceiling, including its advisory header and provenance metadata.
The host receives those packed bytes directly, without duplicate context or an in-prompt
activation trace. Empty grants produce no prompt text. Context-prefetch events report
preparation, not confirmed harness consumption or answer use.

The configured `python_executable` selects the local memory service's Python
environment. Its absolute invocation path is preserved, including virtual-environment
symlinks; resolving such a symlink to the base interpreter would lose that
environment's installed dependencies. A missing configured interpreter fails explicitly.

Exact-grant preparation stops reading record bodies as soon as the eligible record
limit is reached. Scope and lifecycle exclusions do not consume that limit; a zero
limit reads no bodies. The grant index and active-selection order remain authoritative,
and the token packer can further reduce the delivered content.

`MemoryClient.resolve_active_mask` also accepts `expected_revision` and `max_tokens`.
A revision mismatch returns an empty typed resolution; the caller's token ceiling can
reduce the mask's allocation without changing the stored mask. A zero-token warm-up
checks scope and revision without materializing records or querying a backend.

The profile's explicit `recall_max_tokens` is authoritative over inherited
environment settings unless environment overrides are explicitly enabled. Positive
budgets below 64 tokens remain exact; recall requests can reduce a profile budget,
but cannot increase it. Hosts should pass their issued context allowance into the
profile instead of relying on the provider's default.

The provider exposes `um_mem_provider_status`, navigation envelopes, explicit block recall and broad fallback,
closed remember and immutable record amend, full free-topology/typed-edge CRUD, active-mask
lifecycle and resolution, typed maintenance observations, bounded consolidation into an explicit
leaf, promotion proposal, recent typed verification receipts, and read-only graph/Obsidian export.
Replacement status reports the exact registered names and is healthy only when this contract is
complete.
The agent selects a root and non-empty leaf-block list before block recall. An empty active
mask in a new session does not mean that passive memory is empty. Before declaring a past
fact unavailable, the agent navigates and recalls. If evidence is missing or unrelated, it
refines its query and inspects other branches or graph links. When navigation cannot identify
a suitable branch, or scoped attempts remain insufficient, it may explicitly choose the
separate global search. Block IDs and global search can never be mixed in one call.
This is guidance to the model, not an automatic relevance selector or a task-completion gate.
Adds and corrections require an explicitly selected writable leaf, and corrections name the target
record ID. Host observations are retained as hidden raw witnesses without creating semantic
candidates or choosing a leaf from event text. Their receipt may return a placement manifest; the
agent then decides whether the evidence warrants a separate semantic write into an explicit
writable leaf.

Each semantic write should contain one independently changeable fact, or a small connected group,
and may provide typed `child_claims` with their origin segments. These claims and the hidden raw
witness remain recoverable; a short canonical record is an indexable view, not a replacement for
the source evidence. If a remember call is rejected because the selected block is unknown,
index-only, or otherwise unwritable, the receipt may include `suggested_route`. The agent can call
the same tool with `operation="repair"`, the suggested writable block, `repair_of` equal to the
original witness id, and the returned `evidence_refs`. Repair reuses the original witness and does
not create a second source claim or silently choose a semantic block. The canonical record keeps
the same witness reference and a `repair_of` marker for audit and later repair analysis.

The local service records a returned mutation error as a `failed` attempt rather than leaving
its transport receipt `pending`. The compact receipt retains the error type, reason code and
error digest; source text remains in the original host trace. Retrying the same correlation ID
reopens that receipt with the original request fingerprint, timestamp and operation identity.
An unexpected dispatcher failure returns its request ID with an unknown durable outcome and
a typed `dispatcher_unexpected_error`; the service records the internal cause and releases
the request slot. The host must inspect the original operation before any retry.
A loopback transport timeout returns `transport_timeout`, the original `operation_id`,
and `durable_status=unknown`. A host that retries the mutation must reuse that ID.
The client passes one monotonic deadline through its loopback request. If the listener
deadline expires before a mutation's durable receipt claim, it cancels that operation
and reports `durable_status=not_started`. After the claim begins, it reports `unknown`
and lets the worker finish. Both responses retain the original operation ID. Queue
saturation has its own `dispatch_queue_saturated` reason code.
Startup-lock waits, endpoint probes, and lease requests use the same client deadline;
expiry before a call is sent reports `not_started` for the mutation.
Listener shutdown stops accepting new work and returns without waiting for a started
dispatch worker to finish; that worker can still complete and publish its original result.
Dispatch workers are daemon threads so a stalled native call cannot hold process exit.
Background service lanes share a 750 ms shutdown wait; unfinished durable items
remain recoverable after restart.
If exit interrupts projection after the canonical commit, the pending outbox is
reconciled when the backend reopens.
Facade and Hermes exception receipts distinguish connection failures, timeouts, busy store
locks, and access denial. A generic `OSError` is reported as a memory operation failure
without marking it retryable as a transport failure.
Dispatcher exceptions retain the exception type and request ID with a distinct reason code;
their durable outcome remains unknown until the original operation is inspected.
A changed request still conflicts, and a committed success cannot be replaced by a failure.
The conflict carries `reason_code="mutation_receipt_conflict"` and `retryable=false`.
Do not repeat changed arguments under that operation ID. After a confirmed
`route_rejected=true, saved=false` receipt, a corrected model-selected leaf is a
new proposal with a new invocation ID. Persist that ID before dispatch and reuse
it for transport retries. An ambiguous failure is not proof that no record was
committed and does not permit a new creation.
`failed` describes the returned attempt, not a rollback: canonical outboxes and native projection
receipts must still be inspected independently before declaring recovery complete.

`amend` requires the current revision and appends a successor linked by `supersedes`; it never
updates a canonical record in place. A stale expected revision fails closed. Metadata patches
cannot change trust, authority, verification, scope, status, or delivery policy. Trusted or
directive memory continues through the existing correction proposal and tribunal route.
Workspace-owned blocks use integer CAS revisions and service-generated UUIDs. One profile root is
protected; the rest of the tree has no mandatory names or project layer. Active parents must be
index branches, cycles are rejected, and move/revive/archive/delete are separate lifecycle calls.
Split and merge require explicit complete record mappings. Typed graph edges never substitute for
tree ownership. Hermes exposes `um_mem_provider_split_block` and
`um_mem_provider_merge_blocks` with an optional idempotency `operation_id`.
`um_mem_provider_pending_topology_evolutions` lists incomplete generations;
`um_mem_provider_resume_topology_evolution` resumes the exact persisted mapping.
`um_mem_provider_abort_topology_evolution` explicitly abandons an unpublished
mapping with an audit reason when native projection cannot be recovered.
Native target partitions are prepared before an atomic catalog publication.
The agent must relocate active masks and graph references to source leaves first.

`navigation_envelope` and `expand_block` may be called on any block reached from the root map, so
deeper catalogs do not stop at the first child level. The envelope exposes the tree path, explicit
typed graph links, health facts, advisory observations, and proposal options; it never performs a
mutation. Hermes recall automatically includes eligible committed agent-tool candidates,
including durable candidates written in an earlier session. This is provider policy rather than
a model-visible flag; candidates remain advisory and still pass scope, lifecycle, and native
sidecar-integrity checks.

The `deep_harness` profile still requires stable workspace/session identity and authenticated
host observations. Missing lifecycle prerequisites are reported as
`native_lifecycle_unqualified`; they never trigger a hidden retrieval fallback or a quality
claim. Branch and compression sessions retain the root logical run, while reset creates a
new isolated run.
An authenticated `host_hook` can register a conversation source through `observe`
with `event_kind=conversation_turn`, exact `raw_evidence`, event ID, producer,
and optional `speaker` / `source_occurred_at`. The replayable receipt includes
`source_id`, SHA-256, byte length, and witness ID. The model then chooses the
semantic leaf and binds UTF-8 byte ranges through `lossless_atomic` remember;
the host observation itself does not place a semantic record.

Typed test, source, artifact, and reuse receipts may support a promotion proposal, but only the
tribunal can produce an accepted, verified advisory record. System governance categories may guide
authority and retention internally, but they do not define user-visible branches. Repetition and
similarity never raise authority.

The active mask is durable across rounds and resolves to one bounded `ActivationLease` with mask,
catalog, backend, selected-record, lifecycle-exclusion, scope-validation, and token traces. Suspend,
resume, amend, and deactivate are explicit CAS operations. Passive recall remains a distinct tool.

Tool and test observations use exact typed fields such as `observation_kind`; command strings,
filenames, prompts, and tool-output prose are never parsed to infer memory semantics or failures.

### Explicit managed prefetch

A managed host can set selected_records_only=true in its prefetch_only profile. The provider then delivers only pinned read_record_ids and records resolved by the exact active-mask revision, within read_block_ids. An empty selection delivers no ordinary records and does not warm a backend solely because a block is readable. A stale mask cannot broaden the pinned selection. The host keeps its granted catalog for explicit agent recall. The default remains false for existing standalone integrations. Context preparation diagnostics include the exact decision_id for application and outcome callbacks; preparation alone does not prove provider consumption.

## Verified native compaction projections

The optional working_context_enabled setting defaults to false. It requires
the native host CompactionRequest/CompactionProjection hook, durable host frame
persistence, and a configured independent verification_model. working_context_model
selects the author or uses the host compression route; verification_provider can
select a host-configured provider. Instructions are versioned as working-context/1,
working-context-review/1 and candidate-review/1. Disable the setting to return
to standard native compression.

The provider receives the exact replaceable window, frozen system/tool inputs,
profile and session. The model chooses a writable task leaf or proposes one
through ordinary topology tools. The provider preserves an immutable recovery
frame under the native profile's context_frames directory. UM-MEM stores the
advisory view and hashes/references, never a duplicate raw transcript. A native
frame resolver supports exact page-back after provider restart. Accepted views
use the standard historical-summary envelope; protected system text and recent
tool pairs retain host ownership. Rejection or unavailable verification is
diagnosed and leaves standard compression available.

The native producer rejects a source window containing an earlier compressed
summary until its complete transitive raw-source linkage is available. This
includes summaries restored without in-process metadata. Such a window uses
standard native compression; it is never relabelled as raw host evidence.
The core checkpoint contract supports recursive projections only with exact
prior projection references, transitive raw references and preservation proof.

Projection reuse pins request, workspace/profile/task, project path and file
state, model configuration and instruction versions. Semantic report reuse pins
one exact subject and fresh evidence. Candidate authorship comes from the
immutable host-observed authoring tool witness, including CAS amendments; the
current session model cannot stand in for a previous revision's author.
The witness must be bound to the exact candidate revision by its committed
native-operation receipt. CAS amendment acknowledgements are bound to the full
typed request and committed atomically with the canonical correction; recovery
cannot match another operation by copied witness metadata.
The native executor passes the live model with every authoring call, including
after an in-turn model fallback. Without that per-call identity, a candidate
remains advisory and cannot pass independent semantic publication. An original
add witness never authenticates a later revision made through another adapter.

Each provider model operation records actual model/provider, returned usage and
duration without source bodies. Logical call counts do not include unobservable
provider-internal retries. Main-wire/cache claims require host request/response
measurements; estimated token counts are labelled estimates. Enable by default
only after paired held-out evaluation shows no quality regression in any
required class, better aggregate quality and latency, and no higher total model
cost. Keep this setting opt-in when any gate is unproven.

## Store version boundary

Point this host profile only at a fresh store on the current schema. An older
backend index fails closed; no older-schema conversion command is shipped.
Pending projection journals from the current schema must drain before native
recall is considered ready.

Runtime health checks attest service availability, not completion of every native
projection. Hindsight and Mnemosyne expose `strict_partition_validation=false`
on the ordinary health path; explicit `preflight_health()` validates native
partitions exhaustively. Pending indexing must not prevent new durable writes.
Dependent native reads still require their projection barrier. A failed host
witness preserves typed retry diagnostics and reports the semantic write as
`durable_status=not_started`; it must not trigger a new semantic proposal while
the same native operation is awaiting recovery.

The public remember receipt exposes operation/commit/projection fields directly,
including the validated actual leaf ID. A durable `saved=true` is not an indexing
acknowledgement. Hosts must check the matching projection revision before a
dependent native read; benchmark completion uses this barrier independently of
the model's final response. Recall validation errors retain bounded argument
diagnostics; private transport error text remains excluded from compact results.
Remote Hindsight retain and recall use independent HTTP requests, like exact document
reads, so projection workers do not share an SDK event-loop-bound connection.
The native bank, strict block tag and canonical document identity are unchanged.
After the bounded projection retry budget is exhausted, the barrier returns
`native_projection_attention` with `retryable=false`; the pending outbox and
canonical source remain available for explicit repair.
