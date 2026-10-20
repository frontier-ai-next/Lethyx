# Operator guide

Pin one workspace, one native backend, and one lifecycle profile. A release-quality run
starts from a clean-store, verifies the version-exact schema, and fails closed on backend
mismatch or unavailable native infrastructure.

Inspect the content-free root map first, expand only the chosen branch, and pass exact leaf
IDs to recall. Use `search_all=true` only as an explicit agent decision. A request with
neither leaf IDs nor `search_all` is invalid.

Inspect `navigation_envelope` before structural maintenance. Treat health thresholds and
maintenance observations as advisory facts: only an explicit typed create, amend, move,
split, merge, edge, archive, delete, consolidation, or mask call may mutate memory. Verify
integer CAS revisions and exact record coverage before approving the operation.

Resolve the exact-scope active mask on every round. Its `ActivationLease` trace must identify
the mask/catalog/backend revisions, selected and rendered records, excluded lifecycle states,
scope result, and token budget. Active materialization is automatic; recall remains temporary.

Inspect context diagnostics for candidate, selected, and rendered IDs; missing evidence
requirements; token use; reconstruction; and the decision receipt. Do not treat sidecar
recall as proof that an unavailable native backend passed.

Native semantic recall uses the backend's declared score or partition-rank order and is
never filled with sidecar candidates.
Exact record lookups and canonical history can still use the index as their declared path.

For explicit forgetting, distinguish record lifecycle from physical erasure. Public
`forget(record_id, expected_revision=..., reason=...)` retracts the exact current revision;
it does not delete canonical history or raw evidence. Deactivate the exact active mask
before archiving a referenced block, then use the current block revision. Archive retains
records, while `delete_block` requires an archived, unreferenced block without current
records and preserves a tombstone. Protected host roots cannot be removed by ordinary
block mutations. `archive_session_block` is a narrow host operation for an exactly owned
session scaffold; it does not authorize deleting a project or user root.

`MemoryClient.erase` requires a durable host authorization ledger event and a stable
request ID. Its current native purge is limited to MemPalace and returns an exact-ID
certificate; canonical and raw evidence remain retained. There is no public bulk reset
that erases an owner's complete tree, all history and every native backend. A host-wide
reset therefore needs an explicit maintenance policy: stop all users of the selected
store, preserve separately owned settings and source documents, and account for external
native storage. Deleting local catalog files does not erase Hindsight service banks.
Never treat a directory reset or a successful retraction as proof of complete erasure.

MemPalace waits at most 30 seconds to acquire its operation gate. Set
`LETHYX_MEMPALACE_LOCK_TIMEOUT_SECONDS` to a finite number in `(0, 3600]` to
configure that wait before creating a client. Thread and process locks share
one monotonic wait budget. Exhaustion raises public `lethyx.MemoryBusyError`
with code text `memory_operation_lock_timeout`; snapshots report their existing
gate-unavailable outcome. The waiting caller performs no protected operation,
does not change backend and does not cancel the operation holding the lock.
The deadline bounds acquisition only, so a long native operation can still
complete while it owns the gate. A later caller can proceed after release.

Run quality trials sequentially. Parallel execution is a separate stress test because it
changes latency, scheduling, and session causality.
