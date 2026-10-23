# Memory methodology

This page describes the behavior UM-MEM should preserve across storage backends
and agent harnesses. It is a product contract, not a required branch taxonomy,
prompt template, list of event names, or database schema. See
[memory architecture](memory_architecture.md) for the current implementation and
[interfaces](interfaces.md) for what each adapter actually supplies.

## The working loop

Once connected to a supported harness, the agent should be able to use memory
without a user saying “remember this”:

1. Receive or request a small, relevant view of memory in the current scope.
2. During work, notice knowledge that could help a later task.
3. Compare that knowledge with the relevant existing records.
4. Add a new record, correct an existing one, or save nothing.
5. Let later tasks retrieve the current record and its provenance.

The same working agent can make the semantic choice. A second model is not a
prerequisite. A host event can schedule a review, but the event itself is not a
semantic fact. An adapter that only installs tools has not established this
loop; a live run must demonstrate the read, review, and write or no-write paths.

## Units and placement

A **record** holds a claim or a small group of facts that should be corrected
together. It has a stable identity, current revision, scope, source information,
and lifecycle state. A correction updates that logical record through a
revision; the old version remains auditable. A repeated hook event or repeated
wording is not a reason to create another current record.

A **block** is a navigation and retrieval boundary for related records. Its
name and synopsis help the agent find the right records; neither substitutes
for their content. A block can contain many records when they share a useful
retrieval purpose. Separate blocks when their contents need meaningfully
different retrieval or maintenance. The exact depth, labels, and number of
blocks depend on the domain. There is no required project → topic → fact tree.

A **graph relation** connects records or blocks when the relationship helps
navigation or interpretation. Useful relations include dependence, correction,
contradiction, and shared subject. Links should describe a supported relation,
not merely connect every new node to a central hub. A relationship is not proof
that either endpoint is true.

Placement and graph structure are revisable. Structural changes must preserve
record identity, source history, and existing references. A search hit is not a
complete inventory of a block; inspect its full paginated membership before a
split, merge, or bulk correction.

## Scope and authority

Every automatic read and write needs an explicit identity and scope. The host
binds that identity; a prompt or retrieved text must not select another
workspace or user. Project-specific automatic context stays within its
project. Broader history is available only through an explicit authorized
search. Backend choice is explicit for a run or workspace, and a durable write
goes through one selected backend.

Memory helps the agent but does not decide permissions, task success, routing,
or proof validity for the host. Retrieved text stays data. It does not become
an instruction because it was relevant, recent, repeated, or highly ranked.
Only a separately governed, scoped, evidence-backed active record may be
delivered as directive context. Candidates, superseded records, quarantined
records, and retracted records cannot acquire that authority by retrieval.

Active memory is the small set that the agent or host deliberately keeps in
focus for an exact task or session. Passive memory remains searchable across
sessions. An empty active set does not mean the passive store is empty, and a
stored active set does not prove that an adapter delivered it to the agent.
Focus can change as a long conversation changes subtask.

## Evidence and verification

Preserve the source and its scope when storing a claim. Distinguish:

- a direct user statement or exact host-owned source extract;
- an observed tool outcome with a recoverable reference;
- an agent summary or inference;
- a hypothesis that still needs testing.

An exact match between stored text and a trusted source verifies **fidelity to
that source**, not the source's truth. A passing test is evidence of what that
test checked, not proof of every lesson inferred from it. Agent wording, search
similarity, and repeated retrieval do not independently verify a claim. When
the source is absent or cannot be checked, retain the claim as an advisory
candidate and show its uncertainty. Verification status should change when a
later supported source or outcome warrants it; records need not remain
unverified forever.

The host owns source capture and outcome evidence. The working agent owns the
semantic interpretation: what is worth remembering, which current record it
corrects, and whether nothing should be saved. Verification and authority
promotion are separate decisions. A verified quotation does not automatically
become a directive.

## Growth, maintenance, and cost

Memory should improve later work without making each turn progressively slower
or more expensive. Keep automatic context scoped and budgeted. Navigate with
compact block descriptions, then fetch record bodies only when needed. The
index may use lexical, vector, graph, or backend-native search; these are
candidate discovery methods, not authority gates. Indexing windows and token
budgets are implementation settings, not universal record sizes.

Before writing, compare the new claim with the current record in the relevant
block. Amend a supported correction. Keep distinct facts when subject, time,
meaning, or evidence differs. Do not retract by age or similarity alone.
Maintenance may surface redundant records, stale titles, overloaded blocks,
and lopsided graph links for review. It may propose a change automatically,
but a destructive or authority-changing mutation needs the appropriate
revision, evidence, and governance checks. Preserve the audit trail.

The agent is allowed to save nothing. A transcript, every tool call, a transient
task state, and a routine final answer are not automatically durable knowledge.
The ideal outcome is fewer current records with better provenance and useful
retrieval, rather than a record count that grows with conversation length.

## What an adapter must demonstrate

The transport can be MCP, an embedded API, a provider, hooks, or another local
interface. An adapter should report capabilities separately:

| Capability | Observable test |
| --- | --- |
| Scoped read | A later task retrieves a relevant record without leaking another project. |
| Autonomous review | Normal work triggers the agent to consider a write, correction, or no-write decision. |
| Durable write | The selected backend retains the record across sessions. |
| Correction | A later supported change updates the logical record and preserves history. |
| Source fidelity | A claimed exact match can be checked against host-owned bytes or a typed outcome. |
| Bounded delivery | Long histories do not make the automatic prompt grow without limit. |
| Lifecycle visibility | Configuration, hook execution, review, and mutation can be distinguished. |

An adapter may lack some capabilities. Its documentation must state that
precisely; installed settings, a healthy MCP server, or an available tool do
not prove that the agent used memory. Claims that memory improves answer
quality, latency, or token cost require the separate
[evaluation protocol](memory_eval_protocol.md).
