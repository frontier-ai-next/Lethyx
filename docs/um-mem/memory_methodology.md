# Memory Methodology (Backend-Neutral)

This document describes the conceptual model for UM-MEM memory without binding it to any specific storage engine, vector database, archive system, or provider integration. The goal is to define what memory is, how it is structured, how it becomes useful in a run, how it mutates safely, and how multiple memory layers can coexist without creating multiple authorities.

## 1. Core thesis

Memory is not a pile of retrieved text. Memory is a typed, evidence-aware, hierarchical system of blocks that can influence future work only through explicit authority surfaces.

A memory system needs to answer five separate questions:

1. What is remembered?
2. Where does it live in the hierarchy?
3. What is allowed to use it?
4. How much authority does it have?
5. How can it change over time?

The key design rule is separation of authority:

- Active memory can change behavior before the agent asks for retrieval.
- Passive memory can inform, explain, or provide evidence, but it cannot silently become behavior.
- Candidate memory can be proposed freely, but it must pass evidence and promotion gates before it becomes active.
- Current-run state can guide the present run, but it is not automatically durable cross-run knowledge.

This keeps autonomy high while preventing accidental prompt contamination from stale logs, speculative notes, or unverified reflections.

## 2. Memory block as the atomic unit

A memory block is the smallest complete unit of reusable memory behavior. It is not just text. It is a record plus a contract.

Each block should be treated as a ready implementation of some memory capability:

- a rule the agent must obey;
- a lesson from verified work;
- a project summary;
- a code pointer;
- a run-state snapshot;
- a failure pattern;
- an external research capsule;
- a reflection candidate;
- a competency gap;
- a graph node or relationship;
- a retrieval index over some domain.

A block may be physically represented in many ways, but conceptually it needs stable metadata and behaviors.

### 2.1 Required block identity

Every block needs stable identity:

- `id`: canonical identifier for dedupe, audit, and graph links.
- `title`: short human-readable name.
- `kind`: what the block represents, e.g. rule, lesson, idea, run_state, code_pointer, gap, external_capsule.
- `content`: payload or summary.
- `owner`: manager, workspace, phase, agent, supervisor, human, evaluator, importer, or external source.
- `created_at`: creation time.
- `updated_at`: latest mutation time.
- `version`: monotonically increasing version or content hash when mutation is allowed.

Identity must be stable enough that later reports can cite the same block and distinguish mutation from a new block.

### 2.2 Placement metadata

A block is useful only if it knows where it belongs:

- `scope`: global/manager, workspace, project/domain, run, phase, subtask, file, external project, or user/session.
- `workspace_id`: workspace boundary when applicable.
- `run_id`: current run or verified source run.
- `phase_id`: phase that produced or consumes it.
- `subtask_id`: subtask boundary when applicable.
- `path_hint`: logical location, e.g. project/component/module/document area.
- `parent_id`: containment parent if the block is in a tree.
- `graph_refs`: related nodes and typed edges.

The placement metadata is what prevents accidental leakage across unrelated workspaces, stale run state, or wrong projects.

### 2.3 Authority metadata

Every block must declare what it is allowed to do:

- `surface`: how it can appear in context.
- `directive`: whether it can change behavior as an instruction.
- `trust_level`: evidence/trust classification.
- `lifecycle`: active, candidate, quarantined, superseded, deprecated, retracted, archived, transient.
- `evidence_refs`: proof objects, artifact IDs, ledger IDs, test IDs, human approvals, source hashes, or external citations.
- `promotion_state`: none, proposed, under_review, accepted, rejected, needs_more_evidence.
- `risk_level`: low/medium/high/safety-critical.
- `privacy_class`: public, workspace_private, secret_pointer, sensitive_summary, or non-injectable.

Authority is never inferred from good wording. Authority comes from metadata plus promotion gates.

### 2.4 Retrieval and budget metadata

Memory is bounded by prompt budget and relevance:

- `tags`: semantic and operational tags.
- `markers`: deterministic flags used by compilers and gates.
- `keywords`: lexical query helpers.
- `embedding_ready`: whether semantic retrieval can index it.
- `token_cost`: estimated prompt cost.
- `expected_value`: expected usefulness.
- `last_used_at`: latest injection or retrieval use.
- `usage_count`: how often it was used.
- `helped_outcomes`: verified uses that improved outcomes.
- `hurt_outcomes`: verified uses that caused harm or confusion.
- `decay_policy`: how it is downranked or archived.

Retrieval metadata must not override trust. A frequently retrieved unverified note is still unverified.

### 2.5 Mutation metadata

Memory is alive. Blocks may mutate, but only through explicit mutation semantics:

- `mutable`: whether updates are allowed.
- `mutation_policy`: append_only, replace_with_version, summarize, consolidate, archive, supersede, quarantine, retract.
- `mutation_reason`: why it changed.
- `supersedes`: older blocks replaced by this block.
- `superseded_by`: newer block that replaced this one.
- `conflicts_with`: blocks in contradiction.
- `repair_needed`: missing evidence, stale source, broken pointer, failed retrieval, invalid schema.

Mutation should preserve auditability. Silent destructive deletion is not memory management; it is loss of evidence.

## 3. Levels of memory

A robust memory system is hierarchical. Different levels answer different questions and have different lifetimes.

### 3.1 Manager / system level

Purpose: persistent operating principles and cross-workspace policy.

Contains:

- identity and mission;
- safety invariants;
- universal engineering preferences;
- cross-project lessons;
- accepted behavior rules;
- known global pitfalls;
- long-lived capability knowledge.

Authority: can be active if curated and verified.

Risks:

- overgeneralizing a lesson from one workspace to all workspaces;
- letting stale global rules override a more precise workspace rule;
- storing temporary run state as a permanent directive.

### 3.2 Workspace level

Purpose: memory specific to one project/workspace.

Contains:

- project architecture summaries;
- local conventions;
- accepted project-specific behavior rules;
- verified fixes and lessons;
- recurring failure modes;
- code pointers and component maps;
- evaluation expectations;
- external dependency notes.

Authority: can become active inside that workspace only, if promoted.

Risks:

- leaking workspace-specific assumptions into another workspace;
- retaining generated artifacts as if they were source truth;
- losing workspace identity when paths are normalized incorrectly.

### 3.3 Domain / project-family level

Purpose: shared knowledge for related projects without becoming universal.

Contains:

- domain terminology;
- framework or stack conventions;
- reusable patterns;
- integration notes;
- benchmark/evaluation patterns.

Authority: usually supplemental or candidate until accepted in a specific workspace.

Risks:

- treating domain similarity as proof of applicability;
- importing large external contexts into prompt instead of concise capsules.

### 3.4 Run level

Purpose: current run state and verified run outputs.

Contains:

- current goal;
- selected plan;
- active blockers;
- decisions made in the run;
- verification outputs;
- tool results and artifacts;
- stop/abort/retry state.

Authority: active for the current run only; not cross-run durable unless promoted.

Risks:

- stale run state reappearing in a future run;
- treating a failed attempt as a general lesson;
- persisting agent self-report without evidence.

### 3.5 Phase level

Purpose: phase-specific commitments and context.

Contains:

- phase plan;
- phase constraints;
- allowed/forbidden tools;
- phase-specific risks;
- artifacts required for exit;
- watcher feedback;
- remediation notes.

Authority: active only inside the phase boundary.

Risks:

- applying implementation-phase assumptions to verification;
- carrying review warnings as permanent rules without acceptance.

### 3.6 Subtask level

Purpose: isolate worker/subagent context and prevent cross-contamination.

Contains:

- subtask goal;
- assigned files;
- local attempts;
- local evidence;
- completion claim;
- limitations and handoff notes.

Authority: current subtask only unless parent phase promotes it.

Risks:

- a subtask claiming success without parent verification;
- storing local speculative notes as durable project memory.

### 3.7 Artifact / file / code-pointer level

Purpose: point to concrete material without loading everything into prompt.

Contains:

- file paths;
- symbol names;
- summaries;
- line ranges;
- source hashes;
- dependency edges;
- generated artifact IDs;
- validation commands.

Authority: supplemental evidence or archive hint.

Risks:

- code pointer drifting after edits;
- large files becoming prompt spam;
- stale summaries hiding changed source.

### 3.8 External source level

Purpose: retain useful external information with provenance.

Contains:

- documentation capsules;
- research summaries;
- issue/PR references;
- external repositories or APIs;
- citations and fetch metadata.

Authority: never automatically directive; it is evidence or context.

Risks:

- stale external facts;
- missing citation/source hash;
- copying too much raw external content instead of compressing into referenced capsules.

## 4. Surfaces: how memory enters context

The surface is the most important authority marker. It tells the prompt compiler how the block may be rendered.

### 4.1 Directive surface

Directive memory is active behavior memory. It changes what the agent should do before retrieval.

Examples:

- accepted safety invariant;
- verified project-specific rule;
- required workflow constraint;
- human-approved operating preference;
- promoted recurring failure prevention rule.

Requirements:

- explicit lifecycle `active`;
- verified or curated trust;
- bounded scope;
- evidence or human approval;
- conflict resolution;
- prompt budget fit;
- no candidate/quarantined/deprecated/retracted lifecycle.

Directive memory must be rare and high quality.

### 4.2 Current-run-state surface

Current-run-state memory is active only inside the present run.

Examples:

- active plan;
- current phase status;
- open blockers;
- retry count;
- stop request;
- latest verification result.

Requirements:

- run_id binding;
- phase/subtask binding when needed;
- expiration or run boundary;
- cannot become durable without promotion.

### 4.3 Supplemental-evidence surface

Supplemental evidence can be shown to the agent as supporting context, but not as instruction.

Examples:

- verified lesson body;
- external documentation capsule;
- code pointer;
- retrieved prior run observation;
- test output summary;
- graph neighbor.

Requirements:

- source/provenance;
- trust marker;
- bounded token cost;
- clear labeling as evidence, not directive.

### 4.4 Archive-hint surface

Archive hints are pointers to cold/warm data that can be fetched if needed.

Examples:

- log location;
- report path;
- source file pointer;
- prior run ID;
- compressed external repository capsule.

Requirements:

- stable pointer;
- reason to fetch;
- no direct behavior authority.

### 4.5 Reflection-candidate surface

Reflection candidates are proposed memory that may be useful, but not yet accepted.

Examples:

- agent-generated lesson candidate;
- summarizer-generated pattern;
- external archive suggestion;
- failure hypothesis;
- proposed behavior rule.

Requirements:

- candidate lifecycle;
- required evidence list;
- promotion path;
- never rendered as directive.

## 5. Active vs passive memory

### 5.1 Active memory

Active memory is proactively compiled into the agent's working context before normal retrieval. It shapes behavior.

Active memory should be:

- small;
- verified or curated;
- scope-bound;
- conflict-resolved;
- deterministic in selection;
- audited in injection reports.

Active memory includes:

- core identity and mission;
- active behavior rules;
- current run commitments;
- phase-specific constraints;
- selected high-value warnings.

Active memory is dangerous if unmanaged because it can silently steer every decision. Therefore active memory needs the strictest gates.

### 5.2 Passive memory

Passive memory is remembered data that can be retrieved, searched, cited, summarized, or inspected.

Passive memory includes:

- logs;
- run summaries;
- old observations;
- code pointers;
- external capsules;
- unpromoted lessons;
- candidate reflections;
- graph neighborhoods.

Passive memory is high-volume and lower authority. It is useful for recall and evidence repair, but it must not become hidden instruction.

### 5.3 Why both are needed

Only active memory is too brittle: the system forgets long-tail context unless it was pre-promoted.

Only passive memory is too weak: the agent may never search for the memory it needs, or may retrieve it too late.

A balanced system uses:

- active memory for small, verified, high-impact constraints;
- passive memory for broad context and evidence;
- candidate memory for self-improvement proposals;
- current-run memory for live state.

## 6. Hierarchy and graph

Memory is both a tree and a graph.

### 6.1 Tree hierarchy

The tree answers containment:

- manager contains workspaces;
- workspace contains projects/components;
- run contains phases;
- phase contains subtasks;
- subtask contains artifacts;
- artifact contains file/symbol pointers.

Tree placement is useful for scoping, cleanup, budget, and lifecycle boundaries.

### 6.2 Graph relationships

The graph answers relationships that are not simple containment:

- supports;
- contradicts;
- supersedes;
- duplicates;
- depends_on;
- derived_from;
- cites;
- used_in_phase;
- helped_outcome;
- hurt_outcome;
- blocks_promotion;
- repairs_evidence;
- shares_domain_with.

Graph edges are useful for promotion, contradiction detection, evidence repair, and value tracking.

### 6.3 Why both are necessary

Tree without graph loses cross-cutting relationships.

Graph without tree loses scope boundaries.

Together they allow an agent to ask:

- Is this memory allowed here?
- What does it depend on?
- What contradicts it?
- What evidence supports it?
- What replaced it?
- Where should updates propagate?

## 7. Markers, tags, and deterministic routing

Each block must carry explicit labels. The labels are not decoration; they are machine-readable routing.

### 7.1 Tags

Tags are broad retrieval and grouping hints.

Examples:

- `architecture`
- `verification`
- `bug_fix`
- `failure_pattern`
- `external_docs`
- `code_pointer`
- `safety`
- `workspace_rule`

Tags help retrieval and ranking, but should not decide authority alone.

### 7.2 Markers

Markers are deterministic flags used by compilers and gates.

Examples:

- `requires_evidence`
- `do_not_inject`
- `current_run_only`
- `human_approved`
- `verified_by_test`
- `conflict_open`
- `superseded`
- `needs_refresh`
- `candidate_only`
- `secret_pointer_only`

Markers should be normalized strings or typed enum values, not natural-language hints.

### 7.3 Routing fields

Routing fields determine storage, lifecycle, and surface:

- kind;
- scope;
- surface;
- lifecycle;
- trust_level;
- tier;
- owner;
- source;
- required_capabilities;
- allowed_phases;
- denied_phases.

Routing must come from structured fields, not regex over prose.

## 8. Lifecycle and mutation

Memory should move through explicit states.

### 8.1 Candidate

A candidate is proposed memory. It may be useful, but it has not earned authority.

Allowed actions:

- store as candidate;
- attach evidence requirements;
- show as reflection candidate;
- request review;
- search as passive context when explicitly allowed.

Forbidden actions:

- render as directive;
- silently overwrite active memory;
- claim verified status.

### 8.2 Active

Active memory is accepted, scoped, and eligible for directive/current-state injection.

Allowed actions:

- inject within scope;
- track usage and value;
- refresh evidence;
- supersede through gate;
- downrank under budget if not safety-critical.

Forbidden actions:

- silent deletion;
- uncontrolled rewrite;
- cross-scope injection.

### 8.3 Quarantined

Quarantined memory is blocked because it is unsafe, contradicted, stale, suspicious, or evidence-broken.

Allowed actions:

- keep for audit;
- show in reports;
- request evidence repair;
- cite as blocked.

Forbidden actions:

- directive injection;
- normal retrieval as trusted evidence.

### 8.4 Superseded

Superseded memory was replaced by a newer block.

Allowed actions:

- keep backward links;
- cite history;
- show replacement pointer.

Forbidden actions:

- inject old rule when replacement applies.

### 8.5 Deprecated

Deprecated memory is still historically true or useful but no longer preferred.

Allowed actions:

- use as archive hint;
- include if explicitly requested.

Forbidden actions:

- default active injection.

### 8.6 Retracted

Retracted memory was wrong or invalid.

Allowed actions:

- retain as audit trail;
- cite as negative evidence.

Forbidden actions:

- normal recall as advice;
- promotion without new evidence.

### 8.7 Archived

Archived memory is low-frequency or cold.

Allowed actions:

- fetch through archive hint;
- restore if evidence shows renewed value.

Forbidden actions:

- consume active prompt budget by default.

## 9. Promotion model

Promotion is the process of turning low-authority memory into higher-authority memory.

A safe promotion pipeline:

1. Observe or generate candidate.
2. Normalize into typed block.
3. Attach scope and source.
4. Attach evidence refs or mark evidence missing.
5. Detect duplicates and conflicts.
6. Evaluate trust and risk.
7. Review against current active memory.
8. Accept, reject, quarantine, or request more evidence.
9. If accepted, assign surface and lifecycle.
10. Record promotion decision and reason.

Promotion must be explicit. A retrieval result cannot promote itself.

## 10. Evidence model

Memory quality depends on evidence.

Evidence can include:

- tool result IDs;
- test outputs;
- verifier reports;
- ledger/event IDs;
- artifact paths and content hashes;
- human approvals;
- external citations;
- source snapshots;
- before/after comparisons.

Evidence refs should be stable and inspectable. The block should say what each evidence ref proves.

Trust levels should distinguish:

- untrusted speculation;
- agent-reported claim;
- workspace-verified fact;
- externally/publicly verified fact;
- supervisor-verified decision;
- human-approved rule.

Trust is not confidence. Confidence is probabilistic. Trust is procedural evidence.

## 11. Conflict handling

Conflicts are normal in a living memory system.

Conflict types:

- same scope, contradictory behavior;
- newer source contradicts older source;
- global rule conflicts with workspace rule;
- candidate contradicts verified memory;
- two active rules compete for prompt budget;
- source pointer changed after summary;
- test evidence no longer reproduces.

Conflict resolution policy:

1. Prefer narrower scope over broader scope when both are valid.
2. Prefer verified over unverified.
3. Prefer current source hash over stale source hash.
4. Prefer safety invariant over convenience rule.
5. Prefer explicit human/supervisor decision over agent self-report.
6. If unresolved, block directive promotion and surface as conflict.

Conflicts should create graph edges and reports, not silent overwrites.

## 12. Forgetting and decay

Forgetting is controlled lifecycle movement, not disappearance.

Safe forgetting actions:

- downrank;
- move to archive;
- mark deprecated;
- supersede;
- quarantine;
- retract;
- consolidate into a summary;
- reduce prompt priority;
- require refresh before use.

Unsafe forgetting actions:

- deleting verified evidence without audit;
- deleting safety invariants for low usage;
- overwriting source without version history;
- removing negative lessons because they are inconvenient.

Decay should consider:

- age;
- usage;
- verified value;
- risk;
- scope;
- replacement availability;
- source freshness.

Low usage can reduce rank. It should not automatically reduce truth.

## 13. Memory in a run

A run should use memory in stages.

### 13.1 Run start

At run start:

- compile active manager/workspace rules;
- load current task and workspace identity;
- include phase-independent safety constraints;
- retrieve only high-value startup context;
- record what was injected.

The agent should start with enough memory to avoid known mistakes but not enough to drown in old context.

### 13.2 Phase start

At phase start:

- compile phase-specific active memory;
- include allowed/forbidden tools and phase commitments;
- include relevant current-run state;
- provide archive hints for likely-needed passive memory;
- avoid injecting unrelated prior phase context.

### 13.3 During execution

During execution:

- record important observations;
- attach evidence to writes;
- update current-run state;
- refresh passive recall when context drift appears;
- keep run-scoped memory isolated to current run;
- avoid promoting while the evidence is incomplete.

### 13.4 Verification and review

During verification:

- bind claims to artifacts/tests;
- identify which memories helped or hurt;
- mark candidate lessons;
- repair evidence gaps;
- detect stale or contradicted active memory.

### 13.5 End of run

At run end:

- summarize verified outcomes;
- preserve final artifacts and evidence refs;
- propose lessons or rule updates;
- archive transient state;
- keep run state from leaking as durable memory;
- record memory injection/value reports.

## 14. Block contracts

Every memory block type should define a contract.

A block contract specifies:

- schema;
- allowed scopes;
- allowed surfaces;
- required evidence;
- lifecycle rules;
- mutation policy;
- retrieval policy;
- promotion policy;
- budget policy;
- privacy policy;
- conflict policy;
- evaluator/tests;
- cleanup behavior.

This makes each block a complete reusable implementation, not a loose note.

### 14.1 Example contract shape

A backend-neutral block contract can be represented conceptually as:

```yaml
block_type: verified_lesson
schema_version: 1
allowed_scopes: [workspace, manager]
allowed_surfaces: [supplemental_evidence, directive_after_promotion]
required_fields: [id, title, content, scope, trust_level, evidence_refs]
required_evidence: [verification_result]
lifecycle:
  initial: candidate
  promotable_to: [active, archived]
mutation:
  policy: supersede_with_version
retrieval:
  default_surface: supplemental_evidence
  require_query_seed: false
promotion:
  requires_conflict_check: true
  requires_scope_check: true
  requires_human_or_supervisor_gate_for_directive: true
privacy:
  secrets: pointer_only
```

The exact storage format can vary. The contract should not.

## 15. Injection reports and audit

Every prompt-memory injection should be auditable.

An injection report should record:

- run_id;
- phase_id;
- workspace_id;
- input bundle hash;
- active memory included;
- passive memory included;
- candidates queued;
- skipped items and reasons;
- blocked items and reasons;
- token estimates;
- source backends or block families;
- invariants checked;
- contamination/leakage flags.

The report lets operators answer:

- Why did the agent see this?
- Why did it not see that?
- Did retrieval accidentally become directive?
- Did a candidate leak into active rules?
- Was current-run state stale?

## 16. Active/passive contamination rules

The system must prevent contamination between surfaces.

Forbidden contamination:

- retrieval result rendered as instruction;
- candidate rendered as accepted rule;
- stale run state rendered in a new run;
- external documentation rendered as policy;
- unverified reflection rendered as durable lesson;
- low-trust memory included without label;
- deprecated/superseded memory injected without replacement context.

Required labels:

- directive memory must say why it is active;
- supplemental memory must say it is evidence/context;
- archive hints must say fetch is optional;
- reflection candidates must say what evidence is missing;
- current-run state must carry run/phase identity.

## 17. Value learning

Memory should learn which blocks are useful.

Signals:

- injected and used successfully;
- injected and ignored;
- retrieved and cited;
- retrieved and caused wrong action;
- helped verification pass;
- caused conflict or confusion;
- required evidence repair;
- replaced by better block.

Value learning affects ranking and budget. It should not by itself change trust or delete memory.

A high-value block may deserve earlier injection. A low-value block may be archived. A harmful block may be quarantined if evidence supports harm.

## 18. Privacy and secrets

Memory must never turn secrets into prompt content by accident.

Rules:

- store secret references as pointers, not raw values;
- tag sensitive summaries;
- prevent indexing raw credentials;
- redact external API keys and tokens;
- require explicit permission for cross-workspace sharing;
- keep private workspace memory scoped;
- make exported reports safe by default.

Privacy is a memory concern because durable memory outlives the run that created it.

## 19. Multi-agent memory

In multi-agent runs, memory needs ownership and isolation.

Principles:

- subagents get scoped memory, not the whole manager store;
- subagent self-reports are candidate memory until verified;
- parent orchestrator decides what crosses subtask boundaries;
- watcher/supervisor comments are current-run state unless promoted;
- shared memories must carry source agent and evidence refs;
- conflicting subagent claims should create conflict edges.

This prevents one worker's speculative context from becoming global truth.

## 20. External and passive archives

External archives are valuable but should remain passive by default.

External memory should be compressed into capsules:

- source identity;
- fetch time;
- source hash;
- summary;
- relevant excerpts;
- applicability limits;
- refresh policy;
- citation URL/path;
- confidence and trust markers.

The system should avoid dumping entire repositories, websites, or logs into active prompt context. It should store pointers and fetch targeted sections when needed.

## 21. Implementation-independent backend roles

Even without naming any concrete backend, memory systems often contain different physical roles:

- canonical store: durable source of truth;
- hot state store: current run/phase state;
- semantic index: retrieval over text-like blocks;
- graph store: typed relationships;
- archive store: cold logs and large artifacts;
- mirror store: optional copy for experiments or external tools;
- evaluator store: reports and metrics;
- proposal queue: candidate memory awaiting review.

These roles may be implemented by one system or many systems. The methodology requires that authority remains explicit regardless of physical storage.

## 22. Why multiple memory layers can coexist

Multiple memory layers are useful if they have different authority:

- one layer can be canonical;
- another can be a passive archive;
- another can generate reflection candidates;
- another can optimize retrieval;
- another can store graph relationships;
- another can hold transient run state.

Multiple layers become dangerous only when more than one layer can independently declare success, truth, or directive behavior.

The solution is not to ban multiple layers. The solution is to give every layer an explicit role and force all active decisions through typed contracts, evidence gates, and injection reports.

## 23. Minimal invariants

A backend-neutral memory system should enforce these invariants:

1. No directive memory without active lifecycle and accepted authority.
2. No candidate/quarantined/deprecated/retracted block rendered as directive.
3. No passive retrieval result silently changes behavior.
4. No cross-workspace injection without explicit scope permission.
5. No current-run state reused as durable memory without promotion.
6. No verified memory deletion without audit trail.
7. No trust upgrade without evidence or approval.
8. No secret value stored or injected as raw memory.
9. No unbounded prompt growth from archives or external projects.
10. No conflict overwrite without graph/report trail.
11. No promotion based only on agent self-report.
12. No hidden second authority path.

## 24. Practical authoring rules

When an agent writes memory, it should choose the weakest correct authority:

- If it is a guess: candidate idea.
- If it observed a tool output: observation with evidence ref.
- If verification passed: verified outcome or lesson candidate.
- If it should change future behavior: proposed rule, not direct directive.
- If it points to code: code pointer with path/hash.
- If it is current status: current-run state.
- If it is external context: external capsule with citation.
- If it is sensitive: pointer or redacted summary.

Memory authoring should prefer structured fields over prose.

## 25. Operator questions the methodology should answer

A complete memory system should let an operator ask:

- What memory did this run see?
- Which memory was active vs passive?
- Which block caused a behavior change?
- Why was a block promoted?
- Which evidence supports this lesson?
- Is this memory workspace-specific or global?
- Is this block stale, superseded, or contradicted?
- Which memories helped successful runs?
- Which memories caused failures?
- Why is localhost/external archive/mirror involved, if any?
- Can this memory be safely forgotten?
- Can this candidate become a rule?

If the system cannot answer these, memory is not yet operationally safe.

## 26. Summary

The conceptual model is:

- Memory is typed blocks, not text blobs.
- Blocks live in a hierarchy and graph.
- Blocks carry markers for scope, surface, lifecycle, trust, evidence, risk, budget, and mutation.
- Active memory changes behavior and must be rare, verified, and audited.
- Passive memory provides broad context and evidence but must stay labeled.
- Candidate memory enables self-improvement without immediate authority.
- Current-run memory helps execution but must expire or be promoted.
- Mutation is explicit lifecycle movement, not silent overwrite.
- Multiple physical memory layers are acceptable only when authority remains single and typed.

This methodology should remain stable even if the physical memory backend changes completely.
