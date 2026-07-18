# Active Memory Architecture

Lethyx memory has one authority layer and several passive evidence layers.

## Source of truth

Durable writes go through the backend-neutral writer
`lethyx.memory.kernel.writer.write_memory_event`, which validates policy and
then writes to exactly one selected backend from
`lethyx.memory.backends.factory`: `mempalace`, `hindsight`, or `mnemosyne`.
`canonical` and dual-backend runtime modes are removed. Durable behavior rules
are active only after BKB promotion. Hindsight and Mnemosyne are not passive
mirrors when selected; each is the durable backend for that run/workspace, with
native health reported separately from Lethyx adapter health.

## Surfaces and authority

Memory is normalized into five surfaces:

- `directive`: behavior-changing active memory. Only curated core/BKB and accepted promotion output may render here.
- `current_run_state`: current run/phase state, not cross-run behavior.
- `supplemental_evidence`: passive evidence for inspection, reports, and tribunal proof.
- `archive_hint`: pointers to cold/warm records, logs, graph neighbours, or external project chunks.
- `reflection_candidate`: proposed memory awaiting evidence and tribunal.

Retrieval, graph recall, and backend-native recall are supplemental, archive, or
candidate only unless converted into a BKB candidate and accepted by the
tribunal/promotion gate. They cannot enter `[ALWAYS-LOADED MEMORY]` as
directive memory by retrieval alone.

## Lifecycle states

`candidate`, `quarantined`, `superseded`, `deprecated`, and `retracted` are non-directive. Only `active` verified/curated rules with the right scope and evidence are eligible for directive rendering.

## Memory injection report v2

Every persisted LLM bundle writes `.memory/drive/state/memory_injection_report_latest.json` with schema `2`. It records included, skipped, blocked, supplemental evidence, counts, hashes, and invariants:

- `directive_contamination_detected`
- `retrieval_leakage_detected`
- `duplicate_overlay_detected`

## Prospective triggers

BKB rules with `type: prospective_trigger` use deterministic trigger fields such as `phase_any`, `changed_paths_any`, `artifacts_exist_any`, and `run_state_flags_any`. Trigger matching never calls semantic search or Hindsight.

## Logical blocks

Custom memory levels are `MemoryBlock`s over stable stores. For example `github_repo_capsules` uses `palace.codeptr`, tier `warm`, surface `supplemental_evidence`, and promotion policy `typed_evidence -> tribunal_accept`.

## Common pitfalls

- Do not install Mnemosyne/Hindsight provider prompt prefetch as a second prompt-memory authority in Lethyx phases.
- Do not treat utility score as trust. Low utility can downrank, not delete or quarantine.
- Do not load whole GitHub repositories into prompt; store repo capsules/chunk cards as passive code pointers.
