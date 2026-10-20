---
name: organize-memory-topology
description: Organize and evolve long-term memory by the meaning of current work. Use when inspecting or changing a semantic memory tree, graph links, block lifecycle, consolidation targets, or the active-memory mask between rounds.
---

# Organize Memory Topology

Read `docs/lethyx/memory_architecture.md` and `docs/lethyx/data_model.md` when
the repository is available. They define the active/passive boundary and scope
model used by this skill.

Inspect the existing tree, synopses, graph links, active mask, and advisory maintenance observations before changing anything.

- Reuse an existing block when its meaning fits.
- Compare each new source with the synopsis, not only the writable flag. Update stale descriptions and topical tags at meaningful work boundaries.
- Split mixed subjects needing independent retrieval; merge redundant boundaries when the meanings fit. A deliberate no-change decision is valid, and more leaves is not a quality metric.
- Before a full review, save an evidence-grounded keep/split/merge plan. Compare boundaries across leaves, not just descriptions independently. Execute the chosen plan with current revisions, verify published receipts, and report genuinely deferred decisions separately. A wider synopsis alone does not complete a chosen structural change. Do not manufacture split/merge activity for its own sake.
- Enumerate every membership page with `list_records` and inspect exact records before mapping a split/merge. A navigation preview or top-k recall is not a complete record set.
- Create a block only for a stable area that can evolve independently. Keep its synopsis short and let the memory service generate its opaque ID.
- Preserve one ownership parent. Use typed graph links for cross-branch relationships instead of duplicating records.
- Treat machine observations as facts only. Decide from the actual memory content whether to amend, move, split, merge, archive, consolidate, or relink, then use an explicit typed operation.
- Require exact record mappings and revisions for structural mutations. Preserve sources and provenance.
- For large chosen splits, author every record's destination in bounded batches and persist the full mapping before native publication. Unassigned records never receive an automatic default. Distinguish exact record IDs from leaf IDs; the native service checks record revisions.
- Put automatically recurring context in the active mask. Use recall for temporary access to passive memory.
- Treat session and task IDs as scope and provenance. Do not create a visible session or project block unless its meaning warrants one.
- Keep memory advisory: a failed recall or mask materialization must not block the host task.
- Do not require standard branch names, a project level, or a domain taxonomy.
- Do not infer semantic structure, lifecycle, errors, or outcomes from keywords, substrings, regular expressions, prompts, or tool-output prose.
