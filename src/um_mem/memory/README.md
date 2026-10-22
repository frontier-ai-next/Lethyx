# UM-MEM memory runtime

The product memory namespace is backend-neutral:

- canonical contracts live in `um_mem.contracts`;
- durable writes go through `memory.kernel.writer` to exactly one selected
  backend (`mempalace`, `hindsight`, `mnemosyne`, or `zvec`);
- recall uses the typed recall plan, explicit block selection, and the single
  context payload packer;
- working continuity, prospective obligations, outcome telemetry, consolidation,
  and reconstructive retrieval remain evidence-bound and advisory.

The former parallel JSONL lesson/gap/profile and summary/context/reflection
runtime was removed during the Phase 7 cutover. Do not reintroduce a second
durable authority, compatibility facade, or fallback store.
