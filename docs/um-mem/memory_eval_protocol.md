# Memory Evaluation Protocol

UM-MEM is validated by its contract test suite. This tree ships no bundled
benchmark harness, no ablation runner, and no LLM judge: memory quality claims
belong to the host, the benchmark, or the study that makes them.

## What a memory change has to prove

A change to retrieval, promotion, or lifecycle behavior is only complete when
the contract suite still proves the authority boundaries:

- directive contamination: candidate, quarantined, superseded, deprecated, and
  retracted memory never renders as directive;
- retrieval leakage: recall never crosses the workspace, profile, or scope
  chain it was called with;
- workspace isolation: one workspace cannot read another workspace's blocks;
- evidence coverage: curated directive memory carries evidence that a reader
  can trace back to a source record;
- promotion gates: retrieval and backend-native recall stay supplemental until
  evidence and tribunal gates accept them;
- lifecycle state: mutations are recorded, auditable, and reversible through the
  journal;
- backend selection: exactly one durable backend is selected per run and every
  write goes through the backend-neutral writer.

## Commands

```bash
pytest -q
pytest -q tests/test_memory_architecture_invariants.py tests/test_memory_authority_and_isolation.py
pytest -q tests/test_lifecycle_mutation_journal.py
```

The default suite never downloads an embedding model and never reads the
developer's user config. Tests that need a live optional backend are marked and
skipped unless that backend is explicitly available.

## Reporting results

Test output is the verification record. When a change needs a number rather than
a pass/fail boundary, record how the number was produced (corpus, backend,
backend version, embedding model, and command) next to the claim. Do not present
a single-run measurement as a general property of the memory layer.
