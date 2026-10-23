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

## Measuring the product promise

Contract tests show that the boundaries hold; they do not show that an agent
remembers useful things or spends fewer tokens. For a live harness evaluation,
compare the same task sequence with memory disabled, with tools-only MCP, and
with the harness's automatic review enabled. Include short turns where nothing
should be saved, supported corrections of earlier facts, a long history in one
block, dense graph relationships, and a project switch. Keep model, backend,
task order, and context budget comparable; report any host-specific capability
that differs.

Report the full distribution across runs rather than one favorable example:

- later-task answer quality with source support and explicit unknowns;
- useful autonomous writes, correct no-write choices, duplicate current heads,
  stale records, and supported corrections;
- exact-source fidelity checks separately from truth or task-outcome checks;
- cross-project leakage and directive contamination;
- prompt, tool, and review-continuation tokens; model calls; end-to-end latency;
  and total cost over the sequence;
- recall latency, index/projection lag, and storage growth as history and graph
  degree increase.

The one-pass `Stop` review has an immediate token and latency cost. Claim a net
gain only when the later-task benefit exceeds that cost over a stated horizon.
An empty atlas, installed hook, or newly written candidate is not by itself a
quality result.

## Reporting results

Test output is the verification record. When a change needs a number rather than
a pass/fail boundary, record how the number was produced (corpus, backend,
backend version, embedding model, and command) next to the claim. Do not present
a single-run measurement as a general property of the memory layer.
