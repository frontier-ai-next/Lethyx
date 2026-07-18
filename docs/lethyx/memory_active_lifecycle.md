# Active Memory Lifecycle

Active memory in Lethyx means verified behavior memory that can influence the next phase before retrieval. It is intentionally narrower than all remembered data.

## Autonomous actions allowed

The lifecycle manager may:

- update usage/value state;
- propose `downrank_under_budget`, `refresh_needed`, `archive_to_warm`, `supersede_candidate`, `quarantine_candidate`, or `needs_evidence`;
- write lifecycle reports under `.memory/drive/reports/`.

## Autonomous actions forbidden

It must not:

- promote a candidate into active directive memory;
- silently delete verified durable memory;
- remove safety invariants or core identity;
- rewrite BKB/core without the promotion gate.

## Usage, utility, decay, and risk marks

Usage/value stats live in `.memory/drive/state/memory_value_stats.json` and records in `memory_use_records.jsonl`. Decay is implemented as proposal state, not as destructive mutation. Safety invariants are protected from low-usage decay.

## Safe forgetting

Forgetting is tier/downrank/archive/supersede, never silent delete. Contradictions can produce quarantine proposals, but applying active BKB changes remains gated.
