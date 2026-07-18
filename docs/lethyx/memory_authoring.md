# Agent-authored Memory

Agents may propose memory freely through `MemoryWriteProposal`, but the authoring router normalizes every write into `MemoryRouteDecision`.

The router maps kind/tags/requested block into store, tier, scope, surface, lifecycle, and trust. Direct directive writes and core lessons are blocked unless they come through curated promotion.

Reports:

- `.memory/drive/state/memory_authoring_report_latest.json`
- `.memory/drive/state/memory_authoring_events.jsonl`
