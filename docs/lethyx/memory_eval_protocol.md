# Memory Evaluation Protocol

Lethyx includes deterministic, no-LLM static memory evaluation for release gates and research reports.

## Ablation conditions

- `A0_NO_MEMORY`
- `A1_RETRIEVAL_ONLY`
- `A2_PROACTIVE_DIRECTIVE_ONLY`
- `A3_PROACTIVE_PLUS_SUPPLEMENTAL`
- `A4_PROACTIVE_PLUS_HINDSIGHT_CANDIDATES_GATED`
- `A5_PROACTIVE_PLUS_TRIGGERS_PLUS_UTILITY`
- `A6_CANONICAL_MEMPALACE_ONLY`
- `A7_MNEMOSYNE_PASSIVE_RECALL_ONLY`
- `A8_CANONICAL_PLUS_MNEMOSYNE`
- `A9_CANONICAL_PLUS_HINDSIGHT_PLUS_MNEMOSYNE`

## Metrics

Static metrics include first-action alignment, directive precision/recall, contamination rate, retrieval leakage rate, evidence coverage, token ROI, false promotion rate, and candidate yield.

## Commands

```bash
python -m lethyx.memory.cli eval --workspace ws1 --mode static --json
python -m lethyx.memory.cli research-report --workspace ws1
```

Reports are written to `.memory/drive/reports/`, including `memory_ablation_report_latest.json` and `memory_metrics_table.md`.

## Definition of Done for memory changes

A memory change is complete only when authority boundaries are tested, reports are machine-readable, candidate promotion remains gated, and docs distinguish implemented from planned optional live smoke.
