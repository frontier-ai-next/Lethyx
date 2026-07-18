# Memory Taxonomy

The taxonomy routes ideas, projects, logs, lessons, and state over existing stores.

| Level | Store | Tier | Surface | Authority |
|---|---|---|---|---|
| Manager core | `palace.charter` | always_on | directive when curated | not direct agent-writeable |
| Active BKB | core `bkb.yaml` + `palace.lesson` mirror | active/warm | directive | promotion gate required |
| Verified lessons | `palace.lesson` | warm | supplemental_evidence | proof substrate |
| Project ideas | `ideas.jsonl` | cold/warm | reflection_candidate | passive |
| Current run state | `palace.run` | hot | current_run_state | current run only |
| Logs | `palace.transient` | transient | archive_hint | never directive |
| External projects | `palace.codeptr` | warm/cold | supplemental_evidence | never whole repo in prompt |
| Hindsight candidates | proposal queue | candidate | reflection_candidate | gate required |
