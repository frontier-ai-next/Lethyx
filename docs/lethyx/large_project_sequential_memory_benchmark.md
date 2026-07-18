# Large-project sequential Hermes memory benchmark

Purpose: stress Lethyx/Hermes memory profiles on a real, popular, noisy repository rather than compact synthetic facts.

Human-facing Russian report with final 80-card, 67-questions-per-profile all-profile results:
`docs/lethyx/large-langchain-memory-benchmark-ru.md`.

Final 67-row-per-profile run artifacts:
`runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629`.

## Selected project

**langchain-ai/langchain** was selected from live GitHub metadata:

- ~140k stars, ~23k forks at selection time.
- Python monorepo, GitHub API size about 580 MB.
- Local shallow clone under `.external/langchain` at commit recorded by the benchmark.
- Repo has thousands of tracked files and multiple independently versioned packages.

Why this is a stronger benchmark than the small pre-release matrix:

- Many similar package names: `langchain`, `langchain-core`, `langchain-classic`, `langchain-openai`, `langchain-anthropic`, `langchain-tests`.
- Multiple package-level `pyproject.toml` files with overlapping dependency/test groups.
- Real developer guidance in `AGENTS.md`, including monorepo rules, test commands, public API rules, and PR conventions.
- Partner packages create negative-leak traps: `langchain-core` should not be attributed with OpenAI/Anthropic deps even though those packages exist nearby.
- Sequential late-update stage tests whether a profile can retain new memory after an already-large corpus has been seeded.

## Harness

Script:

```text
benchmarks/hermes_memory_profiles/run_large_project_sequence.py
```

It does the following:

1. Uses the existing clean-profile machinery from `run_profile_matrix.py`.
2. Builds deterministic fact cards from the local LangChain checkout:
   - repo inventory card;
   - `AGENTS.md` guideline card;
   - `README.md` overview card;
   - prioritized package `pyproject.toml` cards (`core`, `langchain`, `langchain_v1`, `standard-tests`, `openai`, `anthropic`, then other packages);
   - source cards with AST class/function/import summaries and bounded excerpts.
3. Creates one fresh Hermes profile per provider.
4. Seeds each provider/profile with the same corpus.
5. Asks memory-only questions in sequence in one resumed Hermes session per profile via:

```bash
hermes -p <profile> chat -Q --toolsets memory --source large-langchain-67-session-bench -q <prompt>
hermes -p <profile> chat -Q --toolsets memory --source large-langchain-67-session-bench --resume <session_id> -q <prompt>
```

6. Appends a late update card to the same profile/backend.
7. Asks a follow-up question that requires the late update.
8. Tracks tool-call deltas and context-summary counts from the profile session DB.
9. Writes incremental `results.jsonl`, `progress.json`, `summary.json`, and `report.md`.

## Question types

Current sequence covers:

- monorepo map recall;
- active-vs-legacy package distinction;
- package metadata and dependency recall;
- partner integration local-path/source-link recall;
- rationale recall from comments (`pydantic.mypy` on Anthropic);
- pytest plugin entry-point recall from `standard-tests`;
- negative leakage: whether `langchain-core` depends on OpenAI/Anthropic;
- issue-like planning: which package path and command style to use for OpenAI changes;
- same-session recall probes over earlier user/profile messages;
- source-card symbol/import recall;
- late update recall after the corpus is already seeded.

## Smoke command

Fast validation comparing builtin memory against Lethyx Hindsight on a small but real corpus:

```bash
PYTHONPATH=src python \
  benchmarks/hermes_memory_profiles/run_large_project_sequence.py \
  --providers builtin,lethyx-hindsight \
  --repo-path .external/langchain \
  --max-cards 12 \
  --max-questions 9 \
  --out-dir runs/hermes_memory_profiles/large_langchain_smoke_builtin_vs_lethyx_hindsight_<date>
```

## 67-row all-profile run

Use 80 cards, 67 questions per provider, all Lethyx backend configs, and parallel provider workers:

```bash
label=large_langchain_67rows_per_memory_all_profiles_20260629
out="runs/hermes_memory_profiles/$label"
log="runs/hermes_memory_profiles/${label}.log"
HERMES_PROFILE_BENCH_SEED_TIMEOUT=3600 \
PYTHONPATH=src python \
  benchmarks/hermes_memory_profiles/run_large_project_sequence.py \
  --providers builtin,hindsight,lethyx-hindsight,lethyx-mempalace,lethyx-mnemosyne,lethyx-composite \
  --repo-path .external/langchain \
  --max-cards 80 \
  --provider-workers 6 \
  --query-timeout 720 \
  --source large-langchain-67-session-bench \
  --out-dir "$out" \
  2>&1 | tee "$log"
```

Run the larger command in a tracked background process with `notify_on_complete=true` and a durable shell log, for example `2>&1 | tee runs/hermes_memory_profiles/<label>.log`, then poll `progress.json` and row counts. Direct Hindsight seeding is slow on 80-card corpora; keep Docker Hindsight healthy on `127.0.0.1:8888` and set a long `HERMES_PROFILE_BENCH_SEED_TIMEOUT`.

## Final 67-row summary

Run: `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629`

| Provider | Rows | Avg score | Avg seconds | Max seconds | Failures | Unique tool calls | Context summaries |
|---|---:|---:|---:|---:|---:|---:|---:|
| `builtin` | 67 | 0.9826 | 24.425 | 64.280 | 0 | 3 | 0 |
| `hindsight` | 67 | 0.7896 | 29.930 | 80.982 | 0 | 31 | 0 |
| `lethyx-hindsight` | 67 | 0.9858 | 56.357 | 163.634 | 0 | 78 | 0 |
| `lethyx-mempalace` | 67 | 0.9639 | 30.075 | 57.369 | 1 | 53 | 0 |
| `lethyx-mnemosyne` | 67 | 0.9903 | 34.778 | 167.151 | 0 | 82 | 1 |
| `lethyx-composite` | 67 | 1.0000 | 45.414 | 158.842 | 0 | 62 | 1 |

Tool-call counts above are corrected unique tool-call ids. The generated `summary.json` kept the original lineage-sum counts; see `tool_call_context_analysis.md` and `summary_corrected_tooling.json` under the run directory for the audit.

Primary claim: `lethyx-composite` is the best 67-row profile with perfect score and zero failures; `lethyx-hindsight` substantially beats direct `hindsight` on the same Hindsight backend shape; every Lethyx mode beats direct Hindsight on average score.

## Interpretation rules

- A low score on a question whose supporting card was not seeded is a harness bug, not a memory-quality result. The smoke card order prioritizes required cards first to avoid this.
- Negative-leak questions must score occurrence-by-occurrence: a forbidden term is safe only if every occurrence is inside a local negation/absence span.
- Builtin memory may look good on tiny corpora because everything is injected; this benchmark is designed to show where injection/truncation/noise fails.
- Lethyx wins should be reported only when the same seeded corpus and same memory-only prompts are used across profiles.
- Latency should be reported separately from correctness; Lethyx can be slower but more selective on large/noisy corpora.
- On Windows, do not pass the direct Hindsight seed corpus through one large environment variable. `run_profile_matrix.py` writes `seed_facts.json` under the clean profile instead; this avoids silent pre-progress failures when the all-profile run includes direct `hindsight`.
