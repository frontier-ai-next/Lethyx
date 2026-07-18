# Большой LangChain benchmark памяти Hermes/Lethyx

Дата финального 67-row прогона: 2026-06-29  
Репозиторий-цель: `langchain-ai/langchain`  
Run artifacts: `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629`

## Короткий вывод

Да: в этом benchmark Lethyx показывает более сильную память.

Самый сильный профиль — `lethyx-composite`: он прошёл все 67 вопросов с perfect score `1.0000`, без runtime failures, в одной длинной resumed Hermes-сессии. Для сравнения:

- встроенная память Hermes: `0.9526`;
- прямой Hindsight provider: `0.7896`;
- Lethyx поверх того же Hindsight backend: `0.9858`;
- Lethyx composite: `1.0000`.

Главная честная формулировка:

> На большом noisy LangChain memory-only benchmark Lethyx даёт лучший лучший-профиль результат, все Lethyx режимы существенно превосходят прямой Hindsight, а `lethyx-composite` проходит полный 67-row сценарий без ошибок и без потери точности. Встроенная память Hermes остаётся быстрой и сильной, но не лучшей: она ошиблась на late update и нескольких точных rationale/source вопросах.

Не нужно claim-ить “каждый backend Lethyx всегда лучше builtin”. В 67-row run `lethyx-mempalace` имел один runtime failure, а `lethyx-hindsight`/`lethyx-mnemosyne` были точнее прямого Hindsight и близки/выше builtin. Самый сильный подтверждённый claim — `lethyx-composite` как лучший профиль и Lethyx слой как явное улучшение над direct Hindsight.

## Что проверяли

Проверяли не поиск по файлам, а поведение памяти внутри Hermes-профиля.

Каждый профиль получал одинаковый большой набор фактов о `langchain-ai/langchain`. Затем в рамках одной и той же Hermes-сессии для этого профиля задавались 67 memory-only вопросов. Ответы должны были приходить только из памяти профиля и из предыдущих сообщений этой же resumed session.

Сравнивались 6 профилей:

| Профиль | Что это значит |
|---|---|
| `builtin` | встроенная память Hermes через profile memory / `MEMORY.md` |
| `hindsight` | прямой Hindsight memory provider без Lethyx слоя |
| `lethyx-hindsight` | Hermes → Lethyx provider → Hindsight backend |
| `lethyx-mempalace` | Hermes → Lethyx provider → MemPalace backend |
| `lethyx-mnemosyne` | Hermes → Lethyx provider → Mnemosyne backend |
| `lethyx-composite` | Hermes → Lethyx provider → composite backend |

## Почему выбран LangChain

Для маленьких synthetic facts почти любая память может выглядеть хорошо: всё компактно, мало похожих сущностей, почти нет leakage traps.

`langchain-ai/langchain` — хороший stress case, потому что это реальный популярный Python monorepo с:

- множеством похожих package names: `langchain`, `langchain-core`, `langchain-classic`, `langchain-openai`, `langchain-anthropic`, `langchain-tests`;
- несколькими независимыми `pyproject.toml` в `libs/`;
- partner integrations, которые легко перепутать между собой;
- source files с похожими class/function names;
- естественными negative-leak traps: например, `langchain-core` не должен получить `openai`/`anthropic` зависимости только потому, что рядом есть partner packages;
- rationale facts в comments, которые теряются при обычном TOML parsing.

## Corpus: что загружали в память

Финальный run:

- 80 seed cards;
- 271,786 символов seed-контента;
- 1 late-update card после основной серии;
- 67 вопросов на каждый профиль;
- 6 профилей;
- 402 total result rows.

Seed cards строились детерминированно из локального checkout `langchain-ai/langchain`:

1. repo inventory card;
2. `AGENTS.md` guideline card;
3. `README.md` overview card;
4. приоритетные `pyproject.toml` cards:
   - `libs/core/pyproject.toml`;
   - `libs/langchain/pyproject.toml`;
   - `libs/langchain_v1/pyproject.toml`;
   - `libs/standard-tests/pyproject.toml`;
   - `libs/partners/openai/pyproject.toml`;
   - `libs/partners/anthropic/pyproject.toml`;
   - другие partner/package pyprojects;
5. source cards с AST class/function/import summaries и bounded excerpts.

Важно: для `pyproject.toml` cards сохранялся raw excerpt с комментариями. Это нужно, потому что вопрос про `langchain-anthropic` и `pydantic.mypy` проверяет именно rationale из comments.

## Как обеспечивалась честность сравнения

1. Одинаковый corpus.
   Все профили получили один и тот же набор cards.

2. Чистые профили.
   Для каждого provider создавался свежий Hermes profile, затем память сбрасывалась и seedилась заново.

3. Одна resumed session на профиль.
   Внутри каждого профиля вопросы шли одной цепочкой через `--resume <session_id>`, а не 67 независимыми сессиями. Это проверяет, как память и текущий session context живут вместе в длинной работе.

4. Memory-only режим.
   Вопросы задавались через Hermes CLI с memory toolset:

   ```bash
   hermes -p <profile> chat -Q --toolsets memory --source large-langchain-67-session-bench -q <prompt>
   ```

   После первого вопроса тот же profile продолжался через:

   ```bash
   hermes -p <profile> chat -Q --toolsets memory --source large-langchain-67-session-bench --resume <session_id> -q <prompt>
   ```

   В prompt явно запрещались file, terminal, web и `session_search`.

5. Одинаковый порядок вопросов.
   Все профили отвечали на один и тот же набор 67 questions в одном порядке.

6. Late update.
   Перед последним вопросом в память добавлялась новая update card. Последний вопрос проверял, способен ли профиль вспомнить свежую запись после уже загруженного большого corpus и длинной session history.

7. Session recall probes.
   Внутри ранних prompts добавлялись marker facts вроде `LC67-ORION`, `PROFILE-ACK-ORION`, `clean-profile-single-session`. Позже задавались вопросы, требующие вспомнить эти earlier user/profile messages из той же resumed session.

8. Negative-leak scoring.
   Вопрос `lc_negative_core_openai` проверял, не притянет ли память соседние OpenAI/Anthropic facts к `langchain-core`. Forbidden terms штрафуются только когда они не находятся в локальном отрицательном контексте.

9. Latency отделена от correctness.
   Скорость не смешивалась с score. Lethyx может быть медленнее builtin, но главный показатель здесь — точность recall, scoping и отсутствие leakage.

10. Дополнительные operational metrics.
   Для каждой row сохранялись:
   - `tool_call_count_delta`;
   - cumulative tool calls;
   - `context_summary_count_total`;
   - session id/root id;
   - elapsed seconds;
   - returncode/stderr.

## 67 вопросов: структура набора

67 вопросов на профиль состоят из:

- 8 исходных example questions;
- 17 package/dependency questions по пакетам и partner integrations;
- 4 same-session recall probes;
- 37 source-file symbol/import questions;
- 1 late-update recall question.

### 9 базовых example prompts

Эти 9 запросов показывают форму benchmark-а. Финальный run расширяет их до 67 rows, но принцип тот же: одинаковая память, memory-only ответы, один порядок вопросов.

| ID | Example prompt |
|---|---|
| `lc_monorepo_map` | In `langchain-ai/langchain`, what are the core, classic legacy, active v1, partner integration, and standard-tests directories? |
| `lc_active_vs_legacy` | Which LangChain package directory is actively maintained and which one is legacy/classic? |
| `lc_core_pyproject` | For `langchain-core`, name the package version, Python range, and several key runtime dependencies. |
| `lc_openai_integration` | For the OpenAI partner integration, what is the package name, local path, key runtime deps, and editable source links? |
| `lc_anthropic_mypy_reason` | Why does the Anthropic partner package enable `pydantic.mypy` while most other partner packages do not? |
| `lc_standard_tests_plugin` | What package provides LangChain standard tests and which pytest plugin entry point does it expose? |
| `lc_negative_core_openai` | Does `langchain-core` directly depend on `openai` or `anthropic`? Answer from subject memory only. |
| `lc_issue_like_openai_test_plan` | A PR changes OpenAI chat model behavior. Which package path and local test command style should the developer use before broader checks? |
| `lc_late_update_recall` | After the late benchmark update, what focused command should be used for OpenAI reasoning-summary behavior? |

### Дополнительные сложные проверки

Примеры расширения сверх первых 9:

- `lc_classic_package_sources` — legacy/classic package metadata and source dependencies;
- `lc_v1_package_deps` — active v1 package dependencies and local sources;
- `lc_text_splitters_package` — package + spaCy model source;
- `lc_model_profiles_package` — model profiles package dependencies;
- `lc_chroma_partner_package`, `lc_deepseek_partner_package`, `lc_fireworks_partner_package`, etc. — похожие partner integrations;
- `lc_session_user_codename_recall` — вспомнить earlier user codename `LC67-ORION`;
- `lc_session_profile_ack_recall` — вспомнить earlier assistant marker `PROFILE-ACK-ORION`;
- `lc_session_clean_profile_recall` — вспомнить phrase `clean-profile-single-session` и marker `PROFILE-ACK-CLEAN`;
- `lc_session_metrics_request_recall` — вспомнить, что пользователь просил считать tool calls и context summaries;
- source questions вроде `lc_source_libs_core_langchain_core_runnables_base_py`, `lc_source_libs_core_langchain_core_tools_base_py`, `lc_source_libs_partners_openai_langchain_openai_chat_models_base_py` — path + classes/functions/imports из source cards.

## Результаты 67-row run

| Provider | Rows | Avg score | Avg seconds | Max seconds | Failures | Unique tool calls | Avg unique tools/row
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `builtin` | 67 | 0.9526 | 24.425 | 64.280 | 0 | 3 | 0.045
| `hindsight` | 67 | 0.7896 | 39.930 | 80.982 | 0 | 31 | 0.463
| `lethyx-hindsight` | 67 | 0.9858 | 56.357 | 163.634 | 0 | 78 | 1.164
| `lethyx-mempalace` | 67 | 0.9639 | 30.075 | 57.369 | 1 | 53 | 0.791
| `lethyx-mnemosyne` | 67 | 0.9903 | 34.778 | 167.151 | 0 | 82 | 1.224
| `lethyx-composite` | 67 | 1.0000 | 45.414 | 158.842 | 0 | 62 | 0.925

Note: первоначальный generated `summary.json` завышал `tool_calls` для compressed sessions, потому что суммировал parent+child lineage rows, где Hermes copied tail сохраняет те же tool-call ids. Корректная таблица выше считает unique tool-call ids. Подробный разбор: `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629/tool_call_context_analysis.md`.

Wall-clock runtime: about 65m49s with `--provider-workers 6`.  
Run started around `2026-06-29T02:33:35Z` and finished around `2026-06-29T03:39:24Z`.

## Что именно показывает результат

### 1. Lethyx composite прошёл всё

`lethyx-composite`:

- 67/67 rows;
- avg score `1.0000`;
- 0 failures;
- 1 context summary during the long session;
- late update score `1.0`.

Это самый сильный результат: профиль не только нашёл seeded project facts, но и сохранил точность через длинную resumed session, operational memory tools и context summarization.

### 2. Lethyx поверх Hindsight сильно лучше прямого Hindsight

Самое важное apples-to-apples сравнение:

| Pair | Avg score |
|---|---:|
| direct `hindsight` | 0.7896 |
| `lethyx-hindsight` | 0.9858 |

Разница: +0.1962 absolute score в пользу Lethyx слоя на Hindsight backend.

Это показывает, что выигрывает не просто Hindsight как storage engine. Выигрывает Lethyx слой: scoped records, canonical project facts, backend-neutral write/read shape, rerank/prefetch behavior и более аккуратная работа с evidence.

### 3. Все Lethyx режимы лучше direct Hindsight

| Provider | Avg score |
|---|---:|
| direct `hindsight` | 0.7896 |
| `lethyx-mempalace` | 0.9639 |
| `lethyx-hindsight` | 0.9858 |
| `lethyx-mnemosyne` | 0.9903 |
| `lethyx-composite` | 1.0000 |

Даже самый слабый Lethyx режим в этом run заметно выше direct Hindsight.

### 4. Builtin быстрый, но не лучший

`builtin` был самым быстрым и очень сильным (`0.9526`), но не perfect:

- не идеально ответил на `lc_core_pyproject`;
- потерял часть rationale в `lc_anthropic_mypy_reason`;
- не идеально прошёл один source-card вопрос;
- главное: late update получил только `0.5`, потому что не выдал точную focused command строку.

Для продуктового вывода это важно: встроенная память хороша как fast baseline, но Lethyx composite держит точность лучше на длинном, noisy, sequential benchmark.

### 5. Same-session recall прошёл у всех

4 session recall probes получили 1.0 у всех 6 профилей:

- `lc_session_user_codename_recall`;
- `lc_session_profile_ack_recall`;
- `lc_session_clean_profile_recall`;
- `lc_session_metrics_request_recall`.

Это означает, что harness действительно создавал одну resumed session на профиль и что сами session markers не были сложным differentiator. Differentiator — большой seeded project memory, exact source/package recall, late update и leakage control.

## Low-score rows

| Provider | Low rows | Комментарий |
|---|---:|---|
| `builtin` | 4 | близко к perfect, но потерял точность на `lc_anthropic_mypy_reason`, source-card и late update command |
| `hindsight` | 36 | много partial misses по package/dependency/source questions; direct provider хуже держит noisy corpus |
| `lethyx-hindsight` | 3 | один active/legacy partial, один rationale partial, один source-card partial |
| `lethyx-mempalace` | 4 | один runtime failure на `lc_negative_core_openai`, плюс несколько source-card partials |
| `lethyx-mnemosyne` | 3 | несколько partner-package partials; после scoring correction negative question засчитан корректно |
| `lethyx-composite` | 0 | low-score rows отсутствуют |

Runtime failure был только один: `lethyx-mempalace` row 7 `lc_negative_core_openai`, returncode 15, пустой answer. Это снижает average score MemPalace, но не влияет на composite result.

## Контрольный 9-query run

До 67-row прогона был выполнен меньший контрольный run: 6 профилей × 9 questions = 54 rows.

Артефакты:

`runs/hermes_memory_profiles/large_langchain_all_profiles_80cards_retry_20260628`

| Provider | Rows | Avg score | Avg seconds | Max seconds | Failures |
|---|---:|---:|---:|---:|---:|
| `builtin` | 9 | 0.9444 | 21.981 | 27.086 | 0 |
| `hindsight` | 9 | 0.7778 | 59.458 | 99.832 | 0 |
| `lethyx-hindsight` | 9 | 1.0000 | 50.302 | 97.313 | 0 |
| `lethyx-mempalace` | 9 | 1.0000 | 45.060 | 67.534 | 0 |
| `lethyx-mnemosyne` | 9 | 1.0000 | 51.890 | 68.815 | 0 |
| `lethyx-composite` | 9 | 0.8889 | 52.164 | 87.259 | 0 |

Этот smaller run полезен как sanity check: он уже показывал separation между direct Hindsight и single-backend Lethyx. 67-row run сильнее, потому что проверяет одну длинную session и гораздо больше package/source facts.

## Reproduction command

```bash
label=large_langchain_67rows_per_memory_all_profiles_20260629
out="runs/hermes_memory_profiles/$label"
log="runs/hermes_memory_profiles/${label}.log"
HERMES_PROFILE_BENCH_SEED_TIMEOUT=3600 \
PYTHONPATH=src \
python \
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

## Artifacts

Final 67-row run:

- `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629/results.jsonl`
- `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629/summary.json`
- `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629/summary_corrected_tooling.json`
- `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629/report.md`
- `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629/tool_call_context_analysis.md`
- `runs/hermes_memory_profiles/large_langchain_67rows_per_memory_all_profiles_20260629.log`

Code/docs touched for this benchmark:

- `benchmarks/hermes_memory_profiles/run_large_project_sequence.py`
- `benchmarks/hermes_memory_profiles/run_profile_matrix.py`
- `tests/test_hermes_profile_matrix_benchmark.py`
- `docs/lethyx/large-langchain-memory-benchmark-ru.md`
- `docs/lethyx/large_project_sequential_memory_benchmark.md`

## Verification notes

Validation after implementation:

```bash
PYTHONPATH=src python -m py_compile \
  benchmarks/hermes_memory_profiles/run_profile_matrix.py \
  benchmarks/hermes_memory_profiles/run_large_project_sequence.py \
  tests/test_hermes_profile_matrix_benchmark.py

PYTHONPATH=src python -m pytest \
  tests/test_hermes_profile_matrix_benchmark.py -q --tb=short
```

Result:

```text
3 passed in 0.04s
```

After the run, the strict negative-leak scorer was improved for phrases like “No memory-backed fact ... lists openai or anthropic”. Existing rows were rescored deterministically from saved answers; only one row changed: `lethyx-mnemosyne` / `lc_negative_core_openai` from `0.0` to `1.0`, because the answer was semantically negative and should not have been penalized as leakage.

## Limitations

- Это один крупный repo benchmark, не универсальное доказательство для всех проектов.
- `builtin` показал сильный fast baseline; claim не должен звучать как “builtin плохой”.
- `lethyx-mempalace` имел один runtime failure; это нужно отдельно расследовать.
- `lethyx-composite` лучший по accuracy, но делает больше tool calls и медленнее builtin.
- Метрика score lexical/term-based; она полезна для reproducible сравнения, но не заменяет human review всех ответов.

## Final claim для README / презентации

Коротко:

> В большом LangChain benchmark с 80 memory cards и 67 sequential memory-only questions на профиль Lethyx composite получил perfect `1.0000` score, тогда как встроенная Hermes memory получила `0.9526`, а direct Hindsight — `0.7896`. Lethyx поверх Hindsight (`0.9858`) значительно улучшил тот же backend без Lethyx слоя. Benchmark выполнялся на чистых профилях, в одной resumed session на профиль, с late-update и negative-leak проверками.
