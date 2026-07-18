# Честный Gemma-only прогон `memory_bench` для Lethyx

Дата фиксации: 2026-06-26.

Этот документ описывает локальный режим, в котором сравниваются реализации памяти Lethyx/raw/baseline без подмены модели и без скрытых fallback-метрик.

## Инварианты честного сравнения

1. **Одна модель для reader и judge:** `gemma-4-31B-it`.
2. **Никаких fallback-моделей:** если Gemma недоступна, прогон должен упасть/дать error-row, а не переключиться на другую модель.
3. **Никакого fallback judge:** если LLM judge не отвечает, `LLMJudge` fail-closed; результат не подменяется `F1`.
4. **BM25 не должен превращаться в recency baseline:** пакет `rank-bm25` обязателен, иначе baseline падает fail-closed.
5. **Native backends только native:** MemPalace/Mnemosyne/Hindsight допускаются в native-сравнение только при `native_ok=true`, `strict_ok=true`, `effective_mode=native`.
6. **Hindsight включается только через полноценный Docker service** на `http://localhost:8888`; embedded/sidecar/parity не считаются честным native Hindsight.

## Локальная `.env` конфигурация

Ключи/токены в документации не записываются. В проектной `.env` должны быть выставлены только публичные параметры модели:

```env
LLM_BASE_URL=https://gateway.frontierai.ru/v1
LLM_MODEL=gemma-4-31B-it
MEMORY_BENCH_JUDGE_MODEL=gemma-4-31B-it
MEMORY_BENCH_WORKERS=20
LETHYX_MODEL_FALLBACK_LIST=
LETHYX_MODEL_CATALOG=gemma-4-31B-it|https://gateway.frontierai.ru/v1||FRONTIERAI_GATEWAY_API_KEY
LETHYX_HINDSIGHT_LLM_MODEL=gemma-4-31B-it
LETHYX_HINDSIGHT_BASE_URL=http://localhost:8888
LETHYX_HINDSIGHT_EMBEDDED=false
LETHYX_HINDSIGHT_TIMEOUT_SECONDS=1800
HINDSIGHT_BASE_URL=http://localhost:8888
HINDSIGHT_API_BASE_URL=http://localhost:8888
HINDSIGHT_TIMEOUT_SECONDS=1800
MEMORY_BENCH_RAW_HINDSIGHT_EMBEDDED=false
MEMORY_BENCH_LLM_TIMEOUT_S=90
```

`LETHYX_MODEL_CATALOG` содержит только Gemma. Это не fallback-list: других моделей в catalog быть не должно.

## Зависимости, которые должны быть в активном Python

Бенчи запускаются текущим Hermes Python:

```bash
PYTHONPATH='benchmarks;src' \
python -m memory_bench ...
```

Проверенные зависимости:

```text
openai==2.24.0
chromadb==1.5.9
mempalace==3.5.0
mnemosyne-memory==3.10.1
rank-bm25==0.2.2
hindsight-client==0.8.3
hindsight-all==0.8.3
onnxruntime==1.20.1
```

Почему это важно:

- `chromadb.PersistentClient` нужен для native MemPalace.
- `mempalace` нужен для реального raw/native MemPalace стека.
- `mnemosyne-memory` нужен для raw/native Mnemosyne.
- `hindsight-client`/`hindsight-all` нужны для raw/native Hindsight client path.
- `rank-bm25` нужен для настоящего BM25 baseline; без него baseline теперь fail-closed.

## Docker Hindsight на localhost

Перед Hindsight-enabled запуском должен быть live container:

```bash
docker run -d --name hindsight --restart unless-stopped \
  -p 127.0.0.1:8888:8888 \
  -p 127.0.0.1:9999:9999 \
  -e HINDSIGHT_API_HOST=0.0.0.0 \
  -e HINDSIGHT_API_PORT=8888 \
  -e HINDSIGHT_API_LLM_PROVIDER=openai \
  -e HINDSIGHT_API_LLM_API_KEY=<redacted> \
  -e HINDSIGHT_API_LLM_BASE_URL=https://gateway.frontierai.ru/v1 \
  -e HINDSIGHT_API_LLM_MODEL=gemma-4-31B-it \
  -e HINDSIGHT_API_LLM_TIMEOUT=180 \
  -e HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=true \
  -e TOKENIZERS_PARALLELISM=false \
  -v hindsight-data:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:0.8.3
```

Readiness:

```bash
curl -s http://127.0.0.1:8888/health
curl -s http://127.0.0.1:8888/version
```

Expected:

```text
{"status":"healthy","database":"connected"}
api_version=0.8.3
```

## Preflight перед дорогим прогоном

```bash
PYTHONPATH='benchmarks;src' \
python \
-m memory_bench datasets check
```

Ожидаемо все публичные датасеты на этой машине видны: `locomo`, `longmemeval`, `beam`, `memoryarena`, `evomembench`.

Проверка native health:

```bash
PYTHONPATH='benchmarks;src' \
python - <<'PY'
from pathlib import Path
from lethyx.memory.backends.factory import create_durable_backend
root = Path.cwd()
for mode in ['mempalace', 'mnemosyne', 'hindsight']:
    backend = create_durable_backend(root, workspace_id=f'preflight_{mode}', mode=mode)
    try:
        print(mode, backend.health())
    finally:
        close = getattr(backend, 'close', None)
        if callable(close):
            close()
PY
```

Текущее ожидаемое состояние:

- `mempalace`: `native_ok=true`, `strict_ok=true`, `effective_mode=native`.
- `mnemosyne`: `native_ok=true`, `strict_ok=true`, `effective_mode=native`.
- `hindsight`: `native_ok=true`, `strict_ok=true`, `effective_mode=native`, `base_url=http://localhost:8888`.

## LoCoMo-only матрица

Для честного LoCoMo-only сравнения используется свежий isolated label и новая cache/output директория.

Рекомендуемая матрица:

```toml
[suite]
benchmarks = ["locomo"]

[[provider]]
id = "lethyx-composite-mp-hs-mn-native"
adapter = "lethyx-composite"
backends = "mempalace,hindsight,mnemosyne"
native = true
embedded = false
hindsight_embedded = false
base_url = "http://localhost:8888"
timeout = 1800

[[provider]]
id = "lethyx-mempalace-native"
adapter = "lethyx"
backend = "mempalace"
native = true

[[provider]]
id = "lethyx-hindsight-native"
adapter = "lethyx"
backend = "hindsight"
native = true
embedded = false
base_url = "http://localhost:8888"
timeout = 1800

[[provider]]
id = "lethyx-mnemosyne-native"
adapter = "lethyx"
backend = "mnemosyne"
native = true

[[provider]]
id = "raw-mempalace"
adapter = "raw-mempalace"

[[provider]]
id = "raw-hindsight"
adapter = "raw-hindsight"
base_url = "http://localhost:8888"
timeout = 1800

[[provider]]
id = "raw-mnemosyne"
adapter = "raw-mnemosyne"

[[provider]]
id = "baseline-no-memory"
adapter = "no_memory"

[[provider]]
id = "baseline-full-context"
adapter = "full_context"

[[provider]]
id = "baseline-bm25"
adapter = "bm25"
```

Hindsight ingest на полном LoCoMo медленный: service делает real extraction/embedding, поэтому client timeout для Hindsight provider-ов должен быть большим (`timeout = 1800`), иначе будет честный fail-closed timeout.

## Что сломало последние прогоны

1. В run `memory_bench_seq_twofirst_real_20260626T104521Z` LongMemEval использовал `gpt-4o` judge. Gateway вернул множество `403 Forbidden`, потому что ключ не имел доступа к `gpt-4o`. Старое поведение подменяло judge на `F1`, что делало строки несопоставимыми.
2. До установки/проверки зависимостей MemPalace native падал с ошибкой вида `chromadb has no attribute PersistentClient`. Сейчас `chromadb==1.5.9` и `PersistentClient` доступны, MemPalace health clean.
3. `rank-bm25` отсутствовал, поэтому BM25 мог деградировать в recency baseline. Сейчас `rank-bm25==0.2.2` установлен, а код fail-closed при отсутствии пакета.
4. В `.env` был fallback-list на не-Gemma модели. Сейчас fallback-list пустой, catalog содержит только `gemma-4-31B-it`.
5. Hindsight был установлен как client, но без live service; сейчас используется Docker `ghcr.io/vectorize-io/hindsight:0.8.3` на `localhost:8888`, а UM/raw suites передают external `base_url` и `embedded=false`.

## Минимальная проверка после изменений

```bash
PYTHONPATH='benchmarks;src' \
python \
-m pytest benchmarks/memory_bench/tests -q
```

Ожидаемо: все тесты `benchmarks/memory_bench/tests` зелёные.

Дополнительно перед полным LoCoMo был выполнен smoke на 8 providers × 1 probe:

```text
runs/memory_bench_gemma20_preflight_20260626T142508Z/20260626T142515Z
```

Smoke result: `8` rows, все `status=ok`, manifest фиксирует `model=gemma-4-31B-it`, `llm_judge_model=gemma-4-31B-it`.

## Где искать результат полного LoCoMo

Каждый полный запуск пишет:

```text
runs/memory_bench_seq_<label>/fresh_manifest.json
runs/memory_bench_seq_<label>/locomo.fresh.log
runs/memory_bench_seq_<label>/locomo/<timestamp>/results.jsonl
runs/memory_bench_seq_<label>/locomo/<timestamp>/report.md
runs/memory_bench_seq_<label>/locomo/<timestamp>/manifest.json
```

После завершения проверять:

1. `results.jsonl` должен содержать строки по всем providers.
2. В логах не должно быть `gpt-4o`, `gpt-oss`, `403 Forbidden`, `llm_judge_fallback_f1`, `rank-bm25 not installed`, `native Hindsight benchmark requested`.
3. `manifest.json` должен фиксировать `gemma-4-31B-it` для reader/judge и `max_parallel=20`.
4. Hindsight должен присутствовать в `raw-hindsight`, `lethyx-hindsight-native` и `lethyx-composite-mp-hs-mn-native`.
