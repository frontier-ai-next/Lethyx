# Durable memory backends

UM-MEM memory uses backend-neutral typed contracts. A workspace/run selects exactly one durable backend:

- `mempalace`
- `hindsight`
- `mnemosyne`

The selected backend is the only durable memory authority for that run/workspace. UM-MEM owns the shared semantics: typed memory blocks, lifecycle/trust/evidence fields, promotion gates, graph/value/privacy metadata, and injection audit reports. Backend adapters may use backend-owned sidecars for parity metadata, but they must not mirror to another backend.

Dual memory modes are unsupported. To compare backends, run separate experiments with one selected backend per run and compare exports/reports.

## Operational vs native health

Hindsight and Mnemosyne have two layers:

- the UM-MEM adapter-owned durable index, which is the selected backend path for typed blocks, recall, lifecycle, graph/value/privacy metadata, and promotion evidence;
- the optional native upstream layer, which is reported through `native_ok` / `strict_ok`.

When the native Hindsight service/client or Mnemosyne package is unavailable, the selected adapter can still be `ok=true` through its own SQLite sidecar/parity index while `strict_ok=false` and `effective_mode` is `sqlite_sidecar` or `sqlite_parity`. This is degraded native health, not a silent fallback to MemPalace. Matrix QA should record both facts: operational selected-backend behavior can pass, while native-clean validation requires `strict_ok=true`.

## Configuration

Preferred environment variable:

```dotenv
UM_MEM_MEMORY_BACKEND=mempalace
UM_MEM_MEMORY_BACKEND=hindsight
UM_MEM_MEMORY_BACKEND=mnemosyne
```

Backend-specific settings stay under backend-specific names, for example:

```dotenv
UM_MEM_HINDSIGHT_EMBEDDED=1
UM_MEM_HINDSIGHT_BASE_URL=http://localhost:8888
UM_MEM_MNEMOSYNE_DATA_DIR=workspaces/example/.memory/mnemosyne
```

`localhost:8888` is only a Hindsight non-embedded client/server configuration detail. MemPalace and Mnemosyne do not require it.

## Support matrix

| Capability | MemPalace | Hindsight | Mnemosyne | Evidence |
|---|---:|---:|---:|---|
| Exclusive factory selection | yes | yes | yes | `test_memory_backend_selection_v2.py` |
| Backend-neutral block contract | yes | yes | yes | `test_memory_backend_contract_models.py` |
| Retain/reload/recall parity | yes | yes | yes | `tests/memory_backend_contracts/` |
| Graph/lifecycle/value/privacy parity | yes | yes | yes | `tests/memory_backend_contracts/` |
| UI selection | yes | yes | yes | `test_memory_ui_backend_selection.py` |

Legacy data conversion is handled by migration/export tooling; runtime code should not use legacy JSONL or wing/hall/room readers as fallback authority.
