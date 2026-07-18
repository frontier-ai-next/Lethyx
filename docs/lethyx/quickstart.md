# Lethyx Quickstart

Lethyx is a framework-neutral memory layer for agents. It keeps memory scoped,
evidence-aware, and supplemental by default: host frameworks still own routing,
permissions, and final decisions.

## 1. Install

```bash
python -m pip install lethyx
```

From the private GitLab repository before a PyPI/package-registry publish:

```bash
python -m pip install "lethyx[mcp] @ git+ssh://git@gitlab.frontierai.ru:8022/frontierai/teams/mag/dmas/lethyx.git"
```

For a local clone:

```bash
python -m pip install -e .[dev,mcp]
```

Lethyx requires Python 3.12 or newer.

## 2. Plain Python

```python
from lethyx import MemoryClient

memory = MemoryClient(repo_root=".lethyx", workspace_id="my_project", backend="mempalace")
memory.remember("The project uses pytest for regression tests.", tags=["project", "tests"])

hits = memory.recall("How do we run regressions?", limit=3)
context = memory.build_context("debug a regression", budget=1200)
```

Use a different `workspace_id` per project/profile. Global recall is not mixed
in unless the caller explicitly requests it.

## 3. MCP server

Install the MCP extra first:

```bash
python -m pip install "lethyx[mcp]"
```

```bash
lethyx-mcp
```

Hermes or another MCP-compatible host can add it with a project-scoped runtime
store:

```bash
hermes mcp add lethyx \
  --command lethyx-mcp \
  --env LETHYX_REPO_ROOT="$HOME/.lethyx" \
        LETHYX_WORKSPACE="my_project" \
        LETHYX_BACKEND="mempalace"
```

The MCP tools are `health`, `remember`, `recall`, `build_context`, `promote`,
`forget`, `add_edge`, and `export_obsidian`.

## 4. Hermes native memory provider

Install the thin Hermes plugin wrapper after installing Lethyx:

```bash
python -m lethyx.memory.cli hermes-provider-install --json
hermes config set memory.provider lethyx
hermes config set memory.memory_enabled true
```

By default, the provider stores runtime memory below the Hermes home directory:
`lethyx_store/` for backend state and `lethyx_vault/` for Obsidian export. Set
`LETHYX_REPO_ROOT`, `LETHYX_WORKSPACE`, `LETHYX_BACKEND`, or
`LETHYX_OBSIDIAN_VAULT` only when you need explicit locations.

## 5. Health check

```bash
python -m lethyx.memory.cli backend-health --repo-root "$HOME/.lethyx" --workspace my_project --backend mempalace --json
```

If a selected native backend is unavailable, Lethyx reports that state instead
of silently pretending the backend worked.
