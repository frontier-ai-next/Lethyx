# Install and connect a harness

This is the short, product-shaped path. For the low-level flags and every
backend detail, see [Quickstart](quickstart.md) and
[Backends](backends.md).

## Fast path (npm launcher)

Requires Node 18+ and Python 3.12+.

```console
npm install -g @lethyx/cli
lethyx install
lethyx setup
```

- `lethyx install` creates an isolated Python environment in
  `~/.lethyx/python` and installs the engine with MCP support.
- `lethyx setup` asks for the store, backend, harness and whether to enable
  embeddings, then writes `~/.lethyx/config.json` and updates the harness
  settings.
- Restart the harness after setup.

`lethyx install --semantic` also installs the optional embedding runtime. The
model itself downloads on first use.

Until the npm and Python packages are published to their registries, install
the launcher from a checkout:

```console
npm install -g .\npm
lethyx install --from .
```

Use the absolute checkout path instead of `.` when you run `lethyx install`
from another folder.

## Python-only path

```console
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install "lethyx[mcp]"
lethyx setup
```

Add the optional embedding runtime with `"lethyx[mcp,semantic]"`.

For a source checkout use `python -m pip install -e ".[dev,mcp]"` and run
`lethyx setup` from the checkout. `pipx install "lethyx[mcp]"` works too.

## What `lethyx setup` asks

| Question | Default | Why |
| --- | --- | --- |
| Store folder | `~/.lethyx/store` | All memory, indexes and caches live here. Use an existing Lethyx store to keep your records. |
| Workspace | `default` | A namespace inside the store, for example one per project or one per harness. |
| Profile | `default` | A narrower namespace for experiments or teams. Usually leave it. |
| Backend | `mnemosyne` | See the matrix below. |
| Embeddings | on when installed | Local multilingual similarity; no API key; model downloads on first use. |
| Harness | auto-detected | Writes the MCP entry and, where supported, hooks. |

Everything can be changed later by editing `~/.lethyx/config.json` or running
`lethyx setup` again. CLI flags override the config; environment variables
override the config but not explicit flags.

## Which backend?

| Backend | Choose it when | Trade-off |
| --- | --- | --- |
| `mnemosyne` (default) | You want local durable memory, native recall and the semantic tier. | The richest local path; keeps a projection/index next to the store. |
| `mempalace` | You want the smallest, zero-extra-setup path or are migrating an old store. | Keyword-oriented; no native recall projection. |
| `hindsight` | You already operate a Hindsight service and want memory there. | Needs the `hindsight` extra and a running service; not a local-first setup. |
| `zvec` | You are evaluating an experimental vector backend. | Needs the `zvec` extra; not the default recommendation. |

`lethyx setup` marks the recommended option. `lethyx doctor` checks that the
chosen backend is actually usable before you rely on it.

## Which harness?

| Harness | What setup writes | Notes |
| --- | --- | --- |
| OpenCode | `~/.config/opencode/opencode.json` (`mcp.servers.lethyx`) | Uses the same config file as the desktop app; a `.bak` is kept. |
| Claude Code | `~/.claude.json` and Claude hooks | MCP tools plus hooks. |
| Codex | `~/.codex/config.toml` and Codex hooks | MCP tools plus hooks. |
| Hermes | `~/.hermes/config.yaml` | MCP server entry. |

Connect or reconnect later:

```console
lethyx connect opencode
lethyx connect claude codex
```

The generated environment includes `LETHYX_REPO_ROOT`, `LETHYX_WORKSPACE`,
`LETHYX_PROFILE`, `LETHYX_BACKEND` and, when enabled, the semantic model and
cache path. Keep `lethyx-mcp` on the same Python environment that `lethyx
doctor` checks — the setup command does that automatically.

## Embeddings: on or off?

**Off** is the right default for a first install on a slow machine or a metered
connection. Lethyx still recalls by exact terms, identifiers, tags and the
script-neutral Unicode n-gram fallback.

**On** is worth it when:

- you write memory in more than one language, or
- you want a question phrased differently from the stored record to find it.

The default model is `intfloat/multilingual-e5-large` (about 2.1 GB on first
use, CPU only). It is loaded lazily and released after a period of inactivity.
Change it with `lethyx setup` or:

```console
lethyx setup --semantic --model intfloat/multilingual-e5-small
```

Turning it off later does not delete existing records or the index; it only
stops semantic reordering.

## Everyday commands

```console
lethyx status        # what is configured right now
lethyx doctor        # Python, store, backend and MCP checks
lethyx map           # render and open the memory map
lethyx map --no-open # write the file only
lethyx connect ...   # refresh harness settings
```

`lethyx map` uses the configured store, workspace and backend, so the old
invocation with three explicit flags is no longer necessary:

```console
lethyx map --output .\memory-map.html
```

## Keep an existing store

Point setup at the old store:

```console
lethyx setup --store "C:\path\to\store" --workspace opencode --backend mnemosyne
```

The store is not copied or rewritten. The backend and workspace must match the
records you want to see; the map reads the same store. Run `lethyx doctor`
afterwards, then `lethyx map`.

## Troubleshooting

- **`lethyx` is not found** — reopen the terminal after `npm install -g`, or
  run `lethyx install` from the npm package.
- **`fastembed is not installed`** — reinstall with
  `lethyx install --semantic` or `python -m pip install "lethyx[semantic]"`.
- **The model download is slow** — embeddings are optional; run
  `lethyx setup --no-semantic`, or pre-fetch with `lethyx semantic-model`.
- **The harness shows no tools** — run `lethyx connect <harness>` and restart
  the harness. Check that the path in the MCP entry is the environment from
  `lethyx status`.
- **The map is empty** — confirm the store, workspace and backend with
  `lethyx status`; a different workspace is a different namespace, and a
  different backend can show a different projection.
