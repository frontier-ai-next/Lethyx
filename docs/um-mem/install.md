# Install and connect a harness

This is the short, product-shaped path. For the low-level flags and every
backend detail, see [Quickstart](quickstart.md) and
[Backends](backends.md).

## Fast path (npm launcher)

Requires Node 18+ and Python 3.12+.

```console
npm install -g @um-mem/cli
um-memory install
um-memory setup
```

- `um-memory install` creates an isolated Python environment in
  `~/.um-mem/python` and installs the engine with MCP support.
- `um-memory setup` asks for the store, backend, harness and whether to enable
  embeddings, then writes `~/.um-mem/config.json` and updates the harness
  settings.
- Restart the harness after setup.

`um-memory install --semantic` also installs the optional embedding runtime. The
model itself downloads on first use.

Until the npm and Python packages are published to their registries, install
the launcher from a checkout:

```console
npm install -g .\npm
um-memory install --from .
```

Use the absolute checkout path instead of `.` when you run `um-memory install`
from another folder.

## Python-only path

```console
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install "um-mem[mcp]"
um-memory setup
```

Add the optional embedding runtime with `"um-mem[mcp,semantic]"`.

For a source checkout use `python -m pip install -e ".[dev,mcp]"` and run
`um-memory setup` from the checkout. `pipx install "um-mem[mcp]"` works too.

## What `um-memory setup` asks

| Question | Default | Why |
| --- | --- | --- |
| Store folder | `~/.um-mem/store` | All memory, indexes and caches live here. Use an existing UM-MEM store to keep your records. |
| Workspace | `default` | A namespace inside the store, for example one per project or one per harness. |
| Profile | `default` | A narrower namespace for experiments or teams. Usually leave it. |
| Backend | `mnemosyne` | See the matrix below. |
| Embeddings | on when installed | Local multilingual similarity; no API key; model downloads on first use. |
| Harness | auto-detected | Writes the MCP entry and, where supported, hooks. |

Everything can be changed later by editing `~/.um-mem/config.json` or running
`um-memory setup` again. CLI flags override the config; environment variables
override the config but not explicit flags.

## Which backend?

| Backend | Choose it when | Trade-off |
| --- | --- | --- |
| `mnemosyne` (default) | You want local durable memory, native recall and the semantic tier. | The richest local path; keeps a projection/index next to the store. |
| `mempalace` | You want the smallest, zero-extra-setup path or are migrating an old store. | Keyword-oriented; no native recall projection. |
| `hindsight` | You already operate a Hindsight service and want memory there. | Needs the `hindsight` extra and a running service; not a local-first setup. |
| `zvec` | You are evaluating an experimental vector backend. | Needs the `zvec` extra; not the default recommendation. |

`um-memory setup` marks the recommended option. `um-memory doctor` checks that the
chosen backend is actually usable before you rely on it.

## Which harness?

| Harness | What setup writes | Notes |
| --- | --- | --- |
| OpenCode | `~/.config/opencode/opencode.json` (`mcp.servers.um_mem`) | Portable tools; automatic review needs the separate native adapter. A `.bak` is kept. |
| Claude Code | `~/.claude.json` for MCP and `~/.claude/settings.json` for hooks | Tools and a one-pass same-agent review at `Stop` when hooks run. |
| Codex | `~/.codex/config.toml` and Codex hooks | Tools, scoped prompt read, and a one-pass same-agent review at `Stop` when hooks run. |
| Hermes | `~/.hermes/config.yaml` | MCP tools; use the native provider for pre-round context and agent memory guidance. |

Connect or reconnect later:

```console
um-memory connect opencode
um-memory connect claude codex
```

Connecting a tool is not the same as proving that memory grows. In a new clean
store, test one ordinary task that produces a reusable fact, decision, or
preference. Check that the agent saves or updates a record without a "remember"
command, and check a second task can retrieve it. Then test a trivial turn where
nothing should be saved. Use `um-memory status` for configuration and the atlas
for canonical records; neither alone proves the host review callback ran.

The generated environment includes `UM_MEM_REPO_ROOT`, `UM_MEM_WORKSPACE`,
`UM_MEM_PROFILE`, `UM_MEM_BACKEND` and, when enabled, the semantic model and
cache path. Keep `um-memory-mcp` on the same Python environment that `um-memory
doctor` checks - the setup command does that automatically.

One harness setting holds one identity. To keep several repositories apart, run
`um-memory connect --local` inside the project: it writes `./opencode.json` for
that project only and derives the workspace from the folder name. Only OpenCode
merges a project-local entry; the other hosts refuse `--local`. See
[One harness, several projects](quickstart.md#one-harness-several-projects).

## Embeddings: on or off?

**Off** is the right default for a first install on a slow machine or a metered
connection. UM-MEM still recalls by exact terms, identifiers, tags and the
script-neutral Unicode n-gram fallback.

**On** is worth it when:

- you write memory in more than one language, or
- you want a question phrased differently from the stored record to find it.

The default model is `intfloat/multilingual-e5-large` (about 2.1 GB on first
use, CPU only). It is loaded lazily and released after a period of inactivity.
Change it with `um-memory setup` or:

```console
um-memory setup --semantic --model intfloat/multilingual-e5-small
```

Turning it off later does not delete existing records or the index; it only
stops semantic reordering.

## Everyday commands

```console
um-memory status        # what is configured right now
um-memory doctor        # Python, store, backend and MCP checks
um-memory map           # render and open the memory map
um-memory map --no-open # write the file only
um-memory connect ...   # refresh harness settings
```

`um-memory map` uses the configured store, workspace and backend, so the old
invocation with three explicit flags is no longer necessary:

```console
um-memory map --output .\memory-map.html
```

## Keep an existing store

Point setup at the old store:

```console
um-memory setup --store "C:\path\to\store" --workspace opencode --backend mnemosyne
```

The store is not copied or rewritten. The backend and workspace must match the
records you want to see; the map reads the same store. Run `um-memory doctor`
afterwards, then `um-memory map`.

## Troubleshooting

- **`um-memory` is not found** - reopen the terminal after `npm install -g`, or
  run `um-memory install` from the npm package.
- **`fastembed is not installed`** — reinstall with
  `um-memory install --semantic` or `python -m pip install "um-mem[semantic]"`.
- **The model download is slow** — embeddings are optional; run
  `um-memory setup --no-semantic`, or pre-fetch with `um-memory semantic-model`.
- **The harness shows no tools** — run `um-memory connect <harness>` and restart
  the harness. Check that the path in the MCP entry is the environment from
  `um-memory status`.
- **The map is empty** — confirm the store, workspace and backend with
  `um-memory status`; a different workspace is a different namespace, and a
  different backend can show a different projection.
