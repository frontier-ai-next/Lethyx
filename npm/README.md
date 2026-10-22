# @lethyx/cli

A small npm launcher for [Lethyx](https://github.com/frontier-ai-next/Lethyx), a
local, framework-neutral memory layer for agent harnesses.

The npm package itself contains no memory engine. It gives you one entry point
and installs the Python engine into an isolated environment.

## Install

```console
npm install -g @lethyx/cli
lethyx install
lethyx setup
```

Requirements: Node 18+ and Python 3.12+.

`lethyx install` creates `~/.lethyx/python` and runs
`pip install "lethyx[mcp]"`. Add the optional multilingual embedding tier with:

```console
lethyx install --semantic
```

The model is downloaded on first use, not during installation. Set
`LETHYX_PIP_SPEC` to install a specific build, for example from a checkout or a
Git branch:

```console
$env:LETHYX_PIP_SPEC = "lethyx[mcp,semantic] @ git+https://github.com/frontier-ai-next/Lethyx.git@main"
lethyx install --force
```

On macOS/Linux use `export LETHYX_PIP_SPEC=...`.

Until the package is published, install the launcher from a checkout:

```console
npm install -g .\npm
lethyx install --from .
```

## After install

```console
lethyx setup        guided backend, store, harness and embedding choices
lethyx status       show the active configuration
lethyx doctor       check the store, backend and MCP runtime
lethyx map          render and open the memory map
```

`lethyx setup` writes `~/.lethyx/config.json` and updates the MCP settings of
the harness you choose. Restart the harness afterwards.

## Uninstall

```console
npm uninstall -g @lethyx/cli
```

The Python environment and memory store live in `~/.lethyx`; remove that folder
only if you also want to delete your local memory.
