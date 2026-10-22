# @um-mem/cli

A small npm launcher for [UM-MEM](https://github.com/frontier-ai-next/UM-MEM), a
local, framework-neutral memory layer for agent harnesses.

The npm package itself contains no memory engine. It gives you one entry point
and installs the Python engine into an isolated environment.

## Install

```console
npm install -g @um-mem/cli
um-memory install
um-memory setup
```

Requirements: Node 18+ and Python 3.12+.

`um-memory install` creates `~/.um-mem/python` and runs
`pip install "um-mem[mcp]"`. Add the optional multilingual embedding tier with:

```console
um-memory install --semantic
```

The model is downloaded on first use, not during installation. Set
`UM_MEM_PIP_SPEC` to install a specific build, for example from a checkout or a
Git branch:

```console
$env:UM_MEM_PIP_SPEC = "um-mem[mcp,semantic] @ git+https://github.com/frontier-ai-next/UM-MEM.git@main"
um-memory install --force
```

On macOS/Linux use `export UM_MEM_PIP_SPEC=...`.

Until the package is published, install the launcher from a checkout:

```console
npm install -g .\npm
um-memory install --from .
```

## After install

```console
um-memory setup        guided backend, store, harness and embedding choices
um-memory status       show the active configuration
um-memory doctor       check the store, backend and MCP runtime
um-memory map          render and open the memory map
```

`um-memory setup` writes `~/.um-mem/config.json` and updates the MCP settings of
the harness you choose. Restart the harness afterwards.

## Uninstall

```console
npm uninstall -g @um-mem/cli
```

The Python environment and memory store live in `~/.um-mem`; remove that folder
only if you also want to delete your local memory.
