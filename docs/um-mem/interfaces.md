# MCP and Python interfaces

UM-MEM has one memory model and several ways to reach it. Use the interface that matches the host lifecycle you control.

## Installing host integrations

Claude Code and Codex use two complementary user-level connections: MCP exposes
UM-MEM tools, while native hooks record bounded lifecycle events. Install the
optional MCP dependency in the same Python environment used by the `um-mem`
command, then install the host config:

```console
python -m pip install "um-mem[mcp]"
um-memory claude-install --workspace default --profile default --backend mempalace
um-memory claude-status
um-memory codex-install --workspace default --profile default --backend mempalace
um-memory codex-status
```

The default store is in the operating system's user-data directory, outside the
repository. Pass `--repo-root` when you intentionally want another store. The
installers preserve unrelated host settings and refuse to overwrite an
unmanaged MCP server already named `um-mem`. Claude Code receives MCP settings
in `~/.claude.json` and hooks in `~/.claude/settings.json`; Codex receives MCP
settings in `$CODEX_HOME/config.toml` (or `~/.codex/config.toml`) and hooks in
`$CODEX_HOME/hooks.json` (or `~/.codex/hooks.json`). Restart the host and verify
the MCP server from its own MCP list or panel. Codex also requires reviewing and
trusting newly installed user hooks in `/hooks` before they run.

The status command checks saved configuration only. It does not launch either
host or prove a live tool call. MCP is tools-only: these two integrations do not
deliver an exact active-memory mask before every model round. Hooks are
best-effort, advisory lifecycle observations and cannot block the host task.
Claude/Codex `Stop` events mean a completed response/turn; only `SessionEnd`
means the session ended.

## MCP

Install and run the stdio server:

```console
python -m pip install "um-mem[mcp]"
um-memory-mcp
```

The equivalent module command is:

```console
python -m um_mem.memory.mcp_server
```

The server takes identity from trusted environment variables. Model-generated
arguments cannot switch the configured repository, workspace, or backend.

| Variable | Default | Meaning |
| --- | --- | --- |
| `UM_MEM_REPO_ROOT` | current directory | Runtime storage root |
| `UM_MEM_WORKSPACE` | `default` | Workspace identity |
| `UM_MEM_PROFILE` | `default` | Profile identity |
| `UM_MEM_SESSION_ID` | `default` | Static MCP session scope |
| `UM_MEM_TASK_ID` | empty | Optional exact task scope |
| `UM_MEM_PROJECT_ID` | empty | Optional trusted project scope |
| `UM_MEM_USER_ID` | derived/default | Optional user scope |
| `UM_MEM_BACKEND` | `mempalace` | Selected backend |
| `UM_MEM_MCP_TOOL_TIMEOUT_SECONDS` | `120` | Per-tool timeout |

MCP tools are grouped by purpose:

| Purpose | Tools |
| --- | --- |
| Inspect | `health`, `block_map`, `expand_block`, `navigation_envelope` |
| Evolve topology | `create_block`, `amend_block`, `move_block`, `split_block`, `merge_blocks`, `archive_block`, `revive_block`, `delete_block` |
| Link blocks | `add_block_edge`, `remove_block_edge` |
| Manage active focus | `get_active_mask`, `set_active_mask`, `amend_active_mask`, `transition_active_mask`, `resolve_active_mask` |
| Read and write | `remember`, `amend`, `recall`, `build_context`, `forget`, `add_edge` |
| Maintain | `observe_maintenance`, `consolidate_observations` |
| Export | `export_graph`, `export_obsidian` |

Tool schemas reject unknown arguments. Structural and mask mutations require
explicit reasons and revisions where applicable. These checks protect identity
and consistency; they are not content-quality gates.

Generic MCP is tools-only. It has no host callback that injects an active lease
before every model round. Hermes additionally implements its native memory
provider contract when its Python runtime is compatible; see
[Hermes integration](hermes_runtime_integration_user_guide.md). For Hermes
versions running Python below 3.12, use its native MCP client instead:

```console
python -m pip install "um-mem[mcp]"
um-memory hermes-mcp-install --workspace default --profile default --backend mempalace
um-memory hermes-mcp-status
```

This writes only the `um-mem` MCP entry in Hermes' profile `config.yaml`,
preserving other YAML settings and comments. It exposes tools but does not
replace native memory or provide pre-round active-mask delivery.

## Python API

`MemoryClient` is the main embedded entry point:

```python
from um_mem import MemoryClient

memory = MemoryClient(
    repo_root="./agent-memory",
    workspace_id="alpha",
    profile_id="main-agent",
    session_id="session-1",
    task_id="task-1",
    backend="mempalace",
)
```

The common flow is:

1. inspect `block_map()` and `expand_block()`;
2. create or choose a writable leaf;
3. call `remember()` or `amend()` with that leaf;
4. use `recall()` or `build_context()` for temporary context;
5. use `set_active_mask()` when context should survive later rounds;
6. resolve the mask in the host's pre-round hook;
7. evolve topology through explicit revisioned methods.

The package exports typed mask, lease, recall-explanation, and session objects
alongside `MemoryClient`. Host adapters should pass stable identity into the
client rather than accepting it from model arguments.

## Command-line operations

The `um-mem` command is for installation and diagnostics, not conversational
memory access:

```console
um-memory backend-health --repo-root ./agent-memory --workspace alpha --backend mempalace --json
um-memory explain-recall --repo-root ./agent-memory --workspace alpha --backend mempalace --query "parser decision" --logical-block-id BLOCK_ID --json
um-memory reconcile --repo-root ./agent-memory --workspace alpha --backend mempalace --dry-run --json
um-memory obsidian-export --repo-root ./agent-memory --workspace alpha --backend mempalace --vault-dir ./export --json
```

Run reconcile in dry-run mode first. `--apply` changes backend state and should
be used only after reviewing the report.
