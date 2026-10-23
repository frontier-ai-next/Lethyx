# MCP and Python interfaces

UM-MEM has one memory model and several ways to reach it. Use the interface that matches the host lifecycle you control.

## Portable adapter contract

A harness can use callbacks, native provider methods, command hooks, or another
local transport. The names of its events are adapter details. For autonomous
memory, the adapter needs to establish these capabilities in a live run:

1. Bind workspace, project, session, and the selected backend so writes and
   automatic reads stay in the intended scope.
2. Offer a bounded relevant read before work when the host has that lifecycle
   point. If it cannot, expose an explicit read tool and report the limitation.
3. Give the working agent a review opportunity during work or at a reliable
   boundary. The agent can add, amend, or choose no write without a user command.
4. Supply trusted source bytes or outcomes when available. Verification reports
   what was checked; it does not convert a summary into source truth.
5. Report whether each capability actually ran. Installed configuration, an
   available MCP tool, and a successful service health check are different
   observations.

Storage budgets and maintenance are separate from capture: the adapter should
not create a visible record for every lifecycle event. The selected backend
retains canonical history, while the agent reviews duplicates and stale facts
through current records. Different hosts may provide more or fewer capabilities
without changing the memory schema.

## Installing host integrations

Claude Code and Codex use two complementary user-level connections: MCP exposes
UM-MEM tools, while native hooks report lifecycle signals. A signal alone does
not create a semantic record. The connected working agent must review useful
knowledge and make a scoped write, correction, or no-write decision. Codex and
Claude Code now request this review once at `Stop` when their hooks run.
Install the optional MCP dependency in the same Python environment used by the
`um-mem` command, then install the host config:

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
On Windows, the installer writes `commandWindows` as a PowerShell invocation;
reinstall existing Codex hooks after upgrading so their commands can execute.

The status command checks saved configuration only. It does not launch either
host or prove a live tool call. MCP is tools-only: these two integrations do not
deliver an exact active-memory mask before every model round. Their `Stop`
hooks can continue the same agent once for memory review. The review does not
change the success status of the user's task, and it may produce no memory
mutation.
On `UserPromptSubmit`, the Codex hook first resolves the active mask for that
exact session. If it has no usable context, the hook retrieves bounded memory
from the current project by prompt and passes it as supplemental context. The
project ID is derived from the normalized Git-root path; different worktrees
of one repository currently have different IDs. Candidate records may
appear in the fallback as unverified evidence. Old lifecycle-only hook records,
if present in an existing store, are omitted. This runs at the prompt boundary,
not before every internal model request. An active mask must be deliberately set
for the Codex session; a mask
from another session is not inherited.
Older workspace records remain available through explicit MCP search. The
user-level Codex MCP process has no trusted identity for the current project,
so a new record written through that global MCP process to a workspace leaf
does not enter project-only automatic context. A project-aware host binding is
required for automatic delivery of those new writes; do not set one global
`UM_MEM_PROJECT_ID` for multiple projects.
Ordinary MCP `remember` calls create unverified candidates. The Codex and Claude
Code MCP connections do not run an independent semantic verifier or publish
candidates as verified records; hosts must supply an evidence-bound verification
and promotion flow to do that.
For any host using the Python API, `register_source()` can first capture exact
host-owned source bytes. A `lossless_atomic` write with byte ranges bound to that
registration is automatically checked and retained as advisory `evidence_backed`.
This confirms the extract's fidelity to the registered source, without a model
call or a claim that the source itself is true. Tools-only MCP has no trusted
source-registration surface, so its ordinary writes still remain candidates.
Claude/Codex `Stop` events mean a completed response/turn; only `SessionEnd`
means the session ended.

Automatic review and automatic verification are separate. The Codex and Claude
Code reviews can produce ordinary advisory candidates through MCP. Those paths
do not yet bind the user prompt as a trusted exact source, so they must not
report the candidate as source-verified. A host integrating `register_source()` can verify
an exact extract without a second model. Full live host conformance, including
the no-write outcome and bounded growth over repeated turns, still needs an
end-to-end run before claiming plug-and-play autonomous memory.
The one-pass `Stop` continuation invokes the same model again, adding
latency and tokens. Whether later recall offsets that cost is an evaluation
question, not a property proved by the hook tests.

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
