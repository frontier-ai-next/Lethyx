# Install and quickstart

UM-MEM supports Python 3.12 and newer. Use a dedicated virtual environment so
the backend and host adapter use the same installed version.

## Fast path

```console
npm install -g @um-mem/cli
um-memory install
um-memory setup
um-memory doctor
um-memory map
```

`setup` writes `~/.um-mem/config.json` and connects the harness you choose.
After that, `um-memory map` needs no `--repo-root`, `--workspace` or `--backend`.
See [Install and connect a harness](install.md) for the guided choices.

## One harness, several projects

A workspace is the isolation unit: every workspace has its own records, block
tree and indexes, and recall is bound to the identity the harness starts the
MCP server with. One harness setting can therefore hold only one identity, and
every project needs its own entry with its own workspace.

For OpenCode the entry can live next to the code instead of in the user-level
settings:

```console
cd /path/to/project-alpha
um-memory connect --local
```

This writes `./opencode.json` with the workspace `project-alpha` (the folder
name, normalized) and leaves the user-level settings untouched. Pass
`--workspace parser` to name the workspace explicitly. Restart the harness
afterwards: the identity is bound when the MCP server starts, and the
project-local entry overrides the global one.

Keep one store for all projects and let the workspace separate them. A separate
store (`--store`, or `UM_MEM_REPO_ROOT`) is only needed for an isolated trial or
a different backend.

Claude Code, Codex and Hermes read their MCP settings from the user profile, so
connect them once per identity with an explicit `--workspace`. `--local` refuses
those hosts instead of silently rewriting the user-level entry.

Workspace-scoped records never leave their workspace. Records written with
`user_global` scope are the deliberate exception: they are shared between
workspaces when the same user id is bound.

## Install from GitHub

```console
git clone https://github.com/frontier-ai-next/UM-MEM.git
cd UM-MEM
python -m venv .venv
```

Activate the environment, then install the package:

```console
python -m pip install --upgrade pip
python -m pip install .
```

For a development checkout:

```console
python -m pip install -e ".[dev,mcp]"
```

## Optional: multilingual semantic tier

The default install is small and runs on any machine. It never downloads a
model. A local semantic tier is available as an extra:

```console
python -m pip install "um-mem[semantic]"
```

The tier downloads one multilingual sentence-embedding model the first time it
is used (about 2.1 GB, CPU only, no API key, no network at query time) and
lets a question written in one script reach a record written in another
script. Without the extra, UM-MEM keeps the script-neutral lexical tier and
nothing is downloaded.

| Variable | Meaning |
| --- | --- |
| `UM_MEM_SEMANTIC` | `0` switches the tier off. Default is on when the extra is installed. |
| `UM_MEM_SEMANTIC_MODEL` | Model id. Default `intfloat/multilingual-e5-large`. |
| `UM_MEM_SEMANTIC_CACHE_DIR` | Where the model files live. Default `<um-mem home>/cache/fastembed`. |
| `UM_MEM_SEMANTIC_FLOOR` | Minimum similarity that counts as a hit. Default is chosen per model. |
| `UM_MEM_SEMANTIC_WARMUP` | `0` defers the model load to the first question instead of warming it in the background at service start. |
| `UM_MEM_SEMANTIC_IDLE_SECONDS` | How long an unused model stays loaded. Default 120. `0` keeps it loaded. |

The embeddings only reorder candidates. The stored record text stays the
evidence, and a question with no real match returns nothing instead of a
plausible-looking record.

The model is a few gigabytes of resident memory. It is loaded in the local
service, released again after two idle minutes, and loaded on the next
question, so it is only in memory while it is actually used.

The semantic index covers the whole workspace, not only the newest records.
Every write joins the index in the background, and
`python -m um_mem.memory.cli semantic-index` backfills everything that is
missing. A question in any script therefore reaches old records too.

The relevance floor is measured, not guessed:
`python -m um_mem.memory.cli semantic-calibrate` probes the workspace with
relevant and irrelevant questions in several scripts, reports the background
level and writes the measured floor next to the index. The measured floor
never drops below the model default, and `UM_MEM_SEMANTIC_FLOOR` still
overrides everything.

## Procedures

A procedure is a record of kind `procedure`. It carries the steps, how the
result is verified, and when it was last confirmed:

```python
from um_mem.memory.mcp_facade import um_mem_remember

um_mem_remember(
    "Restart the local service after editing the source.",
    title="Restart the local service",
    kind="procedure",
    tags=["um-mem", "procedure"],
    requested_block_id=leaf_id,
    metadata={
        "steps": ["edit the source", "restart the service", "check health"],
        "verify": "health returns ok",
        "verified_at": "2026-10-22T00:00:00Z",
        "stale_after_days": 60,
    },
)
```

`procedure({ query })` returns them as checklists with a freshness label
(`fresh`, `aging`, `stale`, `unverified`). A procedure that outlived its
environment is reported as stale, because a stale procedure is worse than no
procedure at all. Every procedure is stamped with the environment it was
written against, so a change of the UM-MEM version or the semantic model is
reported as the reason it went stale, not just the number of days.

## Looking at the memory

The workspace can be rendered as one self-contained HTML page: the block tree
as a left-to-right mind map, with the root on the left, blocks as branches, and
records as cards. Semantically related records are joined by faint dashed arcs.

```console
um-memory setup    # once: choose the store, backend and harness
um-memory map      # render and open the map
```

`um-memory map` uses the configured store, workspace and backend. The older
explicit form still works for scripts:

```console
um-memory memory-map --backend mnemosyne --output memory-map.html
```

The page has no external assets, so it opens offline. Click a card to read it,
drag a card to move it (a block carries its records with it), click the chevron
to fold a branch, search or filter by kind, toggle the similarity arcs, and
switch between the light and dark theme. Manual positions are kept per browser;
the reset button (or a double-click on empty space) restores the tidy layout.
The same page is written on demand by the `memory_map` tool, so the snapshot is
never more than one call away.

## Choose one backend

MemPalace is the simplest local starting point and is installed with UM-MEM:

```console
set UM_MEM_BACKEND=mempalace
```

On PowerShell use `$env:UM_MEM_BACKEND = "mempalace"`; on POSIX shells use
`export UM_MEM_BACKEND=mempalace`. See [Backends](backends.md) before choosing
Hindsight, Mnemosyne, or Zvec.

## Create a tree and write one record

```python
from um_mem import MemoryClient

memory = MemoryClient(
    repo_root="./agent-memory",
    workspace_id="project-alpha",
    profile_id="main-agent",
    session_id="session-42",
    task_id="parser-fix",
    backend="mempalace",
)

root_id = memory.block_map()[0]["block_id"]
topic = memory.create_leaf_block(
    name="Parser decisions",
    synopsis="Stable parser decisions and their evidence.",
    parent_id=root_id,
    reason="this subject evolves independently",
)

saved = memory.remember(
    "The parser test is tests/test_parser.py.",
    kind="episode",
    requested_block_id=topic["block_id"],
)
```

The write is an advisory candidate. The caller chose the writable leaf; UM-MEM
did not infer it from `tests/test_parser.py`.

## Keep it active between rounds

```python
mask = memory.set_active_mask(
    selected_block_ids=(topic["block_id"],),
    pinned_record_ids=(saved.memory_id,),
    token_budget=500,
    activation_reason="keep the parser decision between rounds",
)
lease = memory.resolve_active_mask()
print(lease.rendered_context)
```

Calling `resolve_active_mask()` demonstrates materialization. In a native host
adapter, such as the Hermes memory provider, the host resolves and injects the
lease before each round. A generic MCP connection does not have that hook.

## Recall temporarily

```python
context = memory.build_context(
    "Where is the parser test?",
    max_tokens=1200,
    include_candidates=True,
    logical_block_ids=[topic["block_id"]],
)
```

Recall requires selected leaf IDs. Use `search_all=True` only when the caller
explicitly decides that no branch is suitable. Recalled context remains
temporary unless a separate mask mutation selects it.

## Correct instead of overwriting

```python
current_revision = saved.raw["revision_id"]
memory.amend(
    saved.memory_id,
    expected_revision=current_revision,
    content="The parser test moved to tests/parser/test_regression.py.",
    reason="the verified test path changed",
    requested_block_id=topic["block_id"],
)
```

The correction creates an immutable successor. A stale revision fails rather
than overwriting concurrent work.

## Clean-store rule

Stores are version-exact. For an isolated trial, use a clean-store cutover by
pointing `repo_root` at a new
empty directory. Do not reuse a store created by another schema line and do not
delete a live store merely to silence a compatibility error.

Next: install UM-MEM [inside Hermes](hermes_runtime_integration_user_guide.md)
or connect it through [MCP](interfaces.md#mcp).
