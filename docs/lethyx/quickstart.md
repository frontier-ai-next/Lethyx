# Install and quickstart

Lethyx supports Python 3.12 and newer. Use a dedicated virtual environment so
the backend and host adapter use the same installed version.

## Install from GitHub

```console
git clone https://github.com/frontier-ai-next/Lethyx.git
cd Lethyx
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

## Choose one backend

MemPalace is the simplest local starting point and is installed with Lethyx:

```console
set LETHYX_BACKEND=mempalace
```

On PowerShell use `$env:LETHYX_BACKEND = "mempalace"`; on POSIX shells use
`export LETHYX_BACKEND=mempalace`. See [Backends](backends.md) before choosing
Hindsight, Mnemosyne, or Zvec.

## Create a tree and write one record

```python
from lethyx import MemoryClient

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

The write is an advisory candidate. The caller chose the writable leaf; Lethyx
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

Next: install Lethyx [inside Hermes](hermes_runtime_integration_user_guide.md)
or connect it through [MCP](interfaces.md#mcp).
