# Hermes Runtime Integration User Guide

Status: implementation guide / product target.

Lethyx can be used as a project-scoped memory facade for Hermes Agent. It should separate the Lethyx source repository from the runtime memory store.

## Concepts

- **Code root**: where Lethyx source/package lives, e.g. `<repo-root>` during development.
- **Runtime store**: where the user's actual memory lives, e.g. `~/.hermes/lethyx_store`.
- **Workspace**: a project/profile scope such as `admin`, `project_alpha`, `research`, `ops`, `global`.
- **Backend**: selected memory backend for a workspace: `mempalace`, `hindsight`, `mnemosyne`, or a declared `composite` plan.
- **Provider**: native Hermes memory provider `lethyx`, used for auto-prefetch/auto-retain.
- **MCP**: explicit Lethyx tools (`health`, `remember`, `recall`, `build_context`, `promote`, `forget`, `add_edge`, `export_obsidian`).
- **Obsidian vault**: human-facing export/view, not the authority.

## Intended first-run setup flow

The user should not need to know the low-level import/config commands.  The product
flow should be a single setup wizard after Lethyx is installed or selected as the
Hermes memory provider:

```bash
pip install lethyx
hermes memory setup          # choose provider: lethyx
```

or directly:

```bash
hermes lethyx setup
```

The wizard should ask only high-level questions:

1. **Default backend**: `mempalace`, `hindsight`, `mnemosyne`, or `composite`.
2. **Runtime store path**: default `~/.hermes/lethyx_store`.
3. **Project/profile scope**: detect Hermes profiles and current project roots,
   then propose workspace names such as `default`, `global`, `my-project`.
4. **Backend per workspace**: inherit the default unless the user overrides a
   project/profile.
5. **Visualization**: enable Obsidian export, built-in web viewer, both, or none.
6. **Imports**: import existing Hermes built-in memory, session summaries,
   skills index, MCP inventory, and project docs pointers.
7. **Privacy**: confirm no secrets/raw `.env`/credential files are imported.
8. **Apply**: write provider config, register MCP, run health checks, export/open
   the viewer, then tell the user to start a new Hermes session.

The setup should end with Hermes living on Lethyx the same way it lives on a
selected provider such as Hindsight: after choosing the provider/backend once, the
agent automatically uses it through native provider hooks.  Manual commands remain
for inspection and repair, not daily operation.

## Hermes profile config

Each Hermes profile needs:

```yaml
memory:
  provider: lethyx

mcp_servers:
  lethyx:
    command: python
    args: ["-m", "lethyx.memory.mcp_server"]
    enabled: true
    env:
      PYTHONPATH: <code_root>/src
      LETHYX_CODE_ROOT: <code_root>
      LETHYX_REPO_ROOT: <runtime_store>
      LETHYX_WORKSPACE: <workspace_id>
      LETHYX_BACKEND: <backend>
      LETHYX_OBSIDIAN_VAULT: <vault_dir>
```

The native provider config should live at:

```text
<HERMES_HOME>/lethyx/config.json
```

with:

```json
{
  "code_root": "<code_root>",
  "repo_root": "<runtime_store>",
  "workspace_id": "<workspace_id>",
  "backend": "composite",
  "vault_dir": "<vault_dir>",
  "auto_prefetch": true,
  "auto_retain": true,
  "include_passive": false,
  "timeout_seconds": 120
}
```

## Source imports

A full integration should support these import sources as separate adapters:

| Source | Import mode | Notes |
|---|---|---|
| Hermes `MEMORY.md` / `USER.md` | active migrated blocks | stable compact facts |
| Hermes session DB | classified/summarized migration | do not copy raw turns directly into directive memory |
| Skills | `skill_index` pointers | keep source skill files authoritative |
| MCP config | safe inventory blocks | redact env values; store key names and server names |
| Project docs | curated `codeptr`/observation pointers | docs and source files remain authority |
| Obsidian/NotebookLM | export/view only unless explicitly re-imported | imported summaries must be candidate/supplemental |

## Session migration policy

Bulk session migration is acceptable only through the same importance/lifecycle
model used by the rest of Lethyx.  The raw Hermes session DB is an evidence
source, not a memory authority.  A safe migrator should:

1. read sessions from `state.db` using title, source, cwd, git repo/branch,
   token/tool counts, timestamps, and FTS matches;
2. ignore or downrank tool-output-heavy/debug/log-only spans unless they contain
   user decisions or verified outcomes;
3. summarize sessions into candidate blocks with evidence refs pointing back to
   `session_id` and message windows;
4. map importance into Lethyx fields:
   - `tier=always_on|hot|warm|cold|transient`;
   - `lifecycle=candidate|active|archived|deprecated|superseded|quarantined`;
   - `surface=supplemental_evidence|archive_hint|reflection_candidate` by default;
   - `surface=directive` only after human/tribunal promotion;
5. preserve raw transcript recall in `session_search` instead of embedding the
   transcript in prompt memory;
6. run lifecycle/value reports to downrank stale or harmful summaries.

Observed local Hermes DB shape: `state.db` has `sessions`, `messages`, FTS tables,
`active`, `compacted`, and `archived` flags.  It currently has hundreds of
sessions and tens of thousands of messages, so the migrator must be streaming and
summary-first.  Hermes in-place compression also preserves compacted-away rows as
`active=0, compacted=1` and keeps them searchable; those rows should migrate as
low-priority evidence unless promoted.

## Runtime rules

- Default recall is workspace-scoped.
- Global recall is explicit.
- Passive composite blocks are included only when requested.
- Retrieved memory renders as supplemental/non-directive evidence.
- Secrets and raw credentials are never imported or exported.
- Current files, tests, typed contracts, manifests, and latest user request outrank memory.

## Operator commands

```bash
hermes lethyx setup --yes --backend composite --workspace admin --visualization obsidian --export
hermes lethyx status
hermes lethyx config show
hermes lethyx config set backend composite
hermes lethyx recall "query" --workspace project_alpha
hermes lethyx remember "fact" --workspace project_alpha --title "Fact title"
hermes lethyx migrate-builtins --export
hermes lethyx index-skills --workspace auto
hermes lethyx index-mcp
hermes lethyx index-project-docs --workspace project_alpha --path ./project-alpha
hermes lethyx sessions --dry-run --limit-per-profile 1
hermes lethyx sessions
hermes lethyx graph --all-profiles --link-edges
hermes lethyx migrate-all --backend composite --export
hermes lethyx export --all-profiles
hermes lethyx open
```

`migrate-all` runs the full smart pipeline: built-ins, skill pointers, MCP
inventory, project doc pointers, session summaries, source edges,
and Obsidian export.  It remains summary-first: raw session/tool logs and
secret-like values are not copied into prompt memory.  Generated HTML graph
views are intentionally disabled; Obsidian is the canonical visual surface.

## Visualization / viewer

Visualization is a human-facing layer, not the runtime authority.  The viewer
should make the active/passive block model visible:

- project/workspace cards (`global`, profile workspaces, detected repo projects);
- graph of memory blocks and typed edges;
- filters for backend (`mempalace`, `hindsight`, `mnemosyne`, `composite`),
  lifecycle, tier, surface, trust level, tags, source, and privacy class;
- clear badges for active vs passive, directive vs supplemental, candidate vs
  promoted, archived/deprecated/superseded/quarantined;
- click-through evidence refs: source session/message window, file pointer, MCP
  inventory, skill path, or document pointer;
- management actions: promote, quarantine, supersede, archive, forget, re-export;
- Obsidian export and NotebookLM bundle as optional static views.

Technically, Hermes still uses the typed Lethyx blocks and backend recall; the
visual layer is just inspection and curation.

## Verification

After setup:

```bash
hermes memory status
hermes mcp test lethyx
hermes lethyx status
hermes lethyx recall "known project fact" --workspace <workspace>
hermes lethyx export --all-profiles
```

Then restart/new session so native provider hooks are active.
