# UM-MEM for OpenCode V2

Optional OpenCode V2 plugin using the public `session.context` hook and the
existing UM-MEM MCP server. It resolves the active mask before every primary
model request, including continuations after tools. It does not retain the
transcript, infer which records to activate, or start a reflection model.

## Install from this checkout

Requires Node 20+, OpenCode V2 with `@opencode/plugin` 2.1.3-compatible hooks,
and an installed UM-MEM Python environment with the `mcp` extra.

```console
cd integrations/opencode
npm ci
npm run build
```

Add this plugin to the appropriate OpenCode configuration. Set absolute paths
to your Python executable, plugin directory and existing UM-MEM store:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": [{
    "package": "/absolute/path/to/UM-MEM/integrations/opencode",
    "options": {
      "command": ["/absolute/path/to/python", "-m", "um_mem.memory.mcp_server"],
      "env": {
        "UM_MEM_REPO_ROOT": "/absolute/path/to/memory-store",
        "UM_MEM_WORKSPACE": "my-project",
        "UM_MEM_PROFILE": "default",
        "UM_MEM_BACKEND": "mnemosyne"
      },
      "maxTokens": 800
    }
  }]
}
```

Disable the separate `um_mem` MCP server in this host's configuration when using
the plugin. The plugin registers the same tool namespace from the actual MCP
schemas; loading both would introduce competing tools with different scopes.
Windows paths in JSON need escaped backslashes or forward slashes. No global
configuration is changed by building or testing this package.

## Scope and lifecycle

- Host-provided session IDs bind both tool calls and active-context reads.
  The adapter derives a stable project ID from the host working directory's Git
  root and binds it to its MCP processes. Automatic active-context delivery
  accepts only records in that project scope. Put new reusable project knowledge
  in project-scoped blocks; OpenCode automatically routes the four supported
  workspace taxonomy leaves for stable facts, decisions, ideas, and reflection
  candidates to static project-scoped twins with matching policy. Other new
  workspace `remember` writes fail with guidance to choose a supported route or
  create an explicit project-scoped block. User-global, session, task, and
  subtask routes are left to their existing contracts. Old shared workspace
  records stay available through explicit `recall` or `build_context` calls.
  Addressed `amend` and `forget` retain their existing governed exact-record
  paths. A legacy correction submitted through `remember(operation="correct")`
  is rejected rather than retargeted; use `read_exact_record`, then `amend` with
  the record ID, block ID, and expected revision, or `forget` with the exact ID.
  The task ID is explicitly conversation-scoped (`opencode-session:<sessionID>`):
  the public context hook does not expose a stable task ID. Changing subtasks
  inside that conversation has no automatic host signal. When a clear subtask
  change makes the current focus irrelevant, the agent should explicitly call
  `transition_active_mask` with `transition: "deactivate"` and an audit reason,
  then inspect and select a new focus with the memory tools. The next context
  hook confirms the old focus is no longer delivered. The adapter does not infer
  task changes from prompt text, session titles, or agent identity.
- The context hook adds one short reflection reminder: after meaningful work,
  choose between a sourced lasting fact, an explicitly unverified hypothesis,
  and no write. This is an agent decision, not a model callback or an automatic
  `MemoryReviewReceipt`; the host still needs a real boundary to attest it.
- `block_map` and `recall` stay visible in the Code Mode catalog so ordinary
  navigation and reading need no preliminary tool-discovery call. Navigation
  exposes bounded pages; follow `next_cursor` and use `child_count` rather than
  an empty preview to identify a branch.
- Each active session owns a reusable MCP process. Different sessions cannot
  share masks. Passive workspace memory remains shared according to UM-MEM rules.
- Up to 16 simultaneous connections; idle least-recently-used connections are
  evicted at capacity, and inactive connections close after about a minute.
  Reconnecting preserves session identity and durable masks. Plugin
  unload closes owned connections. Native persistence stays in UM-MEM's selected
  backend and shared service, not in the plugin.
- The primary-model hook rematerializes context after compaction without editing
  stored history. It does not intercept title generation or replace summaries.
- A failed active read emits a short degraded diagnostic and lets work continue.
  Mutations are never automatically retried; `remember` receives the host tool-call
  identity (a hash of session, message and call IDs) as `operation_id` unless the
  caller already supplied one. MCP tool errors remain host tool errors.
- `maxTokens=0` disables automatic active-context delivery but keeps scoped tools.
  Raw MCP health still describes the MCP transport, not proof of plugin execution.

## Verification

```console
npm test
```

This type-checks against the published V2 plugin API and tests the registered
hook with real MCP transports: two sessions, correction between rounds,
deactivation, empty focus and advisory failure.

For a native-backend smoke, set `UM_MEM_SMOKE_PYTHON` to an installed UM-MEM Python
executable and run `npm test`. The smoke creates an isolated temporary store,
  requires `selected_native` Mnemosyne, tests pagination, write/recall, focus
  switching, correction, hypothesis retraction and cross-session isolation,
then shuts down its listener and removes only its own temporary store.

The loader test uses OpenCode's `Host.resolve` to check the root `index.js`
entrypoint as well as the built module. On 2026-10-23, a separate live smoke in
OpenCode desktop service 2.1.2 with GPT-6 Astra returned the seeded active value,
then its corrected value, then a newly corrected value after confirmed host
compaction, and `UNKNOWN` in another session without a mask.
It used a temporary project configuration and native Mnemosyne store, then
removed its sessions and store. It did not change the user's permanent setup.

After installation, verify that your host loads the plugin and receives changed
focus on the next model round. These integration checks do not establish memory
quality or latency advantages across other models, providers or workloads.

For isolated live-host QA with a globally installed plugin, verify the reported
workspace before any test write. In the tested desktop 2.1.2 setup, redeclaring
the same package in project configuration still called the global tool instance.
The QA fixture used a distinct plugin ID and namespace plus a workspace preflight;
direct MCP/native tests already bind their own isolated store explicitly.
