# Troubleshooting

Start with the integration status and native backend health. Do not delete a
store or loosen identity checks until the failing layer is known.

## MCP tools work, but active context is not automatic

MCP alone exposes `set_active_mask` and `resolve_active_mask`; it does not
deliver context or make memory decisions on behalf of the host. Codex's
`UserPromptSubmit` hook can supply bounded project context at the prompt
boundary, and its `Stop` hook can ask the same agent for one memory review.
Neither hook runs before every internal model request. Claude Code has a
one-pass same-agent review at `Stop`, but no prompt-time context injection here. Hermes'
native provider has a pre-round `prefetch` lifecycle. Verify the host actually
runs installed hooks; a settings file or MCP health response is not proof.

## Hermes provider is staged but inactive

Run the status command, then complete setup in Hermes:

```console
um-memory hermes-provider-status --json --hermes-home "C:/path/to/hermes-profile" --hermes-python "C:/path/to/hermes-agent/venv/Scripts/python.exe"
hermes memory setup
hermes memory status
```

The UM-MEM installer stages files and declared dependencies only. Hermes must
prepare those dependencies through its package manager and select `um-mem`.
Restart Hermes after preparation. Check `staged`, `active`, and
`runtime_import.importable` independently; `live_callback_proven` remains false
until a live provider receipt is observed. Common causes are:

- missing or incomplete `<HERMES_HOME>/um_mem/config.json`;
- missing provider manifest or dependency preparation;
- wrong `--hermes-python` (it must be Hermes' own interpreter);
- UM-MEM source root is not the repository containing `src/um_mem`;
- provider or observer wrapper missing from the selected profile;
- `memory.provider` not set to `um-mem` after Hermes setup.

The supported install leaves Hermes built-in memory and its notes untouched.
The old `--replacement` hard-cutover option is rejected; older profiles changed
by that option can still use `hermes-provider-restore` if their backup manifest
is intact.

## Hermes cannot import UM-MEM

Confirm that `python_executable` in the profile config exists and can run:

```console
C:/path/to/python.exe -c "import um-mem; print(um_mem.__version__)"
```

For a source checkout, `code_root` must point to the repository directory that
contains `src/um_mem`. For an installed wheel, omit `code_root` and install the
same UM-MEM version into the Python environment Hermes uses.

## Active status is `none`

No live mask exists for the current session/task scope. This is not an error.
Navigate the tree and use temporary recall, or explicitly set a mask if some
context should remain available on later rounds.

## Active status is `unavailable`

The provider could not resolve the mask. Continue the host task and inspect the
lease/service diagnostics. Likely causes include backend downtime, catalog
revision mismatch, an inactive selected block, invalid scope, or service
timeout. Fix the cause or amend the mask explicitly; do not perform hidden
global recall as a substitute.

## `native_projection_rebuild_required`

The selected partition was written by an incompatible projection version or no
longer matches its declared native identity. Use a separate clean store; this
release has no old-store conversion command. Do not replace scoped native
recall with a shared-store scan followed by metadata filtering.

## Recall returns nothing

Check, in order:

1. the selected leaf IDs are current and writable leaves;
2. the record belongs to the same workspace and profile;
3. candidate visibility matches `include_candidates`;
4. lifecycle and temporal filters permit the record;
5. the native partition is healthy and contains the record;
6. token and result limits are non-zero.

Use `um-memory explain-recall` to see exclusions. Use `search_all` only as an
explicit diagnostic or agent decision, not as a permanent routing shortcut.

## Restore refuses to overwrite the Hermes profile

The profile's `config.yaml` changed after replacement. Restore is deliberately
non-destructive in this case. Stop Hermes, compare the current config with the
installer backup named by `<HERMES_HOME>/um_mem/replacement.json`, preserve the
new operator changes, and resolve the file manually. Do not delete the manifest
until the profile has been recovered and verified.
