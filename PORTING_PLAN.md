# UM-MEM Porting Plan

## Current state

The `release/v0.6.0` branch imports the complete upstream `main` snapshot at
`c4a0015535adeba5bdce63e15b60ac7c431b7fe4` (2026-10-20). The Python package,
entry points, docs and tests now use the `um-mem` name. The upstream pipeline and
registry publishing configuration were removed; GitHub Actions CI is used for
validation. UM-MEM research documentation remains in `docs/um-mem/`.

## Product boundary

Memory remains advisory. Host systems own routing, proof validity, permissions,
and task completion. Framework-specific hooks stay in optional integrations.

## Follow-up porting work

- Review the documented backend set against the imported optional Zvec support.
- Document the migration from the former package surface.
- Run the imported CI matrix before treating the release tag as validated.
