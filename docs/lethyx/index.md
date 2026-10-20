# Long-term memory without a growing prompt

An agent can work for a long time only if it can preserve useful state without
replaying its entire history. Lethyx does that with an evolving block tree, a
small navigation layer, and a bounded active slice.

The important idea is separation:

```text
all durable knowledge             what the model sees every round
-----------------------------     -------------------------------
passive tree + graph + records -> active mask -> activation lease
                |
                +----------------> explicit temporary recall
```

The transcript may be compacted or a new session may begin. The passive memory
does not disappear. The next model call receives only the content-free root map
and, when a native host adapter is present, the active slice selected for that
exact session and task.

## What Lethyx is

Lethyx is an orchestration layer over durable memory backends. A logical leaf
is not just a tag placed on one large shared index: it receives its own native
partition in MemPalace, Hindsight, Mnemosyne, or Zvec. The agent can create, rename,
move, split, merge, archive, revive, and connect blocks as its understanding of
the work changes.

Lethyx owns:

- workspace and profile isolation;
- the semantic tree and typed cross-links;
- immutable record history and explicit corrections;
- active-mask lifecycle and bounded materialization;
- backend partitioning, recall, and health;
- typed observations, provenance, and consolidation receipts.

Lethyx does not own task success, agent routing, or the meaning of the work.
Memory failures are reported as degraded operation; they do not turn a valid
agent answer into a failed task. Semantic choices are made by the model through
explicit tools, not by keyword rules hidden in the runtime.

## A tree that follows the work

Every `(workspace, profile)` has one protected root. Everything below it is
free-form. The following is one possible history, not a required schema:

```text
Memory
|-- Host integration evaluation
|   |-- current harness decisions
|   `-- verified run evidence
|-- provider comparison
|   |-- open questions
|   `-- rejected approaches
|-- conversation with Alice
|   `-- agreed constraints
`-- reusable lessons
```

A later experiment can create a sibling branch. A conversation can create a
different-shaped branch. A reusable lesson can be linked to both without being
copied. If a leaf becomes too broad, the agent can split it using an exact
record mapping. Nothing requires a `project`, `code`, `logs`, or `reflection`
level.

## Sessions are scope, not folders

Sessions and tasks are recorded as typed scope and provenance. They are not
automatically turned into visible blocks. The tree persists across sessions
when they use the same workspace and profile. A model may create a
session-shaped block when that has lasting semantic value, but the runtime does
not manufacture one from every transcript or session boundary.

Active masks are narrower: they are bound to an exact session and task. This
prevents one concurrent task's working focus from leaking into another while
both can still navigate the same passive tree.

## Choose an integration

- Use [MCP](interfaces.md#mcp) when you want portable tools and are willing to
  let the agent call them explicitly.
- Use [Claude Code and Codex installers](interfaces.md#installing-host-integrations)
  for user-level MCP tools plus advisory native lifecycle hooks.
- Use the [Hermes memory provider](hermes_runtime_integration_user_guide.md)
  when the host's native provider lifecycle should supply pre-round context.
- Use the [Python API](interfaces.md#python-api) inside another host or control
  plane.

Continue with the [quickstart](quickstart.md), then read the
[mental model](memory_architecture.md) before changing topology or host
integration code.
