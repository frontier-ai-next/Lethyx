# Live local panel

`um-memory dashboard` opens the selected store/workspace/profile on the existing
local service. `um-memory map` is an alias unless an explicit `--output` requests
an offline HTML export. `um-memory memory-map` always exports a static map.
CLI identity flags and the shared config have the same precedence as other
UM-MEM commands. No cloud account, frontend build or additional dependency is
needed; the HTML, stylesheet and JavaScript ship in the Python wheel.

The frontend is off by default, even while agents use memory. Opening it with
the command or desktop shortcut activates a browser session. The power button
or `um-memory dashboard --stop` closes panel sessions without stopping agent memory.
After the last browser stream disconnects, the panel deactivates within about
60 seconds. A new activation clears the transient feed and rotates its access
capability. An unsaved draft remains visible in an open stopped tab for copying;
it is not persisted to local storage.

`um-memory dashboard` also starts a hidden owner when no listener is running,
so its browser session survives the CLI process exiting. For a desktop shortcut
use `pythonw -m um_mem.memory.dashboard` on Windows, or
`python -m um_mem.memory.dashboard` elsewhere. Both launchers read the shared
config and keep a newly started listener's owner alive. The service's normal
idle reaper closes it once browsers and agents stop using it. It reuses an
already running shared listener without taking ownership of it. After updating
Python source, restart the shared listener through its operator lifecycle and
reopen the panel; existing processes do not hot-reload Python modules.

## One memory, multiple views

The panel runs on the loopback HTTP listener used by the MCP clients. It does
not create another memory engine, write database rows directly, or introduce a
second content store. Edits use the existing operator lease and canonical writer.
The shared service publishes bounded content-free events after dispatched
operations, including operations whose caller timed out after commit. Background
index refresh produces a separate invalidation; it does not claim that every
native projection has completed.

The atlas presents each block as an explorable space. Click a block to inspect
it, double-click or press Enter to enter it, and use breadcrumbs or Backspace
to go up. The sidebar provides direct navigation. The first snapshot includes
the full canonical block catalog, service health and up to 64 recent workspace
records. Choosing a block enables pagination in batches of 64. Full record text
and retained history are read only on demand. The graph draws parent/child
ownership, loaded records inside the current space, and explicit catalog links between visible
blocks. It does not invent semantic connections or compute all-pairs similarity.
Coordinates and zoom are browser-local presentation preferences, namespaced by
workspace/profile/block. Dragging a node never changes canonical topology.

Search combines eligible agent-recall results with title/preview matches among
records already loaded for operator inspection. The latter remain inspection
results: they do not bypass candidate eligibility or promote memory for agents.
Returning to the graph clears the search and restores the current space.

The event cursor contains a listener epoch and monotonically increasing sequence.
The last 256 events can be replayed after a connection interruption. A stale or
foreign cursor requests a fresh canonical snapshot. The browser merges events
which arrive during a snapshot and preserves the loaded topic extent, selected
record, graph layout and editing draft. A listener restart changes its port and
access capability; reopen the panel using the launcher. Direct embedded writes
outside the shared listener do not publish panel events; reopen the view after
such maintenance. Automatic cross-client updates apply to clients using the
shared listener.

## Editing and trust

Candidate edits carry the exact expected revision through `amend`. A conflict
preserves the draft, displays the current source, and requires an explicit choice
to continue against that version. Each save has one stable mutation correlation
ID; retrying the same operation after an ambiguous response cannot create a
second record. Editing controls are locked while the save is in flight.

For trusted records, the panel offers a correction proposal through the existing
`remember(operation="correct")` path. An optional `expected_revision` is checked
inside `MemoryClient.correction`, after resolving the canonical head, so queueing
cannot silently retarget a proposal to an unseen revision. The candidate remains
advisory and requires the existing evidence and promotion gates. Historical
versions are inspectable but cannot be edited as if they were current.

## Access and observation boundaries

The CLI authenticates as the local operator before obtaining a random capability
in the URL fragment. The fragment is removed immediately and exchanged for an
HttpOnly, SameSite=Strict cookie scoped to the panel and named per listener port.
API access validates the loopback Host and same Origin; there is no wildcard
CORS, filesystem serving or externally hosted asset. HTML is a static shell
without embedded memory or bootstrap secrets. Session cookies last 12 hours;
live streams are bounded to 16 per listener. Capabilities grant operator access
to this workspace/profile; do not share the launch URL.

The activity feed is a bounded view since panel activation, not a durable
audit ledger. Immutable record history remains in the canonical store. The
context view shows retrieval results and whether a context payload was rendered.
It does not claim that a model read, followed or used a record in its answer.
Explicit session identities are supplied by the host. When absent, each MCP
process gets a unique identity; the legacy `default` and `host-mcp` placeholders
also get this fallback. Clients share durable knowledge while their active masks
stay separate. Hosts that reuse one process for all chats still need to supply
per-chat identities to isolate those chats. Search
similarity is labelled as a score, never as factual confidence.

Semantic retrieval revalidates indexed record IDs against current canonical
membership and query eligibility. Historical/retracted records stay available
for audit but cannot return through stale vectors. Parsed vector caches are
separated by index path, model and profile, with transactionally updated version
tokens so same-second writes from another process invalidate them.
