import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import plugin from "../dist/index.js";

test("native Mnemosyne: pages, hypothesis, correction, focus switch, retraction and cross-session recall", {
  skip: !process.env.UM_MEM_SMOKE_PYTHON,
  timeout: 120_000,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "um-mem-opencode-"));
  const env = { UM_MEM_REPO_ROOT: root, UM_MEM_WORKSPACE: "smoke", UM_MEM_PROFILE: "default", UM_MEM_BACKEND: "mnemosyne", UM_MEM_SEMANTIC: "0" };
  const tools = new Map();
  const hooks = new Map();
  let cleanup;
  try {
    cleanup = await plugin.setup({
      options: { command: [process.env.UM_MEM_SMOKE_PYTHON, "-m", "um_mem.memory.mcp_server"], env },
      location: { directory: root },
      tool: { transform: async (apply) => apply({ namespace() {}, add: (tool) => tools.set(tool.name, tool) }) },
      session: { hook: async (name, hook) => hooks.set(name, hook) },
    });
    let sequence = 0;
    const call = async (name, input, sessionID = "native-a") => {
      const result = await tools.get(name).execute(input, { sessionID, messageID: "native-message", id: `call-${++sequence}`, signal: new AbortController().signal });
      const body = JSON.parse(result.content);
      assert.equal(body.ok, true, JSON.stringify(body));
      return body;
    };
    const health = await call("health", {});
    assert.equal(health.native_ok, true);
    assert.equal(health.effective_mode, "selected_native");
    const map = await call("block_map", {});
    const leaf = await call("create_block", { name: "Constraint", synopsis: "Verified smoke constraint", parent_id: map.roots[0].block_id, reason: "isolated smoke" });
    const secondLeaf = await call("create_block", { name: "Second topic", synopsis: "Independent second constraint", parent_id: map.roots[0].block_id, reason: "separate topic" });
    const distractor = await call("create_block", { name: "Unrelated topic", synopsis: "Unrelated evidence", parent_id: map.roots[0].block_id, reason: "separate topic" });
    let cursor = "";
    const seen = [];
    do {
      const page = await call("expand_block", { block_id: map.roots[0].block_id, limit: 1, cursor });
      assert.equal(page.children.length, 1);
      assert.equal(page.child_count, 3);
      assert.equal(page.root.diagnostics_status, "not_probed");
      seen.push(page.children[0].block_id);
      cursor = page.next_cursor;
    } while (cursor);
    assert.deepEqual(new Set(seen), new Set([leaf.block_id, secondLeaf.block_id, distractor.block_id]));
    const saved = await call("remember", { content: "The opaque constraint is A7K.", title: "Opaque constraint", kind: "user_preference", evidence_kind: "user_statement", requested_block_id: leaf.block_id });
    const recalled = await call("recall", { query: "opaque constraint", logical_block_ids: [leaf.block_id] });
    assert.equal(recalled.items[0].kind, "user_preference");
    await call("set_active_mask", { selected_block_ids: [leaf.block_id], token_budget: 500, activation_reason: "current smoke task" });
    const round = async (sessionID) => {
      const event = { sessionID, system: [], messages: [] };
      await hooks.get("context")(event);
      return event.system.map((part) => part.text).join("\n");
    };
    assert.match(await round("native-a"), /A7K/);
    const updated = await call("amend", { memory_id: saved.memory_id, expected_revision: saved.revision_id, content: "The opaque constraint is B9R.", reason: "source correction", requested_block_id: leaf.block_id });
    assert.ok(updated.new_id);
    const next = await round("native-a");
    assert.match(next, /B9R/);
    assert.doesNotMatch(next, /A7K/);
    assert.doesNotMatch(await round("native-b"), /B9R|A7K/);
    const shared = await call("recall", { query: "opaque constraint", logical_block_ids: [leaf.block_id] }, "native-b");
    assert.match(shared.items[0].text, /B9R/);
    assert.equal(shared.items[0].authority, "advisory");
    const hypothesis = await call("remember", { content: "Unverified hypothesis: the second topic constraint is C5L.", title: "Second-topic hypothesis", requested_block_id: secondLeaf.block_id });
    let mask = await call("get_active_mask", {});
    await call("amend_active_mask", { mask_id: mask.mask_id, expected_revision: mask.revision, selected_block_ids: [secondLeaf.block_id], activation_reason: "switch to the second topic" });
    const switched = await round("native-a");
    assert.match(switched, /C5L/);
    assert.match(switched, /unverified/i);
    assert.doesNotMatch(switched, /A7K|B9R/);
    await call("forget", { memory_id: hypothesis.memory_id, reason: "hypothesis refuted by evidence" });
    assert.doesNotMatch(await round("native-a"), /C5L|A7K|B9R/);
    mask = await call("get_active_mask", {});
    await call("transition_active_mask", { mask_id: mask.mask_id, expected_revision: mask.revision, transition: "deactivate", reason: "task finished" });
    assert.doesNotMatch(await round("native-a"), /B9R|A7K/);
  } finally {
    if (cleanup) await cleanup();
    execFileSync(process.env.UM_MEM_SMOKE_PYTHON, ["-c", "from um_mem.memory.local_service import build_local_service_identity, shutdown_local_service_listener; shutdown_local_service_listener(build_local_service_identity(repo_root=None, workspace_id=None, backend=None, caller_role='operator'))"], { env: { ...process.env, ...env }, stdio: "pipe", timeout: 30_000 });
    await rm(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
  }
});
