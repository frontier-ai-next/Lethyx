import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import plugin from "../dist/index.js";
import { Host } from "@opencode/plugin/host";

test("tools and each model continuation share session focus without leaking to another session", async () => {
  const tools = new Map();
  const hooks = new Map();
  const cleanup = await plugin.setup({
    options: { command: [process.execPath, fileURLToPath(new URL("./memory-server.mjs", import.meta.url))] },
    location: { directory: process.cwd() },
    tool: { transform: async (apply) => apply({ namespace() {}, add: (tool) => tools.set(tool.name, tool) }) },
    session: { hook: async (name, hook) => hooks.set(name, hook) },
  });
  const context = (sessionID, messageID = "message-1") => ({ sessionID, messageID, signal: new AbortController().signal, id: "call-1" });
  const round = async (sessionID) => {
    const event = { sessionID, system: [], messages: [] };
    await hooks.get("context")(event);
    return event.system.map((part) => part.text).join("\n");
  };
  try {
    await tools.get("set_active_mask").execute({ value: "old constraint" }, context("session-a"));
    const focused = await round("session-a");
    assert.match(focused, /old constraint/);
    assert.match(focused, /subtask changes/i);
    assert.match(focused, /transition_active_mask/);
    assert.match(focused, /covers this project only/i);
    assert.match(focused, /or nothing/i);
    assert.match(focused, /stable_project_facts/);
    assert.match(focused, /routes them to matching project twins/i);
    assert.doesNotMatch(await round("session-b"), /old constraint/);
    await tools.get("amend_active_mask").execute({ value: "corrected constraint" }, context("session-a"));
    const next = await round("session-a");
    assert.match(next, /corrected constraint/);
    assert.doesNotMatch(next, /old constraint/);
    assert.match(await round("session-a"), /corrected constraint/); // fresh context after compaction
    await tools.get("transition_active_mask").execute({}, context("session-a"));
    assert.doesNotMatch(await round("session-a"), /corrected constraint/);
    await tools.get("set_active_mask").execute({ value: "FAIL" }, context("session-b"));
    assert.match(await round("session-b"), /unavailable/i);
    assert.doesNotMatch(await round("session-a"), /FAIL/);
    const first = await tools.get("remember").execute({}, context("session-a"));
    assert.deepEqual(first.output, JSON.parse(first.content));
    assert.deepEqual(await tools.get("remember").execute({}, context("session-a")), first);
    assert.notDeepEqual(await tools.get("remember").execute({}, context("session-b")), first);
    assert.notDeepEqual(await tools.get("remember").execute({}, context("session-a", "message-2")), first);
    await assert.rejects(tools.get("fail").execute({}, context("session-a")), /fixture tool rejected/);
    for (let index = 0; index < 18; index++) {
      assert.doesNotMatch(await round(`sequential-${index}`), /unavailable/i);
    }
  } finally {
    await cleanup();
  }
});

test("invalid plugin configuration fails before starting a server", async () => {
  await assert.rejects(plugin.setup({ options: { command: [] } }), /command/);
});

test("OpenCode resolves the built plugin from a local directory", () => {
  const entry = Host.resolve({ directory: fileURLToPath(new URL("..", import.meta.url)) });
  assert.ok(entry.server, "local plugin directory must expose a server entrypoint");
});
