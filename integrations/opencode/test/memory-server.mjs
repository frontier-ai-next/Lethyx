import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "session-memory-fixture", version: "1" });
let value = "";
const result = (body) => ({ content: [{ type: "text", text: JSON.stringify(body) }], structuredContent: body });
for (const name of ["set_active_mask", "amend_active_mask"]) {
  server.registerTool(name, { inputSchema: { value: z.string() } }, ({ value: next }) => {
    value = next;
    return result({ ok: true });
  });
}
server.registerTool("transition_active_mask", { inputSchema: {} }, () => {
  value = "";
  return result({ ok: true });
});
server.registerTool("resolve_active_mask", { inputSchema: { max_tokens: z.number().optional() } }, () => {
  if (!process.env.UM_MEM_PROJECT_ID || process.env.UM_MEM_ACTIVE_SCOPE_PROJECT_ONLY !== "1") {
    return result({ ok: false, error: "project scope missing" });
  }
  if (value === "FAIL") return result({ ok: false, error: "fixture outage" });
  return result({ ok: true, lease: { rendered_context: value, trace: { resolution_status: value ? "resolved" : "not_found" } } });
});
server.registerTool("remember", { inputSchema: { operation_id: z.string() } }, ({ operation_id }) => result({ operation_id }));
server.registerTool("fail", { inputSchema: {} }, () => ({ isError: true, content: [{ type: "text", text: "fixture tool rejected" }] }));
await server.connect(new StdioServerTransport());
