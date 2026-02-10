# ADR-014: MCP ext-apps for Battery Dashboard

**Status**: Deferred
**Date**: 2026-02-09
**Related**:
- ADR-006 (Adapter Resource Ownership)
- ADR-007 (CLI Interface Layer)
- SEP-1865 (MCP Apps Extension)

---

## Context

prompt-prix has three entry points: Gradio UI (humans), CLI (agents via `run_command`), and MCP server (iteration-level primitives). The Gradio UI provides a rich battery results dashboard — interactive grid, cell detail, export to JSON/CSV/PNG. The CLI returns structured JSON. The MCP server exposes fine-grained tools (complete, judge, etc.) but no battery-level orchestration.

MCP ext-apps (SEP-1865), merged January 2026, is the first official MCP extension. It allows servers to declare `ui://` resources containing interactive HTML, associated with tools via metadata. Supporting hosts render the UI in sandboxed iframes alongside tool results. This could unify the human and agent experience: an agent running a battery through MCP would see the same visual dashboard that a human sees in Gradio.

### Research Summary

**What ext-apps provides:**
- Servers declare `ui://` resources (complete HTML documents with bundled JS/CSS)
- Tools link to resources via `_meta.ui.resourceUri`
- Host renders sandboxed iframe with strict CSP, delivers tool results via `postMessage` (`ui/notifications/tool-result`)
- Bidirectional: UI can call server tools back via `callServerTool()`, update model context, request display mode changes
- Lifecycle: tool call → resource fetch → iframe render → data delivery → interactive phase → teardown

**Host support (as of Feb 2026):**
Claude (web), Claude Desktop, VS Code Insiders, Goose, ChatGPT, Postman

**Python MCP SDK 1.26.0 status:**
- No native ext-apps module (`io.modelcontextprotocol/ui` extension)
- Building blocks exist: `_meta` field on `Tool` type, resource system with MIME types, `experimental` capability declaration
- Manual metadata injection possible but fragile — no SDK validation, could break on upgrades

**Gradio MCP integration:**
- Gradio has `@gr.mcp.resource("ui://...", mime_type="text/html+skybridge")` support
- Uses ChatGPT-specific metadata (`openai/outputTemplate`, `openai/resultCanProduceWidget`) — not the standard ext-apps protocol (`_meta.ui.resourceUri`, `text/html;profile=mcp-app`)
- Requires separate HTML/JS — does not wrap existing Gradio components as ext-app resources

---

## Decision

**Deferred.** The integration is architecturally sound but blocked on SDK maturity.

### How It Would Work

```
Agent → MCP Client → tools/call "run_battery" → prompt-prix MCP server
                                                      │
                                                      ├─ Starts BatteryRunner (same as CLI/Gradio)
                                                      ├─ Returns job_id + initial state
                                                      │
MCP Client → resources/read "ui://prompt-prix/battery.html"
                                                      │
                                                      ├─ Server returns bundled HTML dashboard
                                                      ├─ Host renders sandboxed iframe
                                                      │
Host → iframe: ui/notifications/tool-result { job_id, initial_state }
                                                      │
iframe → Host → Server: tools/call "get_battery_status" { job_id }  (polling)
                                                      │
                                                      ├─ Returns current BatteryRun/ConsistencyRun state
                                                      ├─ iframe re-renders grid with progress
                                                      └─ Repeats until complete
```

### What Would Be Required

1. **Stateful MCP server** — Track running batteries by job ID. Significant departure from current stateless tool design. Could use a simple in-process dict (batteries don't survive restarts anyway).

2. **Two new MCP tools:**
   - `run_battery(tests_path, models, runs, ...)` → starts battery, returns job ID
   - `get_battery_status(job_id)` → returns current state snapshot (with `visibility: ["app"]` — hidden from LLM, only callable by UI)

3. **One `ui://` resource** — `ui://prompt-prix/battery.html`, a single-file HTML dashboard that:
   - Receives initial state via `tool-result` notification
   - Polls `get_battery_status` via `callServerTool()`
   - Renders results grid (tests × models, status symbols, latency)
   - Supports cell selection for detail view

4. **Tool metadata injection** — Add `_meta.ui.resourceUri` to `run_battery` tool definition. Currently requires manual dict injection via FastMCP's `meta` parameter.

5. **Capability negotiation** — Declare `io.modelcontextprotocol/ui` in `ServerCapabilities.experimental`. Progressive enhancement: hosts without ext-apps support get text-only results.

### Prerequisites for Revisiting

- [ ] Python MCP SDK adds native ext-apps extension support (module, types, capability negotiation)
- [ ] At least one target host (VS Code, Claude Desktop) confirmed working with Python MCP servers serving `ui://` resources
- [ ] Gradio's `@gr.mcp.resource` converges on standard ext-apps metadata (or we decide to bypass Gradio's MCP layer entirely)

---

## Consequences

### Positive (when unblocked)

- Agents see the same visual battery dashboard as humans — true interface parity
- Progressive enhancement — falls back to structured JSON for non-ext-apps hosts
- Reuses existing BatteryRunner/ConsistencyRunner internals (no new orchestration code)
- `get_battery_status` with `visibility: ["app"]` keeps the LLM's tool list clean

### Negative

- Statefulness in the MCP server adds complexity and a new failure mode (orphaned jobs)
- HTML dashboard is a separate build artifact requiring frontend tooling (Vite + single-file plugin or hand-rolled)
- Two parallel UI codebases (Gradio + ext-app HTML) must stay in sync visually
- SDK gap means early implementation would be fragile

### Near-Term Impact

- None — this is a research capture, not an implementation decision
- CLI (ADR-007) remains the primary agent interface for battery execution
- MCP server remains stateless with fine-grained tools
- Gradio UI remains the primary human interface

---

## References

- [MCP Apps Specification (2026-01-26)](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx)
- [ext-apps GitHub Repository](https://github.com/modelcontextprotocol/ext-apps)
- [SEP-1865 Pull Request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1865)
- [MCP Apps Blog Post](http://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/)
- [Building ChatGPT Apps with Gradio](https://www.gradio.app/guides/building-chatgpt-apps-with-gradio) — Gradio's `ui://` resource pattern
- [Building MCP Server with Gradio](https://www.gradio.app/guides/building-mcp-server-with-gradio) — `@gr.mcp.resource` API
- Python MCP SDK 1.26.0 — `mcp/types.py` (Tool._meta), `mcp/server/fastmcp/server.py` (meta parameter)
