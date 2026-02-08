# Architecture

prompt-prix is a visual fan-out MCP service: identical data dispatched across multiple LLMs simultaneously. Both a Gradio UI (for humans) and an MCP protocol server (for agents) consume the same stateless tool layer.

## Four-Layer Architecture

Per [ADR-006](adr/006-adapter-resource-ownership.md), every import in the codebase follows this strict layer model:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION                            │
│  BatteryRunner │ ConsistencyRunner │ ComparisonSession          │
│                                                                 │
│  • Zero mode awareness — doesn't know react from single-shot   │
│  • Calls execute_test_case(), receives CaseResult              │
│  • Controls concurrency, validation pipeline (refusal → drift) │
│  • NEVER IMPORTS: adapters/*, ServerPool, ConcurrentDispatcher  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ execute_test_case(test, model_id, ...)
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     DISPATCH (react/dispatch.py)                │
│                                                                 │
│  execute_test_case() — the ONLY place that reads test.mode      │
│    mode=None    → _execute_single_shot() → complete_stream()    │
│    mode="react" → _execute_react() → react_step() × N          │
│                                                                 │
│  Returns CaseResult(response, latency_ms, react_trace)          │
│  Raises ReactLoopIncomplete on cycle / max_iterations           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ MCP tool call
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MCP PRIMITIVES                            │
│  complete_stream │ react_step │ judge │ drift │ list_models     │
│  geometry (analyze/generate variants)                           │
│  trajectory (analyze/compare trajectories)                      │
│                                                                 │
│  • Receives adapter via registry (get_adapter())                │
│  • Stateless — no mode awareness                                │
│  • Exposed over JSON-RPC via server.py (prompt-prix-mcp)        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ adapter.stream_completion()
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                       ADAPTER LAYER                             │
│                                                                 │
│  LMStudioAdapter                                                │
│    INTERNAL: ServerPool, ConcurrentDispatcher, httpx            │
│    STRATEGY: Multi-GPU parallel dispatch                        │
│                                                                 │
│  HuggingFaceAdapter                                             │
│    INTERNAL: API client, rate limiter                           │
│    STRATEGY: Rate-limited cloud calls                           │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Import Rules

| Layer | MAY Import | MUST NOT Import |
|-------|------------|-----------------|
| **Orchestration** (BatteryRunner, ConsistencyRunner, ComparisonSession) | `react.dispatch`, `mcp.tools.*`, `mcp.registry` | `adapters/*`, ServerPool, ConcurrentDispatcher |
| **Dispatch** (`react/dispatch.py`) | `mcp.tools.*`, `react.schemas`, `react.cycle_detection` | `adapters/*`, orchestration |
| **MCP Primitives** | `adapters.base.HostAdapter` (protocol), `mcp.registry` | Concrete adapter classes, ServerPool |
| **Adapters** | httpx, internal utilities | Nothing from orchestration or MCP |

> **THE RULE:** ServerPool and ConcurrentDispatcher are INTERNAL to LMStudioAdapter.
> No file outside `adapters/lmstudio.py` may import or reference them.

## Entry Points

Both entry points bootstrap with `register_default_adapter()` and consume `mcp/tools/*`:

| Command | Module | Audience | Transport |
|---------|--------|----------|-----------|
| `prompt-prix` | `main.py` | Humans | Gradio web UI |
| `prompt-prix-mcp` | `mcp/server.py` | Agents | MCP stdio (JSON-RPC) |

`server.py` registers 9 tools with FastMCP via `add_tool()`. Agents (LAS, Claude Desktop, any MCP client) launch `prompt-prix-mcp` as a subprocess.

## Directory Structure

```
prompt_prix/
├── main.py              # Gradio UI entry point (prompt-prix command)
├── ui.py                # Gradio UI definition
├── handlers.py          # Shared event handlers (fetch, stop)
├── state.py             # Global mutable state
├── core.py              # ComparisonSession (orchestration)
├── config.py            # Pydantic models, constants, env loading
├── parsers.py           # Input parsing utilities
├── export.py            # Report generation
├── battery.py           # BatteryRunner (orchestration) - calls execute_test_case()
├── consistency.py       # ConsistencyRunner - multi-run variance testing
├── react/               # ReAct loop execution
│   ├── dispatch.py      # execute_test_case() — single dispatch (ONLY mode reader)
│   ├── schemas.py       # ReActIteration, ToolCall data models
│   └── cycle_detection.py # Stagnation / cycle detection
├── mcp/
│   ├── server.py        # MCP protocol server (FastMCP over stdio) — agent entry point
│   ├── registry.py      # Adapter registry + register_default_adapter()
│   └── tools/
│       ├── complete.py  # complete, complete_stream, latency sentinel utilities
│       ├── react_step.py # Stateless single ReAct iteration primitive
│       ├── drift.py     # Embedding-based semantic drift calculation
│       ├── geometry.py  # Prompt variant generation and distance analysis
│       ├── trajectory.py # Semantic velocity/acceleration analysis
│       ├── judge.py     # LLM-as-judge evaluation
│       ├── list_models.py
│       └── _semantic_chunker.py  # Shared helpers for semantic-chunker tools
├── tabs/
│   ├── battery/
│   │   └── handlers.py  # Battery-specific handlers
│   └── compare/
│       └── handlers.py  # Compare-specific handlers
├── adapters/
│   ├── base.py          # HostAdapter protocol
│   ├── lmstudio.py      # LMStudioAdapter (OWNS ServerPool, ConcurrentDispatcher)
│   └── huggingface.py   # HuggingFaceAdapter (rate-limited cloud calls)
├── semantic_validator.py # Response validation (refusals, tool calls, verdicts)
└── benchmarks/
    ├── base.py          # BenchmarkCase dataclass
    ├── custom_json.py   # CustomJSONLoader (JSON/JSONL)
    └── promptfoo.py     # PromptfooLoader (YAML format)
```

## Adapter Layer

All adapters implement the `HostAdapter` protocol:

```python
class HostAdapter(Protocol):
    async def get_available_models(self) -> list[str]: ...
    async def stream_completion(self, task: InferenceTask) -> AsyncGenerator[str, None]: ...
```

ServerPool and ConcurrentDispatcher are LM Studio concepts. Other backends have different resource models. The adapter encapsulates its internals — orchestration never sees these classes.

## Timeout Contract

A single MCP tool call (e.g. `complete`) passes through three layers, each with different timeout semantics:

```
MCP Client (LAS)          prompt-prix              LM Studio
────────────────          ───────────              ─────────
client timeout_ms    →    no MCP-layer timeout  →  httpx timeout
(client controls)         (FastMCP has none)       (= task.timeout_seconds)
```

| Layer | Timeout | Default | Scope |
|-------|---------|---------|-------|
| **MCP transport** | None | — | FastMCP imposes no timeout. The client (LAS) must set its own `timeout_ms` on the MCP call. |
| **Dispatcher queue** | **Unbounded** | — | `ConcurrentDispatcher.submit()` awaits a server slot with no timeout. If all servers are busy, the call blocks until one frees up. This is intentional: queue wait is excluded from latency measurement. |
| **HTTP inference** | `task.timeout_seconds` | 300s (`complete`), 60s (`InferenceTask` default) | Applied to the httpx client. Covers connection + streaming from LM Studio. |

### Implications for MCP clients

**Single primitive call** (`complete`, `judge`, `react_step`): Wall-clock time = queue wait + inference. With idle servers, queue wait is near-zero and 300s covers even large generations. With busy servers (battery running on the same adapter), queue wait could be minutes — the call blocks in the dispatcher until a slot opens.

**`react_step` in a loop**: Each step is one MCP call. Total wall-clock for an N-step react loop = N × (queue wait + inference). The MCP client controls the loop and can bail out at any point.

**Battery orchestration** (future `run_battery` tool): Would dispatch the entire test matrix internally. Could run 5-30 minutes. A single MCP tool call sitting open that long is architecturally awkward. Options when this becomes needed:
1. **Progress notifications** via MCP notifications (MCP protocol supports `notifications/progress`)
2. **Async pattern**: `start_battery` returns a run ID, `poll_battery` checks status
3. **Keep it out of MCP**: Battery is an orchestration concern — run via Gradio UI or a script, not as an MCP tool

### What happens on MCP connection drop

If the MCP client times out or disconnects while a tool call is in-flight:
- The stdio pipe closes
- FastMCP's event loop exits
- Any in-flight `await` (dispatcher queue or httpx stream) raises `CancelledError`
- `ConcurrentDispatcher.submit()` handles cancellation: if a server was already acquired, it's released back to the pool (lines 184-196 of `lmstudio.py`)
- No orphaned state — the adapter cleans up

### Setting client-side timeout

For LAS or other MCP clients, recommended `timeout_ms`:

| Tool | Recommended | Rationale |
|------|-------------|-----------|
| `list_models` | 30s | Network round-trip to each server |
| `complete` | 600s | 300s inference + up to 300s queue wait |
| `react_step` | 600s | Same as `complete` (one inference call) |
| `judge` | 600s | Same as `complete` (uses LLM inference internally) |
| `calculate_drift` | 10s | Near-instant embedding cosine distance |
| `analyze_variants`, `analyze_trajectory`, `compare_trajectories` | 10s | Embedding-based, no LLM inference |
| `generate_variants` | 600s | Uses LLM inference |

## Battery Execution: Pipelined Judging

When a judge model is selected, BatteryRunner uses **pipelined execution** — judge tasks are submitted eagerly as inference results complete, rather than waiting for all inference to finish first:

```
Without pipelining (original two-phase, ADR-008):
  Phase 1: [inference][inference][inference][inference]
  Phase 2:                                              [judge][judge][judge][judge]

With pipelining:
  GPU0:    [inference][inference][judge][judge][judge]    ← GPU0 idles early, starts judging
  GPU1:    [inference][inference][inference][inference]   ← GPU1 still doing heavy models
```

The `current_model` drain guard on `ServerPool` is the enabler — judge tasks queue in the dispatcher until a server drains its inference model. When no judge model is set, `_execute_inference_phase()` runs directly with no pipelining overhead.

See [ADR-008](adr/ADR-008-judge-scheduling-strategy.md) for the evolution from two-phase to pipelined scheduling.

## ReAct Loop Execution

Tests with `mode="react"` evaluate multi-step tool-use loops. The key design decision: **a react loop is just another way to produce a pass/fail verdict for a (test, model) cell.** React tests flow through the same orchestration pipeline as standard tests — they get drift validation, judge evaluation, and consistency testing for free.

`execute_test_case()` in `react/dispatch.py` is the **only place** that reads `test.mode`. Orchestration above and MCP tools below have zero mode awareness.

The react loop:
1. Calls `react_step()` MCP primitive (stateless — takes trace in, returns one step out)
2. Accumulates `ReActIteration` objects in the trace
3. Checks for stagnation via `detect_cycle_with_pattern()` after each step
4. Completes when the model responds with text only (no tool calls)
5. Raises `ReactLoopIncomplete` on cycle detection or `max_iterations` exhaustion

| Outcome | Result |
|---------|--------|
| Loop completes (final text answer) | `RunResult(COMPLETED)` |
| Cycle detected or max iterations | `RunResult(SEMANTIC_FAILURE)` |
| Infrastructure error | `RunResult(ERROR)` |

## Consistency Testing

`ConsistencyRunner` runs each (test, model) cell N times with different random seeds to identify models that produce inconsistent results.

| Status | Symbol | Meaning |
|--------|--------|---------|
| `CONSISTENT_PASS` | ✓ | N/N runs passed |
| `CONSISTENT_FAIL` | ❌ | 0/N runs passed |
| `INCONSISTENT` | 🟣 3/5 | Some runs passed, some failed |

See [ADR-010](adr/ADR-010-consistency-runner.md) for rationale.

## Semantic Validation

Battery tests validate responses beyond HTTP success (`semantic_validator.py`):

| Check | Trigger | Failure Reason |
|-------|---------|----------------|
| **Empty response** | Response is empty/whitespace | "Empty response" |
| **Refusal detection** | Matches refusal phrases | "Model refused: '{phrase}'" |
| **Tool call required** | `tool_choice: "required"` | "Expected tool call but got text response" |
| **Tool call forbidden** | `tool_choice: "none"` | "Tool call made when tool_choice='none'" |
| **Verdict matching** | `pass_criteria` contains verdict | "Verdict mismatch: expected X, got Y" |

Checks run in order (first failure wins). Verdict matching enables judge competence tests — testing whether a model can correctly judge other outputs.

| Status | Symbol | Meaning |
|--------|--------|---------|
| `COMPLETED` | ✓ | Response passed semantic validation |
| `SEMANTIC_FAILURE` | ❌ | Response received but failed semantic check |
| `ERROR` | ⚠ | Infrastructure error (timeout, connection, etc.) |

## Battery File Formats

**Required fields:** `id`, `user`

**Optional fields:** `name`, `category`, `severity`, `system`, `tools`, `tool_choice`, `mode`, `mock_tools`, `max_iterations`, `expected`, `pass_criteria`, `fail_criteria`, `expected_response`

**Formats:** JSON (with `prompts` array), JSONL (one per line), Promptfoo YAML (with `prompts` + `tests`).

Promptfoo vars extraction:

| Var | BenchmarkCase field | Purpose |
|-----|-------------------|---------|
| `expected_verdict` | `pass_criteria` | Rubric text for LLM judge evaluation |
| `expected_response` | `expected_response` | Exemplar text for embedding drift comparison |
| `category` | `category` | Test category for filtering/grouping |
| `system` | `system` | System message |
| `user` | `user` | User message |

Promptfoo `assert` blocks are logged but **not evaluated** (warning emitted).

## Integration Points

All inference servers must expose OpenAI-compatible endpoints (`GET /v1/models`, `POST /v1/chat/completions`). Supported: LM Studio, Ollama, vLLM, llama.cpp server, any OpenAI-compatible proxy. See [ADR-003](adr/003-openai-compatible-api.md).

## Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [001](adr/001-use-existing-benchmarks.md) | Use existing benchmarks (BFCL, Inspect AI) instead of custom eval schema |
| [002](adr/002-fan-out-pattern-as-core.md) | Fan-out pattern as core architectural abstraction |
| [003](adr/003-openai-compatible-api.md) | OpenAI-compatible API as sole integration layer |
| [006](adr/006-adapter-resource-ownership.md) | Adapters own their resource management (ServerPool internal to LMStudioAdapter) |
| [007](adr/ADR-007-inference-task-schema.md) | InferenceTask schema for adapter interface |
| [008](adr/ADR-008-judge-scheduling-strategy.md) | Pipelined judge scheduling for multi-GPU efficiency |
| [009](adr/ADR-009-interactive-battery-grid.md) | Dismissible dialog for battery grid cell detail |
| [010](adr/ADR-010-consistency-runner.md) | Multi-run consistency analysis (proposed) |
| [011](adr/ADR-011-embedding-based-validation.md) | Embedding-based semantic validation (proposed) |
| [012](adr/ADR-012-compare-to-battery-export.md) | Compare to Battery export pipeline (proposed) |
| [013](adr/ADR-013-semantic-chunker-mcp-primitives.md) | Semantic-chunker MCP primitives (geometry, trajectory) |
| 014 | MCP protocol server — FastMCP over stdio for agent access |
