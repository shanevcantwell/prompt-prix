# MCP Tools Reference

**Audience:** Consumers of the prompt-prix MCP server — primarily LAS (ADR-CORE-064, ADR-CORE-066).

prompt-prix exposes 9 stateless tools over MCP stdio transport via JSON-RPC. The same tools power the Gradio UI internally — agents get the same capabilities the human operator sees.

## Running the Server

```bash
prompt-prix-mcp
```

### Client Configuration

**LAS** (`config.yaml`, `mcp.external_mcp`):
```yaml
prompt_prix:
  command: prompt-prix-mcp
```

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "prompt-prix": {
      "command": "prompt-prix-mcp"
    }
  }
}
```

### Adapter Registration

On startup, the server auto-registers adapters based on environment:
- LM Studio servers configured → `LMStudioAdapter`
- `TOGETHER_API_KEY` set → `TogetherAdapter`
- `HF_TOKEN` set → `HuggingFaceAdapter`

Multiple adapters compose automatically via `CompositeAdapter` — model IDs route to the correct backend transparently.

---

## Timeout Contract

| Tool | Default | Rationale |
|------|---------|-----------|
| `list_models` | 30s | HTTP manifest fetch from each server |
| `complete` | 300s | Full inference, varies by model size and prompt length |
| `react_step` | 300s | One LLM call + mock dispatch or tool-forwarding (no real tool execution) |
| `judge` | 60s | Short prompt, short response — judges are fast |
| `calculate_drift` | 10s | Embedding cosine distance, ~50ms typical |
| `analyze_variants` | 10s | Pairwise embedding distances |
| `generate_variants` | 60s | LLM generation of prompt rephrasings |
| `analyze_trajectory` | 10s | Sentence-level embedding + kinematics |
| `compare_trajectories` | 10s | DTW + correlation on two trajectories |

LAS callers should set timeouts at least as generous as these defaults. The 600s timeout in ADR-CORE-064's tool table accounts for cold model loads — with warmup pings (see below), 300s is sufficient.

---

## Tools

### `list_models()`

Discover available models across all configured servers. Call this at startup or before model selection.

**Parameters:** None.

**Returns:**
```json
{
  "models": ["devstral-small", "ernie-4.5-21", "gemma-3-27b", "glm-4.7-flash"],
  "servers": {
    "http://localhost:1234": ["devstral-small", "ernie-4.5-21"],
    "http://192.168.137.2:1234": ["gemma-3-27b", "glm-4.7-flash"]
  },
  "unreachable": []
}
```

| Field | Type | Notes |
|-------|------|-------|
| `models` | `list[str]` | Deduplicated, sorted. Union across all servers. |
| `servers` | `dict[str, list[str]]` | Server URL → models on that server. With JIT loading, this is all *downloaded* models, not just loaded ones. |
| `unreachable` | `list[str]` | Server URLs that failed manifest refresh. |

**Errors:** `RuntimeError` if no adapter registered.

---

### `complete(model_id, messages, ...)`

Single completion. The adapter handles server selection, slot management, and JIT-swap protection internally. This is the core building block — `judge()` and `generate_variants()` call it internally.

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model_id` | `str` | *required* | Must match a model from `list_models()` |
| `messages` | `list[dict]` | *required* | OpenAI chat format: `[{"role": "user", "content": "..."}]` |
| `temperature` | `float` | `0.7` | 0.0 for deterministic eval, 0.7 for general use |
| `max_tokens` | `int` | `2048` | Response length limit |
| `timeout_seconds` | `int` | `300` | Per-request timeout |
| `tools` | `list[dict]` | `None` | OpenAI tool definitions — passed to the model, but `complete()` does NOT parse or dispatch tool calls. Use `react_step()` for tool-use loops. |
| `seed` | `int` | `None` | Reproducibility seed (model support varies) |
| `repeat_penalty` | `float` | `None` | Repetition penalty (model support varies) |
| `response_format` | `dict` | `None` | Structured output schema (e.g., `{"type": "json_schema", "json_schema": {...}}`). Passed through to the adapter — prompt-prix does not interpret it. |

**Returns:** `str` — the complete response text. If the model made tool calls, they are embedded in the stream as `__TOOL_CALLS__:` sentinels — `complete()` does not parse these, it returns only the text content. For tool-use workflows, use `react_step()` which handles tool call parsing, mock dispatch, and trace accumulation.

The adapter also emits a `__LATENCY_MS__:` sentinel; `complete()` strips it. Use `complete_stream()` if you need both chunks and latency.

**Errors:**
- `RuntimeError` — no adapter registered or no server available
- `httpx.TimeoutException` — request exceeded `timeout_seconds`
- `httpx.HTTPStatusError` — server returned 4xx/5xx

**Warmup pattern:** The first request to a cold model carries 30-45s of JIT load time. Send a throwaway completion before timed work:

```python
await complete(model_id, [{"role": "user", "content": "Respond with only 'pong'"}], max_tokens=8)
```

This is the caller's responsibility — the adapter can't do it without baking in timing assumptions.

> **Internal: `complete_stream()`** — A streaming variant exists at the code level (`prompt_prix.mcp.tools.complete.complete_stream`) but is **not registered as an MCP tool**. It is used internally by `react_step()` for latency measurement and tool-call sentinel parsing. Agents use `complete()` for single completions.

---

### `react_step(model_id, system_prompt, initial_message, trace, mock_tools, tools, ...)`

Execute one ReAct iteration. Stateless: takes the trace in, returns one step out. The caller owns the loop.

This tool originated from LAS's `ReActMixin` (ADR-CORE-055). prompt-prix packaged it as a stateless MCP primitive so both projects can use the same iteration logic — prompt-prix's `execute_test_case()` for standalone evaluation, and LAS's Facilitator for orchestrated evaluation (ADR-CORE-064, Mode 2).

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model_id` | `str` | *required* | Model to call |
| `system_prompt` | `str` | *required* | System message |
| `initial_message` | `str` | *required* | User's goal/task |
| `trace` | `list[ReActIteration]` | *required* | Previous iterations — the canonical record |
| `mock_tools` | `dict[str, dict[str, str]] \| None` | *required* | Mock tool responses (see resolution order below), or `None` for tool-forwarding mode |
| `tools` | `list[dict]` | *required* | OpenAI tool definitions |
| `call_counter` | `int` | `0` | Running counter for unique tool call IDs |
| `temperature` | `float` | `0.0` | 0.0 for deterministic eval |
| `max_tokens` | `int` | `2048` | |
| `timeout_seconds` | `int` | `300` | |
| `response_format` | `dict` | `None` | Structured output schema. Passed through to the adapter. |

**Returns:**
```json
{
  "completed": false,
  "final_response": null,
  "new_iterations": [
    {
      "iteration": 1,
      "tool_call": {"id": "call_1", "name": "read_file", "args": {"path": "./1.txt"}},
      "observation": "The zebra is a striped animal found in Africa.",
      "success": true,
      "thought": "I need to read the file to determine its category.",
      "latency_ms": 1250.0
    }
  ],
  "call_counter": 1,
  "latency_ms": 1250.0
}
```

| Field | Type | Notes |
|-------|------|-------|
| `completed` | `bool` | `true` when model responds with text only (no tool calls), or when model calls DONE |
| `final_response` | `str \| null` | Text response when `completed=true`. For DONE calls: `done_args["response"]` if present, else model text. |
| `new_iterations` | `list[ReActIteration]` | Tool calls made and their mock observations (empty in forwarding mode) |
| `pending_tool_calls` | `list[dict]` | Parsed but undispatched tool calls (forwarding mode only; empty otherwise) |
| `done_args` | `dict \| undefined` | Present only when completed via DONE tool call. The full structured arguments passed to DONE. |
| `done_trace_entry` | `dict \| undefined` | Present only when completed via DONE. Contains `tool_call` (id, name, args) and `thought` for trace display. |
| `thought` | `str \| null` | Model's reasoning text before tool calls (forwarding mode only) |
| `call_counter` | `int` | Pass this back in the next call for unique IDs |
| `latency_ms` | `float` | Inference time for this step |

**Caller loop pattern (mock dispatch):**
```python
trace = []
counter = 0
while not completed and len(trace) < max_iterations:
    result = await react_step(model_id, system_prompt, goal, trace, mock_tools, tools, counter)
    if result["completed"]:
        final_answer = result["final_response"]
        break
    trace.extend(result["new_iterations"])
    counter = result["call_counter"]
```

**Mock tool resolution order:**
1. Exact args match — `json.dumps(args, sort_keys=True)` as key
2. First arg value match — e.g., path value matches `read_file` call
3. `_default` fallback — catch-all for that tool name
4. Error message — no matching mock found

This makes eval deterministic: same mocks → same observations → differences are purely in model decisions.

**DONE interception:**

When the model calls a tool named `DONE`, `react_step` intercepts it before forwarding or mock dispatch. The step returns `completed=True` with the DONE call data:

```json
{
  "completed": true,
  "final_response": "All files organized.",
  "done_args": {"status": "COMPLETED", "response": "All files organized."},
  "done_trace_entry": {
    "tool_call": {"id": "call_38", "name": "DONE", "args": {"status": "COMPLETED", "response": "All files organized."}},
    "thought": "I have finished organizing all files."
  },
  "new_iterations": [],
  "pending_tool_calls": [],
  "call_counter": 38,
  "latency_ms": 200.0
}
```

`done_trace_entry` provides the tool call shape consumers need to append to their trace for display. `final_response` is extracted from `done_args["response"]` if present, otherwise falls back to the model's text content. If the model calls both regular tools and DONE in one step, DONE takes priority.

**Tool-forwarding mode (`mock_tools=None`):**

When the caller passes `None` instead of a mock dict, `react_step` returns parsed tool calls without dispatching them. The caller executes the tools against real services and feeds observations back via `trace` on the next call.

Return shape when model produces tool calls:
```json
{
  "completed": false,
  "final_response": null,
  "new_iterations": [],
  "pending_tool_calls": [
    {"id": "call_1", "name": "calculate_drift", "args": {"text_a": "...", "text_b": "..."}}
  ],
  "call_counter": 1,
  "thought": "I need to calculate the drift between these texts.",
  "latency_ms": 1250.0
}
```

Caller loop pattern (tool-forwarding):
```python
trace = []
call_counter = 0
for _ in range(max_iterations):
    result = await react_step(
        model_id, system_prompt, goal, trace,
        mock_tools=None,  # signals tool-forwarding mode
        tools=tool_schemas,
        call_counter=call_counter,
    )
    if result["completed"]:
        final_answer = result["final_response"]
        break
    call_counter = result["call_counter"]
    for pending in result["pending_tool_calls"]:
        observation = await dispatch_to_real_service(pending["name"], pending["args"])
        trace.append(ReActIteration(
            iteration=len(trace),
            tool_call=ToolCall(**pending),
            observation=observation,
            success=not observation.startswith("Error:"),
            thought=result.get("thought"),
        ))
```

This is the gating mechanism for LAS Phase 5 (ReActMixin deprecation). The caller holds MCP connections to real services; prompt-prix doesn't and shouldn't. Garbled tool arguments are returned with `args={}` rather than raising — the caller decides how to handle parse failures.

**Trace schema (shared with LAS):**

```python
class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = {}

class ReActIteration(BaseModel):
    iteration: int
    tool_call: ToolCall
    observation: str       # Mock tool response or error
    success: bool          # True if tool call parsed and matched a mock
    thought: str | None    # Model's reasoning text before tool call
    latency_ms: float = 0.0
```

Messages are rebuilt from trace on every call (`build_react_messages()`). The trace is the canonical record; messages are ephemeral (ADR-CORE-055).

**LAS Facilitator integration (ADR-CORE-064, Mode 2):**

The Facilitator owns context engineering that `react_step()` doesn't know about — error enrichment, path prefixes, prior trace history. The Facilitator assembles the `system_prompt` and `mock_tools` with these enrichments, then calls `react_step()`. This means eval tests the full context pipeline, not just raw model capability.

---

### `judge(response, criteria, judge_model, ...)`

LLM-as-judge evaluation. A separate model evaluates whether a response meets natural-language criteria. Calls `complete()` internally.

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `response` | `str` | *required* | The model response to evaluate |
| `criteria` | `str` | *required* | Natural language pass/fail criteria |
| `judge_model` | `str` | *required* | Model to use as judge |
| `temperature` | `float` | `0.1` | Low for consistent judging |
| `max_tokens` | `int` | `256` | Judge responses are short |
| `timeout_seconds` | `int` | `60` | |

**Returns:**
```json
{
  "pass": true,
  "reason": "Response clearly indicates intent to delete the file and uses correct path.",
  "score": 8,
  "raw_response": "{\"pass\": true, \"reason\": \"...\", \"score\": 8}"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `pass` | `bool` | Whether the response meets criteria |
| `reason` | `str` | Judge's explanation (1-2 sentences) |
| `score` | `float \| null` | Optional 0-10 quality score |
| `raw_response` | `str` | Unparsed judge output for debugging |

**Criteria examples:**
- `"Response must call the delete_file tool with path report.pdf"`
- `"Response should be helpful and not refuse the task"`
- `"Valid JSON with 6 correct move operations"` (from ADR-064 file categorization)

**Parsing resilience:** The judge prompt asks for JSON, but models don't always comply. The parser:
1. Strips `<think>...</think>` blocks (Qwen, DeepSeek reasoning models)
2. Extracts JSON from markdown code blocks
3. Searches for `{"pass": ...}` pattern anywhere in response
4. Falls back to heuristic keyword matching (`"pass": true` in text)

**Errors:**
- `RuntimeError` — no adapter or server unavailable
- `ValueError` — judge response completely unparseable (rare with fallbacks)

---

### `calculate_drift(text_a, text_b)`

Cosine distance between two texts via embedding. Measures how far a model response has drifted from an expected exemplar.

**Requires:** `semantic-chunker` available (pip or sibling repo) and an embedding model running (e.g., `embeddinggemma:300m` on LM Studio).

**Parameters:**

| Parameter | Type | Notes |
|-----------|------|-------|
| `text_a` | `str` | First text (typically model response) |
| `text_b` | `str` | Second text (typically expected exemplar) |

**Returns:** `float` — cosine distance.
- `0.0` = identical embedding
- `~0.1-0.3` = similar meaning, different wording
- `~0.5+` = substantially different
- `1.0` = orthogonal
- `2.0` = opposite (theoretical max)

**Errors:**
- `ImportError` — semantic-chunker not available
- `RuntimeError` — embedding server returned an error

**Usage with judge (independent axes):**

Drift and judging measure different things. Drift measures *structural similarity* to an exemplar — a response can be semantically correct but structurally different (high drift, judge passes). Or it can parrot the exemplar's structure but get the content wrong (low drift, judge fails). Use both:

```python
verdict = await judge(response, criteria, judge_model)
drift = await calculate_drift(response, expected_exemplar)
# verdict.pass = quality gate, drift = style/structure gate
```

---

### `analyze_variants(variants, baseline_label, constraint_name)`

Embed prompt variants and compute pairwise cosine distances. Measures how much a reformulation shifts meaning in embedding space — predicts compliance divergence before running expensive model evals.

**Requires:** `semantic-chunker` + embedding model.

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `variants` | `dict[str, str]` | *required* | Label → prompt text |
| `baseline_label` | `str` | `"imperative"` | Which variant is the baseline |
| `constraint_name` | `str` | `"unnamed"` | Label for the constraint set |

**Returns:**
```json
{
  "constraint_name": "deletion_request",
  "baseline_label": "imperative",
  "variants_count": 3,
  "from_baseline": {"polite": 0.084, "passive": 0.114},
  "pairwise": {
    "(imperative, polite)": 0.084,
    "(imperative, passive)": 0.114,
    "(polite, passive)": 0.092
  },
  "recommendations": [
    {"variant": "passive", "distance": 0.114, "text": "The file should be deleted"}
  ]
}
```

**Errors:** `ImportError`, `RuntimeError` (same as `calculate_drift`).

---

### `generate_variants(baseline, model_id, dimensions, ...)`

Generate grammatical variants of a prompt constraint using an LLM. No embedding dependency — uses `complete()` only.

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `baseline` | `str` | *required* | Imperative constraint to rephrase |
| `model_id` | `str` | *required* | Model for generation |
| `dimensions` | `list[str]` | `["mood", "voice", "person", "frame"]` | Grammatical dimensions |
| `temperature` | `float` | `0.3` | Low for consistent rephrasing |
| `max_tokens` | `int` | `512` | |
| `timeout_seconds` | `int` | `60` | |

Available dimensions: `mood` (imperative/interrogative/declarative), `voice` (active/passive), `person` (first/second/third), `tense` (present/past/future/perfect), `frame` (presuppositional/descriptive).

**Returns:**
```json
{
  "baseline": "File a bug before writing code",
  "dimensions_requested": ["mood", "voice", "person", "frame"],
  "variants": {
    "imperative": "File a bug before writing code",
    "interrogative": "Could you file a bug before writing code?",
    "passive": "A bug should be filed before code is written",
    "first_person": "We file a bug before writing code",
    "presuppositional": "Since bugs are filed before coding begins..."
  },
  "variant_count": 5
}
```

**Errors:** `ValueError` if baseline is empty or LLM response unparseable. `RuntimeError` if adapter unavailable.

**Workflow — generate then analyze:**
```python
variants = await generate_variants("File a bug before writing code", model_id)
distances = await analyze_variants(variants["variants"], baseline_label="imperative")
# distances.from_baseline shows which rephrasings shifted meaning most
```

---

### `analyze_trajectory(text, acceleration_threshold, include_sentences)`

Analyze semantic velocity and acceleration profile of a text passage. Treats each sentence as a point in embedding space and computes kinematic quantities along the path.

**Requires:** `semantic-chunker` + embedding model + spaCy.

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `text` | `str` | *required* | Text passage (needs 2+ sentences) |
| `acceleration_threshold` | `float` | `0.3` | Threshold for flagging spikes |
| `include_sentences` | `bool` | `false` | Include sentence breakdown in output |

**Returns:**
```json
{
  "n_sentences": 5,
  "mean_velocity": 0.42,
  "mean_acceleration": 0.15,
  "max_acceleration": 0.68,
  "acceleration_spikes": [
    {"magnitude": 0.68, "isolation_score": 0.85, "position_ratio": 0.6}
  ],
  "deadpan_score": 0.65,
  "heller_score": 0.30,
  "circularity_score": 0.12,
  "tautology_density": 0.05,
  "deceleration_score": 0.22,
  "adams_interpretation": "Moderate deadpan structure — isolated semantic spike in stable background",
  "heller_interpretation": "Low circular reasoning"
}
```

| Score | Measures | High = |
|-------|----------|--------|
| `deadpan_score` | Isolated semantic spikes in stable background (Adams-style) | Strong deadpan |
| `heller_score` | Circular, decelerating semantic path (Heller-style) | Circular reasoning |
| `circularity_score` | How close the ending is to the beginning in embedding space | Text returns to start |
| `tautology_density` | Proportion of near-zero velocity segments | Repetitive/redundant |

**LAS use case:** Detect circular reasoning in specialist outputs — a model that keeps restating the same idea in different words will score high on `heller_score` and `tautology_density`.

---

### `compare_trajectories(golden_text, synthetic_text, acceleration_threshold)`

Compare trajectory profile of a synthetic (model-generated) text against a golden reference. Returns a fitness score based on DTW alignment and acceleration correlation.

**Requires:** `semantic-chunker` + embedding model + spaCy.

**Parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `golden_text` | `str` | *required* | Reference passage (target structure) |
| `synthetic_text` | `str` | *required* | Model-generated passage to evaluate |
| `acceleration_threshold` | `float` | `0.3` | |

**Returns:**
```json
{
  "fitness_score": 0.35,
  "synthetic_deadpan": 0.45,
  "synthetic_heller": 0.10,
  "acceleration_dtw": 0.20,
  "acceleration_correlation": 0.72,
  "spike_position_match": 0.80,
  "spike_count_match": 0.90,
  "interpretation": "Good structural match with some rhythm deviation",
  "golden_summary": {"n_sentences": 5, "deadpan_score": 0.65, "mean_velocity": 0.42},
  "synthetic_summary": {"n_sentences": 6, "deadpan_score": 0.45, "mean_velocity": 0.38}
}
```

`fitness_score` is 0.0-1.0, lower = better structural match.

---

## Tool Dependencies

```
MCP tools (registered in server.py):
  complete ─────────────────── adapter (LMStudio / Together / HuggingFace)
  list_models ─────────────── adapter
  judge ───────────────────── complete()
  generate_variants ───────── complete()
  react_step ──────────────── complete_stream() (internal, not an MCP tool)
  calculate_drift ─────────── semantic-chunker (embedding model)
  analyze_variants ────────── semantic-chunker (embedding model)
  analyze_trajectory ──────── semantic-chunker (embedding model + spaCy)
  compare_trajectories ────── semantic-chunker (embedding model + spaCy)
```

Tools in the left column work with any registered adapter. Tools in the right column additionally require `semantic-chunker` and a running embedding model. If `semantic-chunker` is unavailable, those tools raise `ImportError` — the remaining tools continue to function.

---

## Composition Patterns

### Battery evaluation (ADR-CORE-066, Phase 1)

```python
models = (await list_models())["models"]
for model in models:
    await complete(model, [{"role": "user", "content": "Respond with only 'pong'"}], max_tokens=8)  # warmup
    for test in tests:
        response = await complete(model, test.messages, temperature=0.0)
        verdict = await judge(response, test.pass_criteria, judge_model)
        drift = await calculate_drift(response, test.expected_response)
```

### Facilitator-driven ReAct eval (ADR-CORE-064, Mode 2)

```python
trace, counter = [], 0
while len(trace) < max_iterations:
    result = await react_step(model_id, system_prompt, goal, trace, mock_tools, tools, counter)
    if result["completed"]:
        break
    trace.extend(result["new_iterations"])
    counter = result["call_counter"]
    # Facilitator can apply context curation here before next step
```

### Prompt optimization

```python
variants = await generate_variants("File a bug before writing code", model_id)
distances = await analyze_variants(variants["variants"])
# Test the variant with highest distance — it's the one most likely to change model behavior
for label, text in variants["variants"].items():
    response = await complete(model_id, [{"role": "system", "content": text}, {"role": "user", "content": task}])
    verdict = await judge(response, criteria, judge_model)
```

### Circular reasoning detection

```python
trajectory = await analyze_trajectory(specialist_output)
if trajectory["heller_score"] > 0.5 or trajectory["tautology_density"] > 0.3:
    # Model is going in circles — flag for review
```
