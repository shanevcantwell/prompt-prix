# prompt-prix

**MCP toolkit for multi-model testing and agentic self-improvement.**

9 stateless tools for completion, judging, semantic drift, ReAct execution, prompt geometry, and trajectory analysis. Agents call these tools to audition specialist LLMs, measure reliability across quantizations, and drive multi-step tool-use loops. Includes a Gradio UI for human visual comparison.

## UI Tabs

### Compare Tab

Interactive side-by-side comparison. Select 2+ models, type a prompt, and stream responses in parallel.

- Real-time streaming with per-model status indicators
- Multi-turn conversations with separate context per model
- Image attachment for vision model comparison
- Configurable temperature, max tokens, system prompt, seed, and repeat penalty
- Export conversations to Beyond Compare for diff analysis

### Battery Tab

Run test suites across models in a grid. Load a benchmark file, select models, and dispatch all combinations.

- **Grid view**: Model (columns) x Test (rows), with pass/fail/error symbols per cell
- **Semantic validation**: Detects empty responses, refusals, missing tool calls, verdict mismatches
- **LLM-as-judge**: Select a judge model to evaluate responses against `pass_criteria`
- **Embedding drift**: Set a drift threshold to compare responses against `expected_response` exemplars via cosine distance
- **ReAct loop evaluation**: Tests with `mode: "react"` execute multi-step tool-use loops with mock tools, cycle detection, and full trace recording
- **Consistency mode**: Run each test N times with different seeds — grid shows pass rates (e.g., `6/10`) and distinguishes infrastructure errors from semantic failures
- **Cell detail**: Click any grid cell for the full response, latency, drift score, and judge result
- **Export**: Markdown, JSON, or CSV reports

### Grid Symbols

| Symbol | Meaning |
|--------|---------|
| `✓` | Passed — response met all validation criteria |
| `❌` | Semantic failure — response received but didn't meet criteria |
| `⚠` | Infrastructure error — connection failure, timeout, etc. |
| `🟣 6/10` | Inconsistent — passed 6 of 10 consistency runs |
| `🟣 6/10 ⚠3` | Inconsistent with 3 infrastructure errors in the mix |

### Battery File Formats

Test files can be JSON, JSONL, or Promptfoo YAML:

```json
{
  "prompts": [
    {
      "id": "categorize_files",
      "user": "Organize these files by content",
      "system": "You are a file organizer.",
      "mode": "react",
      "tools": [{"type": "function", "function": {"name": "read_file", "description": "...", "parameters": {...}}}],
      "mock_tools": {"read_file": {"./1.txt": "Content about animals"}, "move_file": {"_default": "File moved"}},
      "max_iterations": 20,
      "expected_response": "Files organized into category folders",
      "pass_criteria": "Model must call move_file for each input file"
    }
  ]
}
```

Key fields: `id` and `user` are required. Optional: `system`, `messages` (pre-defined multi-turn history), `tools`, `tool_choice`, `response_format` (structured output schema), `mode` (`"react"` for multi-step), `mock_tools`, `max_iterations`, `expected_response` (drift exemplar), `pass_criteria` / `fail_criteria` (judge input).

---

## MCP Tools

prompt-prix exposes its capabilities as stateless MCP primitives. The UI uses these same tools internally — agents and scripts can call them directly.

### `list_models()`

Discover available models across all configured servers.

```python
result = await list_models()
# {"models": ["qwen2.5-7b", "llama-3.1-8b", ...],
#  "servers": {"http://localhost:1234": ["qwen2.5-7b", ...]},
#  "unreachable": []}
```

### `complete(model_id, messages, ...)`

Single completion. The adapter handles server selection internally.

```python
response = await complete(
    model_id="qwen2.5-7b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.7,
    max_tokens=2048,
)
# "The answer is 4."
```

> Streaming is available at the code level via `complete_stream()`, but it is not registered as an MCP tool. Agents use `complete()` for single completions.

### `react_step(model_id, system_prompt, initial_message, trace, mock_tools, tools, ...)`

Execute one ReAct iteration. Stateless: takes the trace in, returns one step out. The caller manages the loop.

**Mock dispatch mode** (`mock_tools={...}`): tool calls are dispatched against mock responses and returned as `new_iterations`.

```python
result = await react_step(
    model_id="qwen2.5-7b",
    system_prompt="You are a file organizer.",
    initial_message="Organize the files in ./test/",
    trace=[],  # previous ReActIteration objects
    mock_tools={"read_file": {"./1.txt": "Animals content"}},
    tools=[{"type": "function", "function": {"name": "read_file", ...}}],
)
# {"completed": False, "new_iterations": [ReActIteration(...)],
#  "call_counter": 1, "latency_ms": 50.0}
```

**Tool-forwarding mode** (`mock_tools=None`): tool calls are parsed but not dispatched — returned as `pending_tool_calls` for the caller to execute against real services.

```python
result = await react_step(
    model_id="qwen2.5-7b",
    system_prompt="You are a text analyst.",
    initial_message="Analyze drift between these passages",
    trace=[],
    mock_tools=None,  # signals tool-forwarding mode
    tools=[{"type": "function", "function": {"name": "calculate_drift", ...}}],
)
# {"completed": False, "pending_tool_calls": [{"id": "call_1", "name": "calculate_drift", "args": {...}}],
#  "thought": "I need to calculate drift.", "call_counter": 1, "latency_ms": 80.0}
```

When `completed` is `True`, the model has finished and `final_response` contains its text answer. In mock dispatch mode, `new_iterations` contains tool calls and their mock observations — feed them back into `trace`. In tool-forwarding mode, `pending_tool_calls` contains parsed calls for the caller to dispatch and build trace entries from.

### `judge(response, criteria, judge_model, ...)`

LLM-as-judge evaluation. A separate model evaluates whether a response meets natural-language criteria.

```python
result = await judge(
    response="I'll help you delete report.pdf",
    criteria="Response must indicate intent to delete the file",
    judge_model="qwen2.5-7b",
)
# {"pass": True, "reason": "Response clearly states intent to delete", "score": 8}
```

### `calculate_drift(text_a, text_b)`

Cosine distance between two texts via embedding. Requires a running embedding model.

```python
distance = await calculate_drift(
    text_a="Files organized into animals/ and vehicles/ folders",
    text_b="All files categorized by content type into directories",
)
# 0.15  (0.0 = identical, 1.0 = orthogonal)
```

### `analyze_variants(variants, baseline_label, constraint_name)`

Measure embedding distances between prompt variants to understand how reformulations shift meaning.

```python
result = await analyze_variants(
    variants={
        "imperative": "Delete the file report.pdf",
        "polite": "Could you please delete report.pdf?",
        "passive": "The file report.pdf should be deleted",
    },
    baseline_label="imperative",
    constraint_name="deletion_request",
)
# {"from_baseline": {"polite": 0.12, "passive": 0.18},
#  "pairwise": {"(imperative, polite)": 0.12, ...},
#  "recommendations": [...]}
```

### `generate_variants(baseline, model_id, dimensions, temperature)`

Generate prompt variants along semantic dimensions (mood, voice, person, frame) using a specified model.

```python
result = await generate_variants(
    baseline="Delete the file report.pdf",
    model_id="qwen2.5-7b",
    dimensions=["mood", "voice"],
    temperature=0.3,
)
# {"baseline": "Delete the file report.pdf",
#  "variants": {"mood": "Please cheerfully remove report.pdf!", "voice": "report.pdf is to be deleted"},
#  "variant_count": 2}
```

### `analyze_trajectory(text, acceleration_threshold, include_sentences)`

Analyze the semantic velocity and acceleration profile of a text passage. Detects deadpan structure (Adams-style) and circular reasoning (Heller-style).

```python
result = await analyze_trajectory(
    text="Ford suffered through the Vogon poetry. Arthur calmly praised the imagery. The Vogon ejected them into space.",
)
# {"n_sentences": 3, "deadpan_score": 0.65, "heller_score": 0.30,
#  "mean_velocity": 0.42, "acceleration_spikes": [...],
#  "adams_interpretation": "Moderate deadpan structure", ...}
```

### `compare_trajectories(golden_text, synthetic_text, acceleration_threshold)`

Compare the trajectory profile of a synthetic (model-generated) text against a golden reference. Returns a fitness score indicating structural similarity.

```python
result = await compare_trajectories(
    golden_text="Ford suffered. Arthur praised. The Vogon ejected.",
    synthetic_text="The cat sat. The dog barked. Thunder struck.",
)
# {"fitness_score": 0.35, "acceleration_dtw": 0.20,
#  "spike_position_match": 0.80, "interpretation": "Good structure, some rhythm deviation",
#  "golden_summary": {...}, "synthetic_summary": {...}}
```

### Agent Integration Example

An agent can compose these primitives into custom evaluation workflows:

```python
# 1. Discover what's available
models = await list_models()

# 2. Fan out: same prompt to every model
responses = {}
for model in models["models"]:
    responses[model] = await complete(
        model_id=model,
        messages=[{"role": "user", "content": test_prompt}],
    )

# 3. Judge each response
for model, response in responses.items():
    verdict = await judge(
        response=response,
        criteria="Must call the delete_file tool with correct path",
        judge_model="qwen2.5-7b",
    )
    print(f"{model}: {'PASS' if verdict['pass'] else 'FAIL'} — {verdict['reason']}")

# 4. Measure drift from expected exemplar
for model, response in responses.items():
    drift = await calculate_drift(response, expected_exemplar)
    print(f"{model}: drift={drift:.3f}")
```

---

## MCP Server

prompt-prix runs as an MCP protocol server over stdio, exposing the same 9 tools listed above via JSON-RPC. Any MCP client (LAS, Claude Desktop, custom agent) can launch it as a subprocess.

### Running

```bash
prompt-prix-mcp
```

### Client Configuration

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

**Generic MCP client** (Python):

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server = StdioServerParameters(command="prompt-prix-mcp")
async with stdio_client(server) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("list_models", {})
```

The server reads `LM_STUDIO_SERVER_*` and `HF_TOKEN` from environment (or `.env`) at startup — the same configuration as the Gradio UI.

---

## Quick Start

### Prerequisites

- Python 3.10+
- LM Studio running on one or more servers with models loaded

### Setup

```bash
git clone https://github.com/shanevcantwell/prompt-prix.git
cd prompt-prix
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Start the app
prompt-prix
```

### Environment Configuration

```bash
# .env
LM_STUDIO_SERVER_1=http://localhost:1234
LM_STUDIO_SERVER_2=http://192.168.137.2:1234
TOGETHER_API_KEY=...          # Together AI cloud adapter
HF_TOKEN=...                  # HuggingFace Inference API
GRADIO_PORT=7860
```

Multiple adapters can be active simultaneously (LM Studio, Together AI, HuggingFace). The registry auto-detects which are configured and creates a CompositeAdapter that routes by model_id. For local servers, [local-inference-pool](https://github.com/shanevcantwell/local-inference-pool) manages server selection, model routing, and the model-drain guard across GPUs.

---

## Further Reading

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, layer model, and data flow |
| [MCP_TOOLS.md](MCP_TOOLS.md) | MCP tool API reference (9 tools, schemas, timeouts) |
| [adr/](adr/) | Architecture Decision Records |
