# prompt-prix

**Purpose:** Visual fan-out UI for running evaluation prompts across multiple LLMs simultaneously and comparing results side-by-side.

---

## Communication Style

- Calm and level, professional tone
- Avoid premature confidence: "Ready for production!" before testing is overconfident
- Follow the user's pace: "4 tests passed, let's run a full batch!" is unnecessarily urgent

---

## Working Model: Co-Architects

This is a pair programming relationship, not a code generation service. The value is in the conversation that precedes implementation.

**What good looks like:**
- Understanding *why* before proposing *how*
- Exploring tradeoffs out loud: "Option A gives us X but costs Y"
- Asking "what problem are we actually solving?" when requirements seem underspecified
- Reading existing code to understand patterns before writing new code
- Treating the codebase as a long-term asset we're stewarding together

**The goal is working software that remains maintainable** - not code that appears to work, not impressive-looking output, not maximum tokens of plausible implementation. Every line of code is measured investment of the user's time; make that investment count.

When in doubt: discuss the approach first. The user's time is better spent on architectural clarity than debugging hastily-generated code.

---

## Core Concept: Fan-Out Pattern

prompt-prix is NOT an eval framework. It's a visual comparison layer.

```
Input:  One prompt (or benchmark test case)
        ↓
Fan-Out: Dispatch to N models in parallel
        ↓
Output: Side-by-side visual comparison
```

**Positioning:**
| Tool | Purpose |
|------|---------|
| BFCL | Function-calling benchmark |
| Inspect AI | Evaluation framework |
| prompt-prix | Visual fan-out comparison |

---

## Architecture

```
prompt_prix/
├── main.py              # Entry point, Gradio launch, adapter registration
├── ui.py                # Gradio UI composition (imports tab UIs)
├── ui_helpers.py        # CSS, JS constants
├── handlers.py          # Shared async event handlers
├── core.py              # ComparisonSession (orchestration)
├── battery.py           # BatteryRunner (orchestration) - calls MCP tools
├── config.py            # Pydantic models, constants, .env loading
├── parsers.py           # Input parsing utilities
├── export.py            # Markdown/JSON report generation
├── state.py             # Global mutable state
├── semantic_validator.py # Refusal detection, tool call validation
├── mcp/
│   ├── registry.py      # Adapter registry (get_adapter, register_adapter)
│   └── tools/
│       ├── complete.py  # complete, complete_stream primitives
│       ├── fan_out.py   # fan_out primitive
│       └── list_models.py # list_models primitive
├── adapters/
│   ├── base.py          # HostAdapter protocol
│   ├── lmstudio.py      # LMStudioAdapter (OWNS ServerPool, ConcurrentDispatcher)
│   ├── surf_mcp.py      # SurfMcpAdapter (browser automation, TODO)
│   └── hf_inference.py  # HFInferenceAdapter (HuggingFace Spaces, TODO)
├── tabs/
│   ├── battery/
│   │   ├── handlers.py  # Battery tab event handlers
│   │   └── ui.py        # Battery tab Gradio components
│   └── compare/
│       ├── handlers.py  # Compare tab event handlers
│       └── ui.py        # Compare tab Gradio components
└── benchmarks/
    ├── base.py          # TestCase model
    └── custom.py        # CustomJSONLoader (JSON/JSONL/BFCL)
```

### Module Responsibilities

| Module | Layer | Purpose |
|--------|-------|---------|
| `battery.py` | Orchestration | BatteryRunner - calls MCP tools |
| `core.py` | Orchestration | ComparisonSession |
| `mcp/tools/` | MCP | Primitives (complete, fan_out) |
| `mcp/registry.py` | MCP | Adapter registry (get_adapter, register_adapter) |
| `adapters/lmstudio.py` | Adapter | LMStudioAdapter (owns ServerPool, ConcurrentDispatcher) |
| `adapters/base.py` | Adapter | HostAdapter protocol |
| `handlers.py` | UI | Shared async handlers (fetch models, stop) |
| `ui.py` | UI | Gradio app composition |
| `tabs/` | UI | Tab-specific handlers and UI components |
| `config.py` | Shared | Pydantic models, constants, env loading |
| `benchmarks/` | Shared | Test case loading (JSON, JSONL, BFCL) |

---

## Architecture Layers

prompt-prix has three distinct layers with strict import boundaries. **Each layer only talks to the one directly beneath it.**

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION                            │
│  BatteryRunner │ ComparisonSession                              │
│                                                                 │
│  • Defines WHAT to run (test matrix, prompt sequences)          │
│  • Controls concurrency via semaphore (max N concurrent)        │
│  • Calls MCP primitives ONLY — never adapters directly          │
│  • IMPORTS: mcp.tools.complete, mcp.tools.fan_out               │
│  • NEVER IMPORTS: adapters/*, ServerPool, ConcurrentDispatcher  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ MCP tool call
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MCP PRIMITIVES                            │
│  complete │ complete_stream │ fan_out                           │
│                                                                 │
│  • The Universal Contract / Tool Registry                       │
│  • Stateless pass-through                                       │
│  • Receives adapter via registry (get_adapter())                │
│  • IMPORTS: adapters.base.HostAdapter (protocol only)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ adapter.stream_completion()
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                       ADAPTER LAYER                             │
│                                                                 │
│  Each adapter is a BLACK BOX exposing HostAdapter protocol.     │
│  Internal implementation details are ENCAPSULATED.              │
│                                                                 │
│  LMStudioAdapter                                                │
│    INTERNAL: ServerPool, ConcurrentDispatcher, httpx            │
│    STRATEGY: Multi-GPU parallel dispatch                        │
│                                                                 │
│  SurfMcpAdapter                                                 │
│    INTERNAL: browser session                                    │
│    STRATEGY: Sequential (one browser)                           │
│                                                                 │
│  HFInferenceAdapter                                             │
│    INTERNAL: API client, rate limiter                           │
│    STRATEGY: Rate-limited cloud calls                           │
└─────────────────────────────────────────────────────────────────┘
```

> **THE RULE:** ServerPool and ConcurrentDispatcher are INTERNAL to LMStudioAdapter.
> No file outside `adapters/lmstudio.py` may import or reference them.

### Layer Import Rules

| Layer | MAY Import | MUST NOT Import |
|-------|------------|-----------------|
| **Orchestration** (BatteryRunner, ComparisonSession) | `mcp.tools.*`, `mcp.registry` | `adapters/*`, ServerPool, ConcurrentDispatcher |
| **MCP Primitives** | `adapters.base.HostAdapter` (protocol), `mcp.registry` | Concrete adapter classes, ServerPool |
| **Adapters** | httpx, internal utilities | Nothing from orchestration or MCP |

### Why This Matters

**Orchestration is "Capability-Aware", not "Provider-Aware".**

- `BatteryRunner` says: "I have this test case; MCP, please execute it"
- It doesn't know if there are 2 GPUs or 10, local or cloud
- Swapping `LMStudioAdapter` for `SurfMcpAdapter` requires zero changes to orchestration

**Each adapter owns its parallelism strategy:**

| Adapter | Internal Strategy |
|---------|-------------------|
| `LMStudioAdapter` | ConcurrentDispatcher + ServerPool (multi-GPU) |
| `SurfMcpAdapter` | Sequential (one browser) |
| `HFInferenceAdapter` | Rate limiter |

### HostAdapter Protocol

```python
class HostAdapter(Protocol):
    async def get_available_models(self) -> list[str]: ...
    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tools: Optional[list[dict]] = None
    ) -> AsyncGenerator[str, None]: ...
```

### MCP Registry

Adapters are registered at startup and retrieved via registry:

```python
# At startup (main.py)
from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.mcp.registry import register_adapter

adapter = LMStudioAdapter(server_urls=load_servers_from_env())
register_adapter(adapter)

# In MCP tools
from prompt_prix.mcp.registry import get_adapter

async def complete_stream(model_id: str, messages: list[dict], ...):
    adapter = get_adapter()
    async for chunk in adapter.stream_completion(model_id, messages, ...):
        yield chunk

# In BatteryRunner (calls MCP, not adapter)
from prompt_prix.mcp.tools.complete import complete_stream

async for chunk in complete_stream(model_id=item.model_id, ...):
    response += chunk
```

### Current Adapters

| Adapter | Backend | Status |
|---------|---------|--------|
| `LMStudioAdapter` | Local GGUF via OpenAI-compat API | ✓ |
| `SurfMcpAdapter` | Browser automation via surf-mcp | TODO |
| `HFInferenceAdapter` | HuggingFace Spaces / Inference API | TODO |

---

## Tabs

### Battery Tab
Run benchmark test suites across multiple models.
- Load JSON/JSONL test files
- Model × Test grid view
- Parallel execution via ConcurrentDispatcher

### Compare Tab
Interactive side-by-side model comparison.
- Multi-turn conversations
- Per-model context management
- Halt on error capability

---

## Battery File Formats

The Battery tab loads test cases from JSON or JSONL files. Three formats are supported.

### TestCase Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | **Yes** | - | Unique test identifier |
| `user` | string | **Yes** | - | User message content |
| `name` | string | No | `""` | Human-readable display name |
| `category` | string | No | `""` | Test category for grouping |
| `severity` | string | No | `"warning"` | `"critical"` or `"warning"` |
| `system` | string | No | `"You are a helpful assistant."` | System prompt |
| `tools` | array | No | `null` | OpenAI-format tool definitions |
| `tool_choice` | string | No | `null` | `"required"`, `"auto"`, or `"none"` |
| `expected` | object | No | `null` | Expected output (for grading) |
| `pass_criteria` | string | No | `null` | Human description of pass condition |
| `fail_criteria` | string | No | `null` | Human description of fail condition |

### Format 1: JSON (Recommended)

Wrapper object with `prompts` array:

```json
{
  "test_suite": "my_tests_v1",
  "version": "1.0",
  "prompts": [
    {
      "id": "test_basic",
      "name": "Basic Test",
      "category": "sanity",
      "severity": "critical",
      "system": "You are a helpful assistant.",
      "user": "What is 2 + 2?",
      "pass_criteria": "Returns 4"
    },
    {
      "id": "test_tool_call",
      "name": "Tool Call Test",
      "category": "tools",
      "system": "Use the weather tool to answer.",
      "user": "What's the weather in Tokyo?",
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "parameters": {
              "type": "object",
              "properties": {
                "city": {"type": "string"}
              },
              "required": ["city"]
            }
          }
        }
      ],
      "tool_choice": "required"
    }
  ]
}
```

### Format 2: JSONL

One test case per line (auto-detected by `.jsonl` extension or multiple `{` lines):

```jsonl
{"id": "test_1", "user": "What is 2 + 2?", "category": "math"}
{"id": "test_2", "user": "What is 3 + 3?", "category": "math"}
```

### Format 3: BFCL (Berkeley Function Calling Leaderboard)

BFCL format is auto-normalized. Field mappings:

| BFCL Field | Maps To |
|------------|---------|
| `question` (array of messages) | `system` + `user` (extracted) |
| `function` | `tools` |
| `metadata.category` | `category` |
| `metadata.severity` | `severity` |

BFCL example:
```json
{"id": "bfcl_test", "question": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Delete report.pdf"}], "function": [{"name": "delete_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}], "metadata": {"category": "file_ops"}}
```

### Validation

Files are validated on load with fail-fast behavior:
- `id` and `user` fields are required and cannot be empty
- `prompts` array must exist and be non-empty (JSON format)
- Invalid JSON or missing fields raise `ValueError` with line/index info

Example files: `examples/tool_competence_tests.json`, `data/tests/*.jsonl`

---

## Semantic Validation

Battery tests validate responses beyond HTTP success. A model that returns "I'm sorry, but I can't execute scripts" completed the HTTP transaction but semantically failed the task.

### How It Works

After receiving a response, the validator checks:

1. **Refusal Detection** - Matches common refusal phrases
2. **Tool Call Validation** - For tests with `tool_choice: "required"`, verifies tool calls exist

### Test Status Values

| Status | Symbol | Meaning |
|--------|--------|---------|
| `COMPLETED` | ✓ | Response passed semantic validation |
| `SEMANTIC_FAILURE` | ⚠ | Response received but failed semantic check |
| `ERROR` | ❌ | Infrastructure error (timeout, connection, etc.) |

### Refusal Patterns

Defined in `prompt_prix/semantic_validator.py`:

```python
REFUSAL_PATTERNS = [
    r"i(?:'m| am) sorry,? but",
    r"i can(?:'t|not)",
    r"i(?:'m| am) (?:not )?(?:able|unable)",
    r"(?:cannot|can't) (?:execute|run|perform|help with)",
    r"i(?:'m| am) not (?:designed|programmed|able)",
    r"(?:as an ai|as a language model)",
    r"i don't have (?:the ability|access)",
]
```

### Modifying Refusal Patterns

To add new refusal patterns:

1. Edit `REFUSAL_PATTERNS` in `prompt_prix/semantic_validator.py`
2. Add corresponding test case in `tests/test_semantic_validator.py`
3. Run `pytest tests/test_semantic_validator.py -v` to verify

Example - adding a new pattern:
```python
# In semantic_validator.py
REFUSAL_PATTERNS = [
    # ... existing patterns ...
    r"(?:that's|this is) (?:not|beyond) (?:something|what) i",  # NEW
]
```

### Tool Call Validation

For tests with tools defined:

| `tool_choice` | Expected Behavior |
|---------------|-------------------|
| `"required"` | Response MUST contain tool calls → fails if text-only |
| `"none"` | Response must NOT contain tool calls → fails if tools called |
| `"auto"` or unset | Either is valid |

Tool calls are detected by the `**Tool Call:**` marker in formatted responses.

### Extending Validation

The `validate_response_semantic()` function in `semantic_validator.py` returns `(is_valid, failure_reason)`. To add new validation types:

```python
def validate_response_semantic(test: TestCase, response: str) -> Tuple[bool, Optional[str]]:
    # Existing checks run first...

    # Add new validation here:
    if some_condition(test, response):
        return False, "Description of why validation failed"

    return True, None
```

### Example: Detecting Semantic Failure

Test case:
```json
{
  "id": "delete_file",
  "user": "Delete report.pdf",
  "tools": [{"type": "function", "function": {"name": "delete_file", ...}}],
  "tool_choice": "required"
}
```

Model response:
> "I'm sorry, but I can't execute or run scripts. The available API only allows routing tasks to specialists..."

Result: `⚠ Semantic Failure` - "Model refused: 'i'm sorry, but'"

---

## Key Components

### BatteryRunner (Orchestration)
Orchestrates benchmark execution. Calls MCP primitives, never adapters directly.
```python
# BatteryRunner calls MCP tools - doesn't know about servers or adapters
from prompt_prix.mcp.tools.complete import complete_stream

runner = BatteryRunner(tests, models, temperature, max_tokens, timeout)
async for state in runner.run():
    yield state.to_grid()  # Model × Test grid updates
```

### ComparisonSession (Orchestration)
Maintains comparison state:
- Selected models
- Separate conversation context per model
- Configuration (temperature, max tokens, system prompt)
- Halt state

### MCP Registry
Central adapter registration. Tools call `get_adapter()` to retrieve the configured adapter.
```python
from prompt_prix.mcp.registry import get_adapter, register_adapter
```

### LMStudioAdapter (Adapter Layer)
Black box for LM Studio inference. **Internally** manages ServerPool and ConcurrentDispatcher - these are not visible to other layers.
```python
# Orchestration and MCP never see this - it's internal to the adapter
adapter = LMStudioAdapter(server_urls=["http://localhost:1234"])
```

---

## Environment Configuration

```bash
# .env file

# LM Studio Servers (numbered pattern)
LM_STUDIO_SERVER_1=http://192.168.1.10:1234
LM_STUDIO_SERVER_2=http://192.168.1.11:1234

# Gradio UI
GRADIO_PORT=7860

# Optional
BEYOND_COMPARE_PATH=/usr/bin/bcompare
```

---

## Testing

### Philosophy: Integration Tests for Model-Dependent Code

For functionality that depends on LLM behavior (adapters, visual verification, tool use), **integration tests are the meaningful tests**. Unit tests with mocked LLM responses don't validate anything useful—they just test that your mock returns what you told it to return.

Use unit tests for:
- Data parsing (JSON/JSONL loaders)
- Configuration validation
- Pure utility functions
- State management logic

Use integration tests for:
- Adapter implementations (LMStudio, surf-mcp, HF Inference)
- ReAct tool loops
- End-to-end prompt execution

### Running Tests

```bash
# Run unit tests (default, skips integration)
pytest

# Run integration tests (the ones that matter for model code)
pytest -m integration

# Coverage (note: integration tests provide meaningful coverage)
pytest --cov=prompt_prix
```

### Test Markers
- `@pytest.mark.integration` - Requires external services (LM Studio, etc.)
- Default pytest config skips integration tests for CI convenience

### Integration Test Prerequisites

Before running integration tests, ensure your local environment is ready:

1. **LM Studio running** with required models loaded

2. **`.env` configured** with correct server URLs:
   ```bash
   LM_STUDIO_SERVER_1=http://localhost:1234
   ```

### Writing New Integration Tests

When adding model-dependent functionality:
1. Write the integration test first (TDD for real behavior)
2. Mark with `@pytest.mark.integration`
3. Document prerequisites in test docstring
4. Consider adding a "smoke test" that runs quickly for basic verification

### Unit Test Strategy (per ADR-006)

Unit tests mock at layer boundaries, not internal implementation:

| Test Layer | Mocks |
|------------|-------|
| MCP tool tests | Mock adapter interface via registry |
| Orchestration tests | Mock MCP tools |
| Adapter tests | Mock httpx (external boundary) |

Example fixture for MCP tool tests:
```python
from prompt_prix.mcp.registry import register_adapter, clear_adapter

@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.get_available_models = AsyncMock(return_value=["model-1"])
    async def mock_stream(*args, **kwargs):
        yield "response"
    adapter.stream_completion = mock_stream
    register_adapter(adapter)
    yield adapter
    clear_adapter()
```

**Future Work:** Live integration tests for adapters (LMStudioAdapter against real LM Studio, etc.) are not yet implemented. These should test actual HTTP interactions, not mocked responses.

### Adapting Code from Other Projects

When bringing code from other projects (e.g., langgraph-agentic-scaffold):

1. **Check dependencies first** - Does it import packages not in pyproject.toml?
2. **Adapt, don't copy** - Rewrite to use prompt-prix's infrastructure:
   - Use `httpx` not `langchain` for HTTP
   - Use `pydantic` models that match existing patterns
   - Use existing adapters (`LMStudioAdapter`, etc.) not foreign abstractions
3. **Tests must run** - If tests fail due to missing imports, the code isn't ready
4. **No orphan files** - Don't commit code that can't be imported or tested

Code that "looks right" but breaks `pytest` is worse than no code - it creates false confidence and cleanup debt.

---

## Design Principles

### Fail-Fast Validation
Validate servers and models before starting sessions.

### Explicit State Management
- Session state (Python): models, contexts, halted status
- UI state (localStorage): server URLs, selected models

### Separation of Concerns
- `tabs/*/ui.py`: Component definitions
- `tabs/*/handlers.py`: Event logic
- `core.py`: Business logic

---

## Git Operations Safety

### NEVER Use These Commands
```bash
git rm -rf .
git clean -fdx  # without explicit confirmation
```

### Critical Files to Preserve
- `*.code-workspace`
- `.vscode/`
- `.claude/`
- `pyproject.toml`
- `.env`

---

## Bug Fix Workflow (Post-Release)

For bugs discovered after release, follow this sequence:

1. **File the bug** - Create GitHub issue with description, error message, root cause analysis
   ```bash
   gh issue create --title "Bug title" --body "Description..."
   ```

2. **Write tests first** - Add tests that:
   - Demonstrate the current (bad) behavior
   - Define the expected (good) behavior
   - Cover edge cases and error handling

3. **Implement the fix** - Make minimal changes to fix the issue

4. **Run tests** - Verify all tests pass including new ones
   ```bash
   pytest -v --tb=short
   ```

5. **Commit with bug reference** - Use "Fix #N" to auto-close
   ```bash
   git commit -m "Fix #3: Brief description of fix"
   ```

6. **Close the bug** - Add hotfix note for release tracking
   ```bash
   gh issue close 3 --comment "Fixed in <commit>. Hotfix candidate for v0.1.1."
   ```

7. **Push** - Don't update the release tag yet; hotfixes batch into point releases

This workflow ensures:
- Issues are tracked and searchable
- Tests prevent regression
- Commits are traceable to issues
- Release planning has visibility into pending hotfixes

---

## Development Workflow

1. Edit code in `prompt_prix/`
2. Run `pytest` to verify
3. Launch with `prompt-prix` command
4. Test against LM Studio servers
5. Commit with descriptive messages

### Adding a New Adapter

Follow the Adapter pattern (see "Architecture Layers" section above):

1. **Create adapter file:** `prompt_prix/adapters/new_adapter.py`
2. **Implement `HostAdapter` protocol:** `get_available_models()`, `stream_completion()`
3. **Encapsulate backend internals:** Connection pools, sessions, rate limiters belong INSIDE the adapter
4. **Own your parallelism strategy:** If your backend has multiple resources, manage them internally
5. **Export from `__init__.py`:** Add to `prompt_prix/adapters/__init__.py`
6. **Add integration tests:** Mark with `@pytest.mark.integration`

**Critical Rules:**
- Orchestration (BatteryRunner) NEVER imports adapters - it calls MCP tools
- MCP tools get adapter via `get_adapter()` registry
- All backend complexity stays inside the adapter class
