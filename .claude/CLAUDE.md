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
├── main.py              # Entry point, Gradio launch
├── ui.py                # Gradio UI composition (imports tab UIs)
├── ui_helpers.py        # CSS, JS constants
├── handlers.py          # Shared async event handlers
├── core.py              # ServerPool, ComparisonSession, streaming
├── dispatcher.py        # ConcurrentDispatcher (parallel execution)
├── config.py            # Pydantic models, constants, .env loading
├── parsers.py           # Input parsing utilities
├── export.py            # Markdown/JSON report generation
├── state.py             # Global mutable state
├── battery.py           # BatteryRunner, BatteryRun state
├── semantic_validator.py # Refusal detection, tool call validation
├── adapters/
│   ├── base.py          # LLMAdapter protocol
│   ├── lmstudio.py      # LMStudioAdapter (OpenAI-compatible)
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

| Module | Purpose |
|--------|---------|
| `config.py` | ServerConfig, ModelContext, SessionState, env loading |
| `core.py` | ServerPool management, streaming functions |
| `dispatcher.py` | ConcurrentDispatcher for parallel execution |
| `handlers.py` | Shared async handlers (fetch models, stop) |
| `ui.py` | Gradio app composition, imports tab UIs |
| `state.py` | Mutable state shared across handlers |
| `battery.py` | BatteryRunner orchestrator |
| `adapters/` | Provider abstractions (LM Studio, surf-mcp, HF Inference) |
| `tabs/` | Tab-specific handlers and UI components |
| `benchmarks/` | Test case loading (JSON, JSONL, BFCL) |

---

## Adapters (Adapter Pattern)

The adapters layer uses the **Adapter design pattern** to provide a uniform interface to fundamentally different LLM backends.

### The Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP PRIMITIVES                               │
│  complete │ complete_stream │ judge │ fan_out                   │
│                                                                 │
│  Receives adapter via dependency injection.                     │
│  Calls adapter.stream_completion() - doesn't know backend.      │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTER LAYER                                │
│  HostAdapter protocol defines the interface.                    │
│  Each adapter owns its backend-specific internals:              │
│                                                                 │
│  LMStudioAdapter     - owns ServerPool (multi-server mgmt)      │
│  SurfMcpAdapter      - owns browser session                     │
│  HFInferenceAdapter  - owns API client                          │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PROVIDERS                          │
│  LM Studio │ surf-mcp │ HF Spaces │ cloud APIs                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Rules

1. **MCP doesn't instantiate adapters.** MCP receives adapters or calls adapter module functions.
2. **Each adapter encapsulates its backend internals.** ServerPool is an LM Studio concept - it belongs inside LMStudioAdapter, not leaked to MCP.
3. **Adapters expose a uniform interface.** All adapters implement `HostAdapter` protocol regardless of how different their backends are.

### Why This Matters

The backends are **fundamentally different**, not just GGUF variations:
- **LM Studio**: Multiple local servers, availability tracking, OpenAI-compatible API
- **surf-mcp**: Browser automation, no "servers" - automates web UIs
- **HF Inference**: Cloud API, authentication, rate limiting

If MCP knew about `ServerPool`, it could never work with surf-mcp. The Adapter pattern isolates backend-specific concepts.

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

### ServerPool
Manages multiple LM Studio servers:
```python
servers: dict[str, ServerConfig]  # URL → config
find_available_server(model_id)   # Find idle server with model
acquire_server(url)               # Mark busy
release_server(url)               # Mark available
```

### ComparisonSession
Maintains comparison state:
- Selected models
- Separate conversation context per model
- Configuration (temperature, max tokens, system prompt)
- Halt state

### ConcurrentDispatcher (`dispatcher.py`)
Reusable parallel execution strategy:
```python
dispatcher = ConcurrentDispatcher(pool)
async for completed in dispatcher.dispatch(work_items, execute_fn):
    yield state  # UI update opportunity
```

### BatteryRunner (Battery Mode)
Orchestrates benchmark execution:
```python
runner = BatteryRunner(adapter, tests, models, temperature, max_tokens, timeout)
async for state in runner.run():
    yield state.to_grid()  # Model × Test grid updates
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

Follow the Adapter pattern (see "Adapters" section above):

1. **Create adapter file:** `prompt_prix/adapters/new_adapter.py`
2. **Implement `HostAdapter` protocol:** `get_available_models()`, `stream_completion()`
3. **Encapsulate backend internals:** Connection pools, sessions, clients belong INSIDE the adapter
4. **Expose module-level function:** `stream_completion()` for MCP to call without instantiation
5. **Export from `__init__.py`:** Add to `prompt_prix/adapters/__init__.py`
6. **Add integration tests:** Mark with `@pytest.mark.integration`

**Critical:** No adapter instantiation outside `prompt_prix/adapters/`. MCP calls adapter module functions, not classes.
