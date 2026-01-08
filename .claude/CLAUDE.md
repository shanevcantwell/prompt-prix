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
├── dispatcher.py        # WorkStealingDispatcher (parallel execution)
├── config.py            # Pydantic models, constants, .env loading
├── parsers.py           # Input parsing utilities
├── export.py            # Markdown/JSON report generation
├── state.py             # Global mutable state
├── battery.py           # BatteryRunner, BatteryRun state
├── semantic_validator.py # Refusal detection, tool call validation
├── adapters/
│   ├── base.py          # LLMAdapter protocol
│   ├── lmstudio.py      # LMStudioAdapter (OpenAI-compatible)
│   ├── gemini_webui.py  # GeminiWebUIAdapter (DOM-based, deprecated)
│   ├── gemini_visual.py # GeminiVisualAdapter (Fara-based, preferred)
│   └── fara.py          # FaraService (visual element location)
├── tabs/
│   ├── battery/
│   │   ├── handlers.py  # Battery tab event handlers
│   │   └── ui.py        # Battery tab Gradio components
│   ├── compare/
│   │   ├── handlers.py  # Compare tab event handlers
│   │   └── ui.py        # Compare tab Gradio components
│   └── stability/
│       ├── handlers.py  # Stability tab event handlers
│       └── ui.py        # Stability tab Gradio components
└── benchmarks/
    ├── base.py          # TestCase model
    └── custom.py        # CustomJSONLoader (JSON/JSONL/BFCL)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | ServerConfig, ModelContext, SessionState, env loading |
| `core.py` | ServerPool management, streaming functions |
| `dispatcher.py` | WorkStealingDispatcher for parallel execution |
| `handlers.py` | Shared async handlers (fetch models, stop) |
| `ui.py` | Gradio app composition, imports tab UIs |
| `state.py` | Mutable state shared across handlers |
| `battery.py` | BatteryRunner orchestrator |
| `adapters/` | Provider abstractions (LM Studio, Gemini) |
| `tabs/` | Tab-specific handlers and UI components |
| `benchmarks/` | Test case loading (JSON, JSONL, BFCL) |

---

## Adapters

### LMStudioAdapter
Standard OpenAI-compatible adapter for local models via LM Studio.

### GeminiVisualAdapter (Preferred)
Uses Microsoft Fara-7B vision model to interact with Gemini's web UI visually.
- Survives UI redesigns (no brittle DOM selectors)
- Takes screenshots, uses Fara to locate elements
- Executes Playwright actions (click, type, scroll)

```python
adapter = GeminiVisualAdapter()  # Uses env vars for config
result = await adapter.send_prompt("Hello")
result = await adapter.regenerate()
await adapter.close()
```

### FaraService
Visual UI element location using Fara-7B vision model.
- Returns Playwright actions: `left_click`, `type`, `scroll`, etc.
- Handles resolution scaling transparently
- Configured via `FARA_SERVER_URL` and `FARA_MODEL_ID` env vars

```python
fara = FaraService()
result = await fara.locate("The send button", screenshot_b64)
# Returns: {"found": True, "action": "left_click", "x": 640, "y": 480}
```

### GeminiWebUIAdapter (Deprecated)
DOM-based Gemini adapter. Breaks when Gemini UI changes.
Use GeminiVisualAdapter instead.

---

## Tabs

### Battery Tab
Run benchmark test suites across multiple models.
- Load JSON/JSONL test files
- Model × Test grid view
- Parallel execution via WorkStealingDispatcher

### Compare Tab
Interactive side-by-side model comparison.
- Multi-turn conversations
- Per-model context management
- Halt on error capability

### Stability Tab
Analyze regeneration stability for a single model.
- Run same prompt N times
- Capture output variance
- Uses GeminiVisualAdapter for Gemini models

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

### WorkStealingDispatcher (`dispatcher.py`)
Reusable parallel execution strategy:
```python
dispatcher = WorkStealingDispatcher(pool)
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

# Fara Vision Model (for Gemini visual adapter)
FARA_SERVER_URL=http://127.0.0.1:1234
FARA_MODEL_ID=microsoft_fara-7b

# Optional
BEYOND_COMPARE_PATH=/usr/bin/bcompare
```

---

## Gemini Session Management

The Gemini adapters use Playwright browser state for session persistence.

```bash
# CLI for session management
prompt-prix-gemini --on      # Start session (login manually)
prompt-prix-gemini --off     # End session
prompt-prix-gemini --status  # Check session status

# Session stored at: ~/.prompt-prix/gemini_state/state.json
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
- Adapter implementations (LMStudio, Gemini, Fara)
- Visual UI verification (FaraService)
- ReAct tool loops
- End-to-end prompt execution

### Running Tests

```bash
# Run unit tests (default, skips integration)
pytest

# Run integration tests (the ones that matter for model code)
pytest -m integration

# Run specific integration test class
pytest tests/test_gemini_adapter.py::TestGeminiVisualAdapter -m integration -v

# Coverage (note: integration tests provide meaningful coverage)
pytest --cov=prompt_prix
```

### Test Markers
- `@pytest.mark.integration` - Requires external services (LM Studio, Gemini, etc.)
- Default pytest config skips integration tests for CI convenience

### Integration Test Prerequisites

Before running integration tests, ensure your local environment is ready:

1. **LM Studio running** with required models loaded:
   - Primary models for comparison tests
   - `microsoft_fara-7b` for visual adapter tests

2. **Gemini session active** (if testing Gemini adapters):
   ```bash
   prompt-prix-gemini --on   # Opens browser, login manually
   prompt-prix-gemini --status  # Verify session is active
   ```

3. **`.env` configured** with correct server URLs:
   ```bash
   LM_STUDIO_SERVER_1=http://localhost:1234
   FARA_SERVER_URL=http://localhost:1234
   FARA_MODEL_ID=microsoft_fara-7b
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

### Visual-First Automation
Prefer visual element location (Fara) over DOM selectors for web UI automation.

---

## Future Direction: MCP Service

The Fara adapter is evolving toward an MCP (Model Context Protocol) service architecture.

**Current state:** Proof-of-concept working in prompt-prix
**Target state:** Standalone MCP service callable from any client

```
prompt-prix (Gradio UI)
    │
    └── MCP Client ──► fara_service (MCP Service)
                            │
                            ├── Fara-7B (vision)
                            └── Playwright (browser)
```

See: `langgraph-agentic-scaffold/app/src/mcp/services/fara_service.py`

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
1. Create `prompt_prix/adapters/new_adapter.py`
2. Implement required interface (send_prompt, close)
3. Add configuration to `config.py` if needed
4. Wire into appropriate tab handlers
5. Add integration tests with `@pytest.mark.integration`
