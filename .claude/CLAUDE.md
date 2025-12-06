# prompt-prix

**Purpose:** Visual fan-out UI for running evaluation prompts across multiple LLMs simultaneously and comparing results side-by-side.

---

## Communication Style

- Calm and level, professional tone
- Avoid premature confidence: "Ready for production!" before testing is overconfident
- Follow the user's pace: "4 tests passed, let's run a full batch!" is unnecessarily urgent

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

```bash
# Run unit tests (default, skips integration)
pytest

# Run integration tests (require external services)
pytest -m integration

# Run specific test class
pytest tests/test_gemini_adapter.py::TestGeminiVisualAdapter -m integration -v

# Coverage
pytest --cov=prompt_prix
```

### Test Markers
- `@pytest.mark.integration` - Requires LM Studio, Gemini session, etc.
- Default pytest config skips integration tests

### Integration Test Prerequisites
1. LM Studio running with Fara-7B loaded
2. Gemini session active (`prompt-prix-gemini --on`)
3. `.env` configured with correct server URLs

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
