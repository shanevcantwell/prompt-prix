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
├── main.py          # Entry point, Gradio launch
├── ui.py            # Gradio UI components and event bindings
├── ui_helpers.py    # CSS, JS constants
├── handlers.py      # Async event handlers (bridging UI to core)
├── core.py          # ServerPool, ComparisonSession, streaming
├── dispatcher.py    # WorkStealingDispatcher (parallel execution)
├── config.py        # Pydantic models, constants, .env loading
├── parsers.py       # Input parsing utilities
├── export.py        # Markdown/JSON report generation
├── state.py         # Global mutable state
├── battery.py       # BatteryRunner, BatteryRun state
├── adapters/
│   ├── base.py      # LLMAdapter protocol
│   └── lmstudio.py  # LMStudioAdapter implementation
└── benchmarks/
    ├── base.py      # TestCase model
    └── custom.py    # CustomJSONLoader (JSON/JSONL/BFCL)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | ServerConfig, ModelContext, SessionState (Pydantic) |
| `core.py` | ServerPool management, streaming functions |
| `dispatcher.py` | WorkStealingDispatcher for parallel execution |
| `handlers.py` | Async handlers bridging UI to core logic |
| `ui.py` | Gradio components and event bindings |
| `state.py` | Mutable state shared across handlers |
| `battery.py` | BatteryRunner orchestrator, BatteryRun/TestResult state |
| `adapters/` | Provider abstraction (LLMAdapter protocol) |
| `benchmarks/` | Test case loading (JSON, JSONL, BFCL formats) |

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
Work-stealing algorithm:
1. Queue all work items (must have `model_id` property)
2. For each idle server, find work it can run
3. Spawn async task for matched work
4. Yield periodically for UI updates

### BatteryRunner (Battery Mode)
Orchestrates benchmark execution using WorkStealingDispatcher:
```python
runner = BatteryRunner(adapter, tests, models, temperature, max_tokens, timeout)
async for state in runner.run():
    yield state.to_grid()  # Model × Test grid updates in parallel
```

### CustomJSONLoader
Loads test cases from multiple formats:
- JSON with `prompts` array
- JSONL (one test per line, auto-detected)
- BFCL format (auto-normalized: `question[]` → `user`, `function` → `tools`)

---

## Design Principles

### Fail-Fast Validation
Validate servers and models before starting sessions:
```python
# In initialize_session
if not servers_configured:
    return "Error: No servers configured"
if not all_models_available:
    return "Error: Model X not found on any server"
```

### Explicit State Management
- Session state (Python): models, contexts, halted status
- UI state (localStorage): server URLs, selected models, settings

### Separation of Concerns
- `ui.py`: Component definitions only
- `handlers.py`: Event logic only
- `core.py`: Business logic only

### Progressive Error Handling
- Human-readable errors
- Halt on model failures
- Subsequent prompts rejected if halted

---

## Data Flow: Sending a Prompt

```
1. User clicks "Send Prompt"
        ↓
2. handlers.send_single_prompt()
   - Validate session
   - Parse tools JSON
   - Add user message to all contexts
        ↓
3. Work-Stealing Loop
   - Find idle server with queued model
   - Start async task
   - Yield UI updates every 100ms
        ↓
4. Each task: stream_completion()
   - Mark model "streaming"
   - Accumulate chunks
   - On complete: add to context, release server
        ↓
5. Final yield: "✅ All responses complete"
```

---

## Tab Status Colors

| Status | Color | Meaning |
|--------|-------|---------|
| pending | Red | Waiting to start |
| streaming | Yellow | In progress |
| completed | Green | Done |

---

## Environment Configuration

```bash
# .env file
LM_STUDIO_SERVER_1=http://192.168.1.10:1234
LM_STUDIO_SERVER_2=http://192.168.1.11:1234
GRADIO_PORT=7860
BEYOND_COMPARE_PATH=/usr/bin/bcompare  # Optional
```

---

## Integration Points

### Upstream: Benchmark Sources
- **BFCL**: JSONL with `question[]` and `function[]` fields (auto-normalized)
- **Inspect AI**: Export prompts as JSON
- **Custom JSON**: `{"prompts": [{id, user, system?, tools?}]}`
- **Custom JSONL**: One test per line `{id, user, system?, tools?}`

### API Layer: OpenAI-Compatible
All servers must expose:
```
GET  /v1/models
POST /v1/chat/completions
```

Supported: LM Studio, Ollama (OpenAI mode), vLLM, llama.cpp server

---

## Git Operations Safety

### NEVER Use These Commands
```bash
git rm -rf .
git clean -fdx  # without explicit confirmation
```

### SAFE Alternatives
```bash
git checkout HEAD -- .
git stash
git reflog
```

### Critical Files to Preserve
- `*.code-workspace`
- `.vscode/`
- `.claude/`
- `pyproject.toml`

---

## Testing

```bash
# Run tests
pytest

# Run with async support
pytest -v tests/

# Coverage
pytest --cov=prompt_prix
```

### Test Patterns
- Mock HTTP responses for server tests
- Use `pytest-asyncio` for async handlers
- Test Pydantic validation at boundaries

---

## Development Workflow

1. Edit code in `prompt_prix/`
2. Run `pytest` to verify
3. Launch with `prompt-prix` command
4. Test against LM Studio servers
5. Commit with descriptive messages

### Adding a New Feature
1. Check if existing module handles it
2. Update Pydantic models in `config.py` if new state needed
3. Add core logic to appropriate module
4. Wire UI in `ui.py` and handlers in `handlers.py`
5. Add tests
