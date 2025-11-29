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
├── config.py        # Pydantic models, constants, env loading
├── core.py          # ServerPool, ComparisonSession, streaming
├── handlers.py      # Gradio event handlers (async)
├── ui.py            # Gradio component definitions
├── parsers.py       # Text parsing utilities
├── export.py        # Markdown/JSON report generation
├── state.py         # Global mutable state
└── main.py          # Entry point
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | ServerConfig, ModelContext, SessionState (Pydantic) |
| `core.py` | ServerPool management, streaming functions |
| `handlers.py` | Async handlers bridging UI to core logic |
| `ui.py` | Gradio components and event bindings |
| `state.py` | Mutable state shared across handlers |

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

### Work-Stealing Dispatcher
Efficient multi-GPU utilization:
1. Queue all models to process
2. Find idle server that has queued model
3. Execute and stream response
4. Release server for next model

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
- BFCL: JSON with function schemas
- Inspect AI: Export prompts as JSON
- Custom: OpenAI-compatible message format

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
