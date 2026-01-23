# Architecture

This document describes the system architecture of prompt-prix, including module responsibilities, data flow, and key design decisions.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Browser                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Gradio UI (ui.py)                        │   │
│  │  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐ │   │
│  │  │ Config Panel│ │ Prompt Input │ │ Model Output Tabs     │ │   │
│  │  │ • Servers   │ │ • Single     │ │ • Tab 1..10           │ │   │
│  │  │ • Models    │ │ • Batch      │ │ • Streaming display   │ │   │
│  │  │ • System    │ │ • Tools JSON │ │ • Status colors       │ │   │
│  │  │   Prompt    │ └──────────────┘ └───────────────────────┘ │   │
│  │  └─────────────┘                                             │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │ localStorage: servers, models, temperature, etc.        │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Python Backend                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    handlers.py                                │  │
│  │  • fetch_available_models()  → ServerPool.refresh_manifests() │  │
│  │  • initialize_session()      → Create ComparisonSession       │  │
│  │  • send_single_prompt()      → Concurrent dispatcher          │  │
│  │  • export_markdown/json()    → Report generation              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                    │
│  ┌─────────────────────────────┼────────────────────────────────┐  │
│  │                    core.py  │                                 │  │
│  │  ┌─────────────────────┐    │    ┌─────────────────────────┐ │  │
│  │  │    ServerPool       │◄───┴───►│  ComparisonSession      │ │  │
│  │  │  • servers: dict    │         │  • state: SessionState  │ │  │
│  │  │  • refresh_manifest │         │  • send_prompt_to_model │ │  │
│  │  │  • acquire/release  │         │  • get_context_display  │ │  │
│  │  └─────────────────────┘         └─────────────────────────┘ │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │  stream_completion() / get_completion()                  │ │  │
│  │  │  • Async HTTP streaming to LM Studio                     │ │  │
│  │  │  • Yields text chunks or returns full response           │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                │                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    config.py                                  │  │
│  │  Pydantic Models: ServerConfig, ModelContext, SessionState   │  │
│  │  Constants: DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, etc.    │  │
│  │  Environment: load_servers_from_env(), get_gradio_port()     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LM Studio Servers                                │
│  ┌────────────────────────┐    ┌────────────────────────┐          │
│  │  Server 1 (e.g. 3090)  │    │  Server 2 (e.g. 8000)  │          │
│  │  • GET /v1/models      │    │  • GET /v1/models      │          │
│  │  • POST /v1/chat/...   │    │  • POST /v1/chat/...   │          │
│  │  └─ Model A            │    │  └─ Model B, C         │          │
│  │  └─ Model B            │    │                        │          │
│  └────────────────────────┘    └────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### Directory Structure

```
prompt_prix/
├── main.py              # Entry point
├── ui.py                # Gradio UI definition
├── handlers.py          # Shared event handlers (fetch, stop)
├── state.py             # Global mutable state
├── core.py              # ServerPool, ComparisonSession, streaming
├── config.py            # Pydantic models, constants, env loading
├── parsers.py           # Input parsing utilities
├── export.py            # Report generation
├── dispatcher.py        # ConcurrentDispatcher for parallel execution
├── battery.py           # BatteryRunner, TestResult, BatteryRun
├── tabs/
│   ├── __init__.py
│   ├── battery/
│   │   ├── __init__.py
│   │   └── handlers.py  # Battery-specific handlers
│   └── compare/
│       ├── __init__.py
│       └── handlers.py  # Compare-specific handlers
├── adapters/
│   └── lmstudio.py      # LMStudioAdapter
└── benchmarks/
    ├── base.py          # TestCase protocol
    └── custom_json.py   # CustomJSONLoader
```

### config.py - Configuration & Data Models

**Purpose**: Define all Pydantic models for type-safe configuration and state.

| Class | Purpose |
|-------|---------|
| `ServerConfig` | Single LM Studio server state (URL, available_models, is_busy) |
| `ModelConfig` | Model identity and display name |
| `Message` | Single message in a conversation (role, content - supports multimodal) |
| `ModelContext` | Complete conversation history for one model |
| `SessionState` | Full session: models, contexts, system_prompt, halted status |

**Message Multimodal Support**:
The `Message` model supports both text and multimodal content:
```python
# Text-only message
Message(role="user", content="Hello")

# Multimodal message (text + image)
Message(role="user", content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
])

# Helper methods
msg.get_text()   # Extract text content
msg.has_image()  # Check if message contains an image
```

**Key Functions**:
- `load_servers_from_env()` - Read LM_STUDIO_SERVER_N environment variables
- `get_default_servers()` - Return env servers or placeholder defaults
- `get_gradio_port()` - Read GRADIO_PORT or default to 7860
- `encode_image_to_data_url(path)` - Convert image file to base64 data URL
- `build_multimodal_content(text, image_path)` - Build OpenAI-format multimodal content

### core.py - Server Pool & Session Management

**Purpose**: Core business logic for server management and model interactions.

#### ServerPool

Manages multiple LM Studio servers:

```python
class ServerPool:
    servers: dict[str, ServerConfig]  # URL -> config
    _locks: dict[str, asyncio.Lock]   # URL -> lock

    async def refresh_all_manifests()  # GET /v1/models on all servers
    def find_available_server(model_id) -> Optional[str]  # Find idle server with model
    async def acquire_server(url)      # Mark busy, acquire lock
    def release_server(url)            # Mark available, release lock
```

#### ComparisonSession

Manages a comparison session:

```python
class ComparisonSession:
    server_pool: ServerPool
    state: SessionState  # Contains models, contexts, config

    async def send_prompt_to_model(model_id, prompt, on_chunk=None)
    async def send_prompt_to_all(prompt, on_chunk=None)
    def get_context_display(model_id) -> str
```

#### Streaming Functions

```python
async def stream_completion(
    server_url, model_id, messages, temperature, max_tokens,
    timeout_seconds, tools=None, seed=None, repeat_penalty=None
) -> AsyncGenerator[str, None]:
    """Yields text chunks as they arrive via SSE.

    Args:
        seed: Optional int for reproducible outputs (passed to model API)
        repeat_penalty: Optional float to penalize repeated tokens (1.0 = off)
    """

async def get_completion(...) -> str:
    """Non-streaming version, returns full response."""
```

### handlers.py - Shared Event Handlers

**Purpose**: Shared async handlers used across multiple tabs.

| Handler | Purpose | Returns |
|---------|---------|---------|
| `fetch_available_models(servers_text)` | Query all servers for available models | `(status, gr.update(choices=[...]))` |
| `handle_stop()` | Signal cancellation via global state | `status` |
| `_init_pool_and_validate(servers_text, models)` | Initialize ServerPool and validate models | `(pool, error_message)` |

### tabs/battery/handlers.py - Battery Tab Handlers

**Purpose**: Handlers specific to the Battery (benchmark) tab.

| Handler | Trigger | Returns |
|---------|---------|---------|
| `validate_file(file_path)` | File upload | Validation status string |
| `get_test_ids(file_path)` | File upload | List of test IDs |
| `run_handler(file, models, servers, ...)` | "Run Battery" button | Generator yielding `(status, grid_df)` |
| `quick_prompt_handler(prompt, models, ...)` | "Run Prompt" button | Markdown results |
| `export_json()` | "Export JSON" button | `(status, preview)` |
| `export_csv()` | "Export CSV" button | `(status, preview)` |
| `get_cell_detail(model, test)` | Detail dropdown | Markdown detail |
| `refresh_grid(display_mode)` | Display mode change | Updated grid DataFrame |

### tabs/compare/handlers.py - Compare Tab Handlers

**Purpose**: Handlers specific to the Compare (interactive) tab.

| Handler | Trigger | Returns |
|---------|---------|---------|
| `initialize_session(servers, models, system_prompt, ...)` | Auto-init on send | `(status, *model_tabs)` |
| `send_single_prompt(prompt, tools_json, image_path, seed, repeat_penalty)` | "Send to All" button | Generator yielding `(status, tab_states, *model_outputs)` |
| `export_markdown()` | "Export Markdown" button | `(status, preview)` |
| `export_json()` | "Export JSON" button | `(status, preview)` |
| `launch_beyond_compare(model_a, model_b)` | "Open in Beyond Compare" button | `status` |

**Compare Tab Features**:
- **Image Attachment**: Upload images for vision models (encoded as base64 data URLs)
- **Seed Parameter**: Set a seed for reproducible outputs across models
- **Repeat Penalty**: Configurable penalty (1.0-2.0) to reduce repetitive token generation

### dispatcher.py - Concurrent Dispatcher

**Purpose**: Parallel execution across multiple servers.

```python
class ConcurrentDispatcher:
    """Dispatches work items to idle servers in parallel."""

    async def dispatch(
        self,
        work_items: list[WorkItem],
        execute_fn: Callable[[WorkItem, str], Coroutine],
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> dict[str, Any]:
        """Execute work items in parallel across available servers."""
```

The dispatcher:
1. Maintains a queue of work items (model + test case pairs)
2. Finds idle servers that can run each work item
3. Executes items in parallel across all available servers
4. Supports cooperative cancellation via `state.should_stop()`

### ui.py - Gradio UI Definition

**Purpose**: Define all Gradio components and wire up event bindings.

**Key Components**:

| Component | Type | Purpose |
|-----------|------|---------|
| `servers_input` | Textbox | LM Studio server URLs (one per line) |
| `models_checkboxes` | CheckboxGroup | Select models to compare |
| `system_prompt_input` | Textbox (50 lines) | Editable system prompt |
| `temperature_slider` | Slider | Model temperature (0-2) |
| `timeout_slider` | Slider | Request timeout (30-600s) |
| `max_tokens_slider` | Slider | Max tokens (256-8192) |
| `seed_input` | Number | Optional seed for reproducible outputs |
| `repeat_penalty_slider` | Slider | Repeat penalty (1.0-2.0, default 1.1) |
| `prompt_input` | Textbox | User prompt entry |
| `image_input` | Image | Optional image attachment for vision models |
| `tools_input` | Code (JSON) | Tools for function calling |
| `model_outputs[0..9]` | Markdown | Model response tabs |
| `tab_states` | JSON (hidden) | Tab status for color updates |

**Event Bindings**:
- Buttons trigger async handlers
- `tab_states.change` triggers JavaScript for inline style updates
- `app.load` restores state from localStorage

### state.py - Global State

**Purpose**: Holds mutable state shared across handlers.

```python
server_pool: Optional[ServerPool] = None
session: Optional[ComparisonSession] = None
```

**Design Decision**: Separated to avoid circular imports between ui.py and handlers.py.

### parsers.py - Text Parsing Utilities

**Purpose**: Parse user input from UI components.

| Function | Input | Output |
|----------|-------|--------|
| `parse_models_input(text)` | "model1\nmodel2" | `["model1", "model2"]` |
| `parse_servers_input(text)` | "http://...\nhttp://..." | `["http://...", "http://..."]` |
| `parse_prompts_file(content)` | File content | List of prompts |
| `load_system_prompt(file_path)` | Optional file path | System prompt string |
| `get_default_system_prompt()` | - | Default prompt from file or constant |

### export.py - Report Generation

**Purpose**: Generate exportable reports from session state.

```python
def generate_markdown_report(state: SessionState) -> str:
    """Create Markdown with header, system prompt, and all model conversations."""

def generate_json_report(state: SessionState) -> str:
    """Create structured JSON with configuration and conversations."""

def save_report(content: str, filepath: str):
    """Write report to file."""
```

### main.py - Entry Point

**Purpose**: Application entry point and backwards-compatibility exports.

```python
def run():
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=get_gradio_port())
```

## Data Flow: Sending a Prompt

```
1. User types prompt, clicks "Send Prompt"
         │
         ▼
2. ui.py: send_button.click(fn=send_single_prompt, inputs=[prompt, tools])
         │
         ▼
3. handlers.py: send_single_prompt(prompt, tools_json)
   │ - Validate session exists
   │ - Parse tools JSON
   │ - Add user message to all model contexts
   │ - Refresh server manifests
         │
         ▼
4. Concurrent Dispatcher Loop:
   │ ┌─────────────────────────────────────────┐
   │ │ For each idle server:                   │
   │ │   Find model in queue this server has   │
   │ │   If found: start async task            │
   │ └─────────────────────────────────────────┘
   │ │ await asyncio.sleep(0.1)
   │ │ yield (status, tab_states, *outputs)  ──────► UI updates
   │ │ Clean up completed tasks
   │ └─────────── while queue or active_tasks
         │
         ▼
5. Each async task: run_model_on_server(model_id, server_url)
   │ - Mark model as "streaming"
   │ - Call stream_completion() ───────────────────► LM Studio API
   │ - Accumulate chunks in streaming_responses[model_id]
   │ - On complete: add assistant message to context
   │ - Release server
         │
         ▼
6. Final yield: ("✅ All responses complete", final_states, *final_outputs)
```

## State Management

### Session State (Python)

```python
SessionState:
  models: list[str]                    # Selected models
  contexts: dict[str, ModelContext]    # model_id -> conversation
  system_prompt: str
  temperature: float
  timeout_seconds: int
  max_tokens: int
  halted: bool                         # True if any model failed
  halt_reason: Optional[str]
```

### UI State (Browser localStorage)

| Key | Type | Purpose |
|-----|------|---------|
| `promptprix_servers` | string | Server URLs (newline-separated) |
| `promptprix_model_choices` | JSON array | Available models from last fetch |
| `promptprix_models` | JSON array | Selected models |
| `promptprix_temperature` | float | Temperature setting |
| `promptprix_timeout` | int | Timeout setting |
| `promptprix_max_tokens` | int | Max tokens setting |
| `promptprix_tools` | string | Tools JSON |
| `promptprix_system_prompt` | string | System prompt text |

**Persistence**: Only saved when user clicks "Save State" button (explicit save).

## Tab Status Visualization

Tab colors indicate model status during streaming:

| Status | Color | Border |
|--------|-------|--------|
| `pending` | Red gradient (#fee2e2 → #fecaca) | 4px solid #ef4444 |
| `streaming` | Yellow gradient (#fef3c7 → #fde68a) | 4px solid #f59e0b |
| `completed` | Green gradient (#d1fae5 → #a7f3d0) | 4px solid #10b981 |

**Implementation**: Uses inline JavaScript styles (`element.style`) to overcome Gradio theme CSS.

## Error Handling

### Fail-Fast Validation

1. `initialize_session` validates:
   - Servers are configured
   - Models are configured
   - All selected models exist on at least one server

2. `send_single_prompt` validates:
   - Session is initialized
   - Session is not halted
   - Prompt is not empty
   - Tools JSON is valid (if provided)

### Halt-on-Error

If any model fails during `send_prompt_to_all`:
- `state.halted = True`
- `state.halt_reason = "Model {model_id} failed: {error}"`
- Subsequent prompts are rejected

### Human-Readable Errors

The `LMStudioError` exception extracts error messages from LM Studio's JSON responses:

```python
{"error": {"message": "Model not loaded"}}  →  "Model not loaded"
```

## Integration Points

### Upstream: Benchmark Sources

prompt-prix can consume test cases from established benchmark ecosystems:

| Source | Format | Usage |
|--------|--------|-------|
| **BFCL** | JSON with function schemas | Export test cases, load in batch mode |
| **Inspect AI** | Python test definitions | Export prompts, import as JSON |
| **Custom JSON** | OpenAI-compatible messages | Direct load in prompt-prix |

See [ADR-001](adr/001-use-existing-benchmarks.md) for rationale.

### API Layer: OpenAI-Compatible

All inference servers must expose OpenAI-compatible endpoints:

```
GET  /v1/models              → List available models
POST /v1/chat/completions    → Chat completion (streaming)
```

Supported servers:
- LM Studio (native)
- Ollama (OpenAI mode)
- vLLM
- llama.cpp server
- Any OpenAI-compatible proxy

See [ADR-003](adr/003-openai-compatible-api.md) for rationale.

## Fan-Out Dispatcher Pattern

The core abstraction is **fan-out**: one prompt dispatched to N models in parallel.

```
┌─────────────────────────────────────────────────────────────┐
│                     Fan-Out Dispatcher                       │
│                                                              │
│  Input: (prompt, [model_a, model_b, model_c])               │
│                        │                                     │
│         ┌──────────────┼──────────────┐                     │
│         ▼              ▼              ▼                     │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│    │ Model A │    │ Model B │    │ Model C │               │
│    │ Server1 │    │ Server1 │    │ Server2 │               │
│    └────┬────┘    └────┬────┘    └────┬────┘               │
│         │              │              │                     │
│         ▼              ▼              ▼                     │
│    Response A     Response B     Response C                 │
│                                                              │
│  Output: {model_a: resp_a, model_b: resp_b, model_c: resp_c}│
└─────────────────────────────────────────────────────────────┘
```

### Parallel Dispatch

The dispatcher maximizes GPU utilization:

1. **Queue**: All work items (model + test pairs)
2. **Match**: Find idle server that has the required model loaded
3. **Execute**: Stream response, update UI
4. **Release**: Server becomes available for next item

This keeps all GPUs busy even when models are distributed across servers.

See [ADR-002](adr/002-fan-out-pattern-as-core.md) for rationale.

## Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [001](adr/001-use-existing-benchmarks.md) | Use existing benchmarks (BFCL, Inspect AI) instead of custom eval schema |
| [002](adr/002-fan-out-pattern-as-core.md) | Fan-out pattern as core architectural abstraction |
| [003](adr/003-openai-compatible-api.md) | OpenAI-compatible API as sole integration layer |
