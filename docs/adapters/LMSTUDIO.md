# LMStudioAdapter Profile

**Purpose:** Technical specification for the LMStudioAdapter - the multi-GPU resource manager for LM Studio servers.
**Audience:** Developers extending prompt-prix, architects integrating with LAS, AI agents consuming MCP tools.
**Date:** 2026-01-23

---

## Executive Summary

The **LMStudioAdapter** is the concrete implementation of `HostAdapter` for LM Studio servers. It owns all multi-server complexity internally - callers submit work, the adapter handles server selection, queueing, and dispatch.

Key characteristics:
- **Multi-server aware** — manages pool of LM Studio instances (one per GPU)
- **Parallel execution** — different servers run concurrently; same server queues
- **Automatic routing** — finds any available server with the requested model
- **Encapsulated** — internal machinery (`_ServerPool`, `_ConcurrentDispatcher`) not exported

---

## Where LMStudioAdapter Fits

### Layer Architecture (per ADR-006)

```
┌─────────────────────────────────────────────────────────────┐
│                      UI / Agentic Consumer                   │
│            (Gradio tabs, LAS ReActMixin, CLI)               │
└─────────────────────────────┬───────────────────────────────┘
                              │ calls
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCP Primitives                          │
│           complete(), judge(), list_models()                 │
└─────────────────────────────┬───────────────────────────────┘
                              │ via registry
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      HostAdapter Protocol                    │
│     stream_completion(), get_available_models()              │
└─────────────────────────────┬───────────────────────────────┘
                              │ implemented by
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      LMStudioAdapter                         │
│  ┌───────────────┐  ┌─────────────────────────┐            │
│  │ _ServerPool   │  │ _ConcurrentDispatcher   │            │
│  │ (internal)    │  │ (internal)              │            │
│  └───────────────┘  └─────────────────────────┘            │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LM Studio Servers (GPU 0, GPU 1, ...)          │
└─────────────────────────────────────────────────────────────┘
```

### Import Rules

| Layer | MAY Import | MUST NOT Import |
|-------|------------|-----------------|
| UI / Orchestration | `mcp.tools.*`, `mcp.registry` | `adapters.lmstudio` |
| MCP Primitives | `adapters.base.HostAdapter` (protocol) | `LMStudioAdapter` class |
| Adapter | httpx, internal utilities | Nothing from orchestration/MCP |

---

## What the Adapter Actually Does

### Public Interface

```python
class LMStudioAdapter:
    def __init__(self, server_urls: list[str]):
        """Initialize with list of LM Studio server URLs."""

    async def get_available_models(self, only_loaded: bool = False) -> list[str]:
        """Return deduplicated list of all models across all servers."""

    def get_models_by_server(self) -> dict[str, list[str]]:
        """Return models grouped by server URL."""

    def get_unreachable_servers(self) -> list[str]:
        """Return servers that failed model discovery."""

    async def stream_completion(self, task: InferenceTask) -> AsyncGenerator[str, None]:
        """Stream completion tokens from an available server."""
```

Where `InferenceTask` (from `adapters/schema.py`) is:

```python
class InferenceTask(BaseModel):
    model_id: str
    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = -1
    timeout_seconds: float = 60.0
    tools: Optional[List[Dict[str, Any]]] = None
    seed: Optional[int] = None
    repeat_penalty: Optional[float] = None
```

### Server Selection Logic

The adapter automatically routes to any available server that has the requested model loaded, with a **model-drain guard** to prevent VRAM swap mid-stream.

```
Request arrives for model "qwen2.5-7b"
    → find_and_acquire("qwen2.5-7b")
        → Skip servers where current_model ≠ requested AND active_requests > 0
        → Acquire least-loaded server with the model available
        → Set current_model = "qwen2.5-7b", increment active_requests
    → Stream completion
    → release_server(url)
        → Decrement active_requests
        → If active_requests == 0: current_model = None (fully drained)
```

The `current_model` field on `ServerConfig` tracks what model a server is currently serving. This prevents the dispatcher from routing a different model to a server with in-flight requests — which would cause LM Studio's JIT model loading to unload the current model mid-stream, producing "Stream aborted" / "Model unloaded" errors.

### Concurrency Guarantee

```
Task A: stream_completion(model="x")  → Server 0 (first available)
Task B: stream_completion(model="y")  → Server 1 (parallel, different model OK on different server)
                                          ↑ PARALLEL

Task C: stream_completion(model="x")  → Server 0 (same model, shares KV cache slots)
                                          ↑ PARALLEL (LM Studio parallel KV cache)

Task D: stream_completion(model="z")  → Waits until Server 0 or 1 fully drains
                                          ↑ QUEUED (current_model guard blocks)
```

### Output: Streamed Tokens

Yields string chunks as they arrive from the LM Studio SSE stream:

```python
async for chunk in adapter.stream_completion(...):
    print(chunk, end="", flush=True)  # "Hello", " world", "!", ...
```

Tool calls are accumulated and yielded after stream completes:
```
**Tool Call:** `delete_file`
```json
{"path": "report.pdf"}
```
```

---

## Internal Components (Not Exported)

### _ServerPool

Manages server state, slot tracking, and the model-drain guard.

```python
class _ServerPool:
    servers: dict[str, ServerConfig]    # URL → state
    resource_available: asyncio.Event   # Signals when a slot frees up

    def find_and_acquire(model_id: str) → str | None
        # Atomically find server with available slots for this model.
        # Skips servers where current_model ≠ model_id AND active_requests > 0.
        # Returns server_url if successful, None otherwise.
        # Sets current_model = model_id, increments active_requests.

    def release_server(url: str) → None
        # Decrement active_requests.
        # If active_requests == 0: current_model = None (fully drained).
        # Signals resource_available event.
```

### _ConcurrentDispatcher

Routes work to available servers. Spin-waits on `resource_available` event when no server is available.

```python
class _ConcurrentDispatcher:
    async def execute(
        model_id: str,
        work_fn: Callable[[str], Coroutine]  # (server_url) → result
    ) → None:
        """Find server via find_and_acquire, run work_fn, release_server."""
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Server unreachable during discovery | Marked as unreachable, excluded from routing |
| Server unreachable during completion | Raise `LMStudioError` after timeout |
| Model not on any server | Raise immediately |
| All servers busy | Wait until one frees (respects timeout) |
| HTTP 4xx/5xx from server | Raise `LMStudioError` with message from response |

### LMStudioError

Human-readable error from LM Studio API:

```python
class LMStudioError(Exception):
    """Human-readable error from LM Studio API."""
    pass
```

---

## What the Adapter Does NOT Do

Understanding boundaries is critical:

| Capability | Adapter | Who Does It |
|------------|---------|-------------|
| Prompt construction | No | MCP primitives (complete, judge) |
| Response parsing | No | MCP primitives |
| Retry logic | No | Orchestration (BatteryRunner) |
| Progress reporting | No | Orchestration |
| Model selection UI | No | UI layer (Gradio) |
| Server URL configuration | No | Environment / UI layer |

---

## Example Flow: Battery Test Execution

**User action:** Runs battery with 2 models across 2 GPUs

### Step 1: UI Handler

[battery/handlers.py](../prompt_prix/tabs/battery/handlers.py) creates BatteryRunner:

```python
runner = BatteryRunner(
    tests=tests,
    models=["qwen2.5-7b", "llama-3.1-8b"],
    max_concurrent=2,
    judge_model="qwen2.5-7b"
)
```

### Step 2: MCP complete() Primitive

[mcp/tools/complete.py](../prompt_prix/mcp/tools/complete.py) receives call:

```python
response = await complete(
    model_id="qwen2.5-7b",
    messages=[...]
)
```

### Step 3: Adapter Dispatch

```
[DEBUG] Finding server for model qwen2.5-7b
[DEBUG] Acquired server http://gpu0:1234
[DEBUG] Streaming completion...
[DEBUG] Releasing server http://gpu0:1234
```

### Step 4: Parallel Execution

Both models run concurrently on available servers:
- Server selection is automatic based on model availability
- Different servers execute in parallel
- Same server queues requests
- BatteryRunner receives results as they complete

---

## Configuration

### Environment Variables

```bash
# .env file
LM_STUDIO_SERVER_1=http://127.0.0.1:1234      # GPU 0
LM_STUDIO_SERVER_2=http://192.168.137.2:1234  # GPU 1
```

### Adapter Registration

[handlers.py:102](../prompt_prix/handlers.py#L102) registers adapter with servers:

```python
def _ensure_adapter_registered(servers: list[str]) -> None:
    from prompt_prix.adapters.lmstudio import LMStudioAdapter
    from prompt_prix.mcp.registry import register_adapter

    adapter = LMStudioAdapter(server_urls=servers)
    register_adapter(adapter)
```

### MCP Registry

All MCP primitives retrieve adapter via registry:

```python
from prompt_prix.mcp.registry import get_adapter

adapter = get_adapter()  # Returns registered LMStudioAdapter
await adapter.stream_completion(...)
```

---

## Test Coverage

| Scenario | Test Location |
|----------|---------------|
| Model availability | `tests/test_lmstudio_adapter.py::test_model_*` |
| Unreachable detection | `tests/test_lmstudio_adapter.py::test_unreachable_*` |
| Parallel dispatch | `tests/test_lmstudio_adapter.py::TestConcurrentDispatch` |
| Streaming completion | `tests/test_lmstudio_adapter.py::test_stream_*` |
| Model-drain guard | `tests/test_lmstudio_adapter.py::TestServerPool` (current_model transitions) |
| Pipelined judging | `tests/test_battery.py::TestPipelinedJudging` |

Run adapter tests:
```bash
pytest tests/test_lmstudio_adapter.py -v
```

---

## Key Files

| File | Purpose |
|------|---------|
| [adapters/lmstudio.py](../prompt_prix/adapters/lmstudio.py) | Adapter implementation |
| [adapters/base.py](../prompt_prix/adapters/base.py) | HostAdapter protocol |
| [mcp/registry.py](../prompt_prix/mcp/registry.py) | Adapter registration |
| [config.py](../prompt_prix/config.py) | ServerConfig dataclass |
| [tests/test_lmstudio_adapter.py](../tests/test_lmstudio_adapter.py) | Unit tests |

---

## Future: Dispatcher Abstraction

The dispatcher pattern is provider-agnostic. When adding `HFInferenceAdapter`:

1. Extract `_ConcurrentDispatcher` to `adapters/base.py`
2. Parameterize with execution function
3. Both adapters reuse same dispatch algorithm

```python
# Future interface
class ConcurrentDispatcher(Protocol):
    async def execute(
        work_fn: Callable[[str], Coroutine]
    ) -> None
```

---

## Summary

The LMStudioAdapter is a **multi-server resource manager** that:

1. Maintains a pool of LM Studio server connections
2. Automatically routes requests to available servers with the requested model
3. Prevents VRAM swap mid-stream via `current_model` drain guard
4. Enables pipelined judging — judge tasks route to idle GPUs during inference
5. Encapsulates all complexity behind the `HostAdapter` protocol

Callers (MCP primitives) never know about servers, pools, or drain guards. They submit work; the adapter handles dispatch.
