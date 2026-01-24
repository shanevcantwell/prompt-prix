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
- **Server affinity** — optional hint routes to specific server index
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
│        complete(), judge(), list_models(), fan_out()        │
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

    async def get_available_models(self) -> list[str]:
        """Return deduplicated list of all models across all servers."""

    def get_models_by_server(self) -> dict[str, list[str]]:
        """Return models grouped by server URL."""

    def get_unreachable_servers(self) -> list[str]:
        """Return servers that failed model discovery."""

    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tools: list[dict] | None = None,
        seed: int | None = None,
        repeat_penalty: float | None = None,
        server_hint: int | None = None,  # Route to specific server index
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens from an available server."""
```

### Server Selection Logic

```
server_hint provided?
    ├─ YES → Route to servers[hint] (wait if busy)
    └─ NO  → Route to any server with model available
```

### Concurrency Guarantee

```
Task A: stream_completion(model="x", server_hint=0)  → Server 0
Task B: stream_completion(model="y", server_hint=1)  → Server 1
                                                        ↑ PARALLEL

Task C: stream_completion(model="z", server_hint=0)  → Server 0 (waits for A)
                                                        ↑ QUEUED
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

Manages server state and per-server locks.

**Location:** `lmstudio.py:34-110`

```python
class _ServerPool:
    servers: dict[str, ServerConfig]   # URL → state (insertion order = index)
    _locks: dict[str, asyncio.Lock]    # URL → lock (one per server)

    async def acquire_server(url: str) → None    # Blocks until server free
    def release_server(url: str) → None          # Marks server available
    def find_server(model: str, hint: int | None) → str | None
```

**Server indexing:** Order is determined by initialization:
```python
adapter = LMStudioAdapter(["http://gpu0:1234", "http://gpu1:1234"])
# server_hint=0 → gpu0
# server_hint=1 → gpu1
```

### _ConcurrentDispatcher

Routes work to available servers.

**Location:** `lmstudio.py:126-205`

```python
class _ConcurrentDispatcher:
    async def execute(
        model_id: str,
        server_hint: int | None,
        work_fn: Callable[[str], Coroutine]  # (server_url) → result
    ) → None:
        """Find appropriate server, acquire it, run work_fn, release."""
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Server unreachable during discovery | Marked as unreachable, excluded from routing |
| Server unreachable during completion | Raise `LMStudioError` after timeout |
| Model not on hinted server | Raise immediately (no fallback) |
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

[battery/handlers.py:144](../prompt_prix/tabs/battery/handlers.py#L144) creates BatteryRunner:

```python
runner = BatteryRunner(
    tests=tests,
    models=["0:qwen2.5-7b", "1:llama-3.1-8b"],  # Affinity prefixes
    max_concurrent=2,  # One per GPU
    judge_model="qwen2.5-7b"
)
```

### Step 2: MCP complete() Primitive

[mcp/tools/complete.py](../prompt_prix/mcp/tools/complete.py) receives call:

```python
response = await complete(
    model_id="qwen2.5-7b",
    messages=[...],
    server_hint=0  # Parsed from "0:qwen2.5-7b" prefix
)
```

### Step 3: Adapter Dispatch

```
[DEBUG] Server affinity: model=qwen2.5-7b -> server 0
[DEBUG] Acquired server http://gpu0:1234 for model qwen2.5-7b
[DEBUG] Streaming completion...
[DEBUG] Releasing server http://gpu0:1234
```

### Step 4: Parallel Execution

While GPU 0 processes qwen2.5-7b:
- GPU 1 independently processes llama-3.1-8b
- Both complete concurrently
- BatteryRunner receives both results

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
| Server affinity routing | `tests/test_lmstudio_adapter.py::test_server_affinity_*` |
| Invalid hint handling | `tests/test_lmstudio_adapter.py::test_invalid_affinity_*` |
| Model availability | `tests/test_lmstudio_adapter.py::test_model_*` |
| Unreachable detection | `tests/test_lmstudio_adapter.py::test_unreachable_*` |
| Parallel dispatch | `tests/test_lmstudio_adapter.py::TestConcurrentDispatch` |

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
        work_fn: Callable[[str], Coroutine],
        resource_hint: int | None
    ) -> None
```

---

## Summary

The LMStudioAdapter is a **multi-server resource manager** that:

1. Maintains a pool of LM Studio server connections
2. Routes requests to appropriate servers via `server_hint`
3. Ensures parallel execution across GPUs, serial execution per GPU
4. Encapsulates all complexity behind the `HostAdapter` protocol

Callers (MCP primitives) never know about servers, pools, or locks. They submit work; the adapter handles dispatch.
