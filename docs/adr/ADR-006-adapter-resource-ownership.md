# ADR-006: Adapter Resource Ownership

**Status**: Accepted
**Date**: 2025-01-22
**Supersedes**: ADR-002 (partial - resource management implications)

## Context

prompt-prix supports multiple inference backends with fundamentally different resource models:

| Backend | Resource Model |
|---------|----------------|
| LM Studio | Multiple local servers, availability tracking via ServerPool |
| HF Inference | Rate-limited cloud API |
| surf-mcp | Single browser session |

Previous architecture placed `ServerPool` in the MCP/orchestration layer (see ADR-002's "Architecture Implications"), coupling generic orchestration to LM Studio's specific multi-server model. This blocked adding other backends.

**The core insight** (via Gemini 3.0 architectural review):

> "ServerPool is not an orchestration primitive; it's a resource management strategy for a specific class of provider."

## Decision

**Adapters own their resource management strategies.**

- **Orchestration** (BatteryRunner, MCP primitives) manages *what* to run
- **Adapters** manage *how* to run it (including parallelism, pooling, rate limiting)

### Concrete Rules

#### MUST

1. `LMStudioAdapter.__init__` takes `server_urls: list[str]` and creates `ServerPool` internally
2. `ServerPool` is defined in `adapters/lmstudio.py` — not importable from elsewhere
3. `ConcurrentDispatcher` is internal to `LMStudioAdapter`
4. `BatteryRunner` calls MCP primitives (`complete_stream`, `fan_out`) — never adapters directly
5. MCP primitives receive adapter via registry (`get_adapter()`)
6. Orchestration controls concurrency via semaphore; adapter controls server routing

#### MUST NOT

1. **Orchestration** (BatteryRunner, ComparisonSession) MUST NOT import from `adapters/`
2. **Orchestration** MUST NOT know about ServerPool, ConcurrentDispatcher, or server URLs
3. **MCP primitives** MUST NOT instantiate adapters — receive via registry
4. **No module outside `adapters/lmstudio.py`** may reference ServerPool or ConcurrentDispatcher

### The Architecture Layers

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

## Rationale

### The "Black Box" Adapter

Each adapter is a black box that accepts a model ID and messages, and returns a completion stream. The orchestration layer doesn't know or care whether the adapter is managing 10 local servers, calling a rate-limited API, or automating a browser.

This enables:
- Adding new backends without changing orchestration code
- Backend-specific optimizations (pooling, batching, rate limiting) without leaking abstractions
- Clean testing — mock the adapter interface, not internal implementation

### Why Not Pool at Orchestration Level?

If `ServerPool` lives in MCP, then:
- MCP must understand LM Studio's multi-server model
- Adding HF Inference requires MCP to also understand rate limits
- Adding surf-mcp requires MCP to understand browser session management
- MCP becomes a union of all backend complexity — the opposite of abstraction

### The Dependency Injection Pivot

Currently, `BatteryRunner` and MCP tools accept `server_urls`. The fix is to accept an `adapter` object instead:

```python
# Before (coupled to LM Studio)
class BatteryRunner:
    def __init__(self, servers: list[str], ...):
        self.pool = ServerPool(servers)  # LM Studio assumption

# After (backend-agnostic)
class BatteryRunner:
    def __init__(self, adapter: HostAdapter, ...):
        self.adapter = adapter  # Could be any backend
```

## Consequences

### Positive

- Clean separation: orchestration doesn't know backend details
- New adapters are self-contained packages
- Testable: mock adapter interface without mocking httpx/ServerPool internals
- Unblocks #56 (HuggingFace adapter)

### Negative

- `ConcurrentDispatcher` must move into adapter layer or become an internal utility
- More indirection for LM Studio (the common case today)
- Adapter implementations carry more responsibility

### Migration Path

1. `LMStudioAdapter.__init__` takes `server_urls`, creates pool internally
2. Add module-level `stream_completion()` to `lmstudio.py` for simple cases
3. MCP `complete.py` delegates to adapter module function
4. `BatteryRunner` signature changes: `servers` → `adapter`
5. Move `ServerPool` class from `core.py` to `adapters/lmstudio.py`
6. Update ~10 import sites across codebase
7. Move/adapt `ConcurrentDispatcher` into adapter layer

## References

- Gemini 3.0 architectural analysis (2025-01-22)
- Previous planning docs: `harmonic-hatching-shamir.md`, `curried-painting-backus.md`
- Implementation: #95 (Implement ADR-006 adapter resource ownership)
- Related issues: #56 (HuggingFace adapter), #92 (connection error handling)
