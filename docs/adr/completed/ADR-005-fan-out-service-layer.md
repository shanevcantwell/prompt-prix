# ADR-005: prompt-prix as Fan-Out Service Layer

**Status:** Exploratory (Living Document)
**Date:** 2026-01-20
**Context:** Exploring how prompt-prix orchestration could be consumed by langgraph-agentic-scaffold (LAS)

---

## The Insight

> MCP primitives ARE the product. The Gradio UI is a reference consumer.
> LAS could be another consumer - using prompt-prix for multi-model fan-out orchestration.

---

## Problem Statement

LAS (langgraph-agentic-scaffold) implements "Tiered Chat" - a fan-out/join pattern where multiple specialists run in parallel, then synthesize results. However:

- Each specialist is bound to ONE LM Studio server via AdapterFactory
- No ServerPool for multi-server load balancing
- No model discovery across servers
- Parallelism is at the *graph node* level, not *model execution* level

**LAS can parallelize specialists, but each specialist hits a single server.**

---

## What LAS Has

| Component | Purpose | Limitation |
|-----------|---------|------------|
| `GraphOrchestrator.route_to_next_specialist()` | Returns `list[str]` for fan-out | Graph-level only |
| LangGraph array edges | `[alpha, bravo] → synthesizer` | Same server per specialist |
| `LMStudioAdapter` | OpenAI SDK wrapper | Single server binding |
| `AdapterFactory` | Creates adapters per specialist | No pool management |

### LAS Tiered Chat Flow

```
Router (route_to_next_specialist returns list[str])
    ↓
LangGraph conditional_edges (array syntax = parallel)
    ↓
ProgenitorAlpha      ProgenitorBravo
    ↓                    ↓
adapter.invoke()   adapter.invoke()  [PARALLEL via ainvoke()]
    ↓                    ↓
artifacts           artifacts
    ↓                    ↓
TieredSynthesizer (join point, waits for both)
    ↓
messages (permanent storage)
```

Key: Same model, different system prompts. Single server per specialist.

---

## What prompt-prix Has

| Component | Location | Capability |
|-----------|----------|------------|
| `list_models` | `mcp/tools/list_models.py` | Discover models across N servers |
| `complete_stream` | `mcp/tools/complete.py` | Execute on specific server/model |
| `ServerPool` | `core.py` | Multi-server availability tracking |
| `ConcurrentDispatcher` | `dispatcher.py` | Parallel execution with backpressure |
| `BatteryRunner` | `battery.py` | Test × Model matrix orchestration |
| `judge` | `mcp/tools/judge.py` | LLM-as-judge evaluation |

---

## The Inversion

```
LAS Current (Tiered Chat):
  Router → [Alpha, Bravo] → Synthesizer
           ↓       ↓
    Same model, different system prompts
    Single server per specialist

LAS + prompt-prix as service:
  Router → "need multi-model comparison"
           ↓
    prompt-prix.fan_out(prompt, models=[a,b,c])
           ↓
    ServerPool + ConcurrentDispatcher
           ↓
    [model_a@srv1, model_b@srv2, model_c@srv1]
           ↓
    Return all responses → LAS Synthesizer
```

**Different models, same prompt. Multiple servers with load balancing.**

---

## Potential MCP Service Interface

```python
# Hypothetical prompt_prix MCP tools

# Discovery
list_models(
    servers: list[str],
    only_loaded: bool = False
) -> {
    "models": list[str],
    "servers": dict[str, list[str]]  # server → models mapping
}

# Single execution (existing)
complete(
    server: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: int = 120
) -> {
    "response": str,
    "latency_ms": int,
    "tokens": {"prompt": int, "completion": int}
}

# Fan-out execution (NEW - the key primitive)
fan_out(
    servers: list[str],
    models: list[str],          # Which models to query
    messages: list[dict],       # The prompt
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: int = 120
) -> {
    "results": {
        "model_a": {"response": str, "latency_ms": int, "server": str},
        "model_b": {"response": str, "latency_ms": int, "server": str},
        ...
    },
    "errors": {
        "model_c": {"error": str, "server": str}
    }
}

# Evaluation (existing)
judge(
    response: str,
    criteria: str,
    judge_model: str
) -> {
    "score": float,
    "reasoning": str
}
```

---

## Transport Options

### Option A: HTTP API (FastAPI alongside Gradio)

```python
# FastAPI routes
POST /api/v1/models/list
POST /api/v1/complete
POST /api/v1/fan-out
POST /api/v1/judge
```

Pros: Standard REST, easy to consume from any language
Cons: Another port, CORS concerns, stateless by default

### Option B: stdio MCP (subprocess, JSON-RPC)

```python
# LAS ExternalMcpClient pattern
result = await external_mcp.call_tool(
    service_name="prompt_prix",
    tool_name="fan_out",
    arguments={...}
)
```

Pros: Native MCP integration, container-friendly
Cons: Subprocess overhead, stdio buffering

### Option C: Direct Python import (same process)

```python
# Direct import
from prompt_prix.mcp.tools import fan_out
result = await fan_out(servers, models, messages)
```

Pros: Zero overhead, full async support
Cons: Tight coupling, same Python environment required

---

## Questions to Resolve

1. **Transport:** Which option best fits LAS's existing patterns?
   - LAS already has `ExternalMcpClient` for stdio MCP
   - LAS already has HTTP for its FastAPI endpoints

2. **Scope:** What should `fan_out` handle?
   - Just parallel execution → return raw responses
   - Synthesis → aggregate into single response
   - Evaluation → include judge scores

3. **State:** Stateless (single call) or stateful (session)?
   - Stateless: simpler, easier to scale
   - Stateful: enables multi-turn fan-out (like Compare tab)

4. **Backpressure:** How to handle server capacity?
   - `ConcurrentDispatcher` has per-server semaphores
   - Should `fan_out` expose concurrency limits?

---

## Use Cases for LAS Consuming prompt-prix

### 1. Multi-Model Consensus (Evolution of Tiered Chat)

```python
# Instead of Alpha + Bravo (same model, different prompts)
# Use N models (different models, same prompt)
responses = await prompt_prix.fan_out(
    servers=["srv1", "srv2"],
    models=["qwen-32b", "llama-70b", "mistral-22b"],
    messages=[{"role": "user", "content": query}]
)

# LAS TieredSynthesizer combines responses
consensus = await synthesizer.invoke({
    "perspectives": responses["results"]
})
```

### 2. Model Selection (Best-of-N)

```python
# Run same prompt on candidate models
responses = await prompt_prix.fan_out(
    models=["model_a", "model_b", "model_c"],
    messages=task_messages
)

# Use judge to score
scores = {}
for model_id, result in responses["results"].items():
    score = await prompt_prix.judge(
        response=result["response"],
        criteria="accuracy and completeness",
        judge_model="llama-70b"
    )
    scores[model_id] = score

# Select best
best_model = max(scores, key=lambda m: scores[m]["score"])
```

### 3. Latency-Optimized Routing

```python
# Discover what's available and loaded
available = await prompt_prix.list_models(
    servers=["srv1", "srv2", "srv3"],
    only_loaded=True
)

# Route to fastest available model for simple tasks
# Route to most capable for complex tasks
```

---

## Relationship to Existing prompt-prix Components

| prompt-prix Component | Role in Service Layer |
|-----------------------|-----------------------|
| `list_models` MCP | Discovery primitive (already exists) |
| `complete_stream` MCP | Single execution (already exists) |
| `ConcurrentDispatcher` | Parallel execution engine |
| `ServerPool` | Server availability tracking |
| `BatteryRunner` | Reference orchestrator (Test × Model) |
| Gradio UI | Reference consumer #1 |
| **LAS** | **Potential consumer #2** |

---

## Next Steps

1. **Validate need:** Does LAS actually need multi-server/multi-model fan-out?
2. **Prototype `fan_out`:** Thin wrapper over ConcurrentDispatcher
3. **Choose transport:** Likely stdio MCP to match LAS patterns
4. **Test integration:** LAS specialist that calls prompt_prix.fan_out

---

## References

- [LAS ARCHITECTURE.md](/home/shane/github/shanevcantwell/langgraph-agentic-scaffold/docs/ARCHITECTURE.md)
- [LAS tiered_chat.py](/home/shane/github/shanevcantwell/langgraph-agentic-scaffold/app/src/workflow/subgraphs/tiered_chat.py)
- [LAS graph_orchestrator.py](/home/shane/github/shanevcantwell/langgraph-agentic-scaffold/app/src/workflow/graph_orchestrator.py)
- [prompt-prix dispatcher.py](/home/shane/github/shanevcantwell/prompt-prix/prompt_prix/dispatcher.py)
- [prompt-prix mcp/tools/](/home/shane/github/shanevcantwell/prompt-prix/prompt_prix/mcp/tools/)
