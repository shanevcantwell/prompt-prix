# GitHub Issues to File

## Issue 1: Race Condition in LMStudioAdapter Resource Acquisition causes inefficient GPU usage

**Description**
When multiple concurrent requests are made to `LMStudioAdapter.stream_completion` (e.g., from `BatteryRunner`), they race to find an available server. The current implementation lacks atomicity between checking for server availability and acquiring the server lock.

**Location**
`prompt_prix/adapters/lmstudio.py`: `stream_completion` method.

**Steps to Reproduce**
1. Launch 2 concurrent completion requests targeting the same model.
2. Have 2 servers available with that model loaded.
3. Observe that both requests frequently attempt to grab `Server 1`. The second request waits for the first to finish on Server 1, leaving Server 2 idle.

**Root Cause**
The `find_available_server` check and `acquire_server` call are not atomic.
```python
server_url = self._pool.find_available_server(actual_model_id)
# <--- Context switch possible here, or just race condition
if server_url is not None:
    await self._pool.acquire_server(server_url)
```

**Proposed Fix**
Implement an atomic `find_and_acquire` method in `_ServerPool` that internally holds a lock while checking and marking the server as busy.

---

## Issue 2: `_ConcurrentDispatcher` is dead code and should be integrated or removed

**Description**
The `_ConcurrentDispatcher` class in `lmstudio.py` implements a better, queue-based dispatching logic (the "Fan Out" pattern described in Architecture). However, it is completely unused. `BatteryRunner` relies on `stream_completion`, which implements its own inferior while-loop polling mechanism.

**Impact**
Code duplication, confusion, and missing out on the better scheduling logic already implemented in the dispatcher.

**Proposed Fix**
Refactor `LMStudioAdapter` to utilize `_ConcurrentDispatcher` for incoming requests, possibly by having `stream_completion` submit to the dispatcher's queue instead of polling `_ServerPool` directly.

---

## Issue 3: Architectural Coupling and "Stringly-Typed" MCP Communication

**Description**
The communication between `BatteryRunner`/`CompareRunner` (Orchestration) and the Adapter layer relies on "stringly-typed" MCP primitives (`complete_stream(model_id: str, ...)`). This has led to fragile logic where upper layers optimize for these string interfaces.

**Goal**
Decouple the dispatcher in `LMStudioAdapter` from specific runners. The dispatcher should operate on strongly-typed Events or Tasks, not just raw arguments passed through MCP tools.

**Proposed Fix**
Refactor the boundary to use typed schemas (e.g., Pydantic models for `InferenceTask`) and event-based dispatching, reducing the reliance on the broad MCP primitive interface for internal heavy-lifting.

---

## Issue 4: Regression: Dispatcher efficiency worse than v2-simplified

**Description**
The previous `v2-simplified` branch used a global lock around server discovery (`async with self._lock:`), which serialized the "find" step but prevented the race condition where multiple workers pile onto one GPU. The new lock-free implementation introduced to "improve parallelism" has paradoxically degraded scheduling efficiency by introducing the race condition described in Issue 1.

**Reference**
`v2-simplified` branch `old_lmstudio.py` logic vs current `lmstudio.py`.
