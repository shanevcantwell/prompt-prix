# Plan: Refactor Adapter Pattern (#73) - Architecture Design

## Future Requirements (Design Constraints)

1. **Multi-seed testing**: Run each test N times with different seeds
2. **Regeneration/escalation**: Re-run failed tests with variations
3. **Future tabs**: Different testing strategies (stability, regression, etc.)
4. **Multiple backends**: LM Studio, HuggingFace, Ollama, vLLM, etc.

---

## Proposed Architecture: Three Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: Tab Orchestration (Battery, Compare, Future Tabs)      │
│ - Creates tasks from user input (tests × models × seeds)        │
│ - Tab-specific UI logic                                         │
│ - Handles results display                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓ uses
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 2: Task Executor (NEW - unified execution engine)         │
│ - Backend-agnostic                                              │
│ - Manages concurrency via adapter.get_concurrency_limit()       │
│ - Executes Task objects: (model_id, messages, params)           │
│ - Handles retries, seeds, escalation policies                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓ uses
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: Adapters (LMStudio, HuggingFace, Ollama, etc.)         │
│ - Backend-specific streaming and concurrency                    │
│ - acquire/release for resource management                       │
│ - Knows nothing about tests, batches, or tabs                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Adapter Protocol (Updated)

```python
class HostAdapter(Protocol):
    """Backend-agnostic interface for LLM inference."""

    async def get_available_models(self) -> list[str]: ...

    async def stream_completion(
        self, model_id: str, messages: list[dict],
        temperature: float, max_tokens: int, timeout_seconds: int,
        tools: Optional[list[dict]] = None,
        seed: Optional[int] = None  # NEW: for reproducibility
    ) -> AsyncGenerator[str, None]: ...

    def get_concurrency_limit(self) -> int: ...
    async def acquire(self, model_id: str) -> None: ...
    async def release(self, model_id: str) -> None: ...
```

Key: Adapters handle ALL backend-specific logic internally. No leaking of ServerPool, server_url, etc.

---

## Layer 2: Task Executor (NEW)

```python
@dataclass
class Task:
    """Unit of work for the executor."""
    id: str                         # Unique task identifier
    model_id: str                   # Which model to use
    messages: list[dict]            # Conversation to send
    params: dict                    # temperature, max_tokens, etc.
    seed: Optional[int] = None      # For reproducibility
    retry_count: int = 0            # For escalation

@dataclass
class TaskResult:
    """Result of executing a task."""
    task_id: str
    model_id: str
    response: str
    status: Literal["success", "error", "timeout"]
    duration_ms: int
    error: Optional[str] = None

class TaskExecutor:
    """
    Backend-agnostic task execution engine.

    Handles:
    - Concurrent execution respecting adapter limits
    - Streaming results back to caller
    - Retry logic (for escalation)
    - Seed management (for multi-run testing)
    """

    def __init__(self, adapter: HostAdapter):
        self.adapter = adapter
        self._semaphore = asyncio.Semaphore(adapter.get_concurrency_limit())

    async def execute(
        self,
        tasks: list[Task]
    ) -> AsyncGenerator[TaskResult, None]:
        """Execute tasks with managed concurrency."""
        ...

    async def execute_with_seeds(
        self,
        tasks: list[Task],
        seeds: list[int]
    ) -> AsyncGenerator[TaskResult, None]:
        """Execute each task multiple times with different seeds."""
        ...
```

---

## Layer 3: Tab Orchestration

**Battery tab** (tests × models) - uses TaskExecutor:
```python
class BatteryRunner:
    def __init__(self, adapter: HostAdapter, tests: list[TestCase], models: list[str]):
        self.adapter = adapter
        self.tests = tests
        self.models = models

    async def run(self) -> AsyncGenerator[BatteryRun, None]:
        executor = TaskExecutor(self.adapter)
        tasks = [Task(id=f"{t.id}:{m}", model_id=m, ...) for t in tests for m in models]
        async for result in executor.execute(tasks):
            self._update_state(result)
            yield self.state
```

**Compare tab** (interactive prompts) - uses adapter directly for streaming:
```python
class ComparisonSession:
    def __init__(self, adapter: HostAdapter, models: list[str], ...):
        self.adapter = adapter
        self.models = models

    async def send_prompt_to_model(self, model_id, prompt, on_chunk):
        await self.adapter.acquire(model_id)
        try:
            async for chunk in self.adapter.stream_completion(model_id, ...):
                await on_chunk(model_id, chunk)  # Live streaming
        finally:
            await self.adapter.release(model_id)
```

**Future tabs** (regeneration, stability testing):
- Can use TaskExecutor with `seed` parameter for reproducibility
- Multi-seed: `Task(seed=42)` runs same test with specific seed
- Regeneration: Create new Task with modified params on failure

---

## Problem Summary (Original Issues)

Three abstraction leaks prevent HuggingFace (and future backends) from working:

| Issue | Location | Why It Breaks |
|-------|----------|---------------|
| `adapter.pool` access | `battery.py:316` | HF has no pool |
| ServerPool as param | `ComparisonSession` | Bypasses adapters entirely |
| `stream_completion(server_url=...)` | `core.py:70` | LM Studio-specific |

---

## Implementation Phases (Simplified)

### Phase 1: ✅ Finalize Adapter Protocol (done)

Added `get_concurrency_limit()`, `acquire()`, `release()` to HostAdapter.
Updated LMStudioAdapter and HuggingFaceAdapter.

### Phase 2: Create TaskExecutor (Battery only)

**New file:** `prompt_prix/executor.py`

TaskExecutor provides backend-agnostic batch execution for Battery tab.
Compare tab won't use it (uses adapter directly for streaming).

```python
@dataclass
class Task:
    id: str
    model_id: str
    messages: list[dict]
    temperature: Optional[float]
    max_tokens: int
    timeout_seconds: int
    tools: Optional[list[dict]] = None

@dataclass
class TaskResult:
    task_id: str
    model_id: str
    response: str
    status: Literal["success", "error"]
    duration_ms: int
    error: Optional[str] = None

class TaskExecutor:
    def __init__(self, adapter: HostAdapter):
        self.adapter = adapter

    async def execute(self, tasks: list[Task]) -> AsyncGenerator[TaskResult, None]:
        """Run tasks with concurrency managed by adapter."""
        ...
```

### Phase 3: Refactor BatteryRunner

**File:** `prompt_prix/battery.py`

Remove: `BatchRunner(self.adapter.pool)` ← the leak
Remove: `execute_test(test_id, model_id, server_url)` ← LM Studio-specific
Add: Use TaskExecutor with adapter

```python
class BatteryRunner:
    def __init__(self, adapter: HostAdapter, ...):
        self.adapter = adapter  # Not adapter.pool

    async def run(self) -> AsyncGenerator[BatteryRun, None]:
        executor = TaskExecutor(self.adapter)
        tasks = [Task(...) for test in tests for model in models]
        async for result in executor.execute(tasks):
            self._update_state(result)
            yield self.state
```

### Phase 4: Refactor ComparisonSession

**File:** `prompt_prix/core.py`

Remove: `server_pool: ServerPool` parameter ← bypasses adapter
Add: `adapter: HostAdapter` parameter
Use: `adapter.stream_completion()` directly (not TaskExecutor)

```python
class ComparisonSession:
    def __init__(self, adapter: HostAdapter, models: list[str], ...):
        self.adapter = adapter  # Not ServerPool

    async def send_prompt_to_model(self, model_id, prompt, on_chunk):
        await self.adapter.acquire(model_id)
        try:
            async for chunk in self.adapter.stream_completion(...):
                await on_chunk(model_id, chunk)
        finally:
            await self.adapter.release(model_id)
```

### Phase 5: Update handlers

**Files:** `tabs/battery/handlers.py`, `tabs/compare/handlers.py`

```python
# Both handlers
pool = ServerPool(servers)
adapter = LMStudioAdapter(pool)

# Battery
runner = BatteryRunner(adapter=adapter, ...)

# Compare
session = ComparisonSession(adapter=adapter, ...)
```

### Phase 6: Delete dead code

- `scheduler.py`: Remove `BatchRunner` class (replaced by TaskExecutor)
- `core.py`: Keep `stream_completion()` for now (LMStudioAdapter uses it)
- Later: Consider moving `stream_completion()` into LMStudioAdapter

**No HF UI phase** - HF support is for Spaces release, not GitHub release.

---

## Files to Modify

| File | Change | Risk |
|------|--------|------|
| `executor.py` | **NEW** - TaskExecutor | Medium |
| `battery.py` | Remove BatchRunner.pool access, use TaskExecutor | High |
| `core.py` | ComparisonSession takes adapter not ServerPool | High |
| `tabs/compare/handlers.py` | Create adapter, pass to session | Medium |
| `tabs/battery/handlers.py` | Pass adapter to runner | Medium |
| `scheduler.py` | Delete BatchRunner (dead code) | Low |

**Not modified** (deferred to Spaces release):
- `ui.py` - No HF UI for GitHub release
- `adapters/huggingface.py` - Already done, not used in GitHub release

---

## Verification

1. **Unit tests pass:** `pytest tests/`
2. **Battery with LM Studio:** Run tool_competence_tests, verify grid populates correctly
3. **Compare with LM Studio:** Multi-model comparison, streaming works
4. **Manual test:** Start app, run Battery, check grid updates live

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| BatteryRunner refactor breaks retry logic | Keep tenacity retry wrapper, just change how it calls adapter |
| ComparisonSession refactor breaks streaming | Test live streaming after change |
| TaskExecutor concurrency bugs | Keep logic simple, test with multiple models |

---

## Design Decisions (Resolved)

### 1. Streaming: Compare Uses Adapter Directly

**Battery**: Uses TaskExecutor → yields complete TaskResult
**Compare**: Uses adapter.stream_completion() directly → live streaming

Rationale: Simpler. Compare needs live streaming, TaskExecutor is overkill for it.

### 2. No Multi-Adapter Mixing

**GitHub release**: LM Studio only (LMStudioAdapter)
**HF Spaces release**: Will have separate code path for HuggingFace

Rationale: Two deployment targets, two code paths. No composite adapter complexity.

### 3. ServerPool Lifecycle

Handler creates ServerPool, wraps in LMStudioAdapter:
```python
# In handlers
pool = ServerPool(servers)
adapter = LMStudioAdapter(pool)
session = ComparisonSession(adapter=adapter, ...)
```

---

## Rollback Plan

If refactor goes sideways:
- Git revert to pre-refactor state
- Defer HF integration to post-launch

---

## Pre-Refactor Cleanup (27 uncommitted changes)

**Baseline:** 266 passed, 5 failed (export visible issue), 10 deselected

| Commit | Files | Issue |
|--------|-------|-------|
| `chore: Update documentation` | READMEs, CLAUDE.md, TESTING.md, ARCHITECTURE.md, docs/README.md, examples/README.md | - |
| `chore: Move completed ADRs` | Delete 001-003.md, ADR-TAB-001, add docs/adr/completed/ | - |
| `chore: Remove deprecated stability tab` | Delete tabs/stability/*.py | #62 |
| `fix: Export visible state (#41)` | handlers.py (3 lines: 406, 446, 578 → visible=False), tests already updated | #41 |
| `ref #73: Add HostAdapter protocol` | adapters/base.py, lmstudio.py, huggingface.py | #73 |

**Export fix:** Lines 406, 446, 578 in `prompt_prix/tabs/battery/handlers.py` need `visible=True` → `visible=False`

**Test baseline after cleanup should be:** 271 passed, 0 failed
