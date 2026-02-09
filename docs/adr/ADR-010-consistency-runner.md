# ADR-010: Consistency Runner for Multi-Run Statistical Analysis

## Status
Proposed

## Context

Single battery runs hide critical information about model reliability. LLM outputs vary due to:
- Temperature/sampling randomness
- Model non-determinism
- Seed variations

A test that passes once may fail on the next run, or vice versa. Without multi-run data, users cannot distinguish:
- Reliable passes from lucky guesses
- Reliable fails from unlucky samples
- Inherently noisy test/model combinations

Manual re-runs are impractical when battery runs take 20+ minutes.

## Decision

Introduce a **ConsistencyRunner** layer between the UI and BatteryRunner:

```
UI (runs slider, aggregate grid)
       ↓
ConsistencyRunner (new)
       ↓ calls N times with different seeds
BatteryRunner (unchanged)
       ↓
MCP tools → Adapter
```

### New Status Enum

```python
class ConsistencyStatus(Enum):
    CONSISTENT_PASS = "consistent_pass"      # N/N passed
    CONSISTENT_FAIL = "consistent_fail"      # 0/N passed
    INCONSISTENT = "inconsistent"            # 1 to N-1 passed
```

### Grid Display

| Status | Symbol | Tooltip/Detail |
|--------|--------|----------------|
| CONSISTENT_PASS | ✓ | "5/5 passed" |
| CONSISTENT_FAIL | ❌ | "0/5 passed" |
| INCONSISTENT | 🟣 | "3/5 passed" |

### Data Structures

```python
@dataclass
class CellAggregate:
    """Aggregated results for one (test, model) cell across N runs."""
    test_id: str
    model_id: str
    passes: int
    total: int
    results: list[RunResult]  # All individual run results

    @property
    def status(self) -> ConsistencyStatus:
        if self.passes == self.total:
            return ConsistencyStatus.CONSISTENT_PASS
        if self.passes == 0:
            return ConsistencyStatus.CONSISTENT_FAIL
        return ConsistencyStatus.INCONSISTENT

class ConsistencyRun(BaseModel):
    """Source of truth for multi-run battery results."""
    tests: list[str]
    models: list[str]
    runs_per_test: int
    cells: dict[str, CellAggregate]  # key = f"{test_id}:{model_id}"
```

### UI Changes

1. **Slider**: "Runs per test" (1-10, default 1)
2. **Grid**: Show consistency symbols with pass rate
3. **Detail panel**: Expandable list of all N responses
4. **Export**: Include per-run details + aggregate stats

### BatteryRunner Changes

Minimal - just accept and pass through a `seed: int` parameter to completions.

## Consequences

### Positive
- Surfaces reliability information that's currently hidden
- Identifies flaky tests vs flaky models
- Enables data-driven decisions about model selection
- Backwards compatible (runs=1 behaves exactly as today)

### Negative
- N× longer run times (mitigated by parallelization)
- More complex state management
- Larger export files

### Neutral
- New layer adds complexity but follows existing architectural patterns
- ConsistencyRunner can be unit tested independently of BatteryRunner

## Implementation Phases

1. **Data structures**: `CellAggregate`, `ConsistencyRun`, `ConsistencyStatus`
2. **ConsistencyRunner**: Orchestrates N runs, aggregates results
3. **UI**: Slider, grid symbols, detail panel updates
4. **Seed passing**: Thread seed through to adapter completions
