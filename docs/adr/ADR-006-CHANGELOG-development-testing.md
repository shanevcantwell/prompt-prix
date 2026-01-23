# Changelog: development/testing Branch

Historical record of work completed on the `development/testing` branch during the HuggingFace Space release preparation.

---

## 2026-01-23: Test Coverage + Server Affinity

### Commits
- `233847b` - test: Server affinity parsing + LMStudioAdapter boundary tests (35 tests)
- `e977556` - refactor: Rename Test* classes to avoid pytest collection warnings

### Changes
- **Class renames** to fix pytest collection warnings:
  - `TestCase` → `BenchmarkCase`
  - `TestStatus` → `RunStatus`
  - `TestResult` → `RunResult`
- **New test files:**
  - `tests/test_server_affinity.py` - 23 tests for prefix parsing
  - `tests/test_lmstudio_adapter.py` - 12 tests for adapter boundary conditions
- **New module:** `prompt_prix/server_affinity.py` - shared prefix utilities

---

## 2026-01-22: Multi-GPU Parallel Execution

### Commits
- `609b6be` - feat: Complete multi-GPU parallel execution with server affinity
- `faea261` - feat: Column-per-GPU model selector (#96)
- `903dbc7` - fix: Battery loop order + adapter race condition
- `5624532` - fix: Battery serialization + export auto-download

### Features Implemented

**Server Affinity Routing:**
```
UI Selection:
  Server 0 column: [modelA, modelB]     → "0:modelA", "0:modelB"
  Server 1 column: [modelC, modelD]     → "1:modelC", "1:modelD"

BatteryRunner receives prefixed model IDs
Adapter routes to correct server based on prefix
```

**Loop Order Fix (VRAM optimization):**
- Changed from breadth-first to depth-first execution
- Before: test1→model1, test1→model2, test2→model1 (VRAM thrashing)
- After: model1→test1, model1→test2, model2→test1 (model stays loaded)

**Race Condition Fix:**
- Made find_available_server + acquire_server atomic within lock
- Eliminated "No server available" errors during model swapping

**Export Improvements:**
- Timestamped filenames (no browser caching issues)
- CSV uses `csv` module with proper quoting
- Added `failure_reason` field to exports
- Auto-download via Gradio file component

---

## 2026-01-21: ADR-006 Architecture Alignment

### Commits
- `6010b71` - test: ADR-006 test suite alignment
- `e2c4a26` - refactor: Compare tab ADR-006 alignment
- `b79ff2c` - refactor: ADR-006 Battery tab - MCP registry pattern

### Architecture Changes

Implemented three-layer architecture per ADR-006:

```
ORCHESTRATION (BatteryRunner, ComparisonSession)
    │ calls MCP tools only
    ↓
MCP PRIMITIVES (complete, complete_stream, list_models, judge)
    │ stateless, agentic-ready
    ↓
ADAPTER (LMStudioAdapter owns ServerPool, httpx)
```

**Key Rules Enforced:**
- Orchestration NEVER imports from `adapters/*`
- ServerPool and ConcurrentDispatcher are INTERNAL to LMStudioAdapter
- MCP tools receive adapter via registry (`get_adapter()`)

---

## Issues Resolved

| Issue | Description | Commit |
|-------|-------------|--------|
| #96 | Column-per-GPU model selector | `faea261` |
| #97 | Layer violation in handlers.py | `b79ff2c` |
| #101 | Variable naming confusion (model_id vs actual_model_id) | `609b6be` |

---

## Test Suite Status

- **275 unit tests passing**
- **9 integration tests** (require LM Studio, deselected by default)
- Integration tests cover:
  - Basic tool call execution
  - JSON/CSV export format
  - Multi-GPU parallel dispatch timing
  - Tool call fragmentation regression

---

## Files Modified (Summary)

| File | Change Type |
|------|-------------|
| `prompt_prix/battery.py` | Loop order, class renames |
| `prompt_prix/adapters/lmstudio.py` | Server affinity routing, atomic acquire |
| `prompt_prix/tabs/battery/handlers.py` | Export fixes, prefix handling |
| `prompt_prix/tabs/battery/ui.py` | Column-per-GPU selector |
| `prompt_prix/ui.py` | Model prefixing, export download |
| `prompt_prix/server_affinity.py` | NEW - shared utilities |
| `prompt_prix/benchmarks/base.py` | Class renames |
| `tests/test_server_affinity.py` | NEW - 23 tests |
| `tests/test_lmstudio_adapter.py` | NEW - 12 tests |
| `tests/integration/test_battery_integration.py` | Parallelism verification |

---

## Deferred Work

- CLI script tests (`scripts/run_battery.py`)
- ConcurrentDispatcher timing verification
- Export format validation for large datasets
- Judge integration into battery workflow
- HFInferenceAdapter (second provider)
