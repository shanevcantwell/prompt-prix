# ADR-REFACTOR-001: Architecture Cleanup Plan

**Status:** Accepted
**Date:** 2026-01-19
**Supersedes:** ADR-073 (adapter refactor plan)

---

## Context

prompt-prix has grown organically with working functionality but tangled code structure. The adapter pattern exists but leaks implementation details. Adding new adapters (HuggingFace) revealed that the abstraction is incomplete.

Key pain points:
- Battery uses adapter but leaks (`adapter.pool` accessed directly)
- Compare bypasses adapter layer entirely
- Large files with mixed concerns (handlers.py at 615 lines)
- MCP primitives (`complete`, `list_models`) don't exist yet

The project is part of a larger ecosystem where `langgraph-agentic-scaffold` (LAS) is the flagship orchestrator. prompt-prix serves as an **evaluation service** - LAS produces training data, prompt-prix evaluates model performance, results inform LAS routing.

---

## Decision

Align code structure to a clean layered architecture with MCP primitives as the integration contract.

### Architecture Target

```
┌─────────────────────────────────────────┐
│  MCP Tools (the contract)               │
│  complete · judge · list_models         │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Adapters                               │
│  LMStudio · (HuggingFace) · (SurfMCP)   │
└─────────────────────────────────────────┘

Consumers (all call MCP tools, no adapter access):
- Gradio UI (Battery tab, Compare tab)
- Future: LAS and other agentic systems
```

### Design Principles

1. **MCP primitives are the contract** - `complete`, `judge`, `list_models` usable by both Gradio UI and future agents
2. **Orchestration calls primitives** - Battery/Compare are consumers, not peers with privileged access
3. **Broad and shallow** - ~350 lines max per file, single responsibility
4. **Traceable execution** - Follow a Battery run by reading function names, not debugging

---

## Current State Analysis

### What Works
- Flow: URLs → Fetch → Select models → Run battery → Results grid
- GPU parallelization logic is sound
- ~10% dead code estimated

### The Three-Level Problem (Issue #73)

```
LEVEL 1: HostAdapter Protocol    ← What interface says
LEVEL 2: Implementation Reality  ← Code reaches for adapter.pool
LEVEL 3: Hardcoded Assumptions   ← stream_completion() baked for LM Studio
```

### Call Path Reality

**Battery** (uses adapter but leaks):
```
battery/handlers.py → LMStudioAdapter(pool)
    → BatteryRunner(adapter=...)
    → battery.py:316: runner = BatchRunner(self.adapter.pool)  ← LEAK
```

**Compare** (bypasses adapter entirely):
```
compare/handlers.py → ServerPool(servers)  ← NO ADAPTER
    → ComparisonSession(server_pool=...)
    → stream_completion(server_url=...)
```

### MCP Tools Gap

Currently in `mcp/tools/`:
- `judge.py` ✓ exists
- `complete.py` ✗ missing
- `list_models.py` ✗ missing

### Files Over 350-Line Guideline

| File | Lines | Issue |
|------|-------|-------|
| `tabs/battery/handlers.py` | 615 | #61: mixed concerns |
| `battery.py` | 539 | Orchestration + leaky adapter access |
| `tabs/compare/handlers.py` | 424 | Bypasses adapters |

---

## Ecosystem Context

**LAS is the orchestrator. prompt-prix is the evaluation service.**

```
LAS (flagship)
  │ Captures training data from specialist execution
  ▼
prompt-prix (evaluation)
  │ Evaluates model performance
  ▼
Results inform LAS specialist→model bindings
```

**Feedback loop:**
1. LAS runs workflows, captures specialist decisions
2. prompt-prix evaluates: "which model performed best on this task?"
3. Results inform specialist LLM bindings
4. Better routing → better outcomes → better training data

LAS doesn't need real-time orchestration from prompt-prix - it has its own.
It needs **offline batch evaluation**.

---

## MCP Primitive Design

| Primitive | Signature | Consumer Use Case |
|-----------|-----------|-------------------|
| `complete` | `(model_id, messages, params) → response` | Single model completion |
| `judge` | `(response, criteria, judge_model) → {pass, reason}` | Semantic evaluation |
| `list_models` | `(servers, only_loaded?) → model_id[]` | Discovery |
| `run_battery` | `(test_suite, models) → results_grid` | Batch evaluation (TBD) |

The Gradio UI becomes one consumer. LAS (via batch export/import) becomes another.

---

## Implementation Plan

### Recommended Sequence

1. **Spike: `list_models` as MCP tool** - smallest primitive, proves the pattern
2. **Fix #92** - move connection error handling into adapter layer
3. **Split battery/handlers.py** - extract validators, exporters, grid utils per #61

### What NOT to Do

- Don't rewrite from scratch - the orchestration logic is sound
- Don't chase HuggingFace adapter until adapter layer is clean
- Don't add features until MCP primitives exist

---

## Key Files Reference

```
prompt_prix/
├── mcp/tools/         # Where primitives should live (only judge.py exists)
├── adapters/          # LMStudio works, HF blocked by leaky abstraction
├── tabs/battery/handlers.py  # 615 lines, needs split
├── battery.py         # 539 lines, leaks adapter.pool
└── tabs/compare/handlers.py  # 424 lines, bypasses adapters entirely
```

---

## Related Issues

| Issue | Summary | Status |
|-------|---------|--------|
| #4 | Adapter pattern (original vision) | OPEN |
| #61 | battery/handlers.py complexity | OPEN |
| #73 | Consolidate adapter routing | CLOSED (documented, not fixed) |
| #92 | Adapter bypass in handlers | OPEN |

---

## Open Questions

- [ ] Should `run_battery` be an MCP primitive or just orchestration?
- [ ] How does training data flow from LAS → prompt-prix? (file export? MCP call?)

---

## Consequences

**Positive:**
- Clean integration point for ecosystem (MCP primitives)
- Enables multiple adapters without leaky abstractions
- Smaller files are easier to maintain and test
- Clear layering makes debugging tractable

**Negative:**
- Refactoring effort before new features
- May discover hidden coupling during migration

**Neutral:**
- Gradio UI becomes "just another consumer" rather than privileged caller
