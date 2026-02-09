# ADR-002: Fan-Out Pattern as Core Architecture

**Status**: Accepted
**Date**: 2025-11-28

## Context

prompt-prix evolved from a simple model comparison tool into a more general-purpose evaluation viewer. The question arose: what is the **core abstraction** that defines this tool's identity?

## Decision

**The fan-out pattern is the primary architectural abstraction.**

Fan-out means:
- **One input** (prompt, test case, or benchmark item)
- **N outputs** (responses from multiple models in parallel)
- **Visual comparison** (side-by-side display with status indicators)

## Rationale

### Differentiating Pattern
Existing eval frameworks (BFCL, Inspect AI) focus on:
- Running benchmarks at scale
- Computing aggregate metrics
- Producing leaderboards

None of them provide **visual, real-time comparison** of individual responses. The fan-out pattern fills this gap.

### Natural Fit for Model Selection
When selecting a model for an agentic workflow, users need to:
1. See how different models respond to the same prompt
2. Compare qualitative differences (not just pass/fail)
3. Test edge cases interactively

Fan-out directly serves these needs.

### Implementation Simplicity
The fan-out pattern maps cleanly to:
- Concurrent dispatcher (parallel execution)
- Tab-based UI (one tab per model)
- Streaming display (real-time feedback)

## Consequences

### Positive
- Clear product identity: "fan-out comparison UI"
- UI/UX focused on comparison, not eval authoring
- Simple mental model for users

### Negative
- Not suited for batch evaluation with metrics
- No built-in pass/fail grading
- Users must bring their own eval logic for programmatic assessment

### Architecture Implications
- `ServerPool` manages parallel execution across GPUs
- `ComparisonSession` maintains isolated context per model
- Concurrent dispatcher maximizes GPU utilization during fan-out

## Examples

### Typical Fan-Out Flow
```
User prompt: "Get the weather in Tokyo"
    │
    ├──► Model A (Server 1) → Response A
    ├──► Model B (Server 1) → Response B
    ├──► Model C (Server 2) → Response C
    │
    └──► UI displays all three side-by-side
```

### Integration with Benchmarks
```
BFCL test case → prompt-prix → Fan-out to N models → Visual comparison
                                (human judges which responses are best)
```
