# ADR-008: Judge Scheduling Strategy for Multi-GPU Battery Runs

**Status:** Accepted
**Related:** #111, #107
**Implemented:** `battery.py` two-phase execution

## Context

prompt-prix runs battery tests across multiple models on multiple GPUs. When LLM-as-judge evaluation is enabled, each `(model, test)` result must also be evaluated by a judge model.

**Current behavior:**
- Test inferences and judge requests share the same dispatcher queue
- Judge requests compete with test inferences for GPU time
- When GPUs are saturated with test inferences, judge requests timeout

**Observed failure mode:**
- 5 models × 15 tests = 75 test inferences
- 2 GPUs running test inferences continuously
- Judge requests (using `google/gemma-3-12b`) queue behind test requests
- Judge timeout (300s) expires while waiting for GPU availability
- Grid fills with ⚠️ timeout errors for judgment

## Decision Drivers

1. **Throughput**: Minimize total battery runtime
2. **Reliability**: Judge evaluations should not fail due to resource contention
3. **Simplicity**: Avoid complex priority scheduling if simpler solutions exist
4. **Hardware utilization**: Both GPUs should stay busy when work exists
5. **Cognitive sovereignty**: User controls whether judgment happens locally or externally

### Design Philosophy Context

prompt-prix exists to enable informed model selection - structural validation of which open-weights models can handle specific tasks. The judge is another validation layer. Whether that judge runs locally (maintaining full sovereignty) or externally (trading sovereignty for capability/convenience) should be a deliberate user choice, not an accident of GPU contention.

## Options Considered

### Option A: Batch Judging (Two-Phase Execution)

Run all test inferences first, then run all judge evaluations in a second pass.

```
Phase 1: Run all (model, test) inferences → store results
Phase 2: Run all judge evaluations on stored results → update verdicts
```

**Pros:**
- No contention between test and judge requests
- Simpler mental model (clear phases)
- Judge can be on either GPU without blocking tests
- Natural fit for #107 (separate timing metrics)

**Cons:**
- Results appear incomplete until Phase 2 finishes
- Must store all test outputs in memory during Phase 1
- Longer time-to-first-judgment (user sees raw results before verdicts)

### Option B: Dedicated Judge GPU

Reserve one GPU for judge evaluations, use the other for test inferences.

```
GPU 0: Test inferences only
GPU 1: Judge evaluations only (judge model stays loaded)
```

**Pros:**
- Judge never waits for test inference GPU
- Judge model stays hot (no VRAM swapping)
- Real-time verdicts as tests complete

**Cons:**
- Halves test inference throughput (only 1 GPU for tests)
- Requires judge model to fit on single GPU
- Wastes judge GPU when no judgments pending
- Configuration complexity (which GPU for what?)

### Option C: Priority Queue

Add priority levels to dispatcher. Judge requests get higher priority.

```
Queue: [judge_request (high), test_request (normal), test_request (normal), ...]
```

**Pros:**
- Judge requests handled promptly
- Both GPUs available for tests when no judge work
- Minimal architectural change

**Cons:**
- Judge requests can starve test requests
- Priority inversion risk
- More complex dispatcher logic
- Doesn't solve fundamental contention

### Option D: Interleaved Scheduling

After each test inference completes, immediately judge it before starting next test.

```
test1 → judge1 → test2 → judge2 → test3 → judge3 ...
```

**Pros:**
- Guaranteed judge slot after each test
- Predictable timing
- Results complete one at a time

**Cons:**
- Serializes execution (loses parallelism benefit)
- Much slower total runtime
- GPU idle between operations

### Option E: Judge Request Retry with Backoff

Keep current architecture but add retry logic for judge timeouts.

**Pros:**
- Minimal code change
- Eventually succeeds when GPUs free up

**Cons:**
- Doesn't solve root cause
- Unpredictable completion time
- Still poor UX (errors then eventual success)

## Questions for Review

1. **Is two-phase (Option A) acceptable UX?** Users see test results immediately but verdicts populate later. Is "results first, verdicts second" confusing or actually clearer?

2. **How common is the judge model also being a test model?** If `gemma-3-12b` is both judge AND under test, Option B dedicates a GPU to a model that also needs testing.

3. **Should judge be same model family as tests?** Using a different model (e.g., Claude API) for judging would eliminate GPU contention entirely but adds external dependency and cost.

4. **Memory constraints?** Option A requires holding all test outputs in memory. For 100 tests × 10 models × ~4KB output = ~4MB. Probably fine, but worth confirming.

5. **Is real-time verdict feedback valuable?** Or is "run tests, then see all verdicts" acceptable?

## Decision

**Option A (Batch Judging)** selected:
- Eliminates contention without complex scheduling
- Aligns with battery's batch nature
- Supports #107 timing separation naturally
- Maintains architectural purity (no priority logic in dispatcher)
- Keeps `BatteryRunner` as pure orchestrator

### Why NOT Option B (Dedicated GPU)

Option B would force `LMStudioAdapter` to become "Role-Aware" (test GPU vs judge GPU), leaking infrastructure configuration into the orchestration layer. This violates the layer separation established in ADR-006 and would undo the work from #110.

## Implementation

Two-phase execution in `battery.py`:

```python
async def run(self) -> AsyncGenerator[BatteryRun, None]:
    # PHASE 1: Inference (The "Hands")
    async for state in self._execute_inference_phase():
        yield state  # User sees ✓/⚠️/❌ appearing

    # PHASE 2: Judgment (The "Brain")
    if self.judge_model and not app_state.should_stop():
        async for state in self._execute_judgment_phase():
            yield state  # User sees verdicts populating
```

Key methods:
- `_execute_inference_phase()`: Runs all test inferences concurrently
- `_execute_judgment_phase()`: Judges all COMPLETED results with criteria
- `_execute_test()`: Inference + semantic validation only (no judging)
- `_judge_single_result()`: Single judge evaluation via MCP primitive

Design decisions:
- No new `RunStatus` enum needed - `COMPLETED` + `judge_result` field
- Failed tests (semantic validation) are NOT judged
- Results visible immediately, verdicts populate in second pass

## References

- #111: Judge model requests timeout during battery runs
- #107: Separate judge timing from model-under-test latency
- #110: Server affinity removal (current refactor)
