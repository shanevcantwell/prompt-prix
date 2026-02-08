"""Single dispatch function for test case execution.

The ONLY place that reads test.mode. Orchestration above (BatteryRunner,
ConsistencyRunner) has zero mode awareness. MCP tools below have zero
mode awareness.

    mode=None    -> single-shot via complete_stream()
    mode="react" -> react loop via react_step() x N
"""

import logging
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel

from prompt_prix.mcp.tools.complete import complete_stream, parse_latency_sentinel
from prompt_prix.mcp.tools.react_step import react_step
from prompt_prix.react.cycle_detection import detect_cycle_with_pattern
from prompt_prix.react.schemas import ReActIteration

if TYPE_CHECKING:
    from prompt_prix.benchmarks.base import BenchmarkCase

logger = logging.getLogger(__name__)

CYCLE_MIN_REPETITIONS = 3


class CaseResult(BaseModel):
    """Result of executing a single test case (any mode).

    Orchestration consumes response + latency_ms.
    react_trace carries mode-specific details for detail views.
    """
    response: str
    latency_ms: float
    react_trace: Optional[dict] = None


class ReactLoopIncomplete(Exception):
    """Raised when a react loop terminates without completing.

    Carries react_trace so the caller can attach it to RunResult.
    """
    def __init__(self, reason: str, react_trace: dict):
        self.reason = reason
        self.react_trace = react_trace
        super().__init__(reason)


async def execute_test_case(
    test: "BenchmarkCase",
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: int = 300,
    seed: Optional[int] = None,
) -> CaseResult:
    """Execute a single test case, dispatching by mode.

    Args:
        test: BenchmarkCase to execute
        model_id: Model to call
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        timeout_seconds: Timeout per call
        seed: Random seed (consistency runs)

    Returns:
        CaseResult with response, latency, and optional react_trace

    Raises:
        ReactLoopIncomplete: If react loop hits cycle or max_iterations
        Exception: Any infrastructure error from the model call
    """
    if test.mode == "react":
        return await _execute_react(
            test, model_id, temperature, max_tokens, timeout_seconds,
        )
    else:
        return await _execute_single_shot(
            test, model_id, temperature, max_tokens, timeout_seconds, seed,
        )


async def _execute_single_shot(
    test: "BenchmarkCase",
    model_id: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    seed: Optional[int],
) -> CaseResult:
    """Single-shot completion via complete_stream() MCP tool."""
    response = ""
    latency_ms = 0.0

    kwargs = dict(
        model_id=model_id,
        messages=test.to_messages(),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        tools=test.tools,
    )
    if seed is not None:
        kwargs["seed"] = seed

    async for chunk in complete_stream(**kwargs):
        lat = parse_latency_sentinel(chunk)
        if lat is not None:
            latency_ms = lat
        else:
            response += chunk

    return CaseResult(response=response, latency_ms=latency_ms)


async def _execute_react(
    test: "BenchmarkCase",
    model_id: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> CaseResult:
    """React loop via react_step() MCP tool.

    Runs the loop, manages trace accumulation, stagnation detection,
    and max_iterations enforcement. Extracted from ReactRunner._execute_one().
    """
    trace: list[ReActIteration] = []
    call_counter = 0
    total_latency_ms = 0.0
    final_response = None
    termination_reason = None
    cycle_detected = False
    cycle_pattern = None
    valid_iterations = 0
    invalid_iterations = 0

    for _ in range(test.max_iterations):
        step = await react_step(
            model_id=model_id,
            system_prompt=test.system,
            initial_message=test.user,
            trace=trace,
            mock_tools=test.mock_tools or {},
            tools=test.tools or [],
            call_counter=call_counter,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

        call_counter = step["call_counter"]
        total_latency_ms += step["latency_ms"]

        if step["completed"]:
            final_response = step["final_response"]
            break

        # Accumulate iterations from this step
        for new_iter in step["new_iterations"]:
            trace.append(new_iter)
            if new_iter.success:
                valid_iterations += 1
            else:
                invalid_iterations += 1

        # Check stagnation
        if len(trace) >= CYCLE_MIN_REPETITIONS * 2:
            signatures = [
                (s.tool_call.name, tuple(sorted(s.tool_call.args.items())))
                for s in trace
            ]
            period, pattern = detect_cycle_with_pattern(
                signatures, min_repetitions=CYCLE_MIN_REPETITIONS
            )
            if period is not None:
                cycle_detected = True
                cycle_pattern = [
                    {"name": name, "args": dict(args)}
                    for name, args in pattern
                ] if pattern else None
                termination_reason = "cycle_detected"
                logger.info(
                    "Stagnation detected for %s/%s: period=%d",
                    test.id, model_id, period,
                )
                break
    else:
        # Loop exhausted without break -> max iterations
        termination_reason = "max_iterations"

    total_iterations = len(trace)
    react_trace = {
        "completed": final_response is not None,
        "total_iterations": total_iterations,
        "valid_iterations": valid_iterations,
        "invalid_iterations": invalid_iterations,
        "cycle_detected": cycle_detected,
        "cycle_pattern": cycle_pattern,
        "termination_reason": termination_reason,
        "iterations": [step.model_dump() for step in trace],
    }

    if final_response is None:
        # Loop didn't complete — raise so caller maps to SEMANTIC_FAILURE
        reason = f"React loop terminated: {termination_reason}"
        if cycle_detected:
            reason = f"React loop stagnated (cycle period {period})"
        raise ReactLoopIncomplete(reason=reason, react_trace=react_trace)

    return CaseResult(
        response=final_response,
        latency_ms=total_latency_ms,
        react_trace=react_trace,
    )
