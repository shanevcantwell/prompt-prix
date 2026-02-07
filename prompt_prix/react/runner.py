"""ReactRunner — orchestrates ReAct loop evaluation across models.

Mirrors BatteryRunner pattern (ADR-006 orchestration layer):
- Defines WHAT to run (react tests × models)
- Calls react_step() MCP tool in a loop — never adapters directly
- Owns loop control: max_iterations, stagnation detection
- Model-first ordering for VRAM efficiency
- Yields state snapshots for live UI updates
"""

import logging
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, ConfigDict

from prompt_prix.mcp.tools.react_step import react_step
from prompt_prix.react.cycle_detection import detect_cycle_with_pattern
from prompt_prix.react.schemas import ReActIteration

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# STATE MODELS
# ─────────────────────────────────────────────────────────────────────


class ReactResult(BaseModel):
    """Result of a complete ReAct loop for one (test, model) cell."""

    test_id: str = ""
    model_id: str = ""
    completed: bool = False
    final_response: Optional[str] = None
    total_iterations: int = 0
    valid_iterations: int = 0
    invalid_iterations: int = 0
    completion_rate: float = 0.0
    total_latency_ms: float = 0.0
    cycle_detected: bool = False
    cycle_pattern: Optional[list] = None
    termination_reason: Optional[str] = None
    iterations: list[dict] = []
    error: Optional[str] = None

    @property
    def display_cell(self) -> str:
        """Summary for grid cell display."""
        if self.error:
            return f"⚠ Error"

        if self.completed:
            return f"✓ {self.valid_iterations}/{self.total_iterations}"

        if self.cycle_detected:
            return f"⟳ {self.valid_iterations}/{self.total_iterations}"

        if self.termination_reason == "max_iterations":
            return f"⚠ {self.valid_iterations}/{self.total_iterations}"

        return f"? {self.valid_iterations}/{self.total_iterations}"


class ReactRun(BaseModel):
    """State for a complete react battery run. Source of truth for grid UI."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tests: list[str]
    models: list[str]
    results: dict[str, ReactResult] = {}

    def get_key(self, test_id: str, model_id: str) -> str:
        return f"{test_id}:{model_id}"

    def get_result(self, test_id: str, model_id: str) -> Optional[ReactResult]:
        return self.results.get(self.get_key(test_id, model_id))

    def set_result(self, result: ReactResult) -> None:
        key = self.get_key(result.test_id, result.model_id)
        self.results[key] = result

    def to_grid(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame for Gradio display."""
        import pandas as pd

        data = []
        for test_id in self.tests:
            row = {"Test": test_id}
            for model_id in self.models:
                result = self.get_result(test_id, model_id)
                row[model_id] = result.display_cell if result else "—"
            data.append(row)

        return pd.DataFrame(data)

    @property
    def completed_count(self) -> int:
        return len(self.results)

    @property
    def total_count(self) -> int:
        return len(self.tests) * len(self.models)


# ─────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────


class ReactRunner:
    """Orchestrates react_step() across test × model matrix.

    Owns the ReAct loop: calls react_step() (stateless MCP primitive)
    repeatedly, manages trace accumulation, stagnation detection,
    and max_iterations enforcement.

    Model-first ordering: all tests for model A, then model B, etc.
    Minimizes VRAM swaps on multi-GPU setups.
    """

    CYCLE_MIN_REPETITIONS = 3

    def __init__(
        self,
        tests: list,  # list[BenchmarkCase] with mode="react"
        models: list[str],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 300,
    ):
        self.tests = tests
        self.models = models
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

        self.state = ReactRun(
            tests=[t.id for t in tests],
            models=models,
        )

    async def run(self) -> AsyncGenerator[ReactRun, None]:
        """Execute all react tests across models. Yields state snapshots."""

        # Model-first ordering for VRAM efficiency
        for model_id in self.models:
            for test in self.tests:
                await self._execute_one(test, model_id)
                yield self.state

        # Final yield
        yield self.state

    async def _execute_one(self, test, model_id: str) -> None:
        """Execute one (test, model) cell — owns the ReAct loop."""
        try:
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
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
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
                if len(trace) >= self.CYCLE_MIN_REPETITIONS * 2:
                    signatures = [
                        (s.tool_call.name, tuple(sorted(s.tool_call.args.items())))
                        for s in trace
                    ]
                    period, pattern = detect_cycle_with_pattern(
                        signatures, min_repetitions=self.CYCLE_MIN_REPETITIONS
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
                # Loop exhausted without break → max iterations
                termination_reason = "max_iterations"

            total_iterations = len(trace)
            completed = final_response is not None

            self.state.set_result(ReactResult(
                test_id=test.id,
                model_id=model_id,
                completed=completed,
                final_response=final_response,
                total_iterations=total_iterations,
                valid_iterations=valid_iterations,
                invalid_iterations=invalid_iterations,
                completion_rate=valid_iterations / total_iterations if total_iterations > 0 else 0.0,
                total_latency_ms=total_latency_ms,
                cycle_detected=cycle_detected,
                cycle_pattern=cycle_pattern,
                termination_reason=termination_reason,
                iterations=[step.model_dump() for step in trace],
            ))

        except Exception as e:
            logger.error("react loop failed for %s/%s: %s", test.id, model_id, e)
            self.state.set_result(ReactResult(
                test_id=test.id,
                model_id=model_id,
                error=str(e),
            ))
