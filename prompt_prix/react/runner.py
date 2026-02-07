"""ReactRunner — orchestrates ReAct loop evaluation across models.

Mirrors BatteryRunner pattern (ADR-006 orchestration layer):
- Defines WHAT to run (react tests × models)
- Calls react_execute() MCP tool — never adapters directly
- Model-first ordering for VRAM efficiency
- Yields state snapshots for live UI updates
"""

import logging
from typing import AsyncGenerator, Optional

from pydantic import BaseModel, ConfigDict

from prompt_prix.mcp.tools.react_execute import react_execute

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# STATE MODELS
# ─────────────────────────────────────────────────────────────────────


class ReactResult(BaseModel):
    """Result of a single react_execute() call for one (test, model) cell."""

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
    """Orchestrates react_execute() across test × model matrix.

    Model-first ordering: all tests for model A, then model B, etc.
    Minimizes VRAM swaps on multi-GPU setups.
    """

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
        """Execute one (test, model) cell."""
        try:
            raw = await react_execute(
                model_id=model_id,
                system_prompt=test.system,
                initial_message=test.user,
                mock_tools=test.mock_tools or {},
                tools=test.tools or [],
                max_iterations=test.max_iterations,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds,
            )

            self.state.set_result(ReactResult(
                test_id=test.id,
                model_id=model_id,
                completed=raw["completed"],
                final_response=raw.get("final_response"),
                total_iterations=raw["total_iterations"],
                valid_iterations=raw["valid_iterations"],
                invalid_iterations=raw["invalid_iterations"],
                completion_rate=raw["completion_rate"],
                total_latency_ms=raw["total_latency_ms"],
                cycle_detected=raw["cycle_detected"],
                cycle_pattern=raw.get("cycle_pattern"),
                termination_reason=raw.get("termination_reason"),
                iterations=raw["iterations"],
            ))

        except Exception as e:
            logger.error("react_execute failed for %s/%s: %s", test.id, model_id, e)
            self.state.set_result(ReactResult(
                test_id=test.id,
                model_id=model_id,
                error=str(e),
            ))
