"""
ConsistencyRunner - Multi-run variance testing for model reliability.

Runs the same tests N times with different seeds to identify models
that produce inconsistent results. Extends Battery functionality
without modifying BatteryRunner.

Usage:
    runner = ConsistencyRunner(tests, models, runs=5, ...)
    async for state in runner.run():
        grid = state.to_grid()
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from prompt_prix.battery import (
    BatteryWorkItem,
    RunStatus,
    RunResult,
    GridDisplayMode,
    validate_response,
    is_retryable_error,
    CancelledError,
)
from prompt_prix.mcp.tools.complete import complete_stream
from prompt_prix.mcp.tools.judge import judge
from prompt_prix.semantic_validator import validate_response_semantic
from prompt_prix import state as app_state

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from prompt_prix.config import get_retry_attempts, get_retry_min_wait, get_retry_max_wait
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from prompt_prix.benchmarks.base import BenchmarkCase


class ConsistencyStatus(str, Enum):
    """Aggregate status across multiple runs."""
    CONSISTENT_PASS = "consistent_pass"    # N/N passed
    CONSISTENT_FAIL = "consistent_fail"    # 0/N passed
    INCONSISTENT = "inconsistent"          # 1 to N-1 passed
    PENDING = "pending"                    # Not all runs complete


@dataclass
class CellAggregate:
    """Aggregated results for one (test, model) cell across N runs."""
    test_id: str
    model_id: str
    passes: int = 0
    errors: int = 0
    total: int = 0
    results: list[RunResult] = field(default_factory=list)

    @property
    def status(self) -> ConsistencyStatus:
        """Compute aggregate status from results."""
        if self.total == 0 or len(self.results) < self.total:
            return ConsistencyStatus.PENDING
        if self.passes == self.total:
            return ConsistencyStatus.CONSISTENT_PASS
        if self.passes == 0:
            return ConsistencyStatus.CONSISTENT_FAIL
        return ConsistencyStatus.INCONSISTENT

    @property
    def status_symbol(self) -> str:
        """UI symbol for aggregate status."""
        status = self.status
        if status == ConsistencyStatus.CONSISTENT_PASS:
            return "✓"
        elif status == ConsistencyStatus.CONSISTENT_FAIL:
            return "❌"
        elif status == ConsistencyStatus.INCONSISTENT:
            return "🟣"
        else:
            return "⏳"

    @property
    def pass_rate_display(self) -> str:
        """Display string like '3/5' for tooltips."""
        return f"{self.passes}/{self.total}"

    @property
    def avg_latency_ms(self) -> Optional[float]:
        """Average latency across successful runs."""
        latencies = [r.latency_ms for r in self.results if r.latency_ms is not None]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    def get_display(self, mode: GridDisplayMode) -> str:
        """Get display string for the specified mode."""
        if self.status == ConsistencyStatus.PENDING:
            completed = len(self.results)
            return f"⏳ {completed}/{self.total}"

        if mode == GridDisplayMode.LATENCY:
            avg = self.avg_latency_ms
            if avg is not None:
                return f"{avg / 1000:.1f}s"
            return "—"

        # Symbols mode - show status with pass rate for inconsistent
        if self.status == ConsistencyStatus.INCONSISTENT:
            return f"🟣 {self.pass_rate_display}"
        return self.status_symbol


class ConsistencyRun(BaseModel):
    """
    State for a multi-run consistency test.

    Like BatteryRun but stores CellAggregate instead of RunResult.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tests: list[str]
    models: list[str]
    runs_total: int = 1
    runs_completed: int = 0
    aggregates: dict[str, CellAggregate] = {}  # key = "{test_id}:{model_id}"

    # Phase tracking
    phase: str = "inference"
    judge_total: int = 0
    judge_completed: int = 0

    def get_key(self, test_id: str, model_id: str) -> str:
        """Generate key from test and model IDs."""
        return f"{test_id}:{model_id}"

    def get_aggregate(self, test_id: str, model_id: str) -> Optional[CellAggregate]:
        """Get aggregate for a specific cell."""
        return self.aggregates.get(self.get_key(test_id, model_id))

    def ensure_aggregate(self, test_id: str, model_id: str) -> CellAggregate:
        """Get or create aggregate for a cell."""
        key = self.get_key(test_id, model_id)
        if key not in self.aggregates:
            self.aggregates[key] = CellAggregate(
                test_id=test_id,
                model_id=model_id,
                total=self.runs_total
            )
        return self.aggregates[key]

    def add_result(self, result: RunResult) -> None:
        """Add a single run result to its aggregate."""
        agg = self.ensure_aggregate(result.test_id, result.model_id)
        agg.results.append(result)

        if result.status == RunStatus.COMPLETED:
            agg.passes += 1
        elif result.status == RunStatus.ERROR:
            agg.errors += 1
        # SEMANTIC_FAILURE counts as neither pass nor error

    def to_grid(self, display_mode: GridDisplayMode = GridDisplayMode.SYMBOLS) -> "pd.DataFrame":
        """Convert to pandas DataFrame for Gradio display."""
        import pandas as pd

        data = []
        for test_id in self.tests:
            row = {"Test": test_id}
            for model_id in self.models:
                agg = self.get_aggregate(test_id, model_id)
                if agg:
                    row[model_id] = agg.get_display(display_mode)
                else:
                    row[model_id] = "—"
            data.append(row)

        return pd.DataFrame(data)

    @property
    def completed_count(self) -> int:
        """Count of completed cells (all runs done)."""
        return sum(
            1 for agg in self.aggregates.values()
            if len(agg.results) >= agg.total
        )

    @property
    def total_count(self) -> int:
        """Total cells."""
        return len(self.tests) * len(self.models)

    @property
    def total_runs(self) -> int:
        """Total individual runs across all cells."""
        return self.total_count * self.runs_total

    @property
    def completed_runs(self) -> int:
        """Completed individual runs."""
        return sum(len(agg.results) for agg in self.aggregates.values())


class ConsistencyRunner:
    """
    Orchestrates multi-run consistency testing.

    Runs each (test, model) combination N times with different seeds,
    aggregating results to identify inconsistent models.
    """

    def __init__(
        self,
        tests: list["BenchmarkCase"],
        models: list[str],
        runs: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 300,
        judge_model: Optional[str] = None,
    ):
        """
        Initialize consistency runner.

        Args:
            tests: List of BenchmarkCase objects
            models: List of model IDs
            runs: Number of runs per (test, model) cell
            temperature: Sampling temperature (default 0.0)
            max_tokens: Max tokens per response
            timeout_seconds: Timeout per request
            judge_model: Optional model for semantic evaluation
        """
        self.tests = tests
        self.models = models
        self.runs = runs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.judge_model = judge_model

        # Generate seeds upfront for reproducibility
        self.seeds = [random.randint(0, 2**31 - 1) for _ in range(runs)]

        # Initialize state
        self.state = ConsistencyRun(
            tests=[t.id for t in tests],
            models=models,
            runs_total=runs
        )

    async def run(self) -> AsyncGenerator[ConsistencyRun, None]:
        """
        Execute all tests across all models with N runs each.

        Phase 1: All inferences (N runs per cell)
        Phase 2: Judge evaluation (if judge_model set)

        Yields state snapshots for UI updates.
        """
        # PHASE 1: Inference
        async for state in self._execute_inference_phase():
            yield state

        # PHASE 2: Judgment
        if self.judge_model and not app_state.should_stop():
            async for state in self._execute_judgment_phase():
                yield state

        yield self.state

    async def _execute_inference_phase(self) -> AsyncGenerator[ConsistencyRun, None]:
        """Execute all inference runs."""
        # Build work items: (test, model, run_index, seed)
        work_items = []
        for model_id in self.models:
            for test in self.tests:
                for run_idx, seed in enumerate(self.seeds):
                    work_items.append((test, model_id, run_idx, seed))

        active_tasks: set[asyncio.Task] = set()
        yield self.state

        for test, model_id, run_idx, seed in work_items:
            if app_state.should_stop():
                break
            task = asyncio.create_task(
                self._execute_single_run(test, model_id, run_idx, seed)
            )
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks,
                timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    task.result()
                except Exception:
                    pass

            yield self.state

            if app_state.should_stop():
                for task in active_tasks:
                    task.cancel()
                break

    async def _execute_judgment_phase(self) -> AsyncGenerator[ConsistencyRun, None]:
        """Judge all completed results that have criteria."""
        results_to_judge = []
        for agg in self.state.aggregates.values():
            for result in agg.results:
                if result.status == RunStatus.COMPLETED and self._needs_judging(result):
                    results_to_judge.append(result)

        if not results_to_judge:
            return

        self.state.phase = "judging"
        self.state.judge_total = len(results_to_judge)
        self.state.judge_completed = 0

        active_tasks: set[asyncio.Task] = set()

        for result in results_to_judge:
            if app_state.should_stop():
                break
            task = asyncio.create_task(self._judge_single_result(result))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks,
                timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                try:
                    task.result()
                except Exception:
                    pass

            yield self.state

            if app_state.should_stop():
                for task in active_tasks:
                    task.cancel()
                break

    def _needs_judging(self, result: RunResult) -> bool:
        """Check if result needs judge evaluation."""
        test = next((t for t in self.tests if t.id == result.test_id), None)
        if not test:
            return False
        return bool(test.pass_criteria or test.fail_criteria)

    async def _judge_single_result(self, result: RunResult) -> None:
        """Judge a single result and update aggregate."""
        test = next((t for t in self.tests if t.id == result.test_id), None)
        if not test:
            return

        start_time = time.time()
        try:
            criteria = test.pass_criteria or f"Response must NOT: {test.fail_criteria}"
            judge_result = await judge(
                response=result.response,
                criteria=criteria,
                judge_model=self.judge_model,
            )
            judge_latency_ms = (time.time() - start_time) * 1000

            # Update result in place
            result.judge_latency_ms = judge_latency_ms
            result.judge_result = judge_result

            if not judge_result["pass"]:
                # Downgrade from COMPLETED to SEMANTIC_FAILURE
                agg = self.state.get_aggregate(result.test_id, result.model_id)
                if agg and result.status == RunStatus.COMPLETED:
                    agg.passes -= 1  # Undo the pass count
                result.status = RunStatus.SEMANTIC_FAILURE
                result.failure_reason = judge_result["reason"]

        except Exception as e:
            result.error = f"Judge failed: {e}"
            result.status = RunStatus.ERROR
        finally:
            self.state.judge_completed += 1

    async def _execute_single_run(
        self,
        test: "BenchmarkCase",
        model_id: str,
        run_idx: int,
        seed: int
    ) -> None:
        """Execute a single test run with a specific seed."""

        @retry(
            stop=stop_after_attempt(get_retry_attempts()),
            wait=wait_exponential(multiplier=2, min=get_retry_min_wait(), max=get_retry_max_wait()),
            retry=retry_if_exception(is_retryable_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def stream_with_retry() -> tuple[str, float]:
            if app_state.should_stop():
                raise CancelledError("Cancelled by user")

            response = ""
            latency_ms = 0.0

            async for chunk in complete_stream(
                model_id=model_id,
                messages=test.to_messages(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds,
                tools=test.tools,
                seed=seed,
            ):
                if chunk.startswith("__LATENCY_MS__:"):
                    latency_ms = float(chunk.split(":")[1])
                else:
                    response += chunk

                if app_state.should_stop():
                    raise CancelledError("Cancelled by user")

            validate_response(response)
            return response, latency_ms

        try:
            response, latency_ms = await stream_with_retry()

            # Semantic validation
            is_valid, failure_reason = validate_response_semantic(
                test, response, model_id=model_id
            )

            if not is_valid:
                result = RunResult(
                    test_id=test.id,
                    model_id=model_id,
                    status=RunStatus.SEMANTIC_FAILURE,
                    response=response,
                    latency_ms=latency_ms,
                    failure_reason=failure_reason
                )
            else:
                result = RunResult(
                    test_id=test.id,
                    model_id=model_id,
                    status=RunStatus.COMPLETED,
                    response=response,
                    latency_ms=latency_ms
                )

            self.state.add_result(result)

        except Exception as e:
            result = RunResult(
                test_id=test.id,
                model_id=model_id,
                status=RunStatus.ERROR,
                error=str(e),
                latency_ms=None
            )
            self.state.add_result(result)
