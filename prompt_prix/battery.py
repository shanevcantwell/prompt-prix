"""
Battery Engine - orchestrates benchmark test suite execution.

Per ADR-006, BatteryRunner is in the ORCHESTRATION layer:
- Defines WHAT to run (test matrix across models)
- Controls concurrency via asyncio.Semaphore
- Calls MCP primitives ONLY — never adapters directly
- NEVER imports adapters/*, ServerPool, ConcurrentDispatcher

State management per CLAUDE.md:
- Pydantic models for all state (RunResult, BatteryRun)
- Observable by default (yields state snapshots for UI)
- Fail loudly on errors (no swallowing exceptions)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

from prompt_prix.mcp.tools.judge import judge
from prompt_prix.react.dispatch import execute_test_case, ReactLoopIncomplete
from prompt_prix.config import get_retry_attempts, get_retry_min_wait, get_retry_max_wait
from prompt_prix import state as app_state
from prompt_prix.semantic_validator import validate_response_semantic

logger = logging.getLogger(__name__)


class CancelledError(Exception):
    """Raised when battery run is cancelled by user."""
    pass


class EmptyResponseError(Exception):
    """Raised when model returns empty response (possible aborted load)."""
    pass


def is_retryable_error(exception: BaseException) -> bool:
    """
    Determine if an exception is retryable.

    Retryable errors include:
    - Model loading failures (LM Studio swapping models)
    - Connection errors (transient network issues)
    - Timeout errors (server overloaded)
    - Empty responses (aborted model loads)
    """
    # Empty responses are retryable (often caused by model load abort)
    if isinstance(exception, EmptyResponseError):
        return True

    error_msg = str(exception).lower()
    retryable_patterns = [
        "failed to load model",
        "model loading",
        "operation canceled",  # LM Studio aborted model load
        "connection",
        "timeout",
        "server busy",
        "503",
        "502",
    ]
    return any(pattern in error_msg for pattern in retryable_patterns)


def validate_response(response: str) -> None:
    """
    Validate that a response doesn't contain error indicators.

    Raises:
        EmptyResponseError: If response contains error message instead of content

    Note: Empty responses are valid - they mean the model chose to return nothing.
    Aborted streams (model load interrupted) are detected at the adapter layer.
    """
    # Empty response is valid - model chose to return nothing
    if not response or not response.strip():
        return

    # Check for common error indicators in response content
    response_lower = response.lower()
    error_indicators = [
        "failed to load",
        "model not loaded",
        "error loading",
        "internal server error",
    ]
    for indicator in error_indicators:
        if indicator in response_lower and len(response) < 200:
            # Short response with error indicator = likely false positive
            raise EmptyResponseError(f"Response appears to be error message: {response[:100]}")

if TYPE_CHECKING:
    from prompt_prix.benchmarks.base import BenchmarkCase


@dataclass
class BatteryWorkItem:
    """Work item for battery execution."""

    test: "BenchmarkCase"
    model_id: str


class RunStatus(str, Enum):
    """Status of a single (test, model) execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SEMANTIC_FAILURE = "semantic_failure"  # Response received but failed criteria
    ERROR = "error"  # Infrastructure error or couldn't evaluate


class GridDisplayMode(str, Enum):
    """Display mode for battery results grid."""
    SYMBOLS = "symbols"  # ✓, ❌, ⏳, —
    LATENCY = "latency"  # Response time in ms with color


class RunResult(BaseModel):
    """
    Result for one (test_id, model_id) cell in the battery grid.

    Explicit state: tracks status, response, timing, and errors.
    """
    test_id: str
    model_id: str
    status: RunStatus = RunStatus.PENDING
    response: str = ""
    latency_ms: Optional[float] = None  # Inference time only (Phase 1)
    judge_latency_ms: Optional[float] = None  # Judge evaluation time (Phase 2)
    error: Optional[str] = None
    failure_reason: Optional[str] = None  # Explains semantic failures
    judge_result: Optional[dict] = None  # LLM judge evaluation result
    drift_score: Optional[float] = None  # Cosine distance to expected_response
    react_trace: Optional[dict] = None  # ReAct loop metadata (mode="react" only)

    @property
    def status_symbol(self) -> str:
        """UI symbol for this result's status."""
        symbols = {
            RunStatus.PENDING: "—",
            RunStatus.RUNNING: "⏳",
            RunStatus.COMPLETED: "✓",
            RunStatus.SEMANTIC_FAILURE: "❌",
            RunStatus.ERROR: "⚠"
        }
        return symbols.get(self.status, "?")

    @property
    def latency_display(self) -> str:
        """Formatted latency for grid display."""
        if self.status == RunStatus.PENDING:
            return "—"
        elif self.status == RunStatus.RUNNING:
            return "⏳"
        elif self.latency_ms is not None:
            # Format as seconds with 1 decimal for readability
            seconds = self.latency_ms / 1000
            return f"{seconds:.1f}s"
        else:
            return "—"

    def get_display(self, mode: "GridDisplayMode") -> str:
        """Get display string for the specified mode."""
        if mode == GridDisplayMode.LATENCY:
            return self.latency_display
        return self.status_symbol


class BatteryRun(BaseModel):
    """
    Source of truth for battery grid UI.

    Tracks all test IDs (rows), model IDs (columns), and results.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tests: list[str]  # Test IDs (row labels)
    models: list[str]  # Model IDs (column labels)
    results: dict[str, RunResult] = {}  # key = f"{test_id}:{model_id}"

    # Two-phase execution tracking (ADR-008)
    phase: str = "inference"  # "inference" or "judging"
    judge_total: int = 0      # How many results need judging
    judge_completed: int = 0  # How many have been judged

    def get_key(self, test_id: str, model_id: str) -> str:
        """Generate result key from test and model IDs."""
        return f"{test_id}:{model_id}"

    def get_result(self, test_id: str, model_id: str) -> Optional[RunResult]:
        """Get result for a specific (test, model) cell."""
        return self.results.get(self.get_key(test_id, model_id))

    def set_result(self, result: RunResult) -> None:
        """Set result for a specific (test, model) cell."""
        key = self.get_key(result.test_id, result.model_id)
        self.results[key] = result

    def to_grid(
        self, display_mode: GridDisplayMode = GridDisplayMode.SYMBOLS
    ) -> "pd.DataFrame":
        """
        Convert to pandas DataFrame for Gradio display.

        Args:
            display_mode: How to display results (symbols or latency)

        Returns:
            DataFrame with Test column and one column per model.
        """
        import pandas as pd

        data = []
        for test_id in self.tests:
            row = {"Test": test_id}
            for model_id in self.models:
                result = self.get_result(test_id, model_id)
                row[model_id] = result.get_display(display_mode) if result else "—"
            data.append(row)

        return pd.DataFrame(data)

    @property
    def completed_count(self) -> int:
        """Count of completed or errored tests."""
        return sum(
            1 for r in self.results.values()
            if r.status in [RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.SEMANTIC_FAILURE]
        )

    @property
    def total_count(self) -> int:
        """Total number of (test, model) combinations."""
        return len(self.tests) * len(self.models)

    @property
    def progress_percent(self) -> float:
        """Progress as percentage (0-100)."""
        if self.total_count == 0:
            return 100.0
        return (self.completed_count / self.total_count) * 100

    def recalculate_drift_threshold(self, new_threshold: float) -> None:
        """
        Re-evaluate pass/fail for all results based on a new drift threshold.

        Only affects results that have a drift_score (i.e., tests with
        expected_response that completed drift calculation). Results that failed
        semantic validation (refusals) or errored are unaffected.
        """
        for result in self.results.values():
            if result.drift_score is None:
                continue  # No drift data — skip

            if new_threshold <= 0:
                # Threshold disabled — all drift results pass
                if result.status == RunStatus.SEMANTIC_FAILURE and "Drift" in (result.failure_reason or ""):
                    result.status = RunStatus.COMPLETED
                    result.failure_reason = None
            elif result.drift_score > new_threshold:
                if result.status == RunStatus.COMPLETED:
                    result.status = RunStatus.SEMANTIC_FAILURE
                    result.failure_reason = f"Drift {result.drift_score:.3f} exceeds threshold {new_threshold}"
            else:
                # Within threshold — restore to COMPLETED if previously failed on drift
                if result.status == RunStatus.SEMANTIC_FAILURE and "Drift" in (result.failure_reason or ""):
                    result.status = RunStatus.COMPLETED
                    result.failure_reason = None


class BatteryRunner:
    """
    Orchestrates battery execution.

    Per ADR-006 (Orchestration Layer):
    - Defines WHAT to run (test matrix across models)
    - Calls MCP primitives ONLY (complete_stream) — never adapters directly
    - DOES NOT know about servers, ServerPool, or ConcurrentDispatcher

    The adapter (registered in MCP registry) handles server selection and
    concurrency internally via per-server locks.
    """

    def __init__(
        self,
        tests: list["BenchmarkCase"],
        models: list[str],
        temperature: float = 0.0,  # Deterministic for evals
        max_tokens: int = 2048,
        timeout_seconds: int = 300,
        judge_model: Optional[str] = None,  # Model for LLM-as-judge evaluation
        drift_threshold: float = 0.0,  # Cosine distance threshold (0 = disabled)
    ):
        """
        Initialize battery runner.

        Args:
            tests: List of BenchmarkCase objects to run
            models: List of model IDs to test against
            temperature: Sampling temperature (default 0.0 for reproducibility)
            max_tokens: Maximum tokens per response
            timeout_seconds: Timeout per request
            judge_model: Model ID for LLM-as-judge evaluation (optional)
            drift_threshold: Cosine distance threshold for expected_response (0 = disabled)
        """
        self.tests = tests
        self.models = models
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.judge_model = judge_model
        self.drift_threshold = drift_threshold

        # Initialize state
        self.state = BatteryRun(
            tests=[t.id for t in tests],
            models=models
        )

    async def run(self) -> AsyncGenerator[BatteryRun, None]:
        """
        Execute all tests across all models.

        When judge_model is set, uses pipelined execution: judge tasks are
        submitted eagerly as inference results complete. The dispatcher +
        current_model guard routes judge tasks to idle GPUs naturally.

        When no judge_model, runs inference only.

        Yields:
            BatteryRun state snapshot periodically for UI updates
        """
        if self.judge_model:
            async for state in self._execute_pipelined():
                yield state
        else:
            async for state in self._execute_inference_phase():
                yield state

        # Final yield
        yield self.state

    async def _execute_inference_phase(self) -> AsyncGenerator[BatteryRun, None]:
        """
        Phase 1: Execute all test inferences.

        All tasks submitted immediately - adapter handles GPU concurrency.
        No judging in this phase.
        """
        # Build work items: model-first order (depth-first)
        # All tests for model1, then all tests for model2, etc.
        # This minimizes VRAM swapping - each model stays loaded for all its tests
        work_items = [
            BatteryWorkItem(test=test, model_id=model_id)
            for model_id in self.models
            for test in self.tests
        ]

        active_tasks: set[asyncio.Task] = set()

        # Initial yield
        yield self.state

        # Create tasks for all work items
        for item in work_items:
            if app_state.should_stop():
                break
            task = asyncio.create_task(self._execute_test(item))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        # Wait for completion, yielding state periodically
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks,
                timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks (exceptions already caught in _execute_test)
            for task in done:
                try:
                    task.result()
                except Exception:
                    pass  # Errors already recorded in state

            yield self.state

            # Check for cancellation
            if app_state.should_stop():
                for task in active_tasks:
                    task.cancel()
                break

    async def _execute_pipelined(self) -> AsyncGenerator[BatteryRun, None]:
        """
        Pipelined execution: inference + eager judge submission.

        Judge tasks are submitted to the same dispatcher as inference tasks
        as soon as each inference result completes. The current_model drain
        guard (#130) naturally routes judge tasks to idle GPUs without
        displacing active inference models.

        When no GPU idles during inference, judge tasks queue until servers
        drain — same total time as sequential phases.
        """
        # Build work items: model-first order (depth-first)
        work_items = [
            BatteryWorkItem(test=test, model_id=model_id)
            for model_id in self.models
            for test in self.tests
        ]

        inference_tasks: set[asyncio.Task] = set()
        judge_tasks: set[asyncio.Task] = set()

        yield self.state

        for item in work_items:
            if app_state.should_stop():
                break
            task = asyncio.create_task(
                self._inference_then_judge(item, judge_tasks)
            )
            inference_tasks.add(task)

        while inference_tasks or judge_tasks:
            all_current = frozenset(inference_tasks | judge_tasks)
            if not all_current:
                break

            done, _ = await asyncio.wait(
                all_current,
                timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED
            )

            for t in done:
                inference_tasks.discard(t)
                judge_tasks.discard(t)
                try:
                    t.result()
                except Exception:
                    pass  # Errors already recorded in state

            # Phase transition: inference done, judge tasks remain
            if not inference_tasks and judge_tasks and self.state.phase != "judging":
                self.state.phase = "judging"

            yield self.state

            if app_state.should_stop():
                for t in (inference_tasks | judge_tasks):
                    t.cancel()
                break

    async def _inference_then_judge(
        self,
        item: BatteryWorkItem,
        judge_tasks: set[asyncio.Task],
    ) -> None:
        """
        Run inference for a work item, then eagerly submit judge task if needed.

        The judge task is added to the shared judge_tasks set so the outer
        pipelined loop can track it.
        """
        await self._execute_test(item)

        result = self.state.get_result(item.test.id, item.model_id)
        if (result
                and result.status == RunStatus.COMPLETED
                and self._needs_judging(result)):
            self.state.judge_total += 1
            task = asyncio.create_task(self._judge_single_result(result))
            judge_tasks.add(task)

    def _needs_judging(self, result: RunResult) -> bool:
        """Check if a result needs judge evaluation."""
        # Find the test case to check for criteria
        test = next((t for t in self.tests if t.id == result.test_id), None)
        if not test:
            return False
        return bool(test.pass_criteria or test.fail_criteria)

    def _get_test_by_id(self, test_id: str) -> Optional["BenchmarkCase"]:
        """Look up a test case by ID."""
        return next((t for t in self.tests if t.id == test_id), None)

    async def _judge_single_result(self, result: RunResult) -> None:
        """
        Judge a single completed result.

        Updates the result in-place with verdict and timing.
        """
        test = self._get_test_by_id(result.test_id)
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

            if not judge_result["pass"]:
                # Update to SEMANTIC_FAILURE with judge verdict
                self.state.set_result(RunResult(
                    test_id=result.test_id,
                    model_id=result.model_id,
                    status=RunStatus.SEMANTIC_FAILURE,
                    response=result.response,
                    latency_ms=result.latency_ms,
                    judge_latency_ms=judge_latency_ms,
                    failure_reason=judge_result["reason"],
                    judge_result=judge_result
                ))
            else:
                # Keep COMPLETED but add judge_result
                self.state.set_result(RunResult(
                    test_id=result.test_id,
                    model_id=result.model_id,
                    status=RunStatus.COMPLETED,
                    response=result.response,
                    latency_ms=result.latency_ms,
                    judge_latency_ms=judge_latency_ms,
                    judge_result=judge_result
                ))

        except Exception as e:
            judge_latency_ms = (time.time() - start_time) * 1000
            # Judge failed - mark as error
            self.state.set_result(RunResult(
                test_id=result.test_id,
                model_id=result.model_id,
                status=RunStatus.ERROR,
                response=result.response,
                latency_ms=result.latency_ms,
                judge_latency_ms=judge_latency_ms,
                error=f"Judge evaluation failed: {e}"
            ))
        finally:
            # Increment judge progress counter
            self.state.judge_completed += 1

    async def _execute_test(self, item: BatteryWorkItem) -> None:
        """
        Execute a single test via execute_test_case() dispatch.

        Mode-unaware: dispatch handles single-shot vs react internally.
        Inference + semantic validation only. Judging is handled by the
        caller (_inference_then_judge for pipelined, or not at all for
        inference-only runs).
        """
        # Mark as running
        self.state.set_result(RunResult(
            test_id=item.test.id,
            model_id=item.model_id,
            status=RunStatus.RUNNING
        ))

        @retry(
            stop=stop_after_attempt(get_retry_attempts()),
            wait=wait_exponential(multiplier=2, min=get_retry_min_wait(), max=get_retry_max_wait()),
            retry=retry_if_exception(is_retryable_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def dispatch_with_retry():
            """Dispatch test case with retry for transient errors."""
            if app_state.should_stop():
                raise CancelledError("Battery run cancelled by user")

            result = await execute_test_case(
                test=item.test,
                model_id=item.model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds,
            )
            validate_response(result.response)
            return result

        try:
            case_result = await dispatch_with_retry()
            response = case_result.response
            latency_ms = case_result.latency_ms
            react_trace = case_result.react_trace

            # Local semantic checks (fast, free)
            is_valid, failure_reason = validate_response_semantic(
                item.test, response, model_id=item.model_id
            )

            if not is_valid:
                self.state.set_result(RunResult(
                    test_id=item.test.id,
                    model_id=item.model_id,
                    status=RunStatus.SEMANTIC_FAILURE,
                    response=response,
                    latency_ms=latency_ms,
                    failure_reason=failure_reason,
                    react_trace=react_trace,
                ))
                return

            # Drift validation (fast, inline — ~50ms via embedding)
            drift_score = None
            drift_target = item.test.expected_response
            if (self.drift_threshold > 0
                    and drift_target):
                try:
                    from prompt_prix.mcp.tools.drift import calculate_drift
                    drift_score = await calculate_drift(response, drift_target)
                except ImportError:
                    pass  # Logged once by drift module
                except Exception as e:
                    logger.warning(f"Drift calculation failed (fail open): {e}")

            if drift_score is not None and drift_score > self.drift_threshold:
                self.state.set_result(RunResult(
                    test_id=item.test.id,
                    model_id=item.model_id,
                    status=RunStatus.SEMANTIC_FAILURE,
                    response=response,
                    latency_ms=latency_ms,
                    drift_score=drift_score,
                    failure_reason=f"Drift {drift_score:.3f} exceeds threshold {self.drift_threshold}",
                    react_trace=react_trace,
                ))
                return

            # COMPLETED - judging (if needed) happens in Phase 2
            self.state.set_result(RunResult(
                test_id=item.test.id,
                model_id=item.model_id,
                status=RunStatus.COMPLETED,
                response=response,
                latency_ms=latency_ms,
                drift_score=drift_score,
                react_trace=react_trace,
            ))

        except ReactLoopIncomplete as e:
            # React loop didn't complete — model failed the task
            self.state.set_result(RunResult(
                test_id=item.test.id,
                model_id=item.model_id,
                status=RunStatus.SEMANTIC_FAILURE,
                response="",
                failure_reason=e.reason,
                react_trace=e.react_trace,
            ))

        except Exception as e:
            # Mark as error (fail loudly - record error, don't hide it)
            self.state.set_result(RunResult(
                test_id=item.test.id,
                model_id=item.model_id,
                status=RunStatus.ERROR,
                error=str(e),
                latency_ms=None
            ))
