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

from prompt_prix.mcp.tools.complete import complete_stream
from prompt_prix.mcp.tools.judge import judge
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
    Validate that a response is actually valid, not a false positive.

    Raises:
        EmptyResponseError: If response is empty or contains error indicators

    False positives can occur when:
    - Model load is aborted mid-stream
    - Server returns empty response
    - Response contains error message instead of actual content
    """
    if not response or not response.strip():
        raise EmptyResponseError("Empty response received (possible aborted model load)")

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
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    failure_reason: Optional[str] = None  # Explains semantic failures
    judge_result: Optional[dict] = None  # LLM judge evaluation result

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
        """
        self.tests = tests
        self.models = models
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.judge_model = judge_model

        # Initialize state
        self.state = BatteryRun(
            tests=[t.id for t in tests],
            models=models
        )

    async def run(self) -> AsyncGenerator[BatteryRun, None]:
        """
        Execute all tests across all models.

        All tasks are submitted immediately. The adapter handles concurrency
        via per-server locks, enabling parallel execution across GPUs.

        Yields:
            BatteryRun state snapshot periodically for UI updates
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
        # All tasks submitted immediately - the adapter's per-server locks handle concurrency
        for item in work_items:
            if app_state.should_stop():
                break
            task = asyncio.create_task(self._execute_test(item))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        # Wait for completion, yielding state periodically
        while active_tasks:
            # Wait briefly then yield state for UI update
            done, _ = await asyncio.wait(
                active_tasks,
                timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks (exceptions are already caught in _execute_test)
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

        # Final yield with complete state
        yield self.state

    async def _execute_test(self, item: BatteryWorkItem) -> None:
        """Execute a single test with retry logic."""
        # Mark as running
        self.state.set_result(RunResult(
            test_id=item.test.id,
            model_id=item.model_id,
            status=RunStatus.RUNNING
        ))

        start_time = time.time()

        @retry(
            stop=stop_after_attempt(get_retry_attempts()),
            wait=wait_exponential(multiplier=2, min=get_retry_min_wait(), max=get_retry_max_wait()),
            retry=retry_if_exception(is_retryable_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def stream_with_retry() -> str:
            """Stream completion with retry for transient errors."""
            # Check for cancellation before each attempt
            if app_state.should_stop():
                raise CancelledError("Battery run cancelled by user")

            response = ""
            # Call MCP primitive - adapter handles server selection
            async for chunk in complete_stream(
                model_id=item.model_id,
                messages=item.test.to_messages(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_seconds=self.timeout_seconds,
                tools=item.test.tools
            ):
                response += chunk
                # Check for cancellation during streaming
                if app_state.should_stop():
                    raise CancelledError("Battery run cancelled by user")

            # Validate response to catch false positives (empty/error responses)
            validate_response(response)
            return response

        try:
            # 1. Fail-fast: criteria require judge
            has_criteria = item.test.pass_criteria or item.test.fail_criteria
            if has_criteria and not self.judge_model:
                self.state.set_result(RunResult(
                    test_id=item.test.id,
                    model_id=item.model_id,
                    status=RunStatus.ERROR,
                    error="Test has criteria but no judge_model configured"
                ))
                return

            response = await stream_with_retry()
            latency_ms = (time.time() - start_time) * 1000

            # 2. Local semantic checks (fast, free)
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
                    failure_reason=failure_reason
                ))
                return

            # 3. Judge evaluation (if criteria exist)
            judge_result = None
            if self.judge_model and has_criteria:
                criteria = item.test.pass_criteria or f"Response must NOT: {item.test.fail_criteria}"
                judge_result = await judge(
                    response=response,
                    criteria=criteria,
                    judge_model=self.judge_model,
                )
                if not judge_result["pass"]:
                    self.state.set_result(RunResult(
                        test_id=item.test.id,
                        model_id=item.model_id,
                        status=RunStatus.SEMANTIC_FAILURE,
                        response=response,
                        latency_ms=latency_ms,
                        failure_reason=judge_result["reason"],
                        judge_result=judge_result
                    ))
                    return

            # 4. COMPLETED if all checks pass
            self.state.set_result(RunResult(
                test_id=item.test.id,
                model_id=item.model_id,
                status=RunStatus.COMPLETED,
                response=response,
                latency_ms=latency_ms,
                judge_result=judge_result
            ))

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Mark as error (fail loudly - record error, don't hide it)
            self.state.set_result(RunResult(
                test_id=item.test.id,
                model_id=item.model_id,
                status=RunStatus.ERROR,
                error=str(e),
                latency_ms=latency_ms
            ))
