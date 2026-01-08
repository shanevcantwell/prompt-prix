"""
Battery Engine - orchestrates benchmark test suite execution.

State management per CLAUDE.md:
- Pydantic models for all state (TestResult, BatteryRun)
- Observable by default (yields state snapshots for UI)
- Fail loudly on errors (no swallowing exceptions)

Uses BatchRunner for model-batched execution:
- All tests for one model run together on one server
- Minimizes VRAM swapping, ensures fair comparison
- Server affinity prefers where model is already loaded
"""

import logging
import time
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

from prompt_prix.core import stream_completion
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
    from prompt_prix.adapters.lmstudio import LMStudioAdapter
    from prompt_prix.benchmarks.base import TestCase


class TestStatus(str, Enum):
    """Status of a single (test, model) execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SEMANTIC_FAILURE = "semantic_failure"  # Response received but semantically failed
    ERROR = "error"


class GridDisplayMode(str, Enum):
    """Display mode for battery results grid."""
    SYMBOLS = "symbols"  # ✓, ❌, ⏳, —
    LATENCY = "latency"  # Response time in ms with color


class TestResult(BaseModel):
    """
    Result for one (test_id, model_id) cell in the battery grid.

    Explicit state: tracks status, response, timing, and errors.
    """
    test_id: str
    model_id: str
    status: TestStatus = TestStatus.PENDING
    response: str = ""
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    failure_reason: Optional[str] = None  # Explains semantic failures

    @property
    def status_symbol(self) -> str:
        """UI symbol for this result's status."""
        symbols = {
            TestStatus.PENDING: "—",
            TestStatus.RUNNING: "⏳",
            TestStatus.COMPLETED: "✓",
            TestStatus.SEMANTIC_FAILURE: "⚠",
            TestStatus.ERROR: "❌"
        }
        return symbols.get(self.status, "?")

    @property
    def latency_display(self) -> str:
        """Formatted latency for grid display."""
        if self.status == TestStatus.PENDING:
            return "—"
        elif self.status == TestStatus.RUNNING:
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
    results: dict[str, TestResult] = {}  # key = f"{test_id}:{model_id}"

    def get_key(self, test_id: str, model_id: str) -> str:
        """Generate result key from test and model IDs."""
        return f"{test_id}:{model_id}"

    def get_result(self, test_id: str, model_id: str) -> Optional[TestResult]:
        """Get result for a specific (test, model) cell."""
        return self.results.get(self.get_key(test_id, model_id))

    def set_result(self, result: TestResult) -> None:
        """Set result for a specific (test, model) cell."""
        key = self.get_key(result.test_id, result.model_id)
        self.results[key] = result

    def to_grid(
        self, display_mode: GridDisplayMode = GridDisplayMode.SYMBOLS
    ) -> list[list[str]]:
        """
        Convert to gr.Dataframe format.

        Args:
            display_mode: How to display results (symbols or latency)

        Returns:
            List of rows where:
            - First row is headers: ["Test", model1, model2, ...]
            - Data rows are: [test_name, value1, value2, ...]
        """
        # Header row
        grid = [["Test"] + self.models]

        # Data rows
        for test_id in self.tests:
            row = [test_id]
            for model_id in self.models:
                result = self.get_result(test_id, model_id)
                if result:
                    row.append(result.get_display(display_mode))
                else:
                    row.append("—")  # No result yet
            grid.append(row)

        return grid

    @property
    def completed_count(self) -> int:
        """Count of completed or errored tests."""
        return sum(
            1 for r in self.results.values()
            if r.status in [TestStatus.COMPLETED, TestStatus.SEMANTIC_FAILURE, TestStatus.ERROR]
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
    Orchestrates battery execution with model-batched parallelism.

    Design per CLAUDE.md:
    - Dependency injection (adapter passed in, not hardcoded)
    - Observable (yields state snapshots for UI updates)
    - Fail loudly (errors recorded in TestResult, not swallowed)
    - Model-batched execution: all tests for one model run together

    Execution strategy:
    - Tests grouped by model into batches
    - Each batch runs on one server (no mid-batch VRAM swapping)
    - Server affinity prefers where model is already loaded
    - Multiple batches can run in parallel on different GPUs
    """

    def __init__(
        self,
        adapter: "LMStudioAdapter",
        tests: list["TestCase"],
        models: list[str],
        max_tokens: int = 2048,
        timeout_seconds: int = 300,
        temperature: Optional[float] = None
    ):
        """
        Initialize battery runner.

        Args:
            adapter: LMStudioAdapter for model inference (DI)
            tests: List of TestCase objects to run
            models: List of model IDs to test against
            max_tokens: Maximum tokens per response
            timeout_seconds: Timeout per request
            temperature: Sampling temperature. None = use per-model defaults.
        """
        self.adapter = adapter
        self.tests = tests
        self.tests_by_id = {t.id: t for t in tests}  # Lookup for execute_fn
        self.models = models
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

        # Initialize state
        self.state = BatteryRun(
            tests=[t.id for t in tests],
            models=models
        )

    async def run(self) -> AsyncGenerator[BatteryRun, None]:
        """
        Execute all tests across all models using model batching.

        Uses BatchRunner which:
        - Groups tests by model (one batch per model)
        - Runs all tests in a batch sequentially on one server
        - Runs multiple batches in parallel on different servers
        - Prefers servers where model is already loaded

        Yields:
            BatteryRun state snapshot periodically for UI updates
        """
        from prompt_prix.scheduler import BatchRunner

        runner = BatchRunner(self.adapter.pool)

        async def execute_test(test_id: str, model_id: str, server_url: str) -> None:
            """Execute a single test on a specific server with retry logic."""
            test = self.tests_by_id[test_id]

            # Mark as running
            self.state.set_result(TestResult(
                test_id=test_id,
                model_id=model_id,
                status=TestStatus.RUNNING
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
                async for chunk in stream_completion(
                    server_url=server_url,
                    model_id=model_id,
                    messages=test.to_messages(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
                    tools=test.tools
                ):
                    response += chunk
                    # Check for cancellation during streaming
                    if app_state.should_stop():
                        raise CancelledError("Battery run cancelled by user")

                # Validate response to catch false positives (empty/error responses)
                validate_response(response)
                return response

            try:
                response = await stream_with_retry()
                latency_ms = (time.time() - start_time) * 1000

                # Semantic validation: check for refusals and expected tool calls
                # Pass model_id for model-aware tool call parsing
                is_valid, failure_reason = validate_response_semantic(
                    test, response, model_id=model_id
                )

                if is_valid:
                    self.state.set_result(TestResult(
                        test_id=test_id,
                        model_id=model_id,
                        status=TestStatus.COMPLETED,
                        response=response,
                        latency_ms=latency_ms
                    ))
                else:
                    self.state.set_result(TestResult(
                        test_id=test_id,
                        model_id=model_id,
                        status=TestStatus.SEMANTIC_FAILURE,
                        response=response,
                        latency_ms=latency_ms,
                        failure_reason=failure_reason
                    ))

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                # Mark as error (fail loudly - record error, don't hide it)
                self.state.set_result(TestResult(
                    test_id=test_id,
                    model_id=model_id,
                    status=TestStatus.ERROR,
                    error=str(e),
                    latency_ms=latency_ms
                ))

        # Run with BatchRunner - tests batched by model
        test_ids = [t.id for t in self.tests]

        async for progress in runner.run(self.models, test_ids, execute_test):
            # Check for stop request
            if app_state.should_stop():
                runner.request_stop()
            yield self.state

        # Final yield with complete state
        yield self.state
