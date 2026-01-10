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

import asyncio
from collections import defaultdict
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
            TestStatus.SEMANTIC_FAILURE: "❌",  # Model failed the test
            TestStatus.ERROR: "⚠"               # Technical issue (not model's fault)
        }
        return symbols.get(self.status, "?")

    @property
    def latency_display(self) -> str:
        """Formatted latency for grid display.

        Shows status symbol + time (e.g., '✓ 1.2s', '❌ 2.1s').
        """
        if self.status == TestStatus.PENDING:
            return "—"
        elif self.status == TestStatus.RUNNING:
            return "⏳"
        elif self.latency_ms is not None:
            # Format as symbol + seconds with 1 decimal for readability
            seconds = self.latency_ms / 1000
            return f"{self.status_symbol} {seconds:.1f}s"
        else:
            # Completed without latency (shouldn't happen normally)
            return self.status_symbol

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
    ) -> "pd.DataFrame":
        """
        Convert to gr.Dataframe format using pandas DataFrame.

        Args:
            display_mode: How to display results (symbols or latency)

        Returns:
            pandas DataFrame with explicit column headers.
            Columns: ["Test", model1, model2, ...]
            Rows: [test_name, value1, value2, ...]
        """
        import pandas as pd

        headers = ["Test"] + self.models
        rows = []

        for test_id in self.tests:
            row = [test_id]
            for model_id in self.models:
                result = self.get_result(test_id, model_id)
                if result:
                    row.append(result.get_display(display_mode))
                else:
                    row.append("—")  # No result yet
            rows.append(row)

        return pd.DataFrame(rows, columns=headers)

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
        server_hints: Optional[dict[str, str]] = None,
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
            server_hints: Dict mapping model_id → server_url for GPU routing.
                         When provided, enables orchestrated fan-out dispatch.
            max_tokens: Maximum tokens per response
            timeout_seconds: Timeout per request
            temperature: Sampling temperature. None = use per-model defaults.
        """
        self.adapter = adapter
        self.tests = tests
        self.tests_by_id = {t.id: t for t in tests}  # Lookup for execute_fn
        self.models = models
        self.server_hints = server_hints or {}
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
        Execute all tests across all models.

        When server_hints is provided, uses orchestrated fan-out dispatch:
        - Groups tasks by server URL (from user's GPU prefix selection)
        - Runs per-server queues in parallel (GPUs work simultaneously)
        - Each queue runs sequentially (no VRAM thrashing within GPU)

        Without server_hints, falls back to semaphore-based dispatch.

        Yields:
            BatteryRun state snapshot periodically for UI updates
        """
        # Queue for coordinating state updates across parallel server queues
        update_queue: asyncio.Queue = asyncio.Queue()

        async def execute_test(test_id: str, model_id: str) -> None:
            """Execute a single test with retry logic."""
            test = self.tests_by_id[test_id]

            # Mark as running
            self.state.set_result(TestResult(
                test_id=test_id,
                model_id=model_id,
                status=TestStatus.RUNNING
            ))
            await update_queue.put("running")

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
                async for chunk in self.adapter.stream_completion(
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

            await update_queue.put("done")

        async def run_server_queue(server_url: str, queue: list[tuple[str, str]]) -> None:
            """Run all tests for a server sequentially (no VRAM thrashing)."""
            # Set server hint for this queue's models
            for _, model_id in queue:
                self.adapter.set_server_hint(model_id, server_url)

            # Execute tests sequentially within this server's queue
            for test_id, model_id in queue:
                if app_state.should_stop():
                    break
                await execute_test(test_id, model_id)

        # Use orchestrated fan-out when server_hints provided
        if self.server_hints:
            # Group tasks by server URL
            server_queues: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for model_id in self.models:
                server_url = self.server_hints.get(model_id)
                if not server_url:
                    logger.warning(f"No server hint for model {model_id}, using first available")
                    # Fall back to first server for unhinted models
                    server_url = list(self.server_hints.values())[0] if self.server_hints else "default"
                for test in self.tests:
                    server_queues[server_url].append((test.id, model_id))

            # Calculate total tasks for completion tracking
            total_tasks = sum(len(q) for q in server_queues.values())

            # Start all server queues in parallel
            queue_tasks = [
                asyncio.create_task(run_server_queue(url, queue))
                for url, queue in server_queues.items()
            ]

            # Yield state updates as tests complete
            completed = 0
            yield self.state  # Initial state

            while completed < total_tasks:
                if app_state.should_stop():
                    for task in queue_tasks:
                        task.cancel()
                    break

                try:
                    # Wait for update with timeout for responsiveness
                    await asyncio.wait_for(update_queue.get(), timeout=0.1)
                    completed += 0.5  # "running" and "done" each count as 0.5
                except asyncio.TimeoutError:
                    pass
                yield self.state

            # Wait for all queues to finish
            await asyncio.gather(*queue_tasks, return_exceptions=True)

        else:
            # Fallback: semaphore-based dispatch (original behavior)
            semaphore = asyncio.Semaphore(self.adapter.get_concurrency_limit())

            async def execute_with_semaphore(test_id: str, model_id: str) -> None:
                async with semaphore:
                    await execute_test(test_id, model_id)

            # Create tasks for all (test, model) combinations
            tasks = []
            for model_id in self.models:
                for test in self.tests:
                    tasks.append(asyncio.create_task(execute_with_semaphore(test.id, model_id)))

            # Yield state as tasks complete
            pending = set(tasks)
            while pending:
                if app_state.should_stop():
                    for task in pending:
                        task.cancel()
                    break

                done, pending = await asyncio.wait(
                    pending, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                yield self.state

            await asyncio.gather(*tasks, return_exceptions=True)

        yield self.state
