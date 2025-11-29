"""
Battery Engine - orchestrates benchmark test suite execution.

State management per CLAUDE.md:
- Pydantic models for all state (TestResult, BatteryRun)
- Observable by default (yields state snapshots for UI)
- Fail loudly on errors (no swallowing exceptions)

Uses WorkStealingDispatcher for parallel execution across servers.
Includes retry logic with exponential backoff for transient errors.
"""

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

from prompt_prix.core import stream_completion
from prompt_prix.config import get_retry_attempts, get_retry_min_wait, get_retry_max_wait

logger = logging.getLogger(__name__)


def is_retryable_error(exception: BaseException) -> bool:
    """
    Determine if an exception is retryable.

    Retryable errors include:
    - Model loading failures (LM Studio swapping models)
    - Connection errors (transient network issues)
    - Timeout errors (server overloaded)
    """
    error_msg = str(exception).lower()
    retryable_patterns = [
        "failed to load model",
        "model loading",
        "connection",
        "timeout",
        "server busy",
        "503",
        "502",
    ]
    return any(pattern in error_msg for pattern in retryable_patterns)

if TYPE_CHECKING:
    from prompt_prix.adapters.lmstudio import LMStudioAdapter
    from prompt_prix.benchmarks.base import TestCase


@dataclass
class BatteryWorkItem:
    """Work item for battery dispatcher."""

    test: "TestCase"
    model_id: str


class TestStatus(str, Enum):
    """Status of a single (test, model) execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


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

    @property
    def status_symbol(self) -> str:
        """UI symbol for this result's status."""
        symbols = {
            TestStatus.PENDING: "—",
            TestStatus.RUNNING: "⏳",
            TestStatus.COMPLETED: "✓",
            TestStatus.ERROR: "❌"
        }
        return symbols.get(self.status, "?")


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

    def to_grid(self) -> list[list[str]]:
        """
        Convert to gr.Dataframe format.

        Returns:
            List of rows where:
            - First row is headers: ["Test", model1, model2, ...]
            - Data rows are: [test_name, status1, status2, ...]
        """
        # Header row
        grid = [["Test"] + self.models]

        # Data rows
        for test_id in self.tests:
            row = [test_id]
            for model_id in self.models:
                result = self.get_result(test_id, model_id)
                if result:
                    row.append(result.status_symbol)
                else:
                    row.append("—")  # No result yet
            grid.append(row)

        return grid

    @property
    def completed_count(self) -> int:
        """Count of completed or errored tests."""
        return sum(
            1 for r in self.results.values()
            if r.status in [TestStatus.COMPLETED, TestStatus.ERROR]
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
    Orchestrates battery execution with work-stealing parallelism.

    Design per CLAUDE.md:
    - Dependency injection (adapter passed in, not hardcoded)
    - Observable (yields state snapshots for UI updates)
    - Fail loudly (errors recorded in TestResult, not swallowed)
    - Work-stealing dispatcher for parallel execution across servers
    """

    def __init__(
        self,
        adapter: "LMStudioAdapter",
        tests: list["TestCase"],
        models: list[str],
        temperature: float = 0.0,  # Deterministic for evals
        max_tokens: int = 2048,
        timeout_seconds: int = 300
    ):
        """
        Initialize battery runner.

        Args:
            adapter: LMStudioAdapter for model inference (DI)
            tests: List of TestCase objects to run
            models: List of model IDs to test against
            temperature: Sampling temperature (default 0.0 for reproducibility)
            max_tokens: Maximum tokens per response
            timeout_seconds: Timeout per request
        """
        self.adapter = adapter
        self.tests = tests
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
        Execute all tests across all models using work-stealing.

        Uses WorkStealingDispatcher to run tests in parallel across
        available servers, keeping all servers busy.

        Yields:
            BatteryRun state snapshot periodically for UI updates
        """
        from prompt_prix.dispatcher import WorkStealingDispatcher

        # Build work items for all (test, model) combinations
        work_items = [
            BatteryWorkItem(test=test, model_id=model_id)
            for test in self.tests
            for model_id in self.models
        ]

        dispatcher = WorkStealingDispatcher(self.adapter.pool)

        async def execute_test(item: BatteryWorkItem, server_url: str) -> None:
            """Execute a single test on a specific server with retry logic."""
            # Mark as running
            self.state.set_result(TestResult(
                test_id=item.test.id,
                model_id=item.model_id,
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
                response = ""
                async for chunk in stream_completion(
                    server_url=server_url,
                    model_id=item.model_id,
                    messages=item.test.to_messages(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout_seconds=self.timeout_seconds,
                    tools=item.test.tools
                ):
                    response += chunk
                return response

            try:
                response = await stream_with_retry()
                latency_ms = (time.time() - start_time) * 1000

                # Mark as completed
                self.state.set_result(TestResult(
                    test_id=item.test.id,
                    model_id=item.model_id,
                    status=TestStatus.COMPLETED,
                    response=response,
                    latency_ms=latency_ms
                ))

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                # Mark as error (fail loudly - record error, don't hide it)
                self.state.set_result(TestResult(
                    test_id=item.test.id,
                    model_id=item.model_id,
                    status=TestStatus.ERROR,
                    error=str(e),
                    latency_ms=latency_ms
                ))

        # Run dispatcher and yield state on each iteration
        async for _ in dispatcher.dispatch(work_items, execute_test):
            yield self.state

        # Final yield with complete state
        yield self.state
