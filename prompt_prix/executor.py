"""
TaskExecutor - Backend-agnostic task execution engine.

This module provides a unified interface for executing LLM tasks
across any adapter implementing the HostAdapter protocol.

Part of #73 adapter refactor - Layer 2 in the architecture.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_prix.adapters.base import HostAdapter


@dataclass
class Task:
    """
    Unit of work for the executor.

    Represents a single LLM completion request with all necessary parameters.
    """
    id: str
    model_id: str
    messages: list[dict]
    temperature: Optional[float]
    max_tokens: int
    timeout_seconds: int
    tools: Optional[list[dict]] = None


@dataclass
class TaskResult:
    """
    Result of executing a task.

    Captures the outcome, response text, timing, and any errors.
    """
    task_id: str
    model_id: str
    response: str
    status: Literal["success", "error"]
    duration_ms: int
    error: Optional[str] = None


class TaskExecutor:
    """
    Backend-agnostic task execution engine.

    Handles:
    - Concurrent execution respecting adapter limits
    - Error handling
    - Duration tracking

    Resource management is handled internally by adapters (stream_completion).
    Concurrency is managed via semaphore based on adapter.get_concurrency_limit().

    Usage:
        executor = TaskExecutor(adapter)
        async for result in executor.execute(tasks):
            process(result)
    """

    def __init__(self, adapter: "HostAdapter"):
        """
        Initialize executor with an adapter.

        Args:
            adapter: Any HostAdapter implementation (LMStudio, HuggingFace, etc.)
        """
        self.adapter = adapter
        self._semaphore = asyncio.Semaphore(adapter.get_concurrency_limit())

    async def execute(self, tasks: list[Task]) -> AsyncGenerator[TaskResult, None]:
        """
        Execute tasks with managed concurrency.

        Yields TaskResult as each task completes. Tasks run concurrently
        up to the adapter's concurrency limit.

        Args:
            tasks: List of tasks to execute

        Yields:
            TaskResult for each completed task (order not guaranteed)
        """
        # Create a queue to collect results as they complete
        result_queue: asyncio.Queue[TaskResult] = asyncio.Queue()

        async def run_task(task: Task) -> None:
            """Execute a single task and put result in queue."""
            result = await self._execute_single(task)
            await result_queue.put(result)

        # Start all tasks with semaphore-limited concurrency
        async_tasks = [asyncio.create_task(run_task(task)) for task in tasks]

        # Yield results as they complete
        for _ in range(len(tasks)):
            result = await result_queue.get()
            yield result

        # Wait for all tasks to finish (they should already be done)
        await asyncio.gather(*async_tasks, return_exceptions=True)

    async def _execute_single(self, task: Task) -> TaskResult:
        """
        Execute a single task.

        Resource management is handled internally by adapter.stream_completion().
        Concurrency is limited by semaphore.
        """
        start_time = time.perf_counter()

        async with self._semaphore:
            try:
                response_chunks = []
                async for chunk in self.adapter.stream_completion(
                    model_id=task.model_id,
                    messages=task.messages,
                    temperature=task.temperature,
                    max_tokens=task.max_tokens,
                    timeout_seconds=task.timeout_seconds,
                    tools=task.tools
                ):
                    response_chunks.append(chunk)

                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return TaskResult(
                    task_id=task.id,
                    model_id=task.model_id,
                    response="".join(response_chunks),
                    status="success",
                    duration_ms=duration_ms
                )

            except Exception as e:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return TaskResult(
                    task_id=task.id,
                    model_id=task.model_id,
                    response="",
                    status="error",
                    duration_ms=duration_ms,
                    error=str(e)
                )
