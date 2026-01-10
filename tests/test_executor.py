"""
Tests for TaskExecutor - backend-agnostic task execution.

These tests define the expected behavior of TaskExecutor BEFORE implementation.
They should FAIL initially (TaskExecutor doesn't exist yet).

Phase 2 of #73 adapter refactor.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestTaskExecutor:
    """Tests for TaskExecutor - backend-agnostic execution engine."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock HostAdapter for testing."""
        adapter = MagicMock()
        adapter.get_concurrency_limit.return_value = 2

        # Mock stream_completion to yield chunks
        async def mock_stream(*args, **kwargs):
            yield "Hello "
            yield "world!"

        adapter.stream_completion = mock_stream
        return adapter

    @pytest.mark.asyncio
    async def test_execute_single_task(self, mock_adapter):
        """Execute a single task through the adapter."""
        from prompt_prix.executor import TaskExecutor, Task

        executor = TaskExecutor(mock_adapter)
        task = Task(
            id="test-1",
            model_id="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=60
        )

        results = [r async for r in executor.execute([task])]

        assert len(results) == 1
        assert results[0].task_id == "test-1"
        assert results[0].model_id == "model-a"
        assert results[0].status == "success"
        assert results[0].response == "Hello world!"

    @pytest.mark.asyncio
    async def test_execute_multiple_tasks(self, mock_adapter):
        """Execute multiple tasks and get results for each."""
        from prompt_prix.executor import TaskExecutor, Task

        executor = TaskExecutor(mock_adapter)
        tasks = [
            Task(id="t1", model_id="model-a", messages=[{"role": "user", "content": "Hi"}],
                 temperature=0.7, max_tokens=100, timeout_seconds=60),
            Task(id="t2", model_id="model-b", messages=[{"role": "user", "content": "Hey"}],
                 temperature=0.7, max_tokens=100, timeout_seconds=60),
        ]

        results = [r async for r in executor.execute(tasks)]

        assert len(results) == 2
        task_ids = {r.task_id for r in results}
        assert task_ids == {"t1", "t2"}

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_adapter):
        """Errors in stream_completion are captured in TaskResult."""
        from prompt_prix.executor import TaskExecutor, Task

        # Make stream_completion raise an error
        async def failing_stream(*args, **kwargs):
            raise RuntimeError("Model error")
            yield  # unreachable, but makes it a generator

        mock_adapter.stream_completion = failing_stream

        executor = TaskExecutor(mock_adapter)
        task = Task(
            id="test-1",
            model_id="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=60
        )

        results = [r async for r in executor.execute([task])]

        # Task should complete with error status
        assert len(results) == 1
        assert results[0].status == "error"
        assert "Model error" in results[0].error

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self, mock_adapter):
        """Executor respects adapter.get_concurrency_limit()."""
        from prompt_prix.executor import TaskExecutor, Task

        # Track concurrent stream_completion calls
        concurrent_count = 0
        max_concurrent = 0

        async def tracking_stream(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            try:
                await asyncio.sleep(0.05)  # Delay to ensure concurrency is measurable
                yield "response"
            finally:
                concurrent_count -= 1

        mock_adapter.stream_completion = tracking_stream
        mock_adapter.get_concurrency_limit.return_value = 2

        executor = TaskExecutor(mock_adapter)
        tasks = [
            Task(id=f"t{i}", model_id="model-a", messages=[{"role": "user", "content": "Hi"}],
                 temperature=0.7, max_tokens=100, timeout_seconds=60)
            for i in range(5)
        ]

        results = [r async for r in executor.execute(tasks)]

        assert len(results) == 5
        # Max concurrent should not exceed limit
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_result_includes_duration(self, mock_adapter):
        """TaskResult includes execution duration in milliseconds."""
        from prompt_prix.executor import TaskExecutor, Task

        # Add small delay to measure
        async def slow_stream(*args, **kwargs):
            await asyncio.sleep(0.01)
            yield "response"

        mock_adapter.stream_completion = slow_stream

        executor = TaskExecutor(mock_adapter)
        task = Task(
            id="test-1",
            model_id="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=60
        )

        results = [r async for r in executor.execute([task])]

        assert results[0].duration_ms >= 10  # At least 10ms from sleep

    @pytest.mark.asyncio
    async def test_passes_tools_to_adapter(self, mock_adapter):
        """Tools parameter is passed through to adapter."""
        from prompt_prix.executor import TaskExecutor, Task

        captured_kwargs = {}

        async def capturing_stream(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield "response"

        mock_adapter.stream_completion = capturing_stream

        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        executor = TaskExecutor(mock_adapter)
        task = Task(
            id="test-1",
            model_id="model-a",
            messages=[{"role": "user", "content": "Weather?"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=60,
            tools=tools
        )

        async for _ in executor.execute([task]):
            pass

        assert captured_kwargs.get("tools") == tools


class TestTaskDataclass:
    """Tests for Task dataclass structure."""

    def test_task_required_fields(self):
        """Task has required fields."""
        from prompt_prix.executor import Task

        task = Task(
            id="test-1",
            model_id="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=60
        )

        assert task.id == "test-1"
        assert task.model_id == "model-a"
        assert task.messages == [{"role": "user", "content": "Hello"}]
        assert task.temperature == 0.7
        assert task.max_tokens == 100
        assert task.timeout_seconds == 60

    def test_task_optional_fields(self):
        """Task has optional tools field."""
        from prompt_prix.executor import Task

        task = Task(
            id="test-1",
            model_id="model-a",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=60,
            tools=[{"type": "function", "function": {"name": "test"}}]
        )

        assert task.tools == [{"type": "function", "function": {"name": "test"}}]


class TestTaskResultDataclass:
    """Tests for TaskResult dataclass structure."""

    def test_taskresult_success(self):
        """TaskResult captures successful execution."""
        from prompt_prix.executor import TaskResult

        result = TaskResult(
            task_id="test-1",
            model_id="model-a",
            response="Hello world!",
            status="success",
            duration_ms=150
        )

        assert result.task_id == "test-1"
        assert result.model_id == "model-a"
        assert result.response == "Hello world!"
        assert result.status == "success"
        assert result.duration_ms == 150
        assert result.error is None

    def test_taskresult_error(self):
        """TaskResult captures error with message."""
        from prompt_prix.executor import TaskResult

        result = TaskResult(
            task_id="test-1",
            model_id="model-a",
            response="",
            status="error",
            duration_ms=50,
            error="Connection timeout"
        )

        assert result.status == "error"
        assert result.error == "Connection timeout"
