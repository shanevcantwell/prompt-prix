"""Tests for ReactRunner orchestration.

Per ADR-006: orchestration tests mock MCP tools.
ReactRunner calls react_execute(), so we mock that.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock

from prompt_prix.benchmarks.base import BenchmarkCase


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def react_test():
    """A single ReAct-mode BenchmarkCase."""
    return BenchmarkCase(
        id="categorize_files",
        user="Organize the files",
        mode="react",
        system="You are a file organizer.",
        mock_tools={
            "read_file": {"./1.txt": "Content about animals"},
            "move_file": {"_default": "File moved"},
        },
        tools=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "move_file"}},
        ],
        max_iterations=10,
        expected_response="Files organized into folders",
    )


@pytest.fixture
def react_tests(react_test):
    """Multiple ReAct tests."""
    return [
        react_test,
        BenchmarkCase(
            id="sort_documents",
            user="Sort docs by type",
            mode="react",
            mock_tools={"read_file": {"_default": "doc content"}},
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        ),
    ]


def _make_react_result(test_id, model_id, completed=True, total_iterations=3,
                        valid_iterations=3, invalid_iterations=0,
                        cycle_detected=False):
    """Build a mock react_execute() return value."""
    return {
        "model_id": model_id,
        "iterations": [
            {
                "iteration": i + 1,
                "tool_call": {"id": f"call_{i}", "name": "read_file", "args": {}},
                "observation": "data",
                "success": True,
                "thought": None,
                "latency_ms": 50.0,
            }
            for i in range(total_iterations)
        ],
        "completed": completed,
        "final_response": "Done" if completed else None,
        "total_iterations": total_iterations,
        "total_latency_ms": total_iterations * 50.0,
        "cycle_detected": cycle_detected,
        "cycle_pattern": None,
        "termination_reason": None if completed else ("cycle_detected" if cycle_detected else "max_iterations"),
        "valid_iterations": valid_iterations,
        "invalid_iterations": invalid_iterations,
        "completion_rate": valid_iterations / total_iterations if total_iterations > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────
# REACTRUN STATE TESTS
# ─────────────────────────────────────────────────────────────────────

class TestReactRun:
    """Tests for ReactRun state model."""

    def test_create_empty(self):
        from prompt_prix.react.runner import ReactRun
        run = ReactRun(tests=["t1", "t2"], models=["m1", "m2"])
        assert run.total_count == 4
        assert run.completed_count == 0

    def test_set_and_get_result(self):
        from prompt_prix.react.runner import ReactRun, ReactResult
        run = ReactRun(tests=["t1"], models=["m1"])
        result = ReactResult(
            test_id="t1", model_id="m1",
            completed=True, total_iterations=3,
            valid_iterations=3, invalid_iterations=0,
        )
        run.set_result(result)
        got = run.get_result("t1", "m1")
        assert got is not None
        assert got.completed is True
        assert got.total_iterations == 3

    def test_to_grid_completed(self):
        from prompt_prix.react.runner import ReactRun, ReactResult
        run = ReactRun(tests=["t1"], models=["m1"])
        run.set_result(ReactResult(
            test_id="t1", model_id="m1",
            completed=True, total_iterations=5,
            valid_iterations=5, invalid_iterations=0,
        ))
        grid = run.to_grid()
        cell = grid.iloc[0, 1]  # First data column
        assert "✓" in cell
        assert "5" in cell

    def test_to_grid_cycle_detected(self):
        from prompt_prix.react.runner import ReactRun, ReactResult
        run = ReactRun(tests=["t1"], models=["m1"])
        run.set_result(ReactResult(
            test_id="t1", model_id="m1",
            completed=False, total_iterations=8,
            valid_iterations=6, invalid_iterations=2,
            cycle_detected=True,
            termination_reason="cycle_detected",
        ))
        grid = run.to_grid()
        cell = grid.iloc[0, 1]
        assert "⟳" in cell  # Cycle indicator

    def test_to_grid_max_iterations(self):
        from prompt_prix.react.runner import ReactRun, ReactResult
        run = ReactRun(tests=["t1"], models=["m1"])
        run.set_result(ReactResult(
            test_id="t1", model_id="m1",
            completed=False, total_iterations=15,
            valid_iterations=15, invalid_iterations=0,
            termination_reason="max_iterations",
        ))
        grid = run.to_grid()
        cell = grid.iloc[0, 1]
        assert "⚠" in cell  # Max iterations indicator


# ─────────────────────────────────────────────────────────────────────
# REACTRUNNER EXECUTION TESTS
# ─────────────────────────────────────────────────────────────────────

class TestReactRunner:
    """Tests for ReactRunner orchestration."""

    @pytest.mark.asyncio
    async def test_run_completes_all_cells(self, react_tests):
        """ReactRunner executes all (test, model) combinations."""
        models = ["model_a", "model_b"]
        call_log = []

        async def mock_react_execute(**kwargs):
            call_log.append((kwargs["model_id"], kwargs["initial_message"]))
            return _make_react_result(
                test_id="", model_id=kwargs["model_id"],
                completed=True, total_iterations=3,
            )

        with patch("prompt_prix.react.runner.react_execute", side_effect=mock_react_execute):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=react_tests, models=models)

            final_state = None
            async for state in runner.run():
                final_state = state

        # 2 tests × 2 models = 4 cells
        assert final_state.completed_count == 4
        assert len(call_log) == 4

    @pytest.mark.asyncio
    async def test_model_first_ordering(self, react_tests):
        """Work items are ordered model-first for VRAM efficiency."""
        models = ["model_a", "model_b"]
        call_order = []

        async def mock_react_execute(**kwargs):
            call_order.append(kwargs["model_id"])
            return _make_react_result("", kwargs["model_id"])

        with patch("prompt_prix.react.runner.react_execute", side_effect=mock_react_execute):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=react_tests, models=models)

            async for _ in runner.run():
                pass

        # model_a should run all its tests before model_b starts
        assert call_order == ["model_a", "model_a", "model_b", "model_b"]

    @pytest.mark.asyncio
    async def test_run_handles_react_execute_error(self, react_test):
        """Error in react_execute() is captured, not raised."""
        async def mock_react_execute(**kwargs):
            raise RuntimeError("Connection refused")

        with patch("prompt_prix.react.runner.react_execute", side_effect=mock_react_execute):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=[react_test], models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("categorize_files", "model_a")
        assert result is not None
        assert result.error is not None
        assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_yields_state_snapshots(self, react_test):
        """Runner yields state after each cell completes."""
        snapshot_count = 0

        async def mock_react_execute(**kwargs):
            return _make_react_result("", kwargs["model_id"])

        with patch("prompt_prix.react.runner.react_execute", side_effect=mock_react_execute):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=[react_test], models=["model_a", "model_b"])

            async for state in runner.run():
                snapshot_count += 1

        # At least one per cell + final
        assert snapshot_count >= 2
