"""Tests for ReactRunner orchestration.

Per ADR-006: orchestration tests mock MCP tools.
ReactRunner calls react_step(), so we mock that.
"""

import pytest
from unittest.mock import patch

from prompt_prix.benchmarks.base import BenchmarkCase
from prompt_prix.react.schemas import ReActIteration, ToolCall


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


def _make_step_tool_call(name="read_file", args=None, call_counter_start=0):
    """Build a mock react_step() return for a tool call."""
    args = args or {}
    return {
        "completed": False,
        "final_response": None,
        "new_iterations": [
            ReActIteration(
                iteration=1,
                tool_call=ToolCall(
                    id=f"call_{call_counter_start + 1}",
                    name=name,
                    args=args,
                ),
                observation="mock data",
                success=True,
                latency_ms=50.0,
            ),
        ],
        "call_counter": call_counter_start + 1,
        "latency_ms": 50.0,
    }


def _make_step_completed(text="Done.", call_counter=0):
    """Build a mock react_step() return for completion (no tool calls)."""
    return {
        "completed": True,
        "final_response": text,
        "new_iterations": [],
        "call_counter": call_counter,
        "latency_ms": 30.0,
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
    """Tests for ReactRunner orchestration — owns the loop."""

    @pytest.mark.asyncio
    async def test_run_completes_all_cells(self, react_tests):
        """ReactRunner executes all (test, model) combinations."""
        models = ["model_a", "model_b"]
        call_log = []

        async def mock_step(**kwargs):
            call_log.append(kwargs["model_id"])
            return _make_step_completed()

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
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

        async def mock_step(**kwargs):
            call_order.append(kwargs["model_id"])
            return _make_step_completed()

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=react_tests, models=models)

            async for _ in runner.run():
                pass

        # model_a should run all its tests before model_b starts
        assert call_order == ["model_a", "model_a", "model_b", "model_b"]

    @pytest.mark.asyncio
    async def test_run_handles_step_error(self, react_test):
        """Error in react_step() is captured, not raised."""
        async def mock_step(**kwargs):
            raise RuntimeError("Connection refused")

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
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

        async def mock_step(**kwargs):
            return _make_step_completed()

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=[react_test], models=["model_a", "model_b"])

            async for state in runner.run():
                snapshot_count += 1

        # At least one per cell + final
        assert snapshot_count >= 2

    @pytest.mark.asyncio
    async def test_loop_completes_after_tool_calls(self, react_test):
        """Model makes 2 tool calls then text → completed with 2 iterations."""
        step_count = {"n": 0}

        async def mock_step(**kwargs):
            step_count["n"] += 1
            if step_count["n"] <= 2:
                return _make_step_tool_call(
                    name="read_file",
                    args={"path": f"./file{step_count['n']}.txt"},
                    call_counter_start=kwargs.get("call_counter", 0),
                )
            return _make_step_completed(
                text="All files categorized.",
                call_counter=kwargs.get("call_counter", 0),
            )

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=[react_test], models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("categorize_files", "model_a")
        assert result.completed is True
        assert result.total_iterations == 2
        assert result.valid_iterations == 2
        assert result.final_response == "All files categorized."

    @pytest.mark.asyncio
    async def test_max_iterations_enforced(self, react_test):
        """Loop terminates at max_iterations when model keeps calling tools."""
        step_count = {"n": 0}

        async def mock_step(**kwargs):
            # Unique args each call to avoid triggering cycle detection
            step_count["n"] += 1
            return _make_step_tool_call(
                name="read_file",
                args={"path": f"./file_{step_count['n']}.txt"},
                call_counter_start=kwargs.get("call_counter", 0),
            )

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=[react_test], models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("categorize_files", "model_a")
        assert result.completed is False
        assert result.total_iterations == 10  # max_iterations from fixture
        assert result.termination_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_cycle_detection(self):
        """Repeating tool call pattern → stagnation detected."""
        test = BenchmarkCase(
            id="cycle_test",
            user="Do it",
            mode="react",
            mock_tools={"read_file": {"_default": "data"}},
            tools=[{"type": "function", "function": {"name": "read_file"}}],
            max_iterations=20,
        )
        step_count = {"n": 0}

        async def mock_step(**kwargs):
            step_count["n"] += 1
            # Alternate A, B, A, B... to create a cycle
            path = "./a.txt" if step_count["n"] % 2 == 1 else "./b.txt"
            return _make_step_tool_call(
                name="read_file",
                args={"path": path},
                call_counter_start=kwargs.get("call_counter", 0),
            )

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=[test], models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("cycle_test", "model_a")
        assert result.completed is False
        assert result.cycle_detected is True
        assert result.termination_reason == "cycle_detected"

    @pytest.mark.asyncio
    async def test_immediate_text_response(self, react_test):
        """Model responds with text immediately → 0 iterations, completed."""
        async def mock_step(**kwargs):
            return _make_step_completed(text="I already know: 42.")

        with patch("prompt_prix.react.runner.react_step", side_effect=mock_step):
            from prompt_prix.react.runner import ReactRunner
            runner = ReactRunner(tests=[react_test], models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("categorize_files", "model_a")
        assert result.completed is True
        assert result.total_iterations == 0
        assert result.final_response == "I already know: 42."
