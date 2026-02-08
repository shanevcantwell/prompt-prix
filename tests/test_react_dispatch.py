"""Tests for execute_test_case() dispatch function.

The dispatch function is the ONLY place that reads test.mode.
Single-shot tests mock complete_stream, react tests mock react_step.
"""

import pytest
from unittest.mock import patch

from prompt_prix.benchmarks.base import BenchmarkCase
from prompt_prix.react.dispatch import (
    execute_test_case,
    CaseResult,
    ReactLoopIncomplete,
)
from prompt_prix.react.schemas import ReActIteration, ToolCall


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def single_shot_test():
    """Standard single-shot test case (mode=None)."""
    return BenchmarkCase(
        id="simple_test",
        user="What is 2 + 2?",
        system="You are a math tutor.",
    )


@pytest.fixture
def react_test():
    """React-mode test case."""
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


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

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
    """Build a mock react_step() return for completion."""
    return {
        "completed": True,
        "final_response": text,
        "new_iterations": [],
        "call_counter": call_counter,
        "latency_ms": 30.0,
    }


# ─────────────────────────────────────────────────────────────────────
# SINGLE-SHOT DISPATCH TESTS
# ─────────────────────────────────────────────────────────────────────

class TestSingleShotDispatch:
    """Tests for mode=None dispatch (single-shot via complete_stream)."""

    @pytest.mark.asyncio
    async def test_returns_response_and_latency(self, single_shot_test):
        """Single-shot returns CaseResult with response and latency."""
        async def mock_stream(**kwargs):
            yield "The answer is 4."
            yield "__LATENCY_MS__:150"

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_stream):
            result = await execute_test_case(
                test=single_shot_test, model_id="model_a",
            )

        assert isinstance(result, CaseResult)
        assert result.response == "The answer is 4."
        assert result.latency_ms == 150.0
        assert result.react_trace is None  # No trace for single-shot

    @pytest.mark.asyncio
    async def test_passes_seed_for_consistency(self, single_shot_test):
        """Seed is forwarded to complete_stream for consistency runs."""
        captured_kwargs = {}

        async def mock_stream(**kwargs):
            captured_kwargs.update(kwargs)
            yield "Response"
            yield "__LATENCY_MS__:50"

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_stream):
            await execute_test_case(
                test=single_shot_test, model_id="model_a", seed=42,
            )

        assert captured_kwargs["seed"] == 42

    @pytest.mark.asyncio
    async def test_no_seed_omits_param(self, single_shot_test):
        """Without seed, the kwarg is not passed to complete_stream."""
        captured_kwargs = {}

        async def mock_stream(**kwargs):
            captured_kwargs.update(kwargs)
            yield "Response"
            yield "__LATENCY_MS__:50"

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_stream):
            await execute_test_case(
                test=single_shot_test, model_id="model_a",
            )

        assert "seed" not in captured_kwargs

    @pytest.mark.asyncio
    async def test_error_propagates(self, single_shot_test):
        """Infrastructure errors propagate to caller."""
        async def mock_stream(**kwargs):
            raise ConnectionError("Connection refused")
            yield  # unreachable, but makes it an async generator

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_stream):
            with pytest.raises(ConnectionError, match="Connection refused"):
                await execute_test_case(
                    test=single_shot_test, model_id="model_a",
                )


# ─────────────────────────────────────────────────────────────────────
# REACT DISPATCH TESTS
# ─────────────────────────────────────────────────────────────────────

class TestReactDispatch:
    """Tests for mode='react' dispatch (react loop via react_step)."""

    @pytest.mark.asyncio
    async def test_completed_loop_returns_response(self, react_test):
        """React loop completes -> CaseResult with response + react_trace."""
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

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            result = await execute_test_case(
                test=react_test, model_id="model_a",
            )

        assert isinstance(result, CaseResult)
        assert result.response == "All files categorized."
        assert result.latency_ms > 0
        assert result.react_trace is not None
        assert result.react_trace["completed"] is True
        assert result.react_trace["total_iterations"] == 2
        assert result.react_trace["valid_iterations"] == 2

    @pytest.mark.asyncio
    async def test_immediate_text_response(self, react_test):
        """Model responds with text immediately -> 0 iterations, completed."""
        async def mock_step(**kwargs):
            return _make_step_completed(text="I already know: 42.")

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            result = await execute_test_case(
                test=react_test, model_id="model_a",
            )

        assert result.response == "I already know: 42."
        assert result.react_trace["total_iterations"] == 0
        assert result.react_trace["completed"] is True

    @pytest.mark.asyncio
    async def test_max_iterations_raises_incomplete(self, react_test):
        """Loop exhausting max_iterations raises ReactLoopIncomplete."""
        step_count = {"n": 0}

        async def mock_step(**kwargs):
            step_count["n"] += 1
            return _make_step_tool_call(
                name="read_file",
                args={"path": f"./file_{step_count['n']}.txt"},
                call_counter_start=kwargs.get("call_counter", 0),
            )

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            with pytest.raises(ReactLoopIncomplete) as exc_info:
                await execute_test_case(
                    test=react_test, model_id="model_a",
                )

        assert "max_iterations" in exc_info.value.reason
        assert exc_info.value.react_trace["termination_reason"] == "max_iterations"
        assert exc_info.value.react_trace["total_iterations"] == 10

    @pytest.mark.asyncio
    async def test_cycle_detection_raises_incomplete(self):
        """Repeating tool call pattern -> ReactLoopIncomplete with cycle info."""
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
            path = "./a.txt" if step_count["n"] % 2 == 1 else "./b.txt"
            return _make_step_tool_call(
                name="read_file",
                args={"path": path},
                call_counter_start=kwargs.get("call_counter", 0),
            )

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            with pytest.raises(ReactLoopIncomplete) as exc_info:
                await execute_test_case(
                    test=test, model_id="model_a",
                )

        assert exc_info.value.react_trace["cycle_detected"] is True
        assert exc_info.value.react_trace["termination_reason"] == "cycle_detected"

    @pytest.mark.asyncio
    async def test_react_trace_contains_iteration_detail(self, react_test):
        """Completed react loop includes serialized iteration detail in react_trace."""
        step_count = {"n": 0}

        async def mock_step(**kwargs):
            step_count["n"] += 1
            if step_count["n"] == 1:
                return _make_step_tool_call(
                    name="read_file",
                    args={"path": "./1.txt"},
                    call_counter_start=kwargs.get("call_counter", 0),
                )
            return _make_step_completed(text="Done.")

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            result = await execute_test_case(
                test=react_test, model_id="model_a",
            )

        assert "iterations" in result.react_trace
        assert len(result.react_trace["iterations"]) == 1
        assert result.react_trace["iterations"][0]["tool_call"]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_infrastructure_error_propagates(self, react_test):
        """Error in react_step() propagates (not caught by dispatch)."""
        async def mock_step(**kwargs):
            raise RuntimeError("Connection refused")

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            with pytest.raises(RuntimeError, match="Connection refused"):
                await execute_test_case(
                    test=react_test, model_id="model_a",
                )
