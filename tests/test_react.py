"""
Tests for the react package: cycle detection and trace schemas.

Cycle detection tests ported from langgraph-agentic-scaffold
(app/tests/unit/test_cycle_detection.py). Import path adjusted.

Schema tests verify Pydantic model instantiation and exception hierarchy.
"""

import pytest

from prompt_prix.react.cycle_detection import detect_cycle, detect_cycle_with_pattern
from prompt_prix.react.schemas import (
    ToolCall,
    ReActIteration,
    ReActLoopTerminated,
    MaxIterationsExceeded,
    StagnationDetected,
)


# ─────────────────────────────────────────────────────────────────────
# CYCLE DETECTION (ported from LAS)
# ─────────────────────────────────────────────────────────────────────


class TestDetectCycle:
    """Test detect_cycle function."""

    def test_single_item_repeated(self):
        """Period-1 cycle: A-A-A-A."""
        history = ['a', 'a', 'a', 'a']
        assert detect_cycle(history, min_repetitions=2) == 1
        assert detect_cycle(history, min_repetitions=3) == 1
        assert detect_cycle(history, min_repetitions=4) == 1

    def test_two_step_cycle(self):
        """Period-2 cycle: A-B-A-B-A-B."""
        history = ['a', 'b', 'a', 'b', 'a', 'b']
        assert detect_cycle(history, min_repetitions=2) == 2
        assert detect_cycle(history, min_repetitions=3) == 2

    def test_four_step_cycle(self):
        """Period-4 cycle: A-B-C-D-A-B-C-D (batch of 4 files)."""
        history = ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd']
        assert detect_cycle(history, min_repetitions=2) == 4

    def test_four_step_cycle_with_more_repetitions(self):
        """Period-4 cycle repeated 3 times."""
        history = ['a', 'b', 'c', 'd'] * 3
        assert detect_cycle(history, min_repetitions=2) == 4
        assert detect_cycle(history, min_repetitions=3) == 4

    def test_no_cycle_short_history(self):
        """Not enough items to detect cycle."""
        assert detect_cycle(['a'], min_repetitions=2) is None
        assert detect_cycle(['a', 'b'], min_repetitions=2) is None
        assert detect_cycle(['a', 'b', 'c'], min_repetitions=2) is None

    def test_no_cycle_different_items(self):
        """No repeating pattern."""
        history = ['a', 'b', 'c', 'd', 'e', 'f']
        assert detect_cycle(history, min_repetitions=2) is None

    def test_cycle_at_end_only(self):
        """Cycle detection should focus on the end of history."""
        history = ['x', 'y', 'z', 'a', 'b', 'a', 'b']
        assert detect_cycle(history, min_repetitions=2) == 2

    def test_shortest_cycle_found_first(self):
        """When multiple cycles match, shortest period wins."""
        history = ['a', 'a', 'a', 'a']
        assert detect_cycle(history, min_repetitions=2) == 1

    def test_max_period_limit(self):
        """Respect max_period parameter."""
        history = ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd']
        assert detect_cycle(history, min_repetitions=2, max_period=3) is None
        assert detect_cycle(history, min_repetitions=2, max_period=4) == 4

    def test_tool_call_signatures(self):
        """Real-world tool call signature cycle (LAS Issue #78 scenario)."""
        signatures = [
            "read_file:path=sort_by_contents/c.txt",
            "read_file:path=sort_by_contents/k.txt",
            "read_file:path=sort_by_contents/s.txt",
            "read_file:path=sort_by_contents/v.txt",
            "read_file:path=sort_by_contents/c.txt",
            "read_file:path=sort_by_contents/k.txt",
            "read_file:path=sort_by_contents/s.txt",
            "read_file:path=sort_by_contents/v.txt",
        ]
        assert detect_cycle(signatures, min_repetitions=2) == 4

    def test_empty_history(self):
        """Empty history returns None."""
        assert detect_cycle([], min_repetitions=2) is None


class TestDetectCycleWithPattern:
    """Test detect_cycle_with_pattern function."""

    def test_returns_pattern(self):
        """Should return both period and pattern."""
        history = ['a', 'b', 'a', 'b']
        period, pattern = detect_cycle_with_pattern(history, min_repetitions=2)
        assert period == 2
        assert pattern == ['a', 'b']

    def test_four_item_pattern(self):
        """Four-item pattern from batch operation."""
        history = ['c.txt', 'k.txt', 's.txt', 'v.txt', 'c.txt', 'k.txt', 's.txt', 'v.txt']
        period, pattern = detect_cycle_with_pattern(history, min_repetitions=2)
        assert period == 4
        assert pattern == ['c.txt', 'k.txt', 's.txt', 'v.txt']

    def test_no_cycle_returns_none(self):
        """No cycle returns (None, None)."""
        history = ['a', 'b', 'c', 'd']
        period, pattern = detect_cycle_with_pattern(history, min_repetitions=2)
        assert period is None
        assert pattern is None

    def test_single_item_pattern(self):
        """Period-1 cycle has single-item pattern."""
        history = ['a', 'a', 'a']
        period, pattern = detect_cycle_with_pattern(history, min_repetitions=2)
        assert period == 1
        assert pattern == ['a']


class TestMinRepetitionsEdgeCases:
    """Test min_repetitions parameter edge cases."""

    def test_min_repetitions_1_not_useful(self):
        """min_repetitions=1 would match everything, so not typically used."""
        history = ['a', 'b', 'c']
        assert detect_cycle(history, min_repetitions=1) == 1

    def test_high_min_repetitions(self):
        """Need enough history for high min_repetitions."""
        history = ['a', 'b'] * 5  # 10 items
        assert detect_cycle(history, min_repetitions=5) == 2
        assert detect_cycle(history, min_repetitions=6) is None  # Would need 12 items


# ─────────────────────────────────────────────────────────────────────
# SCHEMA TESTS
# ─────────────────────────────────────────────────────────────────────


class TestToolCall:
    """Test ToolCall Pydantic model."""

    def test_basic_instantiation(self):
        tc = ToolCall(id="call_1", name="read_file", args={"path": "./1.txt"})
        assert tc.id == "call_1"
        assert tc.name == "read_file"
        assert tc.args == {"path": "./1.txt"}

    def test_default_args(self):
        tc = ToolCall(id="call_2", name="list_directory")
        assert tc.args == {}

    def test_serialization_roundtrip(self):
        tc = ToolCall(id="call_3", name="move_file", args={"src": "a.txt", "dst": "b/"})
        data = tc.model_dump()
        restored = ToolCall(**data)
        assert restored == tc


class TestReActIteration:
    """Test ReActIteration Pydantic model."""

    def test_successful_iteration(self):
        tc = ToolCall(id="call_1", name="read_file", args={"path": "test.txt"})
        step = ReActIteration(
            iteration=1,
            tool_call=tc,
            observation="File contents here",
            success=True,
            thought="I should read this file first",
            latency_ms=42.5,
        )
        assert step.iteration == 1
        assert step.tool_call.name == "read_file"
        assert step.success is True
        assert step.thought == "I should read this file first"
        assert step.latency_ms == 42.5

    def test_failed_iteration(self):
        tc = ToolCall(id="call_2", name="unknown_tool", args={})
        step = ReActIteration(
            iteration=3,
            tool_call=tc,
            observation="Error: No mock response for unknown_tool({})",
            success=False,
        )
        assert step.success is False
        assert step.thought is None
        assert step.latency_ms == 0.0

    def test_serialization_roundtrip(self):
        tc = ToolCall(id="call_1", name="read_file", args={"path": "x"})
        step = ReActIteration(
            iteration=1, tool_call=tc,
            observation="data", success=True, latency_ms=10.0,
        )
        data = step.model_dump()
        restored = ReActIteration(**data)
        assert restored == step


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_max_iterations_is_loop_terminated(self):
        assert issubclass(MaxIterationsExceeded, ReActLoopTerminated)

    def test_stagnation_is_loop_terminated(self):
        assert issubclass(StagnationDetected, ReActLoopTerminated)

    def test_loop_terminated_is_exception(self):
        assert issubclass(ReActLoopTerminated, Exception)

    def test_catch_base_catches_both(self):
        """Catching ReActLoopTerminated catches both subclasses."""
        for exc_class in (MaxIterationsExceeded, StagnationDetected):
            with pytest.raises(ReActLoopTerminated):
                raise exc_class("test")


class TestPackageImports:
    """Test that package __init__.py exports work."""

    def test_import_from_package(self):
        from prompt_prix.react import (
            detect_cycle,
            detect_cycle_with_pattern,
            ToolCall,
            ReActIteration,
            ReActLoopTerminated,
            MaxIterationsExceeded,
            StagnationDetected,
        )
        # Verify they're the actual classes, not None
        assert callable(detect_cycle)
        assert callable(detect_cycle_with_pattern)
        assert ToolCall is not None
        assert ReActIteration is not None
