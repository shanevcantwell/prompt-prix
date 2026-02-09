"""Tests for react_step() MCP tool — single ReAct iteration primitive.

Per ADR-006: MCP tool tests mock the adapter layer.
react_step() calls complete_stream(), so we mock that.
"""

import json
import pytest
from unittest.mock import patch

from prompt_prix.react.schemas import ToolCall, ReActIteration


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _tool_call_sentinel(name: str, arguments: dict) -> str:
    """Build a __TOOL_CALLS__ sentinel string."""
    return f"__TOOL_CALLS__:{json.dumps([{'name': name, 'arguments': json.dumps(arguments)}])}"


def _make_stream(*chunks):
    """Create an async generator that yields chunks then a latency sentinel."""
    async def stream(**kwargs):
        for chunk in chunks:
            yield chunk
        yield "__LATENCY_MS__:100"
    return stream


# ─────────────────────────────────────────────────────────────────────
# MOCK TOOL DISPATCH TESTS
# ─────────────────────────────────────────────────────────────────────

class TestDispatchMock:
    """Test dispatch_mock() resolution logic."""

    def test_exact_args_match(self):
        from prompt_prix.mcp.tools.react_step import dispatch_mock

        mock_tools = {
            "read_file": {
                json.dumps({"path": "./1.txt"}, sort_keys=True): "File contents here"
            }
        }
        result = dispatch_mock("read_file", {"path": "./1.txt"}, mock_tools)
        assert result == "File contents here"

    def test_first_arg_value_match(self):
        from prompt_prix.mcp.tools.react_step import dispatch_mock

        mock_tools = {
            "read_file": {
                "./1.txt": "File contents here"
            }
        }
        result = dispatch_mock("read_file", {"path": "./1.txt"}, mock_tools)
        assert result == "File contents here"

    def test_default_fallback(self):
        from prompt_prix.mcp.tools.react_step import dispatch_mock

        mock_tools = {
            "move_file": {
                "_default": "File moved"
            }
        }
        result = dispatch_mock("move_file", {"src": "a.txt", "dst": "b/"}, mock_tools)
        assert result == "File moved"

    def test_no_match_returns_error(self):
        from prompt_prix.mcp.tools.react_step import dispatch_mock

        mock_tools = {
            "read_file": {
                "./known.txt": "Known content"
            }
        }
        result = dispatch_mock("read_file", {"path": "./unknown.txt"}, mock_tools)
        assert "Error" in result
        assert "read_file" in result

    def test_unknown_tool_returns_error(self):
        from prompt_prix.mcp.tools.react_step import dispatch_mock

        mock_tools = {}
        result = dispatch_mock("nonexistent", {"arg": "val"}, mock_tools)
        assert "Error" in result


# ─────────────────────────────────────────────────────────────────────
# MESSAGE BUILDING TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBuildMessages:
    """Test build_react_messages() trace serialization."""

    def test_empty_trace(self):
        from prompt_prix.mcp.tools.react_step import build_react_messages

        msgs = build_react_messages("You are helpful.", "Do the thing.", [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_trace_produces_assistant_tool_pairs(self):
        from prompt_prix.mcp.tools.react_step import build_react_messages

        trace = [
            ReActIteration(
                iteration=1,
                tool_call=ToolCall(id="call_1", name="read_file", args={"path": "x"}),
                observation="file data",
                success=True,
                thought="Let me read this file",
            )
        ]
        msgs = build_react_messages("sys", "goal", trace)
        assert len(msgs) == 4  # system, user, assistant, tool
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["tool_calls"][0]["function"]["name"] == "read_file"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["content"] == "file data"
        assert msgs[3]["tool_call_id"] == "call_1"


# ─────────────────────────────────────────────────────────────────────
# REACT_STEP SINGLE-ITERATION TESTS
# ─────────────────────────────────────────────────────────────────────

class TestReactStep:
    """Tests for react_step() — one model call, one result."""

    @pytest.mark.asyncio
    async def test_model_completes_with_text(self):
        """Model responds with text only (no tool calls) → completed."""
        async def mock_stream(**kwargs):
            yield "The answer is 42."
            yield "__LATENCY_MS__:50"

        with patch("prompt_prix.mcp.tools.react_step.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_step import react_step

            result = await react_step(
                model_id="test-model",
                system_prompt="sys",
                initial_message="What is the answer?",
                trace=[],
                mock_tools={},
                tools=[],
            )

        assert result["completed"] is True
        assert result["final_response"] == "The answer is 42."
        assert result["new_iterations"] == []
        assert result["latency_ms"] == 50.0

    @pytest.mark.asyncio
    async def test_model_makes_tool_call(self):
        """Model makes a tool call → returns new iteration, not completed."""
        sentinel = _tool_call_sentinel("read_file", {"path": "./1.txt"})

        async def mock_stream(**kwargs):
            yield "I'll read the file."
            yield sentinel
            yield "__LATENCY_MS__:80"

        with patch("prompt_prix.mcp.tools.react_step.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_step import react_step

            result = await react_step(
                model_id="test-model",
                system_prompt="sys",
                initial_message="Read the file",
                trace=[],
                mock_tools={"read_file": {"./1.txt": "File contents here"}},
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        assert result["completed"] is False
        assert result["final_response"] is None
        assert len(result["new_iterations"]) == 1

        iteration = result["new_iterations"][0]
        assert isinstance(iteration, ReActIteration)
        assert iteration.tool_call.name == "read_file"
        assert iteration.observation == "File contents here"
        assert iteration.success is True
        assert iteration.thought == "I'll read the file."

    @pytest.mark.asyncio
    async def test_garbled_tool_args(self):
        """Model produces unparseable tool args → invalid iteration."""
        async def mock_stream(**kwargs):
            yield "I'll read the file."
            yield '__TOOL_CALLS__:[{"name":"read_file","arguments":"not valid json"}]'
            yield "__LATENCY_MS__:50"

        with patch("prompt_prix.mcp.tools.react_step.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_step import react_step

            result = await react_step(
                model_id="test-model",
                system_prompt="sys",
                initial_message="Read it",
                trace=[],
                mock_tools={"read_file": {"./1.txt": "data"}},
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        assert result["completed"] is False
        assert len(result["new_iterations"]) == 1
        assert result["new_iterations"][0].success is False
        assert "Error" in result["new_iterations"][0].observation

    @pytest.mark.asyncio
    async def test_call_counter_threads_through(self):
        """call_counter increments and returns updated value."""
        sentinel = _tool_call_sentinel("read_file", {"path": "./1.txt"})

        async def mock_stream(**kwargs):
            yield sentinel
            yield "__LATENCY_MS__:50"

        with patch("prompt_prix.mcp.tools.react_step.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_step import react_step

            result = await react_step(
                model_id="test-model",
                system_prompt="sys",
                initial_message="Read",
                trace=[],
                mock_tools={"read_file": {"_default": "data"}},
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                call_counter=5,
            )

        assert result["call_counter"] == 6
        assert result["new_iterations"][0].tool_call.id == "call_6"

    @pytest.mark.asyncio
    async def test_trace_passed_to_message_builder(self):
        """Previous trace entries are included in messages sent to model."""
        existing_trace = [
            ReActIteration(
                iteration=1,
                tool_call=ToolCall(id="call_1", name="list_dir", args={"path": "."}),
                observation="file1.txt\nfile2.txt",
                success=True,
            )
        ]

        async def mock_stream(**kwargs):
            # Verify trace was included in messages
            messages = kwargs.get("messages", [])
            assert len(messages) == 4  # system, user, assistant, tool
            assert messages[2]["role"] == "assistant"
            assert messages[3]["role"] == "tool"
            yield "All done."
            yield "__LATENCY_MS__:30"

        with patch("prompt_prix.mcp.tools.react_step.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_step import react_step

            result = await react_step(
                model_id="test-model",
                system_prompt="sys",
                initial_message="List and report",
                trace=existing_trace,
                mock_tools={},
                tools=[],
                call_counter=1,
            )

        assert result["completed"] is True
        assert result["final_response"] == "All done."
