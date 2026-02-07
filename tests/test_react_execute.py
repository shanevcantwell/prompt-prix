"""Tests for react_execute() MCP tool.

Per ADR-006: orchestration tests mock MCP tools.
react_execute() calls complete_stream(), so we mock that.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock

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
    """Test _dispatch_mock() resolution logic."""

    def test_exact_args_match(self):
        from prompt_prix.mcp.tools.react_execute import _dispatch_mock

        mock_tools = {
            "read_file": {
                json.dumps({"path": "./1.txt"}, sort_keys=True): "File contents here"
            }
        }
        result = _dispatch_mock("read_file", {"path": "./1.txt"}, mock_tools)
        assert result == "File contents here"

    def test_first_arg_value_match(self):
        from prompt_prix.mcp.tools.react_execute import _dispatch_mock

        mock_tools = {
            "read_file": {
                "./1.txt": "File contents here"
            }
        }
        result = _dispatch_mock("read_file", {"path": "./1.txt"}, mock_tools)
        assert result == "File contents here"

    def test_default_fallback(self):
        from prompt_prix.mcp.tools.react_execute import _dispatch_mock

        mock_tools = {
            "move_file": {
                "_default": "File moved"
            }
        }
        result = _dispatch_mock("move_file", {"src": "a.txt", "dst": "b/"}, mock_tools)
        assert result == "File moved"

    def test_no_match_returns_error(self):
        from prompt_prix.mcp.tools.react_execute import _dispatch_mock

        mock_tools = {
            "read_file": {
                "./known.txt": "Known content"
            }
        }
        result = _dispatch_mock("read_file", {"path": "./unknown.txt"}, mock_tools)
        assert "Error" in result
        assert "read_file" in result

    def test_unknown_tool_returns_error(self):
        from prompt_prix.mcp.tools.react_execute import _dispatch_mock

        mock_tools = {}
        result = _dispatch_mock("nonexistent", {"arg": "val"}, mock_tools)
        assert "Error" in result


# ─────────────────────────────────────────────────────────────────────
# MESSAGE BUILDING TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBuildMessages:
    """Test _build_messages() trace serialization."""

    def test_empty_trace(self):
        from prompt_prix.mcp.tools.react_execute import _build_messages

        msgs = _build_messages("You are helpful.", "Do the thing.", [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_trace_produces_assistant_tool_pairs(self):
        from prompt_prix.mcp.tools.react_execute import _build_messages

        trace = [
            ReActIteration(
                iteration=1,
                tool_call=ToolCall(id="call_1", name="read_file", args={"path": "x"}),
                observation="file data",
                success=True,
                thought="Let me read this file",
            )
        ]
        msgs = _build_messages("sys", "goal", trace)
        assert len(msgs) == 4  # system, user, assistant, tool
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["tool_calls"][0]["function"]["name"] == "read_file"
        assert msgs[3]["role"] == "tool"
        assert msgs[3]["content"] == "file data"
        assert msgs[3]["tool_call_id"] == "call_1"


# ─────────────────────────────────────────────────────────────────────
# REACT_EXECUTE INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────

class TestReactExecute:
    """End-to-end tests for react_execute() with mocked complete_stream."""

    @pytest.mark.asyncio
    async def test_model_completes_after_tool_calls(self):
        """Model makes 2 tool calls then responds with text → completed."""
        call_count = {"n": 0}

        async def mock_stream(**kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # Tool call iterations
                sentinel = _tool_call_sentinel(
                    "read_file", {"path": f"./file{call_count['n']}.txt"}
                )
                yield "I'll read the file."
                yield sentinel
                yield "__LATENCY_MS__:50"
            else:
                # Final text response (no tool calls)
                yield "All files have been categorized."
                yield "__LATENCY_MS__:30"

        with patch("prompt_prix.mcp.tools.react_execute.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_execute import react_execute

            result = await react_execute(
                model_id="test-model",
                system_prompt="You are a file organizer.",
                initial_message="Organize these files",
                mock_tools={
                    "read_file": {
                        "./file1.txt": "Content about animals",
                        "./file2.txt": "Content about fruits",
                    }
                },
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                max_iterations=10,
            )

        assert result["completed"] is True
        assert result["total_iterations"] == 2
        assert result["valid_iterations"] == 2
        assert result["invalid_iterations"] == 0
        assert result["cycle_detected"] is False
        assert len(result["iterations"]) == 2
        assert result["iterations"][0]["tool_call"]["name"] == "read_file"
        assert result["final_response"] == "All files have been categorized."

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self):
        """Model keeps calling tools past max_iterations → terminated."""
        async def mock_stream(**kwargs):
            sentinel = _tool_call_sentinel("read_file", {"path": "./loop.txt"})
            yield "Reading again..."
            yield sentinel
            yield "__LATENCY_MS__:50"

        with patch("prompt_prix.mcp.tools.react_execute.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_execute import react_execute

            result = await react_execute(
                model_id="test-model",
                system_prompt="sys",
                initial_message="Do it",
                mock_tools={"read_file": {"_default": "file data"}},
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                max_iterations=3,
            )

        assert result["completed"] is False
        assert result["total_iterations"] == 3
        assert result["termination_reason"] == "max_iterations"

    @pytest.mark.asyncio
    async def test_cycle_detected(self):
        """Model repeats same tool call pattern → stagnation detected."""
        call_count = {"n": 0}

        async def mock_stream(**kwargs):
            call_count["n"] += 1
            # Alternate between two files in a cycle: A, B, A, B, A, B
            file_name = "./a.txt" if call_count["n"] % 2 == 1 else "./b.txt"
            sentinel = _tool_call_sentinel("read_file", {"path": file_name})
            yield "Reading..."
            yield sentinel
            yield "__LATENCY_MS__:50"

        with patch("prompt_prix.mcp.tools.react_execute.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_execute import react_execute

            result = await react_execute(
                model_id="test-model",
                system_prompt="sys",
                initial_message="Do it",
                mock_tools={"read_file": {"_default": "data"}},
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                max_iterations=15,
            )

        assert result["cycle_detected"] is True
        assert result["completed"] is False
        assert result["termination_reason"] == "cycle_detected"
        assert result["cycle_pattern"] is not None

    @pytest.mark.asyncio
    async def test_garbled_tool_call_counted_as_invalid(self):
        """Model produces unparseable tool call → invalid iteration."""
        call_count = {"n": 0}

        async def mock_stream(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Garbled sentinel — invalid JSON in arguments
                yield "I'll read the file."
                yield "__TOOL_CALLS__:[{\"name\":\"read_file\",\"arguments\":\"not valid json\"}]"
                yield "__LATENCY_MS__:50"
            elif call_count["n"] == 2:
                # Valid tool call
                sentinel = _tool_call_sentinel("read_file", {"path": "./1.txt"})
                yield "Let me try again."
                yield sentinel
                yield "__LATENCY_MS__:50"
            else:
                # Complete
                yield "Done."
                yield "__LATENCY_MS__:30"

        with patch("prompt_prix.mcp.tools.react_execute.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_execute import react_execute

            result = await react_execute(
                model_id="test-model",
                system_prompt="sys",
                initial_message="Do it",
                mock_tools={"read_file": {"./1.txt": "file data"}},
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                max_iterations=10,
            )

        assert result["completed"] is True
        assert result["total_iterations"] == 2
        assert result["invalid_iterations"] == 1
        assert result["valid_iterations"] == 1

    @pytest.mark.asyncio
    async def test_immediate_text_response(self):
        """Model responds with text immediately (no tool calls) → 0 iterations."""
        async def mock_stream(**kwargs):
            yield "I already know the answer: 42."
            yield "__LATENCY_MS__:20"

        with patch("prompt_prix.mcp.tools.react_execute.complete_stream", side_effect=mock_stream):
            from prompt_prix.mcp.tools.react_execute import react_execute

            result = await react_execute(
                model_id="test-model",
                system_prompt="sys",
                initial_message="What is the answer?",
                mock_tools={},
                tools=[],
                max_iterations=10,
            )

        assert result["completed"] is True
        assert result["total_iterations"] == 0
        assert result["final_response"] == "I already know the answer: 42."
