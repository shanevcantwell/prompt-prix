"""Tests for prompt_prix.core module.

Per ADR-006: Orchestration tests mock MCP tools.
ServerPool and stream_completion tests are deleted - those are now internal to adapters.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from prompt_prix.core import ComparisonSession, LMStudioError
from prompt_prix.mcp.registry import register_adapter, clear_adapter


@pytest.fixture
def mock_adapter():
    """Create a mock adapter and register it for any tests that need it."""
    adapter = MagicMock()
    adapter.get_available_models = AsyncMock(return_value=["model-1", "model-2"])
    adapter.get_models_by_server = MagicMock(return_value={"http://localhost:1234": ["model-1", "model-2"]})
    adapter.get_unreachable_servers = MagicMock(return_value=[])

    async def default_stream(*args, **kwargs):
        yield "Response text"

    adapter.stream_completion = default_stream

    register_adapter(adapter)
    yield adapter
    clear_adapter()


class TestComparisonSession:
    """Tests for ComparisonSession class.

    Per ADR-006, ComparisonSession is orchestration layer:
    - Calls MCP primitives (complete_stream)
    - Does NOT know about ServerPool
    """

    def test_comparison_session_init(self):
        """Test ComparisonSession initialization."""
        session = ComparisonSession(
            models=["model-1", "model-2"],
            system_prompt="Test prompt",
            temperature=0.5,
            timeout_seconds=60,
            max_tokens=512
        )

        assert session.state.models == ["model-1", "model-2"]
        assert session.state.system_prompt == "Test prompt"
        assert session.state.temperature == 0.5
        assert len(session.state.contexts) == 2

    def test_comparison_session_creates_contexts_for_all_models(self):
        """Test session creates empty context for each model."""
        models = ["model-a", "model-b"]
        session = ComparisonSession(
            models=models,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        for model_id in models:
            assert model_id in session.state.contexts
            assert session.state.contexts[model_id].model_id == model_id
            assert session.state.contexts[model_id].messages == []

    @pytest.mark.asyncio
    async def test_comparison_session_send_single_prompt(self, mock_adapter):
        """Test sending prompt to single model via MCP primitive."""
        session = ComparisonSession(
            models=["model-1"],
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        result = await session.send_prompt_to_model("model-1", "Test prompt")

        assert result == "Response text"
        assert len(session.state.contexts["model-1"].messages) == 2

    @pytest.mark.asyncio
    async def test_comparison_session_send_all_parallel(self, mock_adapter):
        """Test sending prompt to all models."""
        session = ComparisonSession(
            models=["model-1", "model-2"],
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        results = await session.send_prompt_to_all("Test prompt")

        assert len(results) == 2
        assert "model-1" in results
        assert "model-2" in results

    @pytest.mark.asyncio
    async def test_comparison_session_halt_on_error(self, mock_adapter):
        """Test session halts on model error."""
        # Make adapter raise for model-2
        call_count = 0

        async def error_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model_id") == "model-2":
                raise LMStudioError("Server error")
            yield "Response"

        mock_adapter.stream_completion = error_on_second

        session = ComparisonSession(
            models=["model-1", "model-2"],
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        await session.send_prompt_to_all("Test prompt")

        # Session should be halted
        assert session.state.halted is True
        assert session.state.halt_reason is not None
        assert "model-2" in session.state.halt_reason

    def test_comparison_session_get_context_display(self):
        """Test getting display format for a model."""
        session = ComparisonSession(
            models=["model-1"],
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # Add some messages manually
        session.state.contexts["model-1"].add_user_message("Hello")
        session.state.contexts["model-1"].add_assistant_message("Hi there!")

        display = session.get_context_display("model-1")

        assert "**User:** Hello" in display
        assert "**Assistant:** Hi there!" in display

    def test_comparison_session_get_all_contexts(self):
        """Test getting all context displays."""
        session = ComparisonSession(
            models=["model-1", "model-2"],
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        contexts = session.get_all_contexts()

        assert len(contexts) == 2
        assert "model-1" in contexts
        assert "model-2" in contexts

    @pytest.mark.asyncio
    async def test_comparison_session_blocked_when_halted(self, mock_adapter):
        """Test session refuses new prompts after being halted."""
        session = ComparisonSession(
            models=["model-1"],
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # Manually halt
        session.state.halted = True
        session.state.halt_reason = "Previous error"

        with pytest.raises(RuntimeError, match="Session halted"):
            await session.send_prompt_to_all("Another prompt")


class TestLMStudioError:
    """Tests for LMStudioError exception."""

    def test_lmstudio_error_message(self):
        """Test LMStudioError has proper message."""
        error = LMStudioError("Model not found")
        assert str(error) == "Model not found"

    def test_lmstudio_error_is_exception(self):
        """Test LMStudioError is an Exception."""
        error = LMStudioError("Test")
        assert isinstance(error, Exception)
