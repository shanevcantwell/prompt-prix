"""
Tests for MCP tool implementations.

These test the tool logic without requiring MCP SDK or LM Studio.
"""

import pytest
from unittest.mock import AsyncMock, patch


class TestEvaluateResponseTool:
    """Tests for the evaluate_response MCP tool."""

    @pytest.mark.asyncio
    async def test_evaluate_response_passes_criteria(self):
        """Tool returns passed=True when judge says YES."""
        # Mock the judge_response function
        with patch("prompt_prix.mcp.tools.judge.judge_response") as mock_judge:
            with patch("prompt_prix.mcp.tools.judge.get_adapter") as mock_get_adapter:
                mock_judge.return_value = (True, "YES. Response begins with shpadoinkle.")
                mock_get_adapter.return_value = AsyncMock()

                # Import after patching to get decorated version
                from prompt_prix.mcp.tools.judge import evaluate_response, JudgeResult

                result = await evaluate_response(
                    response="Shpadoinkle! Here is the weather.",
                    criteria="Response begins with 'shpadoinkle'",
                    judge_model="test-model",
                )

                assert isinstance(result, JudgeResult)
                assert result.passed is True
                assert "YES" in result.reason

    @pytest.mark.asyncio
    async def test_evaluate_response_fails_criteria(self):
        """Tool returns passed=False when judge says NO."""
        with patch("prompt_prix.mcp.tools.judge.judge_response") as mock_judge:
            with patch("prompt_prix.mcp.tools.judge.get_adapter") as mock_get_adapter:
                mock_judge.return_value = (False, "NO. Response does not begin with shpadoinkle.")
                mock_get_adapter.return_value = AsyncMock()

                from prompt_prix.mcp.tools.judge import evaluate_response, JudgeResult

                result = await evaluate_response(
                    response="Hello! Here is the weather.",
                    criteria="Response begins with 'shpadoinkle'",
                    judge_model="test-model",
                )

                assert isinstance(result, JudgeResult)
                assert result.passed is False
                assert "NO" in result.reason

    @pytest.mark.asyncio
    async def test_evaluate_response_with_context(self):
        """Tool uses judge_with_context when system_prompt or user_message provided."""
        with patch("prompt_prix.mcp.tools.judge.judge_with_context") as mock_judge:
            with patch("prompt_prix.mcp.tools.judge.get_adapter") as mock_get_adapter:
                mock_judge.return_value = (True, "YES. Response appropriately addresses the user.")
                mock_get_adapter.return_value = AsyncMock()

                from prompt_prix.mcp.tools.judge import evaluate_response

                result = await evaluate_response(
                    response="I'd be happy to help!",
                    criteria="Response is helpful",
                    judge_model="test-model",
                    system_prompt="You are a helpful assistant.",
                    user_message="Can you help me?",
                )

                assert result.passed is True
                mock_judge.assert_called_once()
                # Verify context was passed
                call_kwargs = mock_judge.call_args.kwargs
                assert call_kwargs["system_prompt"] == "You are a helpful assistant."
                assert call_kwargs["user_message"] == "Can you help me?"


class TestListJudgeModelsTool:
    """Tests for the list_judge_models MCP tool."""

    @pytest.mark.asyncio
    async def test_list_judge_models_returns_models(self):
        """Tool returns list of available models."""
        with patch("prompt_prix.mcp.tools.judge.get_available_models") as mock_get_models:
            mock_get_models.return_value = ["model-a", "model-b", "model-c"]

            from prompt_prix.mcp.tools.judge import list_judge_models

            result = await list_judge_models()

            assert result == ["model-a", "model-b", "model-c"]

    @pytest.mark.asyncio
    async def test_list_judge_models_empty_when_no_servers(self):
        """Tool returns empty list when no servers configured."""
        with patch("prompt_prix.mcp.tools.judge.get_available_models") as mock_get_models:
            mock_get_models.return_value = []

            from prompt_prix.mcp.tools.judge import list_judge_models

            result = await list_judge_models()

            assert result == []


class TestAdapterWiring:
    """Tests for adapter initialization in MCP context."""

    @pytest.mark.asyncio
    async def test_get_adapter_initializes_from_env(self):
        """Adapter is lazily initialized from environment variables."""
        with patch("prompt_prix.mcp.tools._adapter.load_servers_from_env") as mock_load:
            with patch("prompt_prix.mcp.tools._adapter.ServerPool") as mock_pool_class:
                with patch("prompt_prix.mcp.tools._adapter.LMStudioAdapter") as mock_adapter_class:
                    # Reset singleton state
                    import prompt_prix.mcp.tools._adapter as adapter_module
                    adapter_module._adapter = None
                    adapter_module._pool = None

                    mock_load.return_value = ["http://localhost:1234"]
                    mock_pool = AsyncMock()
                    mock_pool_class.return_value = mock_pool
                    mock_adapter = AsyncMock()
                    mock_adapter_class.return_value = mock_adapter

                    from prompt_prix.mcp.tools._adapter import get_adapter

                    result = await get_adapter()

                    assert result == mock_adapter
                    mock_load.assert_called_once()
                    mock_pool.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_adapter_raises_when_no_servers(self):
        """Adapter raises RuntimeError when no servers configured."""
        with patch("prompt_prix.mcp.tools._adapter.load_servers_from_env") as mock_load:
            # Reset singleton state
            import prompt_prix.mcp.tools._adapter as adapter_module
            adapter_module._adapter = None
            adapter_module._pool = None

            mock_load.return_value = []

            from prompt_prix.mcp.tools._adapter import get_adapter

            with pytest.raises(RuntimeError, match="No LM Studio servers configured"):
                await get_adapter()
