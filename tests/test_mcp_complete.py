"""Tests for prompt_prix.mcp.tools.complete module.

Per ADR-006: Mock at layer boundaries - MCP tests mock the adapter interface.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from prompt_prix.mcp.registry import register_adapter, clear_adapter
from prompt_prix.mcp.tools.complete import complete, complete_stream
from prompt_prix.core import LMStudioError


@pytest.fixture
def mock_adapter():
    """Create a mock adapter and register it with the MCP registry."""
    adapter = MagicMock()

    # Mock get_available_models
    adapter.get_available_models = AsyncMock(return_value=["model-1", "model-2"])

    # Mock stream_completion - default yields a simple response
    async def default_stream(*args, **kwargs):
        yield "Hello "
        yield "World"

    adapter.stream_completion = default_stream

    # For list_models tests (not used in complete tests, but part of adapter)
    adapter.get_models_by_server = MagicMock(return_value={"http://localhost:1234": ["model-1"]})
    adapter.get_unreachable_servers = MagicMock(return_value=[])

    register_adapter(adapter)
    yield adapter
    clear_adapter()


class TestComplete:
    """Tests for complete() MCP tool - non-streaming variant."""

    @pytest.mark.asyncio
    async def test_complete_returns_full_response(self, mock_adapter):
        """Test complete() collects stream and returns full response."""
        result = await complete(
            model_id="model-1",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_complete_passes_model_id(self, mock_adapter):
        """Test complete() passes model_id to adapter."""
        call_args = {}

        async def capture_stream(*args, **kwargs):
            call_args.update(kwargs)
            yield "response"

        mock_adapter.stream_completion = capture_stream

        await complete(
            model_id="qwen2.5-7b",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert call_args["model_id"] == "qwen2.5-7b"

    @pytest.mark.asyncio
    async def test_complete_passes_messages(self, mock_adapter):
        """Test complete() passes messages to adapter."""
        call_args = {}

        async def capture_stream(*args, **kwargs):
            call_args.update(kwargs)
            yield "response"

        mock_adapter.stream_completion = capture_stream

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        await complete(model_id="model-1", messages=messages)

        assert call_args["messages"] == messages

    @pytest.mark.asyncio
    async def test_complete_passes_temperature(self, mock_adapter):
        """Test complete() passes temperature to adapter."""
        call_args = {}

        async def capture_stream(*args, **kwargs):
            call_args.update(kwargs)
            yield "response"

        mock_adapter.stream_completion = capture_stream

        await complete(
            model_id="model-1",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.3,
        )

        assert call_args["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_complete_passes_max_tokens(self, mock_adapter):
        """Test complete() passes max_tokens to adapter."""
        call_args = {}

        async def capture_stream(*args, **kwargs):
            call_args.update(kwargs)
            yield "response"

        mock_adapter.stream_completion = capture_stream

        await complete(
            model_id="model-1",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        assert call_args["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_passes_tools(self, mock_adapter):
        """Test complete() passes tools through to adapter."""
        call_args = {}

        async def capture_stream(*args, **kwargs):
            call_args.update(kwargs)
            yield "response"

        mock_adapter.stream_completion = capture_stream

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }
        ]

        await complete(
            model_id="model-1",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tools,
        )

        assert call_args["tools"] == tools

    @pytest.mark.asyncio
    async def test_complete_raises_on_adapter_error(self, mock_adapter):
        """Test complete() propagates adapter exceptions."""
        async def error_stream(*args, **kwargs):
            raise LMStudioError("API error: model not found")
            yield  # Make it a generator

        mock_adapter.stream_completion = error_stream

        with pytest.raises(LMStudioError, match="API error"):
            await complete(
                model_id="model-1",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_complete_no_adapter_registered(self):
        """Test complete() raises when no adapter registered."""
        # Ensure no adapter is registered
        clear_adapter()

        with pytest.raises(RuntimeError, match="No adapter registered"):
            await complete(
                model_id="model-1",
                messages=[{"role": "user", "content": "Hello"}],
            )


class TestCompleteStream:
    """Tests for complete_stream() MCP tool - streaming variant."""

    @pytest.mark.asyncio
    async def test_complete_stream_yields_chunks(self, mock_adapter):
        """Test complete_stream() yields individual chunks."""
        chunks = []
        async for chunk in complete_stream(
            model_id="model-1",
            messages=[{"role": "user", "content": "Hello"}],
        ):
            chunks.append(chunk)

        assert chunks == ["Hello ", "World"]

    @pytest.mark.asyncio
    async def test_complete_stream_multiple_chunks(self, mock_adapter):
        """Test complete_stream() handles many chunks."""
        async def multi_chunk_stream(*args, **kwargs):
            for word in "The quick brown fox jumps".split():
                yield word + " "

        mock_adapter.stream_completion = multi_chunk_stream

        chunks = []
        async for chunk in complete_stream(
            model_id="model-1",
            messages=[{"role": "user", "content": "Hello"}],
        ):
            chunks.append(chunk)

        assert len(chunks) == 5
        assert "".join(chunks) == "The quick brown fox jumps "

    @pytest.mark.asyncio
    async def test_complete_stream_matches_complete(self, mock_adapter):
        """Test complete_stream() collects to same result as complete()."""
        messages = [{"role": "user", "content": "Hello"}]

        # Non-streaming
        result = await complete(model_id="model-1", messages=messages)

        # Streaming
        chunks = []
        async for chunk in complete_stream(model_id="model-1", messages=messages):
            chunks.append(chunk)
        streamed_result = "".join(chunks)

        assert result == streamed_result

    @pytest.mark.asyncio
    async def test_complete_stream_passes_all_params(self, mock_adapter):
        """Test complete_stream() passes all parameters to adapter."""
        call_args = {}

        async def capture_stream(*args, **kwargs):
            call_args.update(kwargs)
            yield "response"

        mock_adapter.stream_completion = capture_stream

        tools = [{"type": "function", "function": {"name": "test"}}]

        async for _ in complete_stream(
            model_id="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=500,
            timeout_seconds=120,
            tools=tools,
            seed=42,
            repeat_penalty=1.1,
        ):
            pass

        assert call_args["model_id"] == "test-model"
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 500
        assert call_args["timeout_seconds"] == 120
        assert call_args["tools"] == tools
        assert call_args["seed"] == 42
        assert call_args["repeat_penalty"] == 1.1


class TestCompleteIntegration:
    """Integration tests requiring live LM Studio servers.

    These tests are skipped by default. Run with:
        pytest -m integration
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_live(self):
        """Test complete() against real LM Studio server.

        Prerequisites:
        - LM Studio running on configured server
        - At least one model loaded
        """
        from dotenv import load_dotenv
        load_dotenv()

        from prompt_prix.mcp.tools.list_models import list_models
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        from prompt_prix.config import get_default_servers

        servers = get_default_servers()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        adapter = LMStudioAdapter(server_urls=servers)
        register_adapter(adapter)

        try:
            discovery = await list_models()
            if not discovery["models"]:
                pytest.skip("No models available on configured servers")

            model_id = discovery["models"][0]

            result = await complete(
                model_id=model_id,
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
                temperature=0.1,
                max_tokens=50,
            )

            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            clear_adapter()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_stream_live(self):
        """Test complete_stream() against real LM Studio server."""
        from dotenv import load_dotenv
        load_dotenv()

        from prompt_prix.mcp.tools.list_models import list_models
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        from prompt_prix.config import get_default_servers

        servers = get_default_servers()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        adapter = LMStudioAdapter(server_urls=servers)
        register_adapter(adapter)

        try:
            discovery = await list_models()
            if not discovery["models"]:
                pytest.skip("No models available on configured servers")

            model_id = discovery["models"][0]

            chunks = []
            async for chunk in complete_stream(
                model_id=model_id,
                messages=[{"role": "user", "content": "Count from 1 to 5."}],
                temperature=0.1,
                max_tokens=50,
            ):
                chunks.append(chunk)

            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert any(str(i) in full_response for i in range(1, 6))
        finally:
            clear_adapter()
