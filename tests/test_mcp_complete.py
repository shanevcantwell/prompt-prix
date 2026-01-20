"""Tests for prompt_prix.mcp.tools.complete module."""

import pytest
import httpx
import respx

from tests.conftest import (
    MOCK_SERVER_1, MOCK_SERVER_2, MOCK_SERVERS,
    MOCK_MODEL_1, MOCK_MODEL_2,
    MOCK_MANIFEST_RESPONSE,
)


# Helper to create SSE streaming body
def make_streaming_body(content: str) -> bytes:
    """Create SSE streaming response body for given content."""
    chunks = []
    for word in content.split():
        chunks.append(f'data: {{"choices":[{{"delta":{{"content":"{word} "}}}}]}}\n')
    chunks.append('data: [DONE]\n')
    return "".join(chunks).encode()


MOCK_STREAMING_BODY = make_streaming_body("The capital of France is Paris.")


class TestComplete:
    """Tests for complete() MCP tool - non-streaming variant."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_single_server(self):
        """Test complete() with single server returning streamed response."""
        from prompt_prix.mcp.tools.complete import complete

        # Mock manifest endpoint
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        # Mock completion endpoint (streaming)
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=MOCK_STREAMING_BODY)
        )

        result = await complete(
            server_urls=[MOCK_SERVER_1],
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

        assert "capital" in result.lower()
        assert "France" in result
        assert "Paris" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_multiple_servers(self):
        """Test complete() finds model on second server."""
        from prompt_prix.mcp.tools.complete import complete

        # Server 1 has different model
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_2}]})
        )

        # Server 2 has the requested model
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_1}]})
        )

        # Completion from server 2
        respx.post(f"{MOCK_SERVER_2}/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=MOCK_STREAMING_BODY)
        )

        result = await complete(
            server_urls=MOCK_SERVERS,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

        assert "Paris" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        """Test complete() passes tools through to API."""
        from prompt_prix.mcp.tools.complete import complete

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        # Capture the request to verify tools are passed
        request_body = {}

        def capture_request(request):
            nonlocal request_body
            request_body = request.content.decode()
            return httpx.Response(200, content=MOCK_STREAMING_BODY)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

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
            server_urls=[MOCK_SERVER_1],
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tools,
        )

        assert "get_weather" in request_body

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_with_custom_params(self):
        """Test complete() passes temperature, max_tokens through."""
        from prompt_prix.mcp.tools.complete import complete

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        request_body = {}

        def capture_request(request):
            nonlocal request_body
            import json
            request_body = json.loads(request.content.decode())
            return httpx.Response(200, content=MOCK_STREAMING_BODY)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

        await complete(
            server_urls=[MOCK_SERVER_1],
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.3,
            max_tokens=100,
        )

        assert request_body["temperature"] == 0.3
        assert request_body["max_tokens"] == 100

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_no_server_available(self):
        """Test complete() raises when no server has the model."""
        from prompt_prix.mcp.tools.complete import complete

        # Server has different models
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "other-model"}]})
        )

        with pytest.raises(RuntimeError, match="No server available"):
            await complete(
                server_urls=[MOCK_SERVER_1],
                model_id=MOCK_MODEL_1,
                messages=[{"role": "user", "content": "Hello"}],
            )

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_api_error(self):
        """Test complete() raises LMStudioError on API failure."""
        from prompt_prix.mcp.tools.complete import complete
        from prompt_prix.core import LMStudioError

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(
                400,
                json={"error": {"message": "Invalid request: model not found"}}
            )
        )

        with pytest.raises(LMStudioError, match="Invalid request"):
            await complete(
                server_urls=[MOCK_SERVER_1],
                model_id=MOCK_MODEL_1,
                messages=[{"role": "user", "content": "Hello"}],
            )


class TestCompleteStream:
    """Tests for complete_stream() MCP tool - streaming variant."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_stream_yields_chunks(self):
        """Test complete_stream() yields individual chunks."""
        from prompt_prix.mcp.tools.complete import complete_stream

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=MOCK_STREAMING_BODY)
        )

        chunks = []
        async for chunk in complete_stream(
            server_urls=[MOCK_SERVER_1],
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        ):
            chunks.append(chunk)

        # Should have received multiple chunks
        assert len(chunks) > 1

        # Combined should form complete response
        full_response = "".join(chunks)
        assert "Paris" in full_response

    @respx.mock
    @pytest.mark.asyncio
    async def test_complete_stream_matches_complete(self):
        """Test complete_stream() collects to same result as complete()."""
        from prompt_prix.mcp.tools.complete import complete, complete_stream

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=MOCK_STREAMING_BODY)
        )

        messages = [{"role": "user", "content": "What is the capital of France?"}]

        # Non-streaming
        result = await complete(
            server_urls=[MOCK_SERVER_1],
            model_id=MOCK_MODEL_1,
            messages=messages,
        )

        # Reset mocks for streaming call
        respx.reset()
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=MOCK_STREAMING_BODY)
        )

        # Streaming
        chunks = []
        async for chunk in complete_stream(
            server_urls=[MOCK_SERVER_1],
            model_id=MOCK_MODEL_1,
            messages=messages,
        ):
            chunks.append(chunk)

        streamed_result = "".join(chunks)

        # Results should be equivalent
        assert result == streamed_result


class TestCompleteIntegration:
    """Integration tests requiring live LM Studio servers."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_live(self):
        """Test complete() against real LM Studio server.

        Prerequisites:
        - LM Studio running on configured server
        - At least one model loaded

        This test uses the server URLs from .env or defaults.
        """
        from dotenv import load_dotenv
        load_dotenv()

        from prompt_prix.mcp.tools.complete import complete
        from prompt_prix.mcp.tools.list_models import list_models
        from prompt_prix.config import load_servers_from_env

        servers = load_servers_from_env()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        # First discover available models
        discovery = await list_models(servers)
        if not discovery["models"]:
            pytest.skip("No models available on configured servers")

        model_id = discovery["models"][0]

        result = await complete(
            server_urls=servers,
            model_id=model_id,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            temperature=0.1,
            max_tokens=50,
        )

        # Should have received a response
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_stream_live(self):
        """Test complete_stream() against real LM Studio server."""
        from dotenv import load_dotenv
        load_dotenv()

        from prompt_prix.mcp.tools.complete import complete_stream
        from prompt_prix.mcp.tools.list_models import list_models
        from prompt_prix.config import load_servers_from_env

        servers = load_servers_from_env()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        discovery = await list_models(servers)
        if not discovery["models"]:
            pytest.skip("No models available on configured servers")

        model_id = discovery["models"][0]

        chunks = []
        async for chunk in complete_stream(
            server_urls=servers,
            model_id=model_id,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            temperature=0.1,
            max_tokens=50,
        ):
            chunks.append(chunk)

        # Should have received multiple chunks
        assert len(chunks) > 0

        # Combined should form response with numbers
        full_response = "".join(chunks)
        assert any(str(i) in full_response for i in range(1, 6))
