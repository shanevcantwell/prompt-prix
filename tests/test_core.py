"""Tests for prompt_prix.core module."""

import asyncio
import pytest
import httpx
import respx
from unittest.mock import AsyncMock, patch, MagicMock

from tests.conftest import (
    MOCK_SERVER_1, MOCK_SERVER_2, MOCK_SERVERS,
    MOCK_MODEL_1, MOCK_MODEL_2, MOCK_MODELS,
    MOCK_MANIFEST_RESPONSE, MOCK_COMPLETION_RESPONSE, MOCK_STREAMING_CHUNKS,
    MOCK_LOAD_STATE_RESPONSE, MOCK_LOAD_STATE_EMPTY, MOCK_LOAD_STATE_MULTIPLE
)


class TestServerPool:
    """Tests for ServerPool class."""

    def test_server_pool_initialization(self, mock_servers):
        """Test ServerPool initializes with servers."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)

        assert len(pool.servers) == 2
        assert MOCK_SERVER_1 in pool.servers
        assert MOCK_SERVER_2 in pool.servers

    def test_server_pool_servers_have_empty_models(self, mock_servers):
        """Test newly initialized servers have no models."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)

        for server in pool.servers.values():
            assert server.manifest_models == []
            assert server.loaded_models == []
            assert server.is_busy is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_pool_refresh_success(self, mock_servers):
        """Test refreshing fetches both manifest and load state from servers."""
        from prompt_prix.scheduler import ServerPool

        # Mock manifest endpoint for both servers
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        # Mock load state endpoint for both servers
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )

        pool = ServerPool(mock_servers)
        await pool.refresh()

        # Both servers should have the manifest models
        assert MOCK_MODEL_1 in pool.servers[MOCK_SERVER_1].manifest_models
        assert MOCK_MODEL_2 in pool.servers[MOCK_SERVER_1].manifest_models
        # Server 1 should have model 1 loaded
        assert MOCK_MODEL_1 in pool.servers[MOCK_SERVER_1].loaded_models
        # Server 2 should have no model loaded
        assert pool.servers[MOCK_SERVER_2].loaded_models == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_pool_refresh_server_down(self, mock_servers):
        """Test refresh handles server failure gracefully."""
        from prompt_prix.scheduler import ServerPool

        # First server succeeds, second fails
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get(f"{MOCK_SERVER_2}/api/v0/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        pool = ServerPool(mock_servers)
        await pool.refresh()

        # First server should have models
        assert MOCK_MODEL_1 in pool.servers[MOCK_SERVER_1].manifest_models
        # Second server should have empty models (cleared on error)
        assert pool.servers[MOCK_SERVER_2].manifest_models == []
        assert pool.servers[MOCK_SERVER_2].loaded_models == []

    def test_server_pool_find_server_found(self, mock_servers):
        """Test finding server for a model in manifest."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        # Manually set model availability
        pool.servers[MOCK_SERVER_1].manifest_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_2].manifest_models = [MOCK_MODEL_2]

        result = pool.find_server(MOCK_MODEL_1)

        assert result == MOCK_SERVER_1

    def test_server_pool_find_server_not_found(self, mock_servers):
        """Test returns None when model not available."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        # No models in manifest

        result = pool.find_server("nonexistent-model")

        assert result is None

    def test_server_pool_find_server_skips_busy(self, mock_servers):
        """Test skips busy servers when finding available."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        # Both have model but first is busy
        pool.servers[MOCK_SERVER_1].manifest_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_1].is_busy = True
        pool.servers[MOCK_SERVER_2].manifest_models = [MOCK_MODEL_1]

        result = pool.find_server(MOCK_MODEL_1)

        assert result == MOCK_SERVER_2

    def test_server_pool_find_server_prefers_loaded(self, mock_servers):
        """Test find_server prefers server where model is already loaded."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        # Both have model in manifest, but only server 2 has it loaded
        pool.servers[MOCK_SERVER_1].manifest_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_1].loaded_models = [MOCK_MODEL_2]  # different model loaded
        pool.servers[MOCK_SERVER_2].manifest_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_2].loaded_models = [MOCK_MODEL_1]  # target model loaded

        result = pool.find_server(MOCK_MODEL_1)

        # Should prefer server 2 where model is already loaded
        assert result == MOCK_SERVER_2

    def test_server_pool_get_available_models(self, mock_servers):
        """Test getting union of all manifest models."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        pool.servers[MOCK_SERVER_1].manifest_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_2].manifest_models = [MOCK_MODEL_2, "model-c"]

        result = pool.get_available_models()

        assert result == {MOCK_MODEL_1, MOCK_MODEL_2, "model-c"}

    def test_server_pool_get_available_models_only_loaded(self, mock_servers):
        """Test getting only loaded models."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        pool.servers[MOCK_SERVER_1].manifest_models = [MOCK_MODEL_1, MOCK_MODEL_2]
        pool.servers[MOCK_SERVER_1].loaded_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_2].manifest_models = [MOCK_MODEL_1, MOCK_MODEL_2]
        pool.servers[MOCK_SERVER_2].loaded_models = [MOCK_MODEL_2]

        result = pool.get_available_models(only_loaded=True)

        # Should only return the loaded models
        assert result == {MOCK_MODEL_1, MOCK_MODEL_2}

    def test_server_pool_multiple_loaded_models_on_one_server(self, mock_servers):
        """Test that multiple models can be loaded on a single server.

        LM Studio supports loading multiple models into VRAM simultaneously.
        This test verifies we track all of them, not just the first.
        """
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        pool.servers[MOCK_SERVER_1].manifest_models = [MOCK_MODEL_1, MOCK_MODEL_2]
        pool.servers[MOCK_SERVER_1].loaded_models = [MOCK_MODEL_1, MOCK_MODEL_2]  # Both loaded
        pool.servers[MOCK_SERVER_2].manifest_models = [MOCK_MODEL_1, MOCK_MODEL_2]
        pool.servers[MOCK_SERVER_2].loaded_models = []  # None loaded

        result = pool.get_available_models(only_loaded=True)

        # Should return BOTH models from server 1
        assert result == {MOCK_MODEL_1, MOCK_MODEL_2}

    def test_server_pool_find_server_with_multiple_loaded(self, mock_servers):
        """Test find_server works when multiple models are loaded."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)
        pool.servers[MOCK_SERVER_1].manifest_models = [MOCK_MODEL_1, MOCK_MODEL_2]
        pool.servers[MOCK_SERVER_1].loaded_models = [MOCK_MODEL_1, MOCK_MODEL_2]
        pool.servers[MOCK_SERVER_2].manifest_models = [MOCK_MODEL_1, MOCK_MODEL_2]
        pool.servers[MOCK_SERVER_2].loaded_models = []

        # Should find server 1 for both models (they're both loaded there)
        assert pool.find_server(MOCK_MODEL_1) == MOCK_SERVER_1
        assert pool.find_server(MOCK_MODEL_2) == MOCK_SERVER_1

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_pool_refresh_multiple_loaded(self, mock_servers):
        """Test refresh correctly identifies multiple loaded models."""
        from prompt_prix.scheduler import ServerPool

        # Mock both endpoints
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_MULTIPLE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )

        pool = ServerPool(mock_servers)
        await pool.refresh()

        # Server 1 should have both models loaded
        assert MOCK_MODEL_1 in pool.servers[MOCK_SERVER_1].loaded_models
        assert MOCK_MODEL_2 in pool.servers[MOCK_SERVER_1].loaded_models
        # Server 2 should have none loaded
        assert pool.servers[MOCK_SERVER_2].loaded_models == []

    @pytest.mark.asyncio
    async def test_server_pool_acquire_release(self, mock_servers):
        """Test acquiring and releasing server locks."""
        from prompt_prix.scheduler import ServerPool

        pool = ServerPool(mock_servers)

        # Acquire server
        await pool.acquire(MOCK_SERVER_1)
        assert pool.servers[MOCK_SERVER_1].is_busy is True

        # Release server
        pool.release(MOCK_SERVER_1)
        assert pool.servers[MOCK_SERVER_1].is_busy is False


class TestStreamCompletion:
    """Tests for stream_completion function."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_success(self):
        """Test streaming completion yields chunks correctly."""
        from prompt_prix.core import stream_completion

        # Create streaming response content as text
        streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=streaming_content)
        )

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30
        ):
            chunks.append(chunk)

        # Should have received content chunks
        full_response = "".join(chunks)
        assert "capital" in full_response.lower() or "France" in full_response or "Paris" in full_response

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_timeout(self):
        """Test streaming completion handles timeout."""
        from prompt_prix.core import stream_completion

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("Timeout")
        )

        with pytest.raises(httpx.ReadTimeout):
            async for _ in stream_completion(
                server_url=MOCK_SERVER_1,
                model_id=MOCK_MODEL_1,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=1
            ):
                pass

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_with_seed(self):
        """Test streaming completion includes seed in payload when provided."""
        from prompt_prix.core import stream_completion
        import json

        captured_request = None

        def capture_request(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
            return httpx.Response(200, text=streaming_content)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            seed=42
        ):
            chunks.append(chunk)

        # Verify seed was included in request
        assert captured_request is not None
        assert captured_request.get("seed") == 42

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_without_seed(self):
        """Test streaming completion excludes seed when not provided."""
        from prompt_prix.core import stream_completion
        import json

        captured_request = None

        def capture_request(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
            return httpx.Response(200, text=streaming_content)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30
        ):
            chunks.append(chunk)

        # Verify seed was NOT included in request
        assert captured_request is not None
        assert "seed" not in captured_request

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_with_repeat_penalty(self):
        """Test streaming completion includes repeat_penalty in payload when provided."""
        from prompt_prix.core import stream_completion
        import json

        captured_request = None

        def capture_request(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
            return httpx.Response(200, text=streaming_content)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            repeat_penalty=1.2
        ):
            chunks.append(chunk)

        # Verify repeat_penalty was included in request
        assert captured_request is not None
        assert captured_request.get("repeat_penalty") == 1.2

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_without_repeat_penalty(self):
        """Test streaming completion excludes repeat_penalty when set to 1.0."""
        from prompt_prix.core import stream_completion
        import json

        captured_request = None

        def capture_request(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
            return httpx.Response(200, text=streaming_content)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30,
            repeat_penalty=1.0  # Default, should not be sent
        ):
            chunks.append(chunk)

        # Verify repeat_penalty was NOT included in request (1.0 is treated as off)
        assert captured_request is not None
        assert "repeat_penalty" not in captured_request

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_omits_temperature_when_none(self):
        """Test streaming completion excludes temperature when not provided.

        Bug #17: Global temperature overrides LM Studio's per-model settings.
        When temperature is None, it should be omitted from the API payload
        so LM Studio uses the model's configured temperature.
        """
        from prompt_prix.core import stream_completion
        import json

        captured_request = None

        def capture_request(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
            return httpx.Response(200, text=streaming_content)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            timeout_seconds=30
            # Note: no temperature parameter
        ):
            chunks.append(chunk)

        # Verify temperature was NOT included in request
        assert captured_request is not None
        assert "temperature" not in captured_request

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_completion_includes_temperature_when_provided(self):
        """Test streaming completion includes temperature when explicitly set."""
        from prompt_prix.core import stream_completion
        import json

        captured_request = None

        def capture_request(request):
            nonlocal captured_request
            captured_request = json.loads(request.content)
            streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
            return httpx.Response(200, text=streaming_content)

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(side_effect=capture_request)

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
            timeout_seconds=30
        ):
            chunks.append(chunk)

        # Verify temperature was included in request
        assert captured_request is not None
        assert captured_request.get("temperature") == 0.5

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_call_accumulated_not_fragmented(self):
        """Bug #36: Tool call arguments should be accumulated, not fragmented.

        OpenAI streaming sends tool call arguments token-by-token. Each chunk
        should NOT get its own markdown block - arguments should be accumulated
        and formatted once at the end.
        """
        from prompt_prix.core import stream_completion
        import json as json_module

        # Simulate streaming tool call chunks (how OpenAI sends them)
        tool_call_chunks = [
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_weather"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\":\\"Paris"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\"}"}}]}}]}',
            'data: [DONE]'
        ]
        streaming_content = "\n".join(tool_call_chunks) + "\n"

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=streaming_content)
        )

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)

        # Should have exactly ONE **Tool Call:** marker
        assert full_response.count("**Tool Call:**") == 1, \
            f"Expected 1 tool call marker, got {full_response.count('**Tool Call:**')}"

        # Should have exactly ONE json code block
        assert full_response.count("```json") == 1, \
            f"Expected 1 json block, got {full_response.count('```json')}"

        # The JSON should be parseable (not fragmented)
        import re
        json_match = re.search(r"```json\n(.*?)\n```", full_response, re.DOTALL)
        assert json_match, "No JSON block found"
        json_str = json_match.group(1)
        parsed = json_module.loads(json_str)
        assert parsed == {"city": "Paris"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_call_multiple_calls_accumulated(self):
        """Bug #36: Multiple tool calls should each be accumulated separately."""
        from prompt_prix.core import stream_completion

        # Two tool calls, each with fragmented arguments
        tool_call_chunks = [
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_weather"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\":"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "get_time"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\"Tokyo\\"}"}}]}}]}',
            'data: {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\\"tz\\":\\"JST\\"}"}}]}}]}',
            'data: [DONE]'
        ]
        streaming_content = "\n".join(tool_call_chunks) + "\n"

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=streaming_content)
        )

        chunks = []
        async for chunk in stream_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "Weather and time in Tokyo?"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)

        # Should have exactly TWO tool call markers
        assert full_response.count("**Tool Call:**") == 2
        assert "get_weather" in full_response
        assert "get_time" in full_response

        # Should have exactly TWO json code blocks
        assert full_response.count("```json") == 2


class TestGetCompletion:
    """Tests for get_completion function."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_completion_success(self):
        """Test non-streaming completion returns full response."""
        from prompt_prix.core import get_completion

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )

        result = await get_completion(
            server_url=MOCK_SERVER_1,
            model_id=MOCK_MODEL_1,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=30
        )

        assert result == "The capital of France is Paris."

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_completion_http_error(self):
        """Test completion handles HTTP errors with user-friendly message."""
        from prompt_prix.core import get_completion, LMStudioError

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": {"message": "Model context limit exceeded"}})
        )

        with pytest.raises(LMStudioError) as exc_info:
            await get_completion(
                server_url=MOCK_SERVER_1,
                model_id=MOCK_MODEL_1,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=30
            )

        assert "Model context limit exceeded" in str(exc_info.value)
        assert MOCK_MODEL_1 in str(exc_info.value)


class TestComparisonSession:
    """Tests for ComparisonSession class."""

    def test_comparison_session_init(self, mock_servers, mock_models):
        """Test ComparisonSession initialization."""
        from prompt_prix.scheduler import ServerPool
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        from prompt_prix.core import ComparisonSession

        pool = ServerPool(mock_servers)
        adapter = LMStudioAdapter(pool)
        session = ComparisonSession(
            models=mock_models,
            adapter=adapter,
            system_prompt="Test prompt",
            temperature=0.5,
            timeout_seconds=60,
            max_tokens=512
        )

        assert session.state.models == mock_models
        assert session.state.system_prompt == "Test prompt"
        assert session.state.temperature == 0.5
        assert len(session.state.contexts) == 2

    def test_comparison_session_creates_contexts_for_all_models(self, mock_servers, mock_models):
        """Test session creates empty context for each model."""
        from prompt_prix.scheduler import ServerPool
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        from prompt_prix.core import ComparisonSession

        pool = ServerPool(mock_servers)
        adapter = LMStudioAdapter(pool)
        session = ComparisonSession(
            models=mock_models,
            adapter=adapter,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        for model_id in mock_models:
            assert model_id in session.state.contexts
            assert session.state.contexts[model_id].model_id == model_id
            assert session.state.contexts[model_id].messages == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_comparison_session_send_single_prompt(self, mock_servers, mock_models):
        """Test sending prompt to single model."""
        from prompt_prix.scheduler import ServerPool
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        from prompt_prix.core import ComparisonSession

        # Setup mocks - manifest endpoints
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        # Setup mocks - load state endpoints
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )
        respx.get(f"{MOCK_SERVER_2}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )
        # Use streaming format for chat completions (stream=True)
        from tests.conftest import MOCK_STREAMING_CHUNKS
        streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=streaming_content)
        )
        respx.post(f"{MOCK_SERVER_2}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=streaming_content)
        )

        pool = ServerPool(mock_servers)
        adapter = LMStudioAdapter(pool)
        # Pre-populate the server state to avoid infinite retry loop
        await pool.refresh()

        session = ComparisonSession(
            models=[MOCK_MODEL_1],
            adapter=adapter,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        result = await session.send_prompt_to_model(MOCK_MODEL_1, "Test prompt")

        assert result == "The capital of France is Paris."
        assert len(session.state.contexts[MOCK_MODEL_1].messages) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_comparison_session_send_all_parallel(self, mock_servers, mock_models):
        """Test sending prompt to all models."""
        from prompt_prix.scheduler import ServerPool
        from prompt_prix.core import ComparisonSession

        # Setup mocks - manifest endpoints
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        # Setup mocks - load state endpoints
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )
        respx.get(f"{MOCK_SERVER_2}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_2}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )

        pool = ServerPool(mock_servers)
        # Pre-populate the server state to avoid infinite retry loop
        await pool.refresh()

        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        adapter = LMStudioAdapter(pool)
        session = ComparisonSession(
            models=mock_models,
            adapter=adapter,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        results = await session.send_prompt_to_all("Test prompt")

        assert len(results) == 2
        assert MOCK_MODEL_1 in results
        assert MOCK_MODEL_2 in results

    @respx.mock
    @pytest.mark.asyncio
    async def test_comparison_session_halt_on_error(self, mock_servers, mock_models):
        """Test session halts on model error."""
        from prompt_prix.scheduler import ServerPool
        from prompt_prix.core import ComparisonSession

        # Setup mocks - each model only available on one server
        # Server 1 has model 1, Server 2 has model 2
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_1}]})
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_2}]})
        )
        # Setup mocks - load state endpoints
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )
        respx.get(f"{MOCK_SERVER_2}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )
        # Server 1 succeeds, Server 2 fails
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_2}/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )

        pool = ServerPool(mock_servers)
        # Pre-populate the server state to avoid infinite retry loop
        await pool.refresh()

        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        adapter = LMStudioAdapter(pool)
        session = ComparisonSession(
            models=mock_models,
            adapter=adapter,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        await session.send_prompt_to_all("Test prompt")

        # Session should be halted
        assert session.state.halted is True
        assert session.state.halt_reason is not None

    def test_comparison_session_get_context_display(self, mock_servers, mock_models):
        """Test getting display format for a model."""
        from prompt_prix.scheduler import ServerPool
        from prompt_prix.core import ComparisonSession
        from prompt_prix.adapters.lmstudio import LMStudioAdapter

        pool = ServerPool(mock_servers)
        adapter = LMStudioAdapter(pool)
        session = ComparisonSession(
            models=mock_models,
            adapter=adapter,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # Add some messages manually
        session.state.contexts[MOCK_MODEL_1].add_user_message("Hello")
        session.state.contexts[MOCK_MODEL_1].add_assistant_message("Hi there!")

        display = session.get_context_display(MOCK_MODEL_1)

        assert "**User:** Hello" in display
        # Assistant responses are wrapped in code blocks for readability
        assert "**Assistant:**" in display
        assert "Hi there!" in display

    def test_comparison_session_get_all_contexts(self, mock_servers, mock_models):
        """Test getting all context displays."""
        from prompt_prix.scheduler import ServerPool
        from prompt_prix.core import ComparisonSession
        from prompt_prix.adapters.lmstudio import LMStudioAdapter

        pool = ServerPool(mock_servers)
        adapter = LMStudioAdapter(pool)
        session = ComparisonSession(
            models=mock_models,
            adapter=adapter,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        contexts = session.get_all_contexts()

        assert len(contexts) == 2
        assert MOCK_MODEL_1 in contexts
        assert MOCK_MODEL_2 in contexts


class TestCompareTabWithGPUPrefix:
    """Tests for Compare tab initialization with GPU-prefixed model names.

    Bug #29: Compare tab fails to find models when dropdowns contain
    prefixed names like '0: model-name' but server pool has 'model-name'.
    """

    @respx.mock
    @pytest.mark.asyncio
    async def test_initialize_session_strips_gpu_prefix(self):
        """Test that initialize_session works with GPU-prefixed model names."""
        from prompt_prix.tabs.compare.handlers import initialize_session

        # Mock server responses
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_RESPONSE)
        )

        # Call with GPU-prefixed model names (as dropdown provides)
        result = await initialize_session(
            servers_text=MOCK_SERVER_1,
            models_selected=[f"0: {MOCK_MODEL_1}", f"0: {MOCK_MODEL_2}"],
            system_prompt_text="Test prompt",
            timeout=300,
            max_tokens=2048
        )

        status = result[0]
        # Should succeed, not return "Models not found"
        assert "not found" not in status.lower()
        assert "initialized" in status.lower() or "✅" in status

    @respx.mock
    @pytest.mark.asyncio
    async def test_initialize_session_handles_complex_model_paths(self):
        """Test prefix stripping with complex model paths containing slashes."""
        from prompt_prix.tabs.compare.handlers import initialize_session

        complex_model = "openai/gpt-oss-20b-gguf/gpt-oss-20b-router.gguf"
        manifest_with_complex = {
            "data": [{"id": complex_model}]
        }

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=manifest_with_complex)
        )
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_EMPTY)
        )

        result = await initialize_session(
            servers_text=MOCK_SERVER_1,
            models_selected=[f"1: {complex_model}"],
            system_prompt_text="Test",
            timeout=300,
            max_tokens=2048
        )

        status = result[0]
        assert "not found" not in status.lower()

    @respx.mock
    @pytest.mark.asyncio
    async def test_initialize_session_without_prefix_still_works(self):
        """Test that non-prefixed model names still work (backwards compat)."""
        from prompt_prix.tabs.compare.handlers import initialize_session

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_1}/api/v0/models").mock(
            return_value=httpx.Response(200, json=MOCK_LOAD_STATE_RESPONSE)
        )

        # Call without prefix (backwards compatibility)
        result = await initialize_session(
            servers_text=MOCK_SERVER_1,
            models_selected=[MOCK_MODEL_1],
            system_prompt_text="Test",
            timeout=300,
            max_tokens=2048
        )

        status = result[0]
        assert "not found" not in status.lower()


# ─────────────────────────────────────────────────────────────────────
# COMPARISONSESSION ADAPTER INTERFACE TESTS (#73)
# ─────────────────────────────────────────────────────────────────────

class HostAdapterMock:
    """
    Mock adapter implementing ONLY the HostAdapter protocol.

    This mock does NOT have ServerPool methods. Tests using this mock
    verify that ComparisonSession uses HostAdapter interface correctly.

    Part of #73 Phase 6 - tests for adapter refactor.
    """

    def __init__(self, responses: dict[str, str] = None):
        self.responses = responses or {}
        self.calls = []

    async def get_available_models(self) -> list[str]:
        return list(self.responses.keys())

    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tools=None
    ):
        self.calls.append((model_id, messages))
        response = self.responses.get(model_id, "Default response")
        for word in response.split():
            yield word + " "

    def get_concurrency_limit(self) -> int:
        return 2


class TestComparisonSessionWithHostAdapter:
    """
    Tests for ComparisonSession using the HostAdapter interface.

    These tests document the expected behavior after #73 refactor.
    They FAIL initially because ComparisonSession currently takes ServerPool.

    Phase 6 of #73 adapter refactor.
    """

    @pytest.mark.asyncio
    async def test_session_accepts_adapter_not_server_pool(self):
        """ComparisonSession should accept adapter parameter, not server_pool.

        This is the key test for #73 - the abstraction leak is using ServerPool.
        After refactor, ComparisonSession should use adapter interface only.
        """
        from prompt_prix.core import ComparisonSession

        adapter = HostAdapterMock(responses={"model-a": "Hello"})

        # This should work with adapter parameter (not server_pool)
        session = ComparisonSession(
            adapter=adapter,
            models=["model-a"],
            system_prompt="You are helpful.",
            timeout_seconds=60,
            max_tokens=100
        )

        assert session.state.models == ["model-a"]

    @pytest.mark.asyncio
    async def test_session_uses_adapter_stream_completion_for_prompts(self):
        """Session should use adapter.stream_completion for sending prompts.

        Note: For ComparisonSession, stream_completion handles acquire/release
        internally (LMStudioAdapter design). ComparisonSession doesn't call
        acquire/release separately.
        """
        from prompt_prix.core import ComparisonSession

        adapter = HostAdapterMock(responses={"model-a": "Hello world"})

        session = ComparisonSession(
            adapter=adapter,
            models=["model-a"],
            system_prompt="You are helpful.",
            timeout_seconds=60,
            max_tokens=100
        )

        await session.send_prompt_to_model("model-a", "Hi", on_chunk=None)

        # Verify stream_completion was called (not acquire/release separately)
        assert len(adapter.calls) == 1
        assert adapter.calls[0][0] == "model-a"

    @pytest.mark.asyncio
    async def test_session_uses_adapter_stream_completion(self):
        """Session should use adapter.stream_completion(), not core.stream_completion()."""
        from prompt_prix.core import ComparisonSession

        adapter = HostAdapterMock(responses={"model-a": "Response text"})

        session = ComparisonSession(
            adapter=adapter,
            models=["model-a"],
            system_prompt="Test",
            timeout_seconds=60,
            max_tokens=100
        )

        chunks = []

        async def collect_chunks(model_id, chunk):
            chunks.append(chunk)

        await session.send_prompt_to_model("model-a", "Hello", on_chunk=collect_chunks)

        # Verify adapter.stream_completion was called
        assert len(adapter.calls) == 1
        assert adapter.calls[0][0] == "model-a"  # model_id
        # Verify chunks were received
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_session_send_prompt_to_all_uses_adapter(self):
        """send_prompt_to_all should work with adapter-based session."""
        from prompt_prix.core import ComparisonSession

        adapter = HostAdapterMock(responses={
            "model-a": "Response A",
            "model-b": "Response B"
        })

        session = ComparisonSession(
            adapter=adapter,
            models=["model-a", "model-b"],
            system_prompt="Test",
            timeout_seconds=60,
            max_tokens=100
        )

        results = await session.send_prompt_to_all("Hello")

        assert "model-a" in results
        assert "model-b" in results
        # Both models should have had stream_completion called
        model_ids_called = [call[0] for call in adapter.calls]
        assert "model-a" in model_ids_called
        assert "model-b" in model_ids_called
