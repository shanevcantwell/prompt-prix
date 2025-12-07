"""Tests for prompt_prix.core module."""

import asyncio
import pytest
import httpx
import respx
from unittest.mock import AsyncMock, patch, MagicMock

from tests.conftest import (
    MOCK_SERVER_1, MOCK_SERVER_2, MOCK_SERVERS,
    MOCK_MODEL_1, MOCK_MODEL_2, MOCK_MODELS,
    MOCK_MANIFEST_RESPONSE, MOCK_COMPLETION_RESPONSE, MOCK_STREAMING_CHUNKS
)


class TestServerPool:
    """Tests for ServerPool class."""

    def test_server_pool_initialization(self, mock_servers):
        """Test ServerPool initializes with servers."""
        from prompt_prix.core import ServerPool

        pool = ServerPool(mock_servers)

        assert len(pool.servers) == 2
        assert MOCK_SERVER_1 in pool.servers
        assert MOCK_SERVER_2 in pool.servers

    def test_server_pool_servers_have_empty_models(self, mock_servers):
        """Test newly initialized servers have no models."""
        from prompt_prix.core import ServerPool

        pool = ServerPool(mock_servers)

        for server in pool.servers.values():
            assert server.available_models == []
            assert server.is_busy is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_pool_refresh_manifest_success(self, mock_servers):
        """Test refreshing manifest fetches models from servers."""
        from prompt_prix.core import ServerPool

        # Mock both server responses
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        pool = ServerPool(mock_servers)
        await pool.refresh_all_manifests()

        # Both servers should have the models
        assert MOCK_MODEL_1 in pool.servers[MOCK_SERVER_1].available_models
        assert MOCK_MODEL_2 in pool.servers[MOCK_SERVER_1].available_models

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_pool_refresh_manifest_server_down(self, mock_servers):
        """Test manifest refresh handles server failure gracefully."""
        from prompt_prix.core import ServerPool

        # First server succeeds, second fails
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        pool = ServerPool(mock_servers)
        await pool.refresh_all_manifests()

        # First server should have models
        assert MOCK_MODEL_1 in pool.servers[MOCK_SERVER_1].available_models
        # Second server should have empty models (cleared on error)
        assert pool.servers[MOCK_SERVER_2].available_models == []

    def test_server_pool_find_available_server_found(self, mock_servers):
        """Test finding available server for a model."""
        from prompt_prix.core import ServerPool

        pool = ServerPool(mock_servers)
        # Manually set model availability
        pool.servers[MOCK_SERVER_1].available_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_2].available_models = [MOCK_MODEL_2]

        result = pool.find_available_server(MOCK_MODEL_1)

        assert result == MOCK_SERVER_1

    def test_server_pool_find_available_server_not_found(self, mock_servers):
        """Test returns None when model not available."""
        from prompt_prix.core import ServerPool

        pool = ServerPool(mock_servers)
        # No models loaded

        result = pool.find_available_server("nonexistent-model")

        assert result is None

    def test_server_pool_find_available_server_skips_busy(self, mock_servers):
        """Test skips busy servers when finding available."""
        from prompt_prix.core import ServerPool

        pool = ServerPool(mock_servers)
        # Both have model but first is busy
        pool.servers[MOCK_SERVER_1].available_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_1].is_busy = True
        pool.servers[MOCK_SERVER_2].available_models = [MOCK_MODEL_1]

        result = pool.find_available_server(MOCK_MODEL_1)

        assert result == MOCK_SERVER_2

    def test_server_pool_get_all_available_models(self, mock_servers):
        """Test getting union of all available models."""
        from prompt_prix.core import ServerPool

        pool = ServerPool(mock_servers)
        pool.servers[MOCK_SERVER_1].available_models = [MOCK_MODEL_1]
        pool.servers[MOCK_SERVER_2].available_models = [MOCK_MODEL_2, "model-c"]

        result = pool.get_all_available_models()

        assert result == {MOCK_MODEL_1, MOCK_MODEL_2, "model-c"}

    @pytest.mark.asyncio
    async def test_server_pool_acquire_release(self, mock_servers):
        """Test acquiring and releasing server locks."""
        from prompt_prix.core import ServerPool

        pool = ServerPool(mock_servers)

        # Acquire server
        await pool.acquire_server(MOCK_SERVER_1)
        assert pool.servers[MOCK_SERVER_1].is_busy is True

        # Release server
        pool.release_server(MOCK_SERVER_1)
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
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool(mock_servers)
        session = ComparisonSession(
            models=mock_models,
            server_pool=pool,
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
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool(mock_servers)
        session = ComparisonSession(
            models=mock_models,
            server_pool=pool,
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
        from prompt_prix.core import ServerPool, ComparisonSession

        # Setup mocks
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )

        pool = ServerPool(mock_servers)
        # Pre-populate the server manifests to avoid infinite retry loop
        await pool.refresh_all_manifests()

        session = ComparisonSession(
            models=[MOCK_MODEL_1],
            server_pool=pool,
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
        from prompt_prix.core import ServerPool, ComparisonSession

        # Setup mocks
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_2}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )

        pool = ServerPool(mock_servers)
        # Pre-populate the server manifests to avoid infinite retry loop
        await pool.refresh_all_manifests()

        session = ComparisonSession(
            models=mock_models,
            server_pool=pool,
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
        from prompt_prix.core import ServerPool, ComparisonSession

        # Setup mocks - each model only available on one server
        # Server 1 has model 1, Server 2 has model 2
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_1}]})
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_2}]})
        )
        # Server 1 succeeds, Server 2 fails
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_2}/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )

        pool = ServerPool(mock_servers)
        # Pre-populate the server manifests to avoid infinite retry loop
        await pool.refresh_all_manifests()

        session = ComparisonSession(
            models=mock_models,
            server_pool=pool,
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
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool(mock_servers)
        session = ComparisonSession(
            models=mock_models,
            server_pool=pool,
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
        assert "**Assistant:** Hi there!" in display

    def test_comparison_session_get_all_contexts(self, mock_servers, mock_models):
        """Test getting all context displays."""
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool(mock_servers)
        session = ComparisonSession(
            models=mock_models,
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        contexts = session.get_all_contexts()

        assert len(contexts) == 2
        assert MOCK_MODEL_1 in contexts
        assert MOCK_MODEL_2 in contexts
