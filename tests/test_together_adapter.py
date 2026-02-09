"""Tests for TogetherAdapter — protocol compliance, SSE parsing, error handling."""

import json
import pytest
import respx
import httpx

from prompt_prix.adapters.together import TogetherAdapter, TogetherError
from prompt_prix.adapters.schema import InferenceTask


def sse_stream(content: str) -> str:
    """Build SSE stream response."""
    return (
        f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'
        "data: [DONE]\n\n"
    )


def make_task(model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    return InferenceTask(
        model_id=model_id,
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
        max_tokens=100,
        timeout_seconds=5.0,
    )


@pytest.fixture
def adapter():
    return TogetherAdapter(
        models=["meta-llama/Llama-3.3-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
        api_key="test-key-123",
    )


# ─────────────────────────────────────────────────────────────────────
# Protocol compliance
# ─────────────────────────────────────────────────────────────────────


class TestProtocolCompliance:
    @pytest.mark.asyncio
    async def test_get_available_models(self, adapter):
        models = await adapter.get_available_models()
        assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" in models
        assert "Qwen/Qwen2.5-72B-Instruct-Turbo" in models

    def test_get_models_by_server(self, adapter):
        by_server = adapter.get_models_by_server()
        assert "together-ai" in by_server
        assert len(by_server["together-ai"]) == 2

    def test_get_unreachable_servers_always_empty(self, adapter):
        assert adapter.get_unreachable_servers() == []

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="Together API key required"):
            TogetherAdapter(models=["model-a"], api_key=None)


# ─────────────────────────────────────────────────────────────────────
# SSE streaming
# ─────────────────────────────────────────────────────────────────────


class TestSSEStreaming:
    @pytest.mark.asyncio
    @respx.mock
    async def test_streams_content(self, adapter):
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=sse_stream("Hello world"))
        )

        chunks = []
        async for chunk in adapter.stream_completion(make_task()):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if not c.startswith("__")]
        assert "Hello world" in "".join(text_chunks)

    @pytest.mark.asyncio
    @respx.mock
    async def test_yields_latency_sentinel(self, adapter):
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=sse_stream("Hello"))
        )

        chunks = []
        async for chunk in adapter.stream_completion(make_task()):
            chunks.append(chunk)

        latency_chunks = [c for c in chunks if c.startswith("__LATENCY_MS__:")]
        assert len(latency_chunks) == 1


# ─────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    @respx.mock
    async def test_api_error_raises(self, adapter):
        error_body = json.dumps(
            {"error": {"message": "Rate limit exceeded"}}
        ).encode()
        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            return_value=httpx.Response(429, content=error_body)
        )

        with pytest.raises(TogetherError, match="Rate limit exceeded"):
            async for _ in adapter.stream_completion(make_task()):
                pass

    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_auth_header(self, adapter):
        """Verify Authorization header is sent."""
        captured_request = None

        def capture_request(request):
            nonlocal captured_request
            captured_request = request
            return httpx.Response(200, content=sse_stream("ok"))

        respx.post("https://api.together.xyz/v1/chat/completions").mock(
            side_effect=capture_request
        )

        async for _ in adapter.stream_completion(make_task()):
            pass

        assert captured_request is not None
        assert captured_request.headers["Authorization"] == "Bearer test-key-123"
