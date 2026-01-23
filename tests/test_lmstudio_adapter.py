"""Tests for LMStudioAdapter boundary conditions.

Per ADR-006: ServerPool is INTERNAL to LMStudioAdapter. Tests mock at httpx boundary.
"""

import asyncio
import pytest
import respx
import httpx

from prompt_prix.adapters.lmstudio import LMStudioAdapter


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_server_urls():
    """Two server URLs for testing."""
    return ["http://server0:1234", "http://server1:1234"]


def models_response(model_ids: list[str]) -> dict:
    """Build /v1/models response."""
    return {"data": [{"id": m} for m in model_ids]}


def sse_stream(content: str) -> str:
    """Build SSE stream response for chat completions."""
    return (
        f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'
        'data: [DONE]\n\n'
    )


# ─────────────────────────────────────────────────────────────────────
# get_available_models() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGetAvailableModels:
    """Tests for get_available_models() aggregation."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_aggregates_models_from_both_servers(self, two_server_urls):
        """Server 0 has [A,B], server 1 has [B,C] → returns [A,B,C]."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA", "modelB"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB", "modelC"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        models = await adapter.get_available_models()

        assert set(models) == {"modelA", "modelB", "modelC"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_models_from_reachable_server_only(self, two_server_urls):
        """Server 0 unreachable → returns models from server 1 only."""
        respx.get("http://server0:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelC", "modelD"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        models = await adapter.get_available_models()

        assert set(models) == {"modelC", "modelD"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_both_servers_unreachable_returns_empty(self, two_server_urls):
        """Both servers unreachable → returns empty list."""
        respx.get("http://server0:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get("http://server1:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        adapter = LMStudioAdapter(two_server_urls)
        models = await adapter.get_available_models()

        assert models == []


# ─────────────────────────────────────────────────────────────────────
# stream_completion() SERVER INDEX TESTS
# ─────────────────────────────────────────────────────────────────────

class TestStreamCompletionServerIndex:
    """Tests for server index boundary conditions via prefix."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_valid_prefix_routes_to_correct_server(self, two_server_urls):
        """'0:modelA' when server 0 has modelA → success."""
        # Server 0 has modelA
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )
        # Expect completion on server 0
        respx.post("http://server0:1234/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=sse_stream("Hello"))
        )

        adapter = LMStudioAdapter(two_server_urls)
        response = ""
        async for chunk in adapter.stream_completion(
            model_id="0:modelA",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=5
        ):
            response += chunk

        assert "Hello" in response

    @pytest.mark.asyncio
    @respx.mock
    async def test_prefix_server_lacks_model_times_out(self, two_server_urls):
        """'0:modelB' when server 0 lacks modelB → RuntimeError timeout."""
        # Server 0 has modelA (not modelB)
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        adapter = LMStudioAdapter(two_server_urls)

        with pytest.raises(RuntimeError, match="Timeout"):
            async for _ in adapter.stream_completion(
                model_id="0:modelB",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=1  # Short timeout
            ):
                pass

    @pytest.mark.asyncio
    @respx.mock
    async def test_invalid_server_index_times_out(self, two_server_urls):
        """'999:model' with 2 servers → RuntimeError timeout."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        adapter = LMStudioAdapter(two_server_urls)

        with pytest.raises(RuntimeError, match="Timeout"):
            async for _ in adapter.stream_completion(
                model_id="999:modelA",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=1  # Short timeout
            ):
                pass

    @pytest.mark.asyncio
    @respx.mock
    async def test_negative_prefix_treated_as_model_name(self, two_server_urls):
        """'-1:model' → treated as model name (no numeric prefix)."""
        # Neither server has "-1:model" as a model name
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        adapter = LMStudioAdapter(two_server_urls)

        # "-1:model" is treated as a model name, not a server prefix
        # Since no server has this model, it should timeout
        with pytest.raises(RuntimeError, match="Timeout"):
            async for _ in adapter.stream_completion(
                model_id="-1:model",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=1
            ):
                pass


# ─────────────────────────────────────────────────────────────────────
# stream_completion() MODEL AVAILABILITY TESTS
# ─────────────────────────────────────────────────────────────────────

class TestStreamCompletionModelAvailability:
    """Tests for model availability routing."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_not_on_any_server_times_out(self, two_server_urls):
        """Model not on any server → RuntimeError timeout."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        adapter = LMStudioAdapter(two_server_urls)

        with pytest.raises(RuntimeError, match="Timeout"):
            async for _ in adapter.stream_completion(
                model_id="nonexistent",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=1
            ):
                pass

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_on_server_1_only_routes_correctly(self, two_server_urls):
        """Model on server 1 only, no prefix → routes to server 1."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )
        # Expect completion on server 1 (the only one with modelB)
        respx.post("http://server1:1234/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=sse_stream("World"))
        )

        adapter = LMStudioAdapter(two_server_urls)
        response = ""
        async for chunk in adapter.stream_completion(
            model_id="modelB",  # No prefix, should find on server 1
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
            max_tokens=100,
            timeout_seconds=5
        ):
            response += chunk

        assert "World" in response


# ─────────────────────────────────────────────────────────────────────
# get_models_by_server() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGetModelsByServer:
    """Tests for get_models_by_server() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_models_grouped_by_server(self, two_server_urls):
        """Returns models grouped by server URL."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA", "modelB"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelC"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        await adapter.get_available_models()  # Refresh manifests

        by_server = adapter.get_models_by_server()

        assert set(by_server["http://server0:1234"]) == {"modelA", "modelB"}
        assert set(by_server["http://server1:1234"]) == {"modelC"}


# ─────────────────────────────────────────────────────────────────────
# get_unreachable_servers() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGetUnreachableServers:
    """Tests for get_unreachable_servers() method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_unreachable_servers(self, two_server_urls):
        """Returns servers that returned no models."""
        respx.get("http://server0:1234/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        await adapter.get_available_models()  # Refresh manifests

        unreachable = adapter.get_unreachable_servers()

        assert unreachable == ["http://server0:1234"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_reachable_returns_empty(self, two_server_urls):
        """All servers reachable → returns empty list."""
        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelA"]))
        )
        respx.get("http://server1:1234/v1/models").mock(
            return_value=httpx.Response(200, json=models_response(["modelB"]))
        )

        adapter = LMStudioAdapter(two_server_urls)
        await adapter.get_available_models()

        unreachable = adapter.get_unreachable_servers()

        assert unreachable == []
