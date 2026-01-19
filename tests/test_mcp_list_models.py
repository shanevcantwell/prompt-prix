"""Tests for prompt_prix.mcp.tools.list_models module."""

import pytest
import httpx
import respx

from tests.conftest import (
    MOCK_SERVER_1, MOCK_SERVER_2, MOCK_SERVERS,
    MOCK_MODEL_1, MOCK_MODEL_2,
    MOCK_MANIFEST_RESPONSE,
)


class TestListModels:
    """Tests for list_models MCP tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_single_server(self):
        """Test list_models with single server returning models."""
        from prompt_prix.mcp.tools.list_models import list_models

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        result = await list_models([MOCK_SERVER_1])

        assert "models" in result
        assert "servers" in result
        assert "unreachable" in result

        assert MOCK_MODEL_1 in result["models"]
        assert MOCK_MODEL_2 in result["models"]
        assert len(result["unreachable"]) == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_multiple_servers(self):
        """Test list_models with multiple servers, deduplicates models."""
        from prompt_prix.mcp.tools.list_models import list_models

        # Both servers return same models - should deduplicate
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        result = await list_models(MOCK_SERVERS)

        # Models should be deduplicated
        assert len(result["models"]) == 2
        assert MOCK_MODEL_1 in result["models"]
        assert MOCK_MODEL_2 in result["models"]

        # Servers should both have model lists
        assert MOCK_SERVER_1 in result["servers"]
        assert MOCK_SERVER_2 in result["servers"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_multiple_servers_different_models(self):
        """Test list_models with servers having different models."""
        from prompt_prix.mcp.tools.list_models import list_models

        # Server 1 has model 1, Server 2 has model 2
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_1}]})
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": MOCK_MODEL_2}]})
        )

        result = await list_models(MOCK_SERVERS)

        # Should have both models from both servers
        assert len(result["models"]) == 2
        assert MOCK_MODEL_1 in result["models"]
        assert MOCK_MODEL_2 in result["models"]

        # Each server should report its own models
        assert result["servers"][MOCK_SERVER_1] == [MOCK_MODEL_1]
        assert result["servers"][MOCK_SERVER_2] == [MOCK_MODEL_2]

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_server_unreachable(self):
        """Test list_models gracefully handles unreachable server."""
        from prompt_prix.mcp.tools.list_models import list_models

        # First server succeeds, second fails
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await list_models(MOCK_SERVERS)

        # Should still have models from working server
        assert MOCK_MODEL_1 in result["models"]
        assert MOCK_MODEL_2 in result["models"]

        # Second server should appear in unreachable
        assert MOCK_SERVER_2 in result["unreachable"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_all_servers_unreachable(self):
        """Test list_models when all servers are unreachable."""
        from prompt_prix.mcp.tools.list_models import list_models

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await list_models(MOCK_SERVERS)

        assert result["models"] == []
        assert len(result["unreachable"]) == 2
        assert MOCK_SERVER_1 in result["unreachable"]
        assert MOCK_SERVER_2 in result["unreachable"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_empty_server(self):
        """Test list_models with server that has no models loaded."""
        from prompt_prix.mcp.tools.list_models import list_models

        # Server responds but has no models
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        result = await list_models([MOCK_SERVER_1])

        assert result["models"] == []
        # Server with no models appears as unreachable (proxy for "no models")
        assert MOCK_SERVER_1 in result["unreachable"]

    @pytest.mark.asyncio
    async def test_list_models_empty_input(self):
        """Test list_models with empty server list."""
        from prompt_prix.mcp.tools.list_models import list_models

        result = await list_models([])

        assert result == {
            "models": [],
            "servers": {},
            "unreachable": [],
        }

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models_returns_sorted(self):
        """Test list_models returns models in sorted order."""
        from prompt_prix.mcp.tools.list_models import list_models

        # Return models in reverse order
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"id": "zebra-model"},
                    {"id": "alpha-model"},
                    {"id": "beta-model"},
                ]
            })
        )

        result = await list_models([MOCK_SERVER_1])

        # Should be sorted
        assert result["models"] == ["alpha-model", "beta-model", "zebra-model"]


class TestListModelsIntegration:
    """Integration tests requiring live LM Studio servers."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_list_models_live(self):
        """Test list_models against real LM Studio server.

        Prerequisites:
        - LM Studio running on configured server
        - At least one model loaded

        This test uses the server URLs from .env or defaults.
        """
        from dotenv import load_dotenv
        load_dotenv()

        from prompt_prix.mcp.tools.list_models import list_models
        from prompt_prix.config import load_servers_from_env

        servers = load_servers_from_env()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        result = await list_models(servers)

        # Should have structured response
        assert "models" in result
        assert "servers" in result
        assert "unreachable" in result
        assert isinstance(result["models"], list)

        # If we have reachable servers, should have some models
        reachable_servers = [
            url for url in servers
            if url not in result["unreachable"]
        ]
        if reachable_servers:
            assert len(result["models"]) > 0, "Expected at least one model from reachable servers"
