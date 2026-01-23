"""Tests for prompt_prix.mcp.tools.list_models module.

Per ADR-006: Mock at layer boundaries - MCP tests mock the adapter interface.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from prompt_prix.mcp.registry import register_adapter, clear_adapter
from prompt_prix.mcp.tools.list_models import list_models


@pytest.fixture
def mock_adapter():
    """Create a mock adapter and register it with the MCP registry."""
    adapter = MagicMock()

    # Default mock returns - can be overridden in individual tests
    adapter.get_available_models = AsyncMock(return_value=["model-1", "model-2"])
    adapter.get_models_by_server = MagicMock(return_value={
        "http://localhost:1234": ["model-1", "model-2"]
    })
    adapter.get_unreachable_servers = MagicMock(return_value=[])

    # stream_completion for complete tests
    async def default_stream(*args, **kwargs):
        yield "response"
    adapter.stream_completion = default_stream

    register_adapter(adapter)
    yield adapter
    clear_adapter()


class TestListModels:
    """Tests for list_models MCP tool."""

    @pytest.mark.asyncio
    async def test_list_models_returns_models(self, mock_adapter):
        """Test list_models returns available models from adapter."""
        result = await list_models()

        assert "models" in result
        assert "servers" in result
        assert "unreachable" in result

        assert "model-1" in result["models"]
        assert "model-2" in result["models"]
        assert len(result["unreachable"]) == 0

    @pytest.mark.asyncio
    async def test_list_models_multiple_servers(self, mock_adapter):
        """Test list_models returns per-server model mapping."""
        mock_adapter.get_available_models = AsyncMock(
            return_value=["model-1", "model-2"]
        )
        mock_adapter.get_models_by_server = MagicMock(return_value={
            "http://server1:1234": ["model-1", "model-2"],
            "http://server2:1234": ["model-1", "model-2"],
        })

        result = await list_models()

        # Models should be deduplicated (adapter does this)
        assert len(result["models"]) == 2

        # Servers should both have model lists
        assert "http://server1:1234" in result["servers"]
        assert "http://server2:1234" in result["servers"]

    @pytest.mark.asyncio
    async def test_list_models_different_models_per_server(self, mock_adapter):
        """Test list_models with servers having different models."""
        mock_adapter.get_available_models = AsyncMock(
            return_value=["model-1", "model-2"]
        )
        mock_adapter.get_models_by_server = MagicMock(return_value={
            "http://server1:1234": ["model-1"],
            "http://server2:1234": ["model-2"],
        })

        result = await list_models()

        assert len(result["models"]) == 2
        assert "model-1" in result["models"]
        assert "model-2" in result["models"]

        # Each server should report its own models
        assert result["servers"]["http://server1:1234"] == ["model-1"]
        assert result["servers"]["http://server2:1234"] == ["model-2"]

    @pytest.mark.asyncio
    async def test_list_models_server_unreachable(self, mock_adapter):
        """Test list_models reports unreachable servers."""
        mock_adapter.get_available_models = AsyncMock(
            return_value=["model-1", "model-2"]
        )
        mock_adapter.get_models_by_server = MagicMock(return_value={
            "http://server1:1234": ["model-1", "model-2"],
        })
        mock_adapter.get_unreachable_servers = MagicMock(return_value=[
            "http://server2:1234"
        ])

        result = await list_models()

        # Should still have models from working server
        assert "model-1" in result["models"]
        assert "model-2" in result["models"]

        # Second server should appear in unreachable
        assert "http://server2:1234" in result["unreachable"]

    @pytest.mark.asyncio
    async def test_list_models_all_servers_unreachable(self, mock_adapter):
        """Test list_models when all servers are unreachable."""
        mock_adapter.get_available_models = AsyncMock(return_value=[])
        mock_adapter.get_models_by_server = MagicMock(return_value={})
        mock_adapter.get_unreachable_servers = MagicMock(return_value=[
            "http://server1:1234",
            "http://server2:1234",
        ])

        result = await list_models()

        assert result["models"] == []
        assert len(result["unreachable"]) == 2
        assert "http://server1:1234" in result["unreachable"]
        assert "http://server2:1234" in result["unreachable"]

    @pytest.mark.asyncio
    async def test_list_models_empty_server(self, mock_adapter):
        """Test list_models with adapter that has no models loaded."""
        mock_adapter.get_available_models = AsyncMock(return_value=[])
        mock_adapter.get_models_by_server = MagicMock(return_value={})
        mock_adapter.get_unreachable_servers = MagicMock(return_value=[
            "http://server1:1234"  # Server with no models appears as unreachable
        ])

        result = await list_models()

        assert result["models"] == []
        assert "http://server1:1234" in result["unreachable"]

    @pytest.mark.asyncio
    async def test_list_models_no_adapter_registered(self):
        """Test list_models raises when no adapter registered."""
        clear_adapter()

        with pytest.raises(RuntimeError, match="No adapter registered"):
            await list_models()

    @pytest.mark.asyncio
    async def test_list_models_returns_sorted(self, mock_adapter):
        """Test list_models returns models in sorted order."""
        mock_adapter.get_available_models = AsyncMock(
            return_value=["zebra-model", "alpha-model", "beta-model"]
        )

        result = await list_models()

        # list_models() sorts the result
        assert result["models"] == ["alpha-model", "beta-model", "zebra-model"]


class TestListModelsIntegration:
    """Integration tests requiring live LM Studio servers.

    These tests are skipped by default. Run with:
        pytest -m integration
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_list_models_live(self):
        """Test list_models against real LM Studio server.

        Prerequisites:
        - LM Studio running on configured server
        - At least one model loaded
        """
        from dotenv import load_dotenv
        load_dotenv()

        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        from prompt_prix.config import get_default_servers

        servers = get_default_servers()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        adapter = LMStudioAdapter(server_urls=servers)
        register_adapter(adapter)

        try:
            result = await list_models()

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
        finally:
            clear_adapter()
