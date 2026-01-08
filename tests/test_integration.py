"""
Integration tests for LM Studio connectivity.

These tests hit REAL LM Studio endpoints - no mocks.
Run with: pytest -m integration -v

Prerequisites:
- LM Studio running on localhost:1234
- At least one model loaded in VRAM
"""

import pytest
from prompt_prix.core import ServerPool


@pytest.mark.integration
class TestServerPoolIntegration:
    """Integration tests for ServerPool against live LM Studio."""

    @pytest.fixture
    def pool(self):
        """Create a ServerPool pointing at real LM Studio."""
        return ServerPool(["http://localhost:1234"])

    @pytest.mark.asyncio
    async def test_refresh_populates_manifest_models(self, pool):
        """refresh() should populate manifest_models from /v1/models."""
        await pool.refresh()

        server = pool.servers["http://localhost:1234"]
        assert len(server.manifest_models) > 0, "No models found in manifest"
        print(f"\nManifest models: {server.manifest_models[:5]}...")

    @pytest.mark.asyncio
    async def test_refresh_populates_loaded_models(self, pool):
        """refresh() should populate loaded_models from /api/v0/models."""
        await pool.refresh()

        server = pool.servers["http://localhost:1234"]
        # loaded_models may be empty if nothing is in VRAM
        print(f"\nLoaded models: {server.loaded_models}")
        # This test documents current behavior - loaded_models should exist
        assert hasattr(server, 'loaded_models')

    @pytest.mark.asyncio
    async def test_get_available_models_returns_all(self, pool):
        """get_available_models(only_loaded=False) returns all manifest models."""
        await pool.refresh()

        all_models = pool.get_available_models(only_loaded=False)
        assert len(all_models) > 0, "No models available"
        print(f"\nAll available models: {len(all_models)}")

    @pytest.mark.asyncio
    async def test_get_available_models_only_loaded_filters(self, pool):
        """get_available_models(only_loaded=True) returns only loaded models."""
        await pool.refresh()

        server = pool.servers["http://localhost:1234"]
        loaded = pool.get_available_models(only_loaded=True)
        all_models = pool.get_available_models(only_loaded=False)

        print(f"\nLoaded: {loaded}")
        print(f"All: {len(all_models)}")

        # Key assertion: loaded should be subset of or equal to all
        assert len(loaded) <= len(all_models)

        # If models are loaded, they should appear in loaded list
        if server.loaded_models:
            assert len(loaded) > 0, "Models in VRAM but only_loaded returned empty"

    @pytest.mark.asyncio
    async def test_find_server_for_manifest_model(self, pool):
        """find_server() should find server for any manifest model."""
        await pool.refresh()

        server_config = pool.servers["http://localhost:1234"]
        if not server_config.manifest_models:
            pytest.skip("No manifest models available")

        test_model = server_config.manifest_models[0]
        found = pool.find_server(test_model)

        assert found == "http://localhost:1234", f"Couldn't find server for {test_model}"

    @pytest.mark.asyncio
    async def test_find_server_for_loaded_model(self, pool):
        """find_server() should find server for loaded model."""
        await pool.refresh()

        server_config = pool.servers["http://localhost:1234"]
        if not server_config.loaded_models:
            pytest.skip("No models currently loaded in VRAM")

        test_model = server_config.loaded_models[0]
        found = pool.find_server(test_model)

        assert found == "http://localhost:1234", f"Couldn't find server for loaded model {test_model}"

    @pytest.mark.asyncio
    async def test_find_server_returns_none_for_unknown(self, pool):
        """find_server() should return None for unknown model."""
        await pool.refresh()

        found = pool.find_server("nonexistent-model-xyz-123")
        assert found is None

    @pytest.mark.asyncio
    async def test_model_id_format_consistency(self, pool):
        """Model IDs from /v1/models should match those from /api/v0/models."""
        await pool.refresh()

        server = pool.servers["http://localhost:1234"]

        # If we have loaded models, they should be findable in manifest
        for loaded_model in server.loaded_models:
            assert loaded_model in server.manifest_models, \
                f"Loaded model '{loaded_model}' not in manifest_models. " \
                f"Format mismatch between /v1/models and /api/v0/models?"


@pytest.mark.integration
class TestOnlyLoadedFilterIntegration:
    """Integration tests for the Only Loaded checkbox behavior."""

    @pytest.mark.asyncio
    async def test_only_loaded_returns_vram_models(self):
        """The only_loaded filter should return exactly the models in VRAM."""
        import httpx

        # Get ground truth from LM Studio API directly
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:1234/api/v0/models")
            data = resp.json()

        actual_loaded = [
            m["id"] for m in data["data"]
            if m.get("state") == "loaded"
        ]

        # Now test ServerPool
        pool = ServerPool(["http://localhost:1234"])
        await pool.refresh()

        pool_loaded = pool.get_available_models(only_loaded=True)

        print(f"\nAPI reports loaded: {actual_loaded}")
        print(f"ServerPool reports: {pool_loaded}")

        # They should match exactly
        assert set(pool_loaded) == set(actual_loaded), \
            f"Mismatch! API: {actual_loaded}, Pool: {pool_loaded}"
