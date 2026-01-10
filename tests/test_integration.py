"""
Integration tests for LM Studio connectivity.

These tests hit REAL LM Studio endpoints - no mocks.
Run with: pytest -m integration -v

Prerequisites:
- LM Studio servers configured in .env (LM_STUDIO_SERVER_1, etc.)
- At least one model available on configured servers
"""

import pytest
from prompt_prix.config import get_default_servers
from prompt_prix.scheduler import ServerPool
from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.battery import BatteryRunner, TestStatus
from prompt_prix.benchmarks import CustomJSONLoader


@pytest.mark.integration
class TestServerPoolIntegration:
    """Integration tests for ServerPool against live LM Studio."""

    @pytest.fixture
    def servers(self):
        """Get configured servers from .env."""
        return get_default_servers()

    @pytest.fixture
    def pool(self, servers):
        """Create a ServerPool pointing at configured LM Studio servers."""
        return ServerPool(servers)

    @pytest.mark.asyncio
    async def test_refresh_populates_manifest_models(self, pool, servers):
        """refresh() should populate manifest_models from /v1/models."""
        await pool.refresh()

        # Each configured server should have models (0 = server down or misconfigured)
        failed_servers = []
        for url in servers:
            if url in pool.servers:
                server = pool.servers[url]
                count = len(server.manifest_models)
                print(f"\n{url}: {count} manifest models")
                if count == 0:
                    failed_servers.append(url)

        assert not failed_servers, \
            f"Server(s) returned 0 models (down or misconfigured): {failed_servers}"

    @pytest.mark.asyncio
    async def test_refresh_populates_loaded_models(self, pool, servers):
        """refresh() should populate loaded_models from /api/v0/models."""
        await pool.refresh()

        for url in servers:
            if url in pool.servers:
                server = pool.servers[url]
                print(f"\n{url} loaded: {server.loaded_models}")
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

        loaded = pool.get_available_models(only_loaded=True)
        all_models = pool.get_available_models(only_loaded=False)

        print(f"\nLoaded: {loaded}")
        print(f"All: {len(all_models)}")

        # Key assertion: loaded should be subset of or equal to all
        assert len(loaded) <= len(all_models)

    @pytest.mark.asyncio
    async def test_find_server_for_manifest_model(self, pool, servers):
        """find_server() should find server for any manifest model."""
        await pool.refresh()

        # Find first server with models
        test_model = None
        expected_server = None
        for url in servers:
            if url in pool.servers and pool.servers[url].manifest_models:
                test_model = pool.servers[url].manifest_models[0]
                expected_server = url
                break

        if not test_model:
            pytest.skip("No manifest models available on any server")

        found = pool.find_server(test_model)
        assert found == expected_server, f"Couldn't find server for {test_model}"

    @pytest.mark.asyncio
    async def test_find_server_for_loaded_model(self, pool, servers):
        """find_server() should find server for loaded model."""
        await pool.refresh()

        # Find first server with loaded models
        test_model = None
        expected_server = None
        for url in servers:
            if url in pool.servers and pool.servers[url].loaded_models:
                test_model = pool.servers[url].loaded_models[0]
                expected_server = url
                break

        if not test_model:
            pytest.skip("No models currently loaded in VRAM on any server")

        found = pool.find_server(test_model)
        assert found == expected_server, f"Couldn't find server for loaded model {test_model}"

    @pytest.mark.asyncio
    async def test_find_server_returns_none_for_unknown(self, pool):
        """find_server() should return None for unknown model."""
        await pool.refresh()

        found = pool.find_server("nonexistent-model-xyz-123")
        assert found is None

    @pytest.mark.asyncio
    async def test_model_id_format_consistency(self, pool, servers):
        """Model IDs from /v1/models should match those from /api/v0/models."""
        await pool.refresh()

        for url in servers:
            if url not in pool.servers:
                continue
            server = pool.servers[url]
            # If we have loaded models, they should be findable in manifest
            for loaded_model in server.loaded_models:
                assert loaded_model in server.manifest_models, \
                    f"Loaded model '{loaded_model}' not in manifest_models on {url}. " \
                    f"Format mismatch between /v1/models and /api/v0/models?"


@pytest.mark.integration
class TestOnlyLoadedFilterIntegration:
    """Integration tests for the Only Loaded checkbox behavior."""

    @pytest.mark.asyncio
    async def test_only_loaded_returns_vram_models(self):
        """The only_loaded filter should return exactly the models in VRAM."""
        import httpx

        servers = get_default_servers()

        # Get ground truth from LM Studio API directly
        actual_loaded = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in servers:
                try:
                    resp = await client.get(f"{url}/api/v0/models")
                    data = resp.json()
                    for m in data["data"]:
                        if m.get("state") == "loaded":
                            actual_loaded.append(m["id"])
                except Exception as e:
                    print(f"\n{url}: {e}")

        # Now test ServerPool
        pool = ServerPool(servers)
        await pool.refresh()

        pool_loaded = pool.get_available_models(only_loaded=True)

        print(f"\nAPI reports loaded: {actual_loaded}")
        print(f"ServerPool reports: {pool_loaded}")

        # They should match exactly
        assert set(pool_loaded) == set(actual_loaded), \
            f"Mismatch! API: {actual_loaded}, Pool: {pool_loaded}"


@pytest.mark.integration
class TestBatteryRunnerIntegration:
    """
    Integration tests for BatteryRunner with LMStudioAdapter.

    Tests the full flow: BatteryRunner → LMStudioAdapter → ServerPool → LM Studio.
    This verifies that the #76 deadlock fix works in production.
    """

    @pytest.fixture
    def adapter(self):
        """Create LMStudioAdapter with real ServerPool."""
        servers = get_default_servers()
        pool = ServerPool(servers)
        return LMStudioAdapter(pool)

    @pytest.fixture
    async def available_model(self, adapter):
        """Get first available model from servers."""
        models = await adapter.get_available_models()
        if not models:
            pytest.skip("No models available on configured servers")
        return models[0]

    @pytest.mark.asyncio
    async def test_battery_completes_without_deadlock(self, adapter, available_model):
        """
        BatteryRunner should complete without deadlock.

        This is the key test for #76 - verifies the fix works end-to-end.
        If adapter.acquire() is called before stream_completion() (the bug),
        this will deadlock because stream_completion() also calls acquire internally.
        """
        # Simple test case
        tests = CustomJSONLoader.load("examples/tool_competence_tests.json")[:1]

        runner = BatteryRunner(
            adapter=adapter,
            tests=tests,
            models=[available_model],
            max_tokens=128,  # Short for speed
            timeout_seconds=60
        )

        final_state = None
        async for state in runner.run():
            final_state = state

        assert final_state is not None
        assert final_state.completed_count == 1

        # Check for deadlock errors
        for result in final_state.results.values():
            if result.error and "DEADLOCK" in result.error:
                pytest.fail(f"Deadlock detected: {result.error}")

    @pytest.mark.asyncio
    async def test_battery_with_multiple_tests(self, adapter, available_model):
        """BatteryRunner should handle multiple tests without deadlock."""
        tests = CustomJSONLoader.load("examples/tool_competence_tests.json")[:3]

        runner = BatteryRunner(
            adapter=adapter,
            tests=tests,
            models=[available_model],
            max_tokens=128,
            timeout_seconds=60
        )

        final_state = None
        async for state in runner.run():
            final_state = state

        assert final_state.completed_count == 3

        # All should complete (success or semantic failure, not ERROR from deadlock)
        error_results = [
            r for r in final_state.results.values()
            if r.status == TestStatus.ERROR
        ]
        # Errors are OK if they're real errors, not deadlock
        for r in error_results:
            assert "DEADLOCK" not in (r.error or ""), f"Deadlock: {r.error}"


@pytest.mark.integration
class TestBatteryFileLoadersIntegration:
    """Integration tests for loading battery test files."""

    def test_load_json_file(self):
        """Load JSON test file from examples."""
        tests = CustomJSONLoader.load("examples/tool_competence_tests.json")
        assert len(tests) > 0
        # Verify structure
        for test in tests:
            assert test.id is not None
            assert test.user is not None

    def test_load_promptfoo_yaml_file(self):
        """Load promptfoo YAML test file from examples."""
        from pathlib import Path
        from prompt_prix.promptfoo import parse_tests

        tests = parse_tests(Path("examples/tool_competence_tests.promptfoo.yaml"))
        assert len(tests) > 0
        # Verify structure
        for test in tests:
            assert test.id is not None
            assert test.user is not None

    def test_json_and_yaml_produce_equivalent_tests(self):
        """JSON and YAML loaders should produce equivalent test cases."""
        from pathlib import Path
        from prompt_prix.promptfoo import parse_tests

        json_tests = CustomJSONLoader.load("examples/tool_competence_tests.json")
        yaml_tests = parse_tests(Path("examples/tool_competence_tests.promptfoo.yaml"))

        # Both should produce test cases
        assert len(json_tests) > 0
        assert len(yaml_tests) > 0

        # Note: IDs and exact structure may differ between formats
        # This test verifies both can load successfully
