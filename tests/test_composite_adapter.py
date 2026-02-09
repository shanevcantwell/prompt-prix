"""Tests for CompositeAdapter — routing, merging, collision handling."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from prompt_prix.adapters.composite import CompositeAdapter
from prompt_prix.adapters.schema import InferenceTask


def make_mock_adapter(models, server_name="mock-server"):
    """Create a mock adapter with given models."""
    adapter = AsyncMock()
    adapter.get_available_models = AsyncMock(return_value=list(models))
    adapter.get_models_by_server = MagicMock(
        return_value={server_name: list(models)}
    )
    adapter.get_unreachable_servers = MagicMock(return_value=[])

    async def stream_chunks(task):
        yield f"response from {server_name} for {task.model_id}"
        yield "__LATENCY_MS__:100"

    adapter.stream_completion = stream_chunks
    return adapter


def make_task(model_id="modelA"):
    return InferenceTask(
        model_id=model_id,
        messages=[{"role": "user", "content": "Hi"}],
    )


# ─────────────────────────────────────────────────────────────────────
# Single adapter passthrough
# ─────────────────────────────────────────────────────────────────────


class TestSingleAdapter:
    @pytest.mark.asyncio
    async def test_single_adapter_returns_models(self):
        adapter = make_mock_adapter(["modelA", "modelB"], "server1")
        composite = CompositeAdapter({"lmstudio": adapter})

        models = await composite.get_available_models()
        assert sorted(models) == ["modelA", "modelB"]

    @pytest.mark.asyncio
    async def test_single_adapter_routes_correctly(self):
        adapter = make_mock_adapter(["modelA"], "server1")
        composite = CompositeAdapter({"lmstudio": adapter})

        await composite.get_available_models()

        chunks = []
        async for chunk in composite.stream_completion(make_task("modelA")):
            chunks.append(chunk)

        assert any("server1" in c for c in chunks)


# ─────────────────────────────────────────────────────────────────────
# Multi-adapter merging
# ─────────────────────────────────────────────────────────────────────


class TestMultiAdapterMerging:
    @pytest.mark.asyncio
    async def test_merges_unique_models(self):
        a1 = make_mock_adapter(["modelA"], "lmstudio")
        a2 = make_mock_adapter(["modelB"], "together")
        composite = CompositeAdapter({"lmstudio": a1, "together": a2})

        models = await composite.get_available_models()
        assert sorted(models) == ["modelA", "modelB"]

    @pytest.mark.asyncio
    async def test_routes_to_correct_adapter(self):
        a1 = make_mock_adapter(["modelA"], "lmstudio")
        a2 = make_mock_adapter(["modelB"], "together")
        composite = CompositeAdapter({"lmstudio": a1, "together": a2})

        await composite.get_available_models()

        chunks = []
        async for chunk in composite.stream_completion(make_task("modelB")):
            chunks.append(chunk)

        assert any("together" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_merges_models_by_server(self):
        a1 = make_mock_adapter(["modelA"], "lmstudio")
        a2 = make_mock_adapter(["modelB"], "together")
        composite = CompositeAdapter({"lmstudio": a1, "together": a2})

        by_server = composite.get_models_by_server()
        assert "lmstudio" in by_server
        assert "together" in by_server

    @pytest.mark.asyncio
    async def test_merges_unreachable_servers(self):
        a1 = make_mock_adapter(["modelA"], "lmstudio")
        a2 = AsyncMock()
        a2.get_available_models = AsyncMock(return_value=[])
        a2.get_models_by_server = MagicMock(return_value={})
        a2.get_unreachable_servers = MagicMock(
            return_value=["http://down:1234"]
        )
        composite = CompositeAdapter({"lmstudio": a1, "down": a2})

        unreachable = composite.get_unreachable_servers()
        assert "http://down:1234" in unreachable


# ─────────────────────────────────────────────────────────────────────
# Collision handling
# ─────────────────────────────────────────────────────────────────────


class TestCollisionHandling:
    @pytest.mark.asyncio
    async def test_collision_prefixes_with_adapter_name(self):
        """Same model on two adapters gets prefixed."""
        a1 = make_mock_adapter(["shared-model"], "lmstudio")
        a2 = make_mock_adapter(["shared-model"], "together")
        composite = CompositeAdapter({"lmstudio": a1, "together": a2})

        models = await composite.get_available_models()
        assert sorted(models) == [
            "lmstudio/shared-model",
            "together/shared-model",
        ]

    @pytest.mark.asyncio
    async def test_prefixed_model_routes_correctly(self):
        """Prefixed model routes to correct adapter with prefix stripped."""
        a1 = make_mock_adapter(["shared-model"], "lmstudio")
        a2 = make_mock_adapter(["shared-model"], "together")
        composite = CompositeAdapter({"lmstudio": a1, "together": a2})

        await composite.get_available_models()

        chunks = []
        async for chunk in composite.stream_completion(
            make_task("together/shared-model")
        ):
            chunks.append(chunk)

        # The adapter receives "shared-model" (prefix stripped)
        assert any("together" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_mix_of_unique_and_colliding(self):
        """Unique models stay bare, collisions get prefixed."""
        a1 = make_mock_adapter(["unique-a", "shared"], "lmstudio")
        a2 = make_mock_adapter(["unique-b", "shared"], "together")
        composite = CompositeAdapter({"lmstudio": a1, "together": a2})

        models = await composite.get_available_models()
        assert "unique-a" in models  # Bare
        assert "unique-b" in models  # Bare
        assert "lmstudio/shared" in models  # Prefixed
        assert "together/shared" in models  # Prefixed


# ─────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_model_raises(self):
        adapter = make_mock_adapter(["modelA"], "server1")
        composite = CompositeAdapter({"lmstudio": adapter})

        await composite.get_available_models()

        with pytest.raises(ValueError, match="Unknown model"):
            async for _ in composite.stream_completion(make_task("nonexistent")):
                pass


# ─────────────────────────────────────────────────────────────────────
# Registry integration
# ─────────────────────────────────────────────────────────────────────


class TestRegistryIntegration:
    def test_single_adapter_returns_directly(self):
        """With one adapter, get_adapter() returns it directly (no composite)."""
        from prompt_prix.mcp.registry import register_adapter, get_adapter, clear_adapter

        mock = make_mock_adapter(["modelA"], "server1")
        register_adapter(mock, name="lmstudio")

        try:
            result = get_adapter()
            assert result is mock  # Direct, not wrapped
        finally:
            clear_adapter()

    def test_multiple_adapters_returns_composite(self):
        """With multiple adapters, get_adapter() returns CompositeAdapter."""
        from prompt_prix.mcp.registry import register_adapter, get_adapter, clear_adapter

        a1 = make_mock_adapter(["modelA"], "lmstudio")
        a2 = make_mock_adapter(["modelB"], "together")
        register_adapter(a1, name="lmstudio")
        register_adapter(a2, name="together")

        try:
            result = get_adapter()
            assert isinstance(result, CompositeAdapter)
        finally:
            clear_adapter()
