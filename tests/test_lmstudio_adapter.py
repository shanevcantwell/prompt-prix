"""Tests for LMStudioAdapter backwards-compatibility layer.

Verifies that the thin subclass and re-exports work correctly.
Bulk adapter tests live in test_pooled_local_adapter.py.
"""

import pytest
import respx
import httpx

from prompt_prix.adapters.pooled_local import PooledLocalInferenceAdapter, LocalInferenceError


class TestLMStudioBackwardsCompat:
    """Verify LMStudioAdapter is a proper subclass with working re-exports."""

    def test_subclass_relationship(self):
        """LMStudioAdapter is a subclass of PooledLocalInferenceAdapter."""
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        assert issubclass(LMStudioAdapter, PooledLocalInferenceAdapter)

    def test_error_alias(self):
        """LMStudioError is LocalInferenceError."""
        from prompt_prix.adapters.lmstudio import LMStudioError
        assert LMStudioError is LocalInferenceError

    def test_stream_completion_reexport(self):
        """stream_completion function is re-exported from lmstudio module."""
        from prompt_prix.adapters.lmstudio import stream_completion
        from prompt_prix.adapters.pooled_local import stream_completion as canonical
        assert stream_completion is canonical

    def test_normalize_tools_reexport(self):
        """_normalize_tools_for_openai is re-exported from lmstudio module."""
        from prompt_prix.adapters.lmstudio import _normalize_tools_for_openai
        from prompt_prix.adapters.pooled_local import _normalize_tools_for_openai as canonical
        assert _normalize_tools_for_openai is canonical

    @pytest.mark.asyncio
    @respx.mock
    async def test_instantiation_and_models(self):
        """LMStudioAdapter can be instantiated and used like the generic adapter."""
        from prompt_prix.adapters.lmstudio import LMStudioAdapter

        respx.get("http://server0:1234/v1/models").mock(
            return_value=httpx.Response(
                200, json={"data": [{"id": "modelA"}]}
            )
        )

        adapter = LMStudioAdapter(["http://server0:1234"])
        models = await adapter.get_available_models()
        assert "modelA" in models

    def test_isinstance_check(self):
        """LMStudioAdapter instances pass isinstance for PooledLocalInferenceAdapter."""
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        adapter = LMStudioAdapter(["http://localhost:1234"])
        assert isinstance(adapter, PooledLocalInferenceAdapter)

    def test_core_error_alias(self):
        """core.py LMStudioError is the same as LocalInferenceError."""
        from prompt_prix.core import LMStudioError
        assert LMStudioError is LocalInferenceError
