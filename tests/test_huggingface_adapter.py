"""Tests for HuggingFaceAdapter.

Mocks AsyncInferenceClient to test adapter logic without real API calls.
Integration tests (marked @pytest.mark.integration) require HF_TOKEN.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional

from prompt_prix.adapters.huggingface import HuggingFaceAdapter, HuggingFaceError
from prompt_prix.adapters.schema import InferenceTask


# ─────────────────────────────────────────────────────────────────────
# MOCK HELPERS
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MockDelta:
    """Mock for ChatCompletionStreamOutputDelta."""
    content: Optional[str] = None
    role: Optional[str] = None


@dataclass
class MockChoice:
    """Mock for ChatCompletionStreamOutputChoice."""
    delta: MockDelta
    finish_reason: Optional[str] = None
    index: int = 0


@dataclass
class MockStreamOutput:
    """Mock for ChatCompletionStreamOutput."""
    choices: list[MockChoice]


def make_stream_chunks(tokens: list[str]) -> list[MockStreamOutput]:
    """Build mock stream output chunks from token list."""
    chunks = []
    for token in tokens:
        chunks.append(MockStreamOutput(
            choices=[MockChoice(delta=MockDelta(content=token))]
        ))
    # Final chunk with finish_reason
    chunks.append(MockStreamOutput(
        choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]
    ))
    return chunks


async def async_iter(items):
    """Convert list to async iterator."""
    for item in items:
        yield item


# ─────────────────────────────────────────────────────────────────────
# INITIALIZATION TESTS
# ─────────────────────────────────────────────────────────────────────

class TestHuggingFaceAdapterInit:
    """Tests for adapter initialization."""

    def test_requires_token(self):
        """Raises ValueError if no token provided and HF_TOKEN not set."""
        # Ensure HF_TOKEN is not set
        with patch.dict(os.environ, {}, clear=True):
            # Remove HF_TOKEN if it exists
            os.environ.pop("HF_TOKEN", None)

            with pytest.raises(ValueError, match="HuggingFace token required"):
                HuggingFaceAdapter(models=["test/model"])

    def test_accepts_token_parameter(self):
        """Accepts token via parameter."""
        with patch("prompt_prix.adapters.huggingface.AsyncInferenceClient"):
            adapter = HuggingFaceAdapter(
                models=["test/model"],
                token="hf_test_token"
            )
            assert adapter._token == "hf_test_token"

    def test_uses_env_token_fallback(self):
        """Falls back to HF_TOKEN env var."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_env_token"}):
            with patch("prompt_prix.adapters.huggingface.AsyncInferenceClient"):
                adapter = HuggingFaceAdapter(models=["test/model"])
                assert adapter._token == "hf_env_token"

    def test_stores_model_list(self):
        """Stores provided model list."""
        with patch("prompt_prix.adapters.huggingface.AsyncInferenceClient"):
            adapter = HuggingFaceAdapter(
                models=["model/a", "model/b"],
                token="hf_test"
            )
            assert adapter._models == ["model/a", "model/b"]


# ─────────────────────────────────────────────────────────────────────
# get_available_models() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGetAvailableModels:
    """Tests for model list methods."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked client."""
        with patch("prompt_prix.adapters.huggingface.AsyncInferenceClient"):
            return HuggingFaceAdapter(
                models=["meta-llama/Llama-3.2-3B", "mistral/Mistral-7B"],
                token="hf_test"
            )

    @pytest.mark.asyncio
    async def test_returns_configured_models(self, adapter):
        """Returns the models provided at initialization."""
        models = await adapter.get_available_models()
        assert models == ["meta-llama/Llama-3.2-3B", "mistral/Mistral-7B"]

    def test_get_models_by_server(self, adapter):
        """Returns models under huggingface-inference key."""
        result = adapter.get_models_by_server()
        assert result == {
            "huggingface-inference": ["meta-llama/Llama-3.2-3B", "mistral/Mistral-7B"]
        }

    def test_get_unreachable_servers_always_empty(self, adapter):
        """HF is cloud API - always returns empty list."""
        assert adapter.get_unreachable_servers() == []

    def test_add_model(self, adapter):
        """add_model() appends to list."""
        adapter.add_model("new/model")
        assert "new/model" in adapter._models

    def test_add_model_no_duplicate(self, adapter):
        """add_model() doesn't add duplicates."""
        adapter.add_model("meta-llama/Llama-3.2-3B")
        assert adapter._models.count("meta-llama/Llama-3.2-3B") == 1

    def test_remove_model(self, adapter):
        """remove_model() removes from list."""
        adapter.remove_model("meta-llama/Llama-3.2-3B")
        assert "meta-llama/Llama-3.2-3B" not in adapter._models


# ─────────────────────────────────────────────────────────────────────
# stream_completion() TESTS
# ─────────────────────────────────────────────────────────────────────

class TestStreamCompletion:
    """Tests for streaming completion."""

    @pytest.fixture
    def mock_client(self):
        """Create mock AsyncInferenceClient."""
        return MagicMock()

    @pytest.fixture
    def adapter(self, mock_client):
        """Create adapter with mocked client."""
        with patch("prompt_prix.adapters.huggingface.AsyncInferenceClient", return_value=mock_client):
            return HuggingFaceAdapter(
                models=["test/model"],
                token="hf_test"
            )

    @pytest.mark.asyncio
    async def test_streams_content_tokens(self, adapter, mock_client):
        """Yields content tokens from stream."""
        chunks = make_stream_chunks(["Hello", " ", "world", "!"])
        mock_client.chat.completions.create = AsyncMock(
            return_value=async_iter(chunks)
        )

        task = InferenceTask(
            model_id="test/model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )

        result = []
        async for chunk in adapter.stream_completion(task):
            result.append(chunk)

        # Should have content + latency sentinel
        assert "Hello" in result
        assert " " in result
        assert "world" in result
        assert "!" in result
        assert any(c.startswith("__LATENCY_MS__:") for c in result)

    @pytest.mark.asyncio
    async def test_yields_latency_sentinel(self, adapter, mock_client):
        """Yields latency sentinel at end of stream."""
        chunks = make_stream_chunks(["Test"])
        mock_client.chat.completions.create = AsyncMock(
            return_value=async_iter(chunks)
        )

        task = InferenceTask(
            model_id="test/model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )

        result = []
        async for chunk in adapter.stream_completion(task):
            result.append(chunk)

        # Last chunk should be latency sentinel
        latency_chunk = [c for c in result if c.startswith("__LATENCY_MS__:")]
        assert len(latency_chunk) == 1

        # Should be a valid float
        latency_ms = float(latency_chunk[0].split(":")[1])
        assert latency_ms >= 0

    @pytest.mark.asyncio
    async def test_passes_parameters_to_client(self, adapter, mock_client):
        """Passes task parameters to HF client."""
        chunks = make_stream_chunks(["OK"])
        mock_create = AsyncMock(return_value=async_iter(chunks))
        mock_client.chat.completions.create = mock_create

        task = InferenceTask(
            model_id="custom/model",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.5,
            max_tokens=100,
            seed=42
        )

        async for _ in adapter.stream_completion(task):
            pass

        # Verify call arguments
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "custom/model"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["seed"] == 42
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_skips_max_tokens_when_negative(self, adapter, mock_client):
        """Does not pass max_tokens when set to -1 (unlimited)."""
        chunks = make_stream_chunks(["OK"])
        mock_create = AsyncMock(return_value=async_iter(chunks))
        mock_client.chat.completions.create = mock_create

        task = InferenceTask(
            model_id="test/model",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=-1  # Unlimited
        )

        async for _ in adapter.stream_completion(task):
            pass

        call_kwargs = mock_create.call_args.kwargs
        assert "max_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_handles_empty_response(self, adapter, mock_client):
        """Empty response yields only latency sentinel."""
        # Stream with no content, just finish
        chunks = [MockStreamOutput(
            choices=[MockChoice(delta=MockDelta(), finish_reason="stop")]
        )]
        mock_client.chat.completions.create = AsyncMock(
            return_value=async_iter(chunks)
        )

        task = InferenceTask(
            model_id="test/model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )

        result = []
        async for chunk in adapter.stream_completion(task):
            result.append(chunk)

        # Should only have latency sentinel
        assert len(result) == 1
        assert result[0].startswith("__LATENCY_MS__:")


# ─────────────────────────────────────────────────────────────────────
# ERROR HANDLING TESTS
# ─────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def mock_client(self):
        """Create mock AsyncInferenceClient."""
        return MagicMock()

    @pytest.fixture
    def adapter(self, mock_client):
        """Create adapter with mocked client."""
        with patch("prompt_prix.adapters.huggingface.AsyncInferenceClient", return_value=mock_client):
            return HuggingFaceAdapter(
                models=["test/model"],
                token="hf_test"
            )

    @pytest.mark.asyncio
    async def test_wraps_timeout_error(self, adapter, mock_client):
        """Wraps InferenceTimeoutError in HuggingFaceError."""
        from huggingface_hub.errors import InferenceTimeoutError

        mock_client.chat.completions.create = AsyncMock(
            side_effect=InferenceTimeoutError("Timeout")
        )

        task = InferenceTask(
            model_id="test/model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )

        with pytest.raises(HuggingFaceError, match="timeout"):
            async for _ in adapter.stream_completion(task):
                pass

    @pytest.mark.asyncio
    async def test_wraps_http_error(self, adapter, mock_client):
        """Wraps HfHubHTTPError in HuggingFaceError."""
        from huggingface_hub.errors import HfHubHTTPError

        mock_client.chat.completions.create = AsyncMock(
            side_effect=HfHubHTTPError("HTTP 429 Too Many Requests")
        )

        task = InferenceTask(
            model_id="test/model",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )

        with pytest.raises(HuggingFaceError, match="API error"):
            async for _ in adapter.stream_completion(task):
                pass


# ─────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS (require HF_TOKEN)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestHuggingFaceIntegration:
    """Integration tests with real HF API. Require HF_TOKEN env var."""

    @pytest.fixture
    def hf_token(self):
        """Get HF token from environment."""
        token = os.environ.get("HF_TOKEN")
        if not token:
            pytest.skip("HF_TOKEN not set")
        return token

    @pytest.mark.asyncio
    async def test_real_streaming_completion(self, hf_token):
        """Real streaming completion with small model."""
        adapter = HuggingFaceAdapter(
            models=["HuggingFaceH4/zephyr-7b-beta"],
            token=hf_token
        )

        task = InferenceTask(
            model_id="HuggingFaceH4/zephyr-7b-beta",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            temperature=0.1,
            max_tokens=10
        )

        result = []
        async for chunk in adapter.stream_completion(task):
            result.append(chunk)

        # Should have content and latency
        content = "".join(c for c in result if not c.startswith("__LATENCY_MS__:"))
        assert len(content) > 0
        assert any(c.startswith("__LATENCY_MS__:") for c in result)
