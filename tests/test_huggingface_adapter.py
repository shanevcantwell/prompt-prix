"""Tests for HuggingFace adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHuggingFaceAdapter:
    """Tests for HuggingFaceAdapter."""

    def test_init_with_token_and_models(self):
        """Adapter initializes with token and model list."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            token="hf_test_token",
            models=["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1"]
        )

        assert adapter._token == "hf_test_token"
        assert len(adapter._models) == 2

    def test_init_with_env_token(self, monkeypatch):
        """Adapter reads token from HF_TOKEN env var if not provided."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        monkeypatch.setenv("HF_TOKEN", "hf_env_token")

        adapter = HuggingFaceAdapter(models=["meta-llama/Llama-2-7b-chat-hf"])

        assert adapter._token == "hf_env_token"

    def test_init_without_token_raises(self, monkeypatch):
        """Adapter raises if no token provided and HF_TOKEN not set."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        monkeypatch.delenv("HF_TOKEN", raising=False)

        with pytest.raises(ValueError, match="HuggingFace token required"):
            HuggingFaceAdapter(models=["meta-llama/Llama-2-7b-chat-hf"])

    @pytest.mark.asyncio
    async def test_get_available_models_returns_user_list(self):
        """get_available_models returns the user-provided model list."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            token="hf_test_token",
            models=["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1"]
        )

        models = await adapter.get_available_models()

        assert models == ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1"]

    def test_add_model(self):
        """add_model adds to the model list."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(token="hf_test_token", models=[])
        adapter.add_model("meta-llama/Llama-2-7b-chat-hf")

        assert "meta-llama/Llama-2-7b-chat-hf" in adapter._models

    def test_add_model_no_duplicates(self):
        """add_model doesn't add duplicates."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            token="hf_test_token",
            models=["meta-llama/Llama-2-7b-chat-hf"]
        )
        adapter.add_model("meta-llama/Llama-2-7b-chat-hf")

        assert len(adapter._models) == 1

    def test_remove_model(self):
        """remove_model removes from the model list."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            token="hf_test_token",
            models=["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1"]
        )
        adapter.remove_model("meta-llama/Llama-2-7b-chat-hf")

        assert "meta-llama/Llama-2-7b-chat-hf" not in adapter._models
        assert len(adapter._models) == 1

    @pytest.mark.asyncio
    async def test_stream_completion_yields_content(self):
        """stream_completion yields text chunks from HF API."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            token="hf_test_token",
            models=["meta-llama/Llama-2-7b-chat-hf"]
        )

        # Mock the InferenceClient.chat_completion
        mock_chunk_1 = MagicMock()
        mock_chunk_1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        mock_chunk_2 = MagicMock()
        mock_chunk_2.choices = [MagicMock(delta=MagicMock(content=" world"))]

        with patch.object(adapter, '_client') as mock_client:
            mock_client.chat_completion.return_value = iter([mock_chunk_1, mock_chunk_2])

            chunks = []
            async for chunk in adapter.stream_completion(
                model_id="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=60
            ):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_completion_handles_none_content(self):
        """stream_completion skips chunks with None content."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            token="hf_test_token",
            models=["meta-llama/Llama-2-7b-chat-hf"]
        )

        mock_chunk_1 = MagicMock()
        mock_chunk_1.choices = [MagicMock(delta=MagicMock(content=None))]
        mock_chunk_2 = MagicMock()
        mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="Hello"))]

        with patch.object(adapter, '_client') as mock_client:
            mock_client.chat_completion.return_value = iter([mock_chunk_1, mock_chunk_2])

            chunks = []
            async for chunk in adapter.stream_completion(
                model_id="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=60
            ):
                chunks.append(chunk)

        assert chunks == ["Hello"]

    @pytest.mark.asyncio
    async def test_stream_completion_with_tools(self):
        """stream_completion passes tools to API."""
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter(
            token="hf_test_token",
            models=["meta-llama/Llama-2-7b-chat-hf"]
        )

        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        with patch.object(adapter, '_client') as mock_client:
            mock_client.chat_completion.return_value = iter([])

            async for _ in adapter.stream_completion(
                model_id="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=100,
                timeout_seconds=60,
                tools=tools
            ):
                pass

            # Verify tools were passed
            call_kwargs = mock_client.chat_completion.call_args[1]
            assert call_kwargs.get("tools") == tools


class TestHuggingFaceAdapterIntegration:
    """Integration tests requiring real HF API access."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_completion(self):
        """Test real completion against HF Inference API."""
        import os
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter

        token = os.environ.get("HF_TOKEN")
        if not token:
            pytest.skip("HF_TOKEN not set")

        adapter = HuggingFaceAdapter(
            token=token,
            models=["meta-llama/Llama-3.2-1B-Instruct"]
        )

        chunks = []
        async for chunk in adapter.stream_completion(
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=0.1,
            max_tokens=10,
            timeout_seconds=30
        ):
            chunks.append(chunk)

        response = "".join(chunks)
        assert len(response) > 0
