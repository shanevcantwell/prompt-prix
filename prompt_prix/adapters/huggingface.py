"""
HuggingFace Inference API adapter.

Implements HostAdapter protocol for HuggingFace serverless inference.
Uses huggingface_hub.InferenceClient for API access.

Note: Unlike LMStudioAdapter, HF has no discovery endpoint.
Models are user-provided, not discovered.
"""

import os
from typing import AsyncGenerator, Optional

from huggingface_hub import InferenceClient


class HuggingFaceAdapter:
    """
    HuggingFace implementation of HostAdapter.

    Design decisions:
    - No discovery: Models are user-provided (HF has no list endpoint)
    - Token from env: HF_TOKEN env var (Spaces-compatible)
    - Streaming: Uses InferenceClient.chat_completion with stream=True

    Concurrency model:
    - High concurrency (HF manages rate limiting)
    - acquire/release are no-ops
    """

    def __init__(
        self,
        models: list[str],
        token: Optional[str] = None,
    ):
        """
        Initialize adapter with model list.

        Args:
            models: List of HuggingFace model IDs (e.g., "meta-llama/Llama-2-7b-chat-hf")
            token: HuggingFace API token. Falls back to HF_TOKEN env var.

        Raises:
            ValueError: If no token provided and HF_TOKEN not set
        """
        self._token = token or os.environ.get("HF_TOKEN")
        if not self._token:
            raise ValueError(
                "HuggingFace token required. "
                "Provide token parameter or set HF_TOKEN environment variable."
            )

        self._models = list(models)
        self._client = InferenceClient(token=self._token)

    async def get_available_models(self) -> list[str]:
        """
        Return list of user-provided model IDs.

        Note: HuggingFace has no discovery endpoint.
        This returns the models the user configured.
        """
        return list(self._models)

    def add_model(self, model_id: str) -> None:
        """Add a model to the available list."""
        if model_id not in self._models:
            self._models.append(model_id)

    def remove_model(self, model_id: str) -> None:
        """Remove a model from the available list."""
        if model_id in self._models:
            self._models.remove(model_id)

    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tools: Optional[list[dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from HuggingFace Inference API.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
            messages: OpenAI-format messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout_seconds: Request timeout (passed to client)
            tools: Optional tool definitions

        Yields:
            Text chunks as they arrive

        Note:
            HuggingFace uses synchronous iteration but we yield async
            for interface compatibility with HostAdapter protocol.
        """
        kwargs = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools

        # InferenceClient.chat_completion returns sync iterator when stream=True
        # We wrap in async generator for protocol compatibility
        response = self._client.chat_completion(**kwargs)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_concurrency_limit(self) -> int:
        """
        Return high concurrency limit.

        HuggingFace handles rate limiting server-side.
        We allow many concurrent requests.
        """
        return 10

    async def acquire(self, model_id: str) -> None:
        """No-op: HuggingFace manages concurrency server-side."""
        pass

    async def release(self, model_id: str) -> None:
        """No-op: HuggingFace manages concurrency server-side."""
        pass
