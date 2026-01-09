"""
HostAdapter Protocol - defines the contract for LLM inference backends.

This is the WHAT (interface), not the HOW (implementation).
See lmstudio.py for concrete implementation.
"""

from typing import Protocol, AsyncGenerator, Optional


class HostAdapter(Protocol):
    """
    Contract for LLM inference backends.

    Implementations must provide:
    - Model discovery (get_available_models)
    - Streaming completion (stream_completion)
    - Concurrency management (get_concurrency_limit, acquire, release)

    Design rationale (from CLAUDE.md):
    - Separation of Concerns: Protocol defines capability, adapters define implementation
    - Provider-Agnostic: Same interface works for LM Studio, Ollama, vLLM, HuggingFace, etc.

    Concurrency model:
    - Local backends (LM Studio): Limited by GPU/server count, requires acquire/release
    - Cloud backends (HuggingFace): High concurrency, acquire/release are no-ops
    """

    async def get_available_models(self) -> list[str]:
        """
        Return list of model IDs available on this host.

        Returns:
            List of model identifiers (e.g., ["llama-3.2-3b", "qwen2.5-7b"])
        """
        ...

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
        Stream completion chunks from the model.

        Args:
            model_id: Model identifier
            messages: OpenAI-format messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout_seconds: Request timeout
            tools: Optional OpenAI-format tool definitions

        Yields:
            Text chunks as they arrive from the model

        Raises:
            Exception on model error (fail loudly per CLAUDE.md)
        """
        ...

    def get_concurrency_limit(self) -> int:
        """
        Maximum concurrent requests this adapter can handle.

        Returns:
            Number of concurrent requests allowed.
            - Local backends: Typically 1 per server/GPU
            - Cloud backends: Higher (e.g., 10+), rate limiting handled externally
        """
        ...

    async def acquire(self, model_id: str) -> None:
        """
        Acquire a slot before making a request.

        For local backends: Blocks until a server is available, handles model loading.
        For cloud backends: No-op (cloud handles concurrency).

        Args:
            model_id: The model that will be used for the request
        """
        ...

    async def release(self, model_id: str) -> None:
        """
        Release a slot after completing a request.

        For local backends: Frees the server for other requests.
        For cloud backends: No-op.

        Args:
            model_id: The model that was used for the request
        """
        ...
