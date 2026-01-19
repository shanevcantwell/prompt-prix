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
    - Concurrency limit (get_concurrency_limit)

    Design rationale (from CLAUDE.md):
    - Separation of Concerns: Protocol defines capability, adapters define implementation
    - Provider-Agnostic: Same interface works for LM Studio, Ollama, vLLM, HuggingFace, etc.

    Resource management:
    - stream_completion() handles all resource management internally
    - Callers just call stream_completion() - no acquire/release needed
    - This follows the pattern of database pools and HTTP clients
    - Concurrency limiting is done by callers using get_concurrency_limit() with a semaphore
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

        Resource management is handled internally by the adapter.
        Callers should use get_concurrency_limit() with a semaphore to
        limit concurrent calls, but do not need to manage resources.

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

        Callers should use this with asyncio.Semaphore to limit concurrent
        stream_completion() calls.

        Returns:
            Number of concurrent requests allowed.
            - Local backends: Typically 1 per server/GPU
            - Cloud backends: Higher (e.g., 10+), rate limiting handled externally
        """
        ...

    async def refresh(self) -> None:
        """
        Refresh server state and model availability.

        Call this before get_available_models() or stream_completion()
        if you need fresh server state. Some adapters may call this
        internally, but explicit refresh ensures current state.
        """
        ...

    def get_connection_errors(self) -> list[tuple[str, str]]:
        """
        Return list of servers that failed to connect.

        Returns:
            List of (url, error_message) tuples for servers with errors.
            Empty list if all servers connected successfully.
        """
        ...
