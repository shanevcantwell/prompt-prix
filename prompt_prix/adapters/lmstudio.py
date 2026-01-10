"""
LMStudioAdapter - wraps existing ServerPool for HostAdapter interface.

Implementation detail: This adapter does NOT modify core.py.
It wraps the existing ServerPool and stream_completion functions.
"""

import asyncio
from typing import AsyncGenerator, Optional

from prompt_prix.scheduler import ServerPool
from prompt_prix.core import stream_completion


def _strip_gpu_prefix(model_id: str) -> str:
    """Strip GPU prefix for API calls.

    Converts "0: llama-3.2" -> "llama-3.2" for server/API compatibility.
    Prefixed IDs are used internally for uniqueness (same model on multiple GPUs),
    but server APIs expect the raw model name.
    """
    if ': ' in model_id:
        prefix, rest = model_id.split(': ', 1)
        if prefix.isdigit():
            return rest
    return model_id


class LMStudioAdapter:
    """
    LM Studio implementation of HostAdapter.

    Wraps ServerPool for server discovery and stream_completion for inference.
    Uses asyncio.Lock for thread-safety (state hygiene per CLAUDE.md).

    Resource management:
    - stream_completion() handles server acquire/release internally
    - One concurrent request per server (GPU memory constraint)
    - Server hints allow GPU prefix routing (prefer specific server for a model)
    """

    def __init__(self, server_pool: ServerPool):
        """
        Initialize adapter with existing ServerPool.

        Args:
            server_pool: Pre-configured ServerPool instance
        """
        self._pool = server_pool
        self._lock = asyncio.Lock()
        self._server_hints: dict[str, str] = {}  # model_id → preferred server_url

    def set_server_hint(self, model_id: str, server_url: str) -> None:
        """Set preferred server for a model (for GPU prefix routing)."""
        self._server_hints[model_id] = server_url

    def get_concurrency_limit(self) -> int:
        """Return number of servers (one request per server)."""
        return max(1, len(self._pool.servers))

    async def get_available_models(self) -> list[str]:
        """Return list of all models available across all servers."""
        async with self._lock:
            await self._pool.refresh()
            return list(self._pool.get_available_models())

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
        Stream completion from LM Studio server.

        Finds an available server with the model, acquires it,
        streams the completion, then releases the server.

        Args:
            model_id: Model identifier, may be prefixed (e.g., "0: llama-3.2")
                     Prefix is used for hint lookup, stripped for API calls.
        """
        # Strip GPU prefix for server lookup and API calls
        # Keep prefixed ID for hint lookup (same model on different GPUs)
        api_model_id = _strip_gpu_prefix(model_id)

        # Find available server (respecting hints for GPU prefix routing)
        server_url = None
        async with self._lock:
            await self._pool.refresh()
            # Use full prefixed ID for hint lookup (uniqueness)
            preferred = self._server_hints.get(model_id)
            # Use stripped ID for server lookup (servers know "llama" not "0: llama")
            server_url = self._pool.find_server(api_model_id, preferred_url=preferred)

        if server_url is None:
            raise RuntimeError(f"No server available for model: {model_id}")

        # Acquire and stream
        await self._pool.acquire(server_url)
        try:
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=api_model_id,  # Use stripped ID for API
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                tools=tools
            ):
                yield chunk
        finally:
            self._pool.release(server_url)

