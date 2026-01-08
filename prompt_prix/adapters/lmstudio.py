"""
LMStudioAdapter - wraps existing ServerPool for HostAdapter interface.

Implementation detail: This adapter does NOT modify core.py.
It wraps the existing ServerPool and stream_completion functions.
"""

import asyncio
from typing import AsyncGenerator, Optional

from prompt_prix.scheduler import ServerPool
from prompt_prix.core import stream_completion


class LMStudioAdapter:
    """
    LM Studio implementation of HostAdapter.

    Wraps ServerPool for server discovery and stream_completion for inference.
    Uses asyncio.Lock for thread-safety (state hygiene per CLAUDE.md).
    """

    def __init__(self, server_pool: ServerPool):
        """
        Initialize adapter with existing ServerPool.

        Args:
            server_pool: Pre-configured ServerPool instance
        """
        self._pool = server_pool
        self._lock = asyncio.Lock()

    @property
    def pool(self) -> ServerPool:
        """Expose ServerPool for BatchRunner access."""
        return self._pool

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
        """
        # Find available server
        server_url = None
        async with self._lock:
            await self._pool.refresh()
            server_url = self._pool.find_server(model_id)

        if server_url is None:
            raise RuntimeError(f"No server available for model: {model_id}")

        # Acquire and stream
        await self._pool.acquire(server_url)
        try:
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                tools=tools
            ):
                yield chunk
        finally:
            self._pool.release(server_url)
