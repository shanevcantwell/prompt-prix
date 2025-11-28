"""
LMStudioAdapter - wraps existing ServerPool for HostAdapter interface.

Implementation detail: This adapter does NOT modify core.py.
It wraps the existing ServerPool and stream_completion functions.
"""

import asyncio
from typing import AsyncGenerator, Optional

from prompt_prix.core import ServerPool, stream_completion


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

    async def get_available_models(self) -> list[str]:
        """Return list of all models available across all servers."""
        async with self._lock:
            await self._pool.refresh_all_manifests()
            return list(self._pool.get_all_available_models())

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
            await self._pool.refresh_all_manifests()
            server_url = self._pool.find_available_server(model_id)

        if server_url is None:
            raise RuntimeError(f"No server available for model: {model_id}")

        # Acquire and stream
        await self._pool.acquire_server(server_url)
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
            self._pool.release_server(server_url)
