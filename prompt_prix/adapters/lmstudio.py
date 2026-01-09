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

    Concurrency model:
    - One concurrent request per server (GPU memory constraint)
    - acquire/release manage server slots
    - _active_servers tracks which server is handling each model
    """

    def __init__(self, server_pool: ServerPool):
        """
        Initialize adapter with existing ServerPool.

        Args:
            server_pool: Pre-configured ServerPool instance
        """
        self._pool = server_pool
        self._lock = asyncio.Lock()
        self._active_servers: dict[str, str] = {}  # model_id → server_url
        self._server_hints: dict[str, str] = {}    # model_id → preferred server_url

    @property
    def pool(self) -> ServerPool:
        """Expose ServerPool for BatchRunner access (deprecated - use acquire/release)."""
        return self._pool

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

    async def acquire(self, model_id: str) -> None:
        """
        Acquire a server slot for this model.

        Finds an available server (respecting hints), acquires it,
        and tracks it for later release.

        Args:
            model_id: The model that will be used

        Raises:
            RuntimeError: If no server is available for this model
        """
        async with self._lock:
            await self._pool.refresh()
            preferred = self._server_hints.get(model_id)
            server_url = self._pool.find_server(model_id, preferred_url=preferred)

        if server_url is None:
            raise RuntimeError(f"No server available for model: {model_id}")

        await self._pool.acquire(server_url)
        self._active_servers[model_id] = server_url

    async def release(self, model_id: str) -> None:
        """
        Release the server slot for this model.

        Args:
            model_id: The model to release
        """
        server_url = self._active_servers.pop(model_id, None)
        if server_url:
            self._pool.release(server_url)

    def get_active_server(self, model_id: str) -> Optional[str]:
        """Get the server URL currently handling this model (after acquire)."""
        return self._active_servers.get(model_id)
