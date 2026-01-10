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
        """
        # Find available server (respecting hints for GPU prefix routing)
        server_url = None
        async with self._lock:
            await self._pool.refresh()
            preferred = self._server_hints.get(model_id)
            server_url = self._pool.find_server(model_id, preferred_url=preferred)

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

