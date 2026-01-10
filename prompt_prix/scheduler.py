"""
Server pool management for LM Studio servers.

Manages multiple servers with manifest and load state tracking.
- Server affinity: prefer servers where model is already loaded
- Explicit state: manifest vs loaded is not conflated
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    """
    State of a single LM Studio server.

    Separates what CAN run (manifest) from what IS running (loaded).
    LM Studio supports loading multiple models into VRAM simultaneously.
    """
    url: str
    manifest_models: list[str] = field(default_factory=list)  # From /v1/models
    loaded_models: list[str] = field(default_factory=list)  # From /api/v0/models (currently in VRAM)
    is_busy: bool = False


class ServerPool:
    """
    Manages multiple LM Studio servers with explicit load state tracking.

    Queries both the OpenAI-compatible manifest endpoint and LM Studio's
    native API for load state.
    """

    def __init__(self, server_urls: list[str]):
        self.servers: dict[str, ServerState] = {
            url: ServerState(url=url) for url in server_urls
        }
        self._locks: dict[str, asyncio.Lock] = {
            url: asyncio.Lock() for url in server_urls
        }

    async def refresh(self) -> None:
        """Refresh both manifest and load state for all servers."""
        tasks = [self._refresh_server(url) for url in self.servers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh_server(self, server_url: str) -> None:
        """Refresh manifest and load state for a single server."""
        import httpx

        server = self.servers[server_url]

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get manifest (OpenAI-compatible endpoint)
            try:
                resp = await client.get(f"{server_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    server.manifest_models = [m["id"] for m in data.get("data", [])]
            except Exception as e:
                logger.debug(f"Failed to get manifest from {server_url}: {e}")
                server.manifest_models = []

            # Get load state (LM Studio native API)
            try:
                resp = await client.get(f"{server_url}/api/v0/models")
                if resp.status_code == 200:
                    data = resp.json()
                    # Collect ALL loaded models (LM Studio can load multiple into VRAM)
                    loaded = [
                        model.get("id")
                        for model in data.get("data", [])
                        if model.get("state") == "loaded" and model.get("id")
                    ]
                    server.loaded_models = loaded
            except Exception as e:
                logger.debug(f"Failed to get load state from {server_url}: {e}")
                server.loaded_models = []

    def find_server(
        self,
        model_id: str,
        require_loaded: bool = False,
        preferred_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Find best server for a model.

        Priority:
        1. Preferred URL if specified and model available there
        2. Idle server with model already loaded (no swap needed)
        3. Idle server with model in manifest (will JIT load)

        Args:
            model_id: Model to find server for
            require_loaded: If True, only return servers where model is loaded
            preferred_url: If set, force this server (for GPU prefix routing)

        Returns:
            Server URL or None if no suitable server available
        """
        # If preferred URL specified, use it if model is available there
        if preferred_url and preferred_url in self.servers:
            server = self.servers[preferred_url]
            if not server.is_busy:
                if model_id in server.loaded_models or model_id in server.manifest_models:
                    return preferred_url

        # First pass: prefer server where model is already loaded
        for url, server in self.servers.items():
            if server.is_busy:
                continue
            if model_id in server.loaded_models:
                return url

        # Second pass: any server with model in manifest (unless require_loaded)
        if not require_loaded:
            for url, server in self.servers.items():
                if server.is_busy:
                    continue
                if model_id in server.manifest_models:
                    return url

        return None

    def get_available_models(self, only_loaded: bool = False) -> set[str]:
        """
        Get all models that can be run.

        Args:
            only_loaded: If True, only return models currently loaded

        Returns:
            Set of model IDs
        """
        if only_loaded:
            result = set()
            for server in self.servers.values():
                result.update(server.loaded_models)
            return result
        else:
            result = set()
            for server in self.servers.values():
                result.update(server.manifest_models)
            return result

    async def acquire(self, server_url: str) -> None:
        """Mark server as busy."""
        await self._locks[server_url].acquire()
        self.servers[server_url].is_busy = True

    def release(self, server_url: str) -> None:
        """Mark server as available."""
        self.servers[server_url].is_busy = False
        try:
            self._locks[server_url].release()
        except RuntimeError:
            pass  # Lock wasn't held
