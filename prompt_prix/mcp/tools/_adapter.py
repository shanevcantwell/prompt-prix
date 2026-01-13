"""
Adapter management for MCP tools.

Handles lazy initialization of LMStudioAdapter from environment.
Singleton pattern for MCP server lifetime.
"""

import logging
from typing import Optional

from prompt_prix.adapters import LMStudioAdapter
from prompt_prix.scheduler import ServerPool
from prompt_prix.config import load_servers_from_env

logger = logging.getLogger(__name__)

_adapter: Optional[LMStudioAdapter] = None
_pool: Optional[ServerPool] = None


async def get_adapter() -> LMStudioAdapter:
    """
    Get or create LMStudioAdapter.

    Lazy initialization from LM_STUDIO_SERVER_* env vars.
    Adapter is singleton for MCP server lifetime.

    Returns:
        LMStudioAdapter instance

    Raises:
        RuntimeError: If no LM Studio servers configured
    """
    global _adapter, _pool

    if _adapter is None:
        servers = load_servers_from_env()
        if not servers:
            raise RuntimeError(
                "No LM Studio servers configured. "
                "Set LM_STUDIO_SERVER_1, LM_STUDIO_SERVER_2, etc. in environment."
            )

        logger.info(f"Initializing MCP adapter with {len(servers)} server(s)")
        _pool = ServerPool(servers)
        await _pool.refresh()
        _adapter = LMStudioAdapter(_pool)

    return _adapter


async def get_available_models() -> list[str]:
    """
    Get list of available models from configured servers.

    Returns:
        List of model IDs
    """
    adapter = await get_adapter()
    return await adapter.get_available_models()
