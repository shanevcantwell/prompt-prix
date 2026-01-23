"""
MCP Adapter Registry - Central point for adapter dependency injection.

Per ADR-006, orchestration and MCP tools should not instantiate adapters directly.
Instead, adapters are registered at startup and retrieved via this registry.

Usage:
    # At startup (main.py)
    from prompt_prix.adapters.lmstudio import LMStudioAdapter
    from prompt_prix.mcp.registry import register_adapter

    adapter = LMStudioAdapter(server_urls=load_servers_from_env())
    register_adapter(adapter)

    # In MCP tools
    from prompt_prix.mcp.registry import get_adapter

    adapter = get_adapter()
    async for chunk in adapter.stream_completion(...):
        yield chunk
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_prix.adapters.base import HostAdapter

_adapter: Optional["HostAdapter"] = None


def register_adapter(adapter: "HostAdapter") -> None:
    """
    Register the active adapter instance.

    Called once at application startup. The adapter should be fully
    initialized with server URLs or other configuration.

    Args:
        adapter: A configured HostAdapter implementation
    """
    global _adapter
    _adapter = adapter


def get_adapter() -> "HostAdapter":
    """
    Get the registered adapter.

    Raises:
        RuntimeError: If no adapter has been registered

    Returns:
        The registered HostAdapter instance
    """
    if _adapter is None:
        raise RuntimeError(
            "No adapter registered. Call register_adapter() at startup."
        )
    return _adapter


def clear_adapter() -> None:
    """
    Clear the registered adapter.

    Primarily useful for testing to reset state between tests.
    """
    global _adapter
    _adapter = None
