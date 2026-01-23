"""MCP tool: list_models - discover available models from configured servers.

Adapter is retrieved via registry - callers don't manage servers.

Usage:
    result = await list_models()
    print(result["models"])  # ["model-a", "model-b", ...]
"""

from prompt_prix.mcp.registry import get_adapter


async def list_models() -> dict:
    """
    List available models from configured LM Studio servers.

    This is an MCP primitive - a self-contained operation that can be called
    by Gradio UI, CLI, or agentic systems.

    The adapter is retrieved from the registry and contains all server
    configuration. Callers don't need to know about server URLs.

    Returns:
        {
            "models": ["model-a", "model-b", ...],  # Deduplicated, sorted
            "servers": {
                "http://...": ["model-a", "model-b"],
                "http://...": ["model-c"],
            },
            "unreachable": ["http://..."],  # Servers that returned no models
        }

    Raises:
        RuntimeError: If no adapter has been registered
    """
    adapter = get_adapter()

    # Fetch models (triggers HTTP calls to all servers)
    models = await adapter.get_available_models()

    return {
        "models": sorted(models),
        "servers": adapter.get_models_by_server(),
        "unreachable": adapter.get_unreachable_servers(),
    }
