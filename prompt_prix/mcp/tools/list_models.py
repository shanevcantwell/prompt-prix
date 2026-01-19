"""MCP tool: list_models - discover available models from configured servers.

This is the first MCP primitive in prompt-prix, establishing the pattern
for future tools (complete, judge).

Usage:
    result = await list_models(["http://192.168.1.10:1234", "http://192.168.1.11:1234"])
    print(result["models"])  # ["model-a", "model-b", ...]
"""

from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.core import ServerPool


async def list_models(server_urls: list[str]) -> dict:
    """
    List available models from configured LM Studio servers.

    This is an MCP primitive - a self-contained operation that can be called
    by Gradio UI, CLI, or agentic systems.

    Args:
        server_urls: List of OpenAI-compatible server URLs
            e.g., ["http://192.168.1.10:1234", "http://localhost:1234"]

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
        No exceptions - errors are captured in the response structure.
    """
    if not server_urls:
        return {
            "models": [],
            "servers": {},
            "unreachable": [],
        }

    # Create adapter stack: URLs -> ServerPool -> LMStudioAdapter
    pool = ServerPool(server_urls)
    adapter = LMStudioAdapter(pool)

    # Fetch models (triggers HTTP calls to all servers)
    models = await adapter.get_available_models()

    return {
        "models": sorted(models),
        "servers": adapter.get_models_by_server(),
        "unreachable": adapter.get_unreachable_servers(),
    }
