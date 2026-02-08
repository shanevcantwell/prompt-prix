"""MCP tool: calculate_drift — semantic distance via embedding cosine distance.

Wraps semantic-chunker's calculate_drift for inline use during battery
execution. Direct import (not MCP subprocess) for ~50ms performance.

Shared StateManager and importability check via _semantic_chunker module.
"""

from prompt_prix.mcp.tools._semantic_chunker import ensure_importable, get_manager


async def calculate_drift(text_a: str, text_b: str) -> float:
    """
    Calculate cosine distance between two texts via embedding.

    Returns:
        Float 0.0 (identical) to 1.0 (orthogonal) to 2.0 (opposite).

    Raises:
        ImportError: If semantic-chunker is not available.
        RuntimeError: If embedding server returns an error.
    """
    if not ensure_importable():
        raise ImportError("semantic-chunker is not available")

    from semantic_chunker.mcp.commands.embeddings import calculate_drift as _calculate_drift
    result = await _calculate_drift(get_manager(), {"text_a": text_a, "text_b": text_b})
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["drift"]
