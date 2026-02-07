"""
Drift calculation tool — semantic distance via embedding cosine distance.

Wraps semantic-chunker's calculate_drift for inline use during battery
execution. Direct import (not MCP subprocess) for ~50ms performance.

Env config (read by semantic-chunker's StateManager):
    EMBEDDING_SERVER_URL: LM Studio API URL (default: http://localhost:1234/v1)
    EMBEDDING_MODEL: Model name (e.g., embeddinggemma:300m)
    EMBEDDING_BACKEND: "lmstudio" (default)
"""

import logging

logger = logging.getLogger(__name__)

_manager = None


def _get_manager():
    """Lazy-init semantic-chunker StateManager singleton."""
    global _manager
    if _manager is None:
        from semantic_chunker.mcp.state_manager import StateManager
        _manager = StateManager()
    return _manager


async def calculate_drift(text_a: str, text_b: str) -> float:
    """
    Calculate cosine distance between two texts via embedding.

    Returns:
        Float 0.0 (identical) to 1.0 (orthogonal) to 2.0 (opposite).

    Raises:
        ImportError: If semantic-chunker is not installed.
        Exception: If embedding server is unreachable.
    """
    from semantic_chunker.mcp.commands.embeddings import calculate_drift as _calculate_drift
    result = await _calculate_drift(_get_manager(), {"text_a": text_a, "text_b": text_b})
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["drift"]
