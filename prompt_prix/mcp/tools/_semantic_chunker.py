"""Internal: shared lazy-init for semantic-chunker StateManager.

Used by drift.py, geometry.py, trajectory.py. Not exported in __init__.py.

Falls back to sibling repo path if semantic-chunker isn't pip-installed.
Logs once on unavailability rather than per-call.

Env config (read by semantic-chunker's StateManager):
    EMBEDDING_SERVER_URL: LM Studio API URL (default: http://localhost:1234/v1)
    EMBEDDING_MODEL: Model name (e.g., embeddinggemma:300m)
    EMBEDDING_BACKEND: "lmstudio" (default)
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

_manager = None
_available: bool | None = None  # None = not checked yet

# Sibling repo path for local development (no pip install needed)
_SIBLING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "semantic-chunker"
)


def ensure_importable() -> bool:
    """Check (once) whether semantic-chunker is importable.

    Tries pip-installed package first, then sibling repo path.
    Logs a single info message if unavailable.
    """
    global _available
    if _available is not None:
        return _available

    try:
        import semantic_chunker  # noqa: F401
        _available = True
        return True
    except ImportError:
        pass

    # Try sibling repo path
    sibling = os.path.normpath(_SIBLING_PATH)
    if os.path.isdir(os.path.join(sibling, "semantic_chunker")):
        sys.path.insert(0, sibling)
        try:
            import semantic_chunker  # noqa: F401
            _available = True
            logger.info("semantic-chunker loaded from sibling repo: %s", sibling)
            return True
        except ImportError:
            pass

    _available = False
    logger.info("semantic-chunker not found — tools unavailable")
    return False


def get_manager():
    """Lazy-init semantic-chunker StateManager singleton."""
    global _manager
    if _manager is None:
        from semantic_chunker.mcp.state_manager import StateManager
        _manager = StateManager()
    return _manager
