"""
Global state for the prompt-prix application.

This module holds mutable state that is shared across handlers and UI.
Keeping it separate avoids circular import issues.
"""

from typing import Optional, TYPE_CHECKING
from prompt_prix.core import ServerPool, ComparisonSession

if TYPE_CHECKING:
    from prompt_prix.battery import BatteryRun


# These will be initialized when the app starts
server_pool: Optional[ServerPool] = None
session: Optional[ComparisonSession] = None

# Battery state - persists after run for detail retrieval
battery_run: Optional["BatteryRun"] = None

# Source filename for export naming
battery_source_file: Optional[str] = None

# Cancellation flag - checked by long-running handlers
stop_requested: bool = False

# Server index mapping for GPU prefix feature
server_index_map: dict[int, str] = {}  # idx → URL
server_hints: dict[str, str] = {}      # model_id → forced URL


def request_stop():
    """Signal that the current operation should stop."""
    global stop_requested
    stop_requested = True


def clear_stop():
    """Clear the stop flag (call at start of new operation)."""
    global stop_requested
    stop_requested = False


def should_stop() -> bool:
    """Check if stop was requested."""
    return stop_requested


def set_server_map(mapping: dict[int, str]) -> None:
    """Set server index → URL mapping (from fetch)."""
    global server_index_map
    server_index_map = mapping


def get_server_url(index: int) -> Optional[str]:
    """Get server URL by index."""
    return server_index_map.get(index)


def set_server_hints(hints: dict[str, str]) -> None:
    """Set model_id → forced server URL hints."""
    global server_hints
    server_hints = hints


def get_server_hint(model_id: str) -> Optional[str]:
    """Get forced server URL for a model, if set."""
    return server_hints.get(model_id)
