"""
Global state for the prompt-prix application.

This module holds mutable state that is shared across handlers and UI.
Keeping it separate avoids circular import issues.
"""

from typing import Optional, TYPE_CHECKING
from prompt_prix.core import ServerPool, ComparisonSession

if TYPE_CHECKING:
    from prompt_prix.battery import BatteryRun
    from prompt_prix.adapters.huggingface import HuggingFaceAdapter


# These will be initialized when the app starts
server_pool: Optional[ServerPool] = None
session: Optional[ComparisonSession] = None

# HuggingFace adapter (initialized on first use)
hf_adapter: Optional["HuggingFaceAdapter"] = None

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


def get_hf_adapter() -> Optional["HuggingFaceAdapter"]:
    """Get the HuggingFace adapter, if initialized."""
    return hf_adapter


def init_hf_adapter(token: Optional[str] = None) -> "HuggingFaceAdapter":
    """Initialize or return existing HuggingFace adapter."""
    global hf_adapter
    if hf_adapter is None:
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter
        hf_adapter = HuggingFaceAdapter(models=[], token=token)
    return hf_adapter


def add_hf_model(model_id: str, token: Optional[str] = None) -> None:
    """Add a model to the HuggingFace adapter."""
    adapter = init_hf_adapter(token)
    adapter.add_model(model_id)


def remove_hf_model(model_id: str) -> None:
    """Remove a model from the HuggingFace adapter."""
    if hf_adapter is not None:
        hf_adapter.remove_model(model_id)


def get_hf_models() -> list[str]:
    """Get list of HuggingFace models."""
    if hf_adapter is None:
        return []
    return list(hf_adapter._models)
