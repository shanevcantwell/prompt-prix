"""
Global state for the prompt-prix application.

This module holds mutable state that is shared across handlers and UI.
Keeping it separate avoids circular import issues.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_prix.battery import BatteryRun
    from prompt_prix.consistency import ConsistencyRun
    from prompt_prix.core import ComparisonSession


# Compare tab session state
session: Optional["ComparisonSession"] = None

# Battery state - persists after run for detail retrieval
battery_run: Optional["BatteryRun"] = None

# Consistency run state - for multi-run variance testing
consistency_run: Optional["ConsistencyRun"] = None

# Source filename for export naming
battery_source_file: Optional[str] = None

# Current display mode for battery grid (prevents mode reset on cell click)
battery_display_mode: str = "Symbols (✓/❌)"

# Cancellation flag - checked by long-running handlers
stop_requested: bool = False


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
