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
