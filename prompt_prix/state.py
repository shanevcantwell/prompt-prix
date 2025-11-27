"""
Global state for the prompt-prix application.

This module holds mutable state that is shared across handlers and UI.
Keeping it separate avoids circular import issues.
"""

from typing import Optional
from prompt_prix.core import ServerPool, ComparisonSession


# These will be initialized when the app starts
server_pool: Optional[ServerPool] = None
session: Optional[ComparisonSession] = None
