"""
Battery tab module.

Provides benchmark test suite execution across multiple models.
"""

from prompt_prix.tabs.battery.ui import create_tab
from prompt_prix.tabs.battery import handlers

__all__ = ["create_tab", "handlers"]
