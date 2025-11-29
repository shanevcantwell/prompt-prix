"""
Compare tab module.

Provides interactive multi-turn model comparison.
"""

from prompt_prix.tabs.compare.ui import create_tab
from prompt_prix.tabs.compare import handlers

__all__ = ["create_tab", "handlers"]
