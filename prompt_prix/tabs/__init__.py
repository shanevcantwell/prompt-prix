"""
Tab modules for prompt-prix UI.

Each tab is a self-contained module with its own handlers and UI definition.
This allows for easy addition of new tabs and keeps code organized by feature.
"""

from prompt_prix.tabs import battery, compare

__all__ = ["battery", "compare"]
