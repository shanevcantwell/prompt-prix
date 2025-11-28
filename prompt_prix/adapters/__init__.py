"""
Adapters for LLM inference backends.

Provider-agnostic architecture: Protocol defines WHAT, implementations define HOW.
"""

from .base import HostAdapter
from .lmstudio import LMStudioAdapter

__all__ = ["HostAdapter", "LMStudioAdapter"]
