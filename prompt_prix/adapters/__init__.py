"""
Adapters for LLM inference backends.

Provider-agnostic architecture: Protocol defines WHAT, implementations define HOW.
"""

from .base import HostAdapter
from .huggingface import HuggingFaceAdapter

# LMStudioAdapter requires local-inference-pool (not available on HF Spaces)
try:
    from .lmstudio import LMStudioAdapter
except ImportError:
    LMStudioAdapter = None  # type: ignore[assignment,misc]

__all__ = ["HostAdapter", "LMStudioAdapter", "HuggingFaceAdapter"]
