"""
Adapters for LLM inference backends.

Provider-agnostic architecture: Protocol defines WHAT, implementations define HOW.
"""

from .base import HostAdapter
from .pooled_local import PooledLocalInferenceAdapter, LocalInferenceError
from .lmstudio import LMStudioAdapter
from .huggingface import HuggingFaceAdapter

__all__ = [
    "HostAdapter",
    "PooledLocalInferenceAdapter",
    "LocalInferenceError",
    "LMStudioAdapter",
    "HuggingFaceAdapter",
]
