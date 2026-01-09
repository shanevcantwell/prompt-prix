"""
Benchmark loaders and test case models.

Supports loading test cases from various formats:
- CustomJSONLoader: For JSON/JSONL format
- Future: promptfoo YAML, Inspect AI formats
"""

from .base import TestCase
from .custom import CustomJSONLoader

__all__ = ["TestCase", "CustomJSONLoader"]
