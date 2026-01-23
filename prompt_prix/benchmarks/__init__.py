"""
Benchmark loaders and test case models.

Supports loading test cases from various formats:
- CustomJSONLoader: For tool_competence_tests.json format
- Future: BFCL, Inspect AI formats
"""

from .base import BenchmarkCase
from .custom import CustomJSONLoader

__all__ = ["BenchmarkCase", "CustomJSONLoader"]
