"""
CustomJSONLoader - loads test cases from JSON or JSONL format.

Supports two formats:
- JSON: {"prompts": [{...}, {...}]}
- JSONL: One test case per line (auto-detected)

Fail-fast validation per CLAUDE.md: Invalid data is rejected immediately.
"""

import json
from pathlib import Path
from typing import Union

from .base import TestCase


class CustomJSONLoader:
    """
    Load test cases from JSON or JSONL format.

    JSON format:
    {
        "test_suite": "...",
        "version": "...",
        "prompts": [
            {"id": "...", "user": "...", ...},
            ...
        ]
    }

    JSONL format (one test per line):
    {"id": "test-1", "user": "...", ...}
    {"id": "test-2", "user": "...", ...}
    """

    @staticmethod
    def load(file_path: Union[str, Path]) -> list[TestCase]:
        """
        Load test cases from JSON or JSONL file.

        Args:
            file_path: Path to JSON/JSONL file

        Returns:
            List of TestCase objects

        Raises:
            ValueError: If file format is invalid (fail-fast)
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Detect format: JSONL if file starts with { and has multiple lines
        # or has .jsonl extension
        is_jsonl = path.suffix.lower() == ".jsonl"
        if not is_jsonl:
            # Auto-detect: if first non-whitespace char is { and there are
            # multiple JSON objects (one per line), treat as JSONL
            stripped = content.strip()
            if stripped.startswith("{") and "\n{" in stripped:
                is_jsonl = True

        if is_jsonl:
            return CustomJSONLoader._load_jsonl(content, path)
        else:
            return CustomJSONLoader._load_json(content, path)

    @staticmethod
    def _load_json(content: str, path: Path) -> list[TestCase]:
        """Load from JSON format with prompts array."""
        data = json.loads(content)

        # Fail-fast validation
        if not isinstance(data, dict):
            raise ValueError("Benchmark file must contain a JSON object")

        if "prompts" not in data:
            raise ValueError("Benchmark file missing required 'prompts' key")

        prompts = data["prompts"]
        if not isinstance(prompts, list):
            raise ValueError("'prompts' must be an array")

        if len(prompts) == 0:
            raise ValueError("'prompts' array cannot be empty")

        # Parse each prompt into TestCase (Pydantic validates fields)
        test_cases = []
        for i, prompt in enumerate(prompts):
            try:
                test_cases.append(TestCase(**prompt))
            except Exception as e:
                raise ValueError(f"Invalid test case at index {i}: {e}") from e

        return test_cases

    @staticmethod
    def _load_jsonl(content: str, path: Path) -> list[TestCase]:
        """Load from JSONL format (one test case per line)."""
        test_cases = []
        lines = content.strip().split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                data = json.loads(line)
                test_cases.append(TestCase(**data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i + 1}: {e}") from e
            except Exception as e:
                raise ValueError(f"Invalid test case on line {i + 1}: {e}") from e

        if len(test_cases) == 0:
            raise ValueError("JSONL file contains no valid test cases")

        return test_cases

    @staticmethod
    def validate(file_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Pre-validate benchmark file before enabling Run button.

        Args:
            file_path: Path to JSON file

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            cases = CustomJSONLoader.load(file_path)
            categories = set(c.category for c in cases if c.category)
            return True, f"✅ Valid: {len(cases)} tests in {len(categories)} categories"
        except FileNotFoundError as e:
            return False, f"❌ File not found: {e}"
        except json.JSONDecodeError as e:
            return False, f"❌ Invalid JSON: {e}"
        except ValueError as e:
            return False, f"❌ Validation failed: {e}"
        except Exception as e:
            return False, f"❌ Unexpected error: {e}"
