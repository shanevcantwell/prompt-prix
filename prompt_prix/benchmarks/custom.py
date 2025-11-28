"""
CustomJSONLoader - loads test cases from tool_competence_tests.json format.

Fail-fast validation per CLAUDE.md: Invalid JSON is rejected immediately.
"""

import json
from pathlib import Path
from typing import Union

from .base import TestCase


class CustomJSONLoader:
    """
    Load test cases from custom JSON format.

    Expected format:
    {
        "test_suite": "...",
        "version": "...",
        "prompts": [
            {"id": "...", "user": "...", ...},
            ...
        ]
    }
    """

    @staticmethod
    def load(file_path: Union[str, Path]) -> list[TestCase]:
        """
        Load test cases from JSON file.

        Args:
            file_path: Path to JSON file

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
            data = json.load(f)

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
