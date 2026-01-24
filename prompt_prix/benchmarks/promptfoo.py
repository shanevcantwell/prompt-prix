"""
PromptfooLoader - loads test cases from promptfoo YAML config files.

Parses promptfoo YAML format and transforms to BenchmarkCase objects.
Includes Pydantic schema validation for safe public deployment.

Reference: https://www.promptfoo.dev/docs/configuration/guide/
"""

import re
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

from .base import BenchmarkCase


# --- Pydantic models for input validation ---

class PromptfooAssertion(BaseModel):
    """Single assertion in a promptfoo test."""
    type: str
    value: Optional[str] = None


class PromptfooTest(BaseModel):
    """Single test case in promptfoo config."""
    description: Optional[str] = None
    vars: Optional[dict[str, str]] = None
    assert_: Optional[list[PromptfooAssertion]] = Field(None, alias="assert")
    options: Optional[dict] = None
    metadata: Optional[dict] = None

    model_config = {"populate_by_name": True}


class PromptfooConfig(BaseModel):
    """Top-level promptfoo configuration file."""
    prompts: list[str]
    tests: Optional[list[PromptfooTest]] = None
    providers: Optional[list] = None  # ignored, just validated
    description: Optional[str] = None


# --- Helper functions ---

def _slugify(text: str) -> str:
    """
    Convert text to safe ID.

    Example: 'Basic Tool Call (Sanity Check)' -> 'basic-tool-call-sanity-check'
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')[:50]  # Cap length for readability


def _substitute_vars(template: str, vars_dict: dict[str, str]) -> str:
    """
    Substitute {{var}} placeholders in template string.

    Args:
        template: String with {{var}} placeholders
        vars_dict: Variable name -> value mapping

    Returns:
        String with placeholders replaced
    """
    result = template
    for var_name, var_value in vars_dict.items():
        result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
    return result


def _assertions_to_criteria(assertions: list[PromptfooAssertion]) -> Optional[str]:
    """
    Convert promptfoo assertions to pass_criteria string.

    Args:
        assertions: List of PromptfooAssertion objects

    Returns:
        Semicolon-separated criteria string, or None if empty
    """
    if not assertions:
        return None

    criteria_parts = []
    for assertion in assertions:
        if assertion.value:
            criteria_parts.append(f"{assertion.type}: {assertion.value}")
        else:
            criteria_parts.append(assertion.type)

    return "; ".join(criteria_parts) if criteria_parts else None


# --- Main loader class ---

class PromptfooLoader:
    """
    Load test cases from promptfoo YAML config files.

    Promptfoo format example:
    ```yaml
    prompts:
      - "{{system}}\n\nUser: {{user}}"

    tests:
      - description: "Basic test"
        vars:
          system: "You are helpful."
          user: "What is 2+2?"
        assert:
          - type: contains
            value: "4"
    ```
    """

    @staticmethod
    def load(file_path: Union[str, Path]) -> list[BenchmarkCase]:
        """
        Load test cases from promptfoo YAML file.

        Args:
            file_path: Path to promptfoo YAML config

        Returns:
            List of BenchmarkCase objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is malformed or fails validation
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Promptfoo config not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            try:
                raw_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML syntax: {e}") from e

        if not raw_config:
            raise ValueError("Empty YAML file")

        # Validate with Pydantic
        try:
            config = PromptfooConfig(**raw_config)
        except ValidationError as e:
            # Extract first error for cleaner message
            first_error = e.errors()[0]
            field = ".".join(str(loc) for loc in first_error["loc"])
            msg = first_error["msg"]
            raise ValueError(f"Invalid promptfoo config: {field} - {msg}") from e

        return PromptfooLoader._transform_to_cases(config)

    @staticmethod
    def _transform_to_cases(config: PromptfooConfig) -> list[BenchmarkCase]:
        """
        Transform validated config to BenchmarkCase objects.

        Handles:
        - Simple prompts (no tests array)
        - Tests with variable substitution
        - Tests with assertions -> pass_criteria
        """
        test_cases = []
        seen_ids: set[str] = set()

        def unique_id(base_id: str, fallback_idx: int) -> str:
            """Ensure ID is unique by appending suffix if needed."""
            if not base_id:
                base_id = f"promptfoo_{fallback_idx}"
            if base_id not in seen_ids:
                seen_ids.add(base_id)
                return base_id
            # Append numeric suffix for duplicates
            suffix = 2
            while f"{base_id}-{suffix}" in seen_ids:
                suffix += 1
            unique = f"{base_id}-{suffix}"
            seen_ids.add(unique)
            return unique

        # If no tests defined, create one test per prompt
        if not config.tests:
            for idx, prompt in enumerate(config.prompts):
                base_id = _slugify(prompt[:80]) if prompt else ""
                test_id = unique_id(base_id, idx + 1)
                test_cases.append(BenchmarkCase(
                    id=test_id,
                    user=prompt,
                    name=f"Test {idx + 1}"
                ))
            return test_cases

        # Process tests with prompts
        test_idx = 0
        for test in config.tests:
            test_idx += 1
            vars_dict = test.vars or {}

            # Process each prompt with this test's vars
            for prompt in config.prompts:
                prompt_text = prompt

                # Substitute variables
                if vars_dict:
                    prompt_text = _substitute_vars(prompt_text, vars_dict)

                # Extract pass_criteria from assertions
                pass_criteria = None
                if test.assert_:
                    pass_criteria = _assertions_to_criteria(test.assert_)

                # Get description and metadata for ID
                description = test.description or ""
                metadata_id = ""
                if test.metadata and isinstance(test.metadata, dict):
                    metadata_id = test.metadata.get("test_id", "")

                # Use metadata.test_id if available, else slugified description
                base_id = metadata_id or (_slugify(description) if description else "")
                test_id = unique_id(base_id, test_idx)

                # Extract user message from vars if present
                user_content = vars_dict.get("user", prompt_text)
                system_content = vars_dict.get("system", "You are a helpful assistant.")

                test_cases.append(BenchmarkCase(
                    id=test_id,
                    user=user_content,
                    system=system_content,
                    name=description or f"Test {test_idx}",
                    pass_criteria=pass_criteria
                ))

        return test_cases

    @staticmethod
    def validate(file_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Pre-validate promptfoo YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            cases = PromptfooLoader.load(file_path)
            categories = set(c.category for c in cases if c.category)
            cat_info = f" in {len(categories)} categories" if categories else ""
            return True, f"✅ Valid promptfoo config: {len(cases)} tests{cat_info}"
        except FileNotFoundError as e:
            return False, f"❌ File not found: {e}"
        except ValueError as e:
            return False, f"❌ {e}"
        except Exception as e:
            return False, f"❌ Unexpected error: {e}"
