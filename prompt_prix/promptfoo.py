"""
promptfoo configuration parser.

Parses promptfoo YAML config files to extract:
- Provider list (model IDs for HF integration)
- Test cases (for Battery import)

Reference: https://www.promptfoo.dev/docs/configuration/guide/
"""

import re
from pathlib import Path
from typing import Optional

import yaml

from prompt_prix.benchmarks.base import TestCase


def _slugify(text: str) -> str:
    """Convert text to safe ID.

    Example: 'Basic Tool Call (Sanity Check)' -> 'basic-tool-call-sanity-check'
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')[:50]  # Cap length for readability


def parse_providers(
    config_path: Path,
    filter_prefix: Optional[str] = "huggingface"
) -> list[str]:
    """
    Extract provider model IDs from promptfoo config.

    Args:
        config_path: Path to promptfooconfig.yaml
        filter_prefix: Only return providers with this prefix (e.g., "huggingface").
                      Set to None to return all providers.

    Returns:
        List of model IDs (without the provider prefix)

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config:
        return []

    providers = config.get("providers", [])
    if not providers:
        return []

    model_ids = []
    for provider in providers:
        # Handle both string and dict formats
        if isinstance(provider, str):
            provider_id = provider
        elif isinstance(provider, dict):
            provider_id = provider.get("id", "")
        else:
            continue

        if not provider_id:
            continue

        # Extract model ID from "provider:model_id" format
        if ":" in provider_id:
            prefix, model_id = provider_id.split(":", 1)
            if filter_prefix is None or prefix == filter_prefix:
                model_ids.append(model_id)
        elif filter_prefix is None:
            # No prefix, include if not filtering
            model_ids.append(provider_id)

    return model_ids


def parse_tests(config_path: Path) -> list[TestCase]:
    """
    Extract test cases from promptfoo config.

    Handles:
    - Simple prompts array
    - Tests with variable substitution
    - Tests with assertions

    Args:
        config_path: Path to promptfooconfig.yaml

    Returns:
        List of TestCase objects

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config:
        return []

    prompts = config.get("prompts", [])
    tests = config.get("tests", [])

    if not prompts:
        return []

    test_cases = []
    seen_ids: set[str] = set()  # Track used IDs to prevent collisions

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
    if not tests:
        for idx, prompt in enumerate(prompts):
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            # Use slugified prompt text as ID, fall back to numbered ID
            base_id = _slugify(prompt_text[:80]) if prompt_text else ""
            test_id = unique_id(base_id, idx + 1)
            test_cases.append(TestCase(
                id=test_id,
                user=prompt_text,
                name=f"Test {idx + 1}"
            ))
        return test_cases

    # Process tests with prompts
    test_idx = 0
    for test in tests:
        test_idx += 1

        # Get variables for substitution
        vars_dict = test.get("vars", {}) if isinstance(test, dict) else {}

        # Process each prompt with this test's vars
        for prompt in prompts:
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)

            # Substitute variables ({{var}} pattern)
            if vars_dict:
                for var_name, var_value in vars_dict.items():
                    prompt_text = prompt_text.replace(
                        f"{{{{{var_name}}}}}",
                        str(var_value)
                    )

            # Extract assertions as pass_criteria
            pass_criteria = None
            if isinstance(test, dict) and "assert" in test:
                assertions = test["assert"]
                criteria_parts = []
                for assertion in assertions:
                    if isinstance(assertion, dict):
                        assert_type = assertion.get("type", "")
                        assert_value = assertion.get("value", "")
                        criteria_parts.append(f"{assert_type}: {assert_value}")
                if criteria_parts:
                    pass_criteria = "; ".join(criteria_parts)

            # Get description as name and use it for ID (#75)
            description = ""
            if isinstance(test, dict):
                description = test.get("description", "")

            # Use slugified description as ID, ensuring uniqueness
            base_id = _slugify(description) if description else ""
            test_id = unique_id(base_id, test_idx)

            test_cases.append(TestCase(
                id=test_id,
                user=prompt_text,
                name=description or f"Test {test_idx}",
                pass_criteria=pass_criteria
            ))

    return test_cases


def parse_config(config_path: Path) -> dict:
    """
    Parse full promptfoo config and return structured data.

    Args:
        config_path: Path to promptfooconfig.yaml

    Returns:
        Dict with 'providers' and 'tests' keys
    """
    return {
        "providers": parse_providers(config_path, filter_prefix=None),
        "huggingface_models": parse_providers(config_path, filter_prefix="huggingface"),
        "tests": parse_tests(config_path)
    }
