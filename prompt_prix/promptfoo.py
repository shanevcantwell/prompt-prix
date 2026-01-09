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

    # If no tests defined, create one test per prompt
    if not tests:
        for idx, prompt in enumerate(prompts):
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            test_cases.append(TestCase(
                id=f"promptfoo_{idx + 1}",
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

            # Get description as name
            name = ""
            if isinstance(test, dict):
                name = test.get("description", "")

            test_cases.append(TestCase(
                id=f"promptfoo_{test_idx}",
                user=prompt_text,
                name=name or f"Test {test_idx}",
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
