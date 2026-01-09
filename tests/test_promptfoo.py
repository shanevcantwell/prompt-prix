"""Tests for promptfoo config parser."""

import tempfile
from pathlib import Path

import pytest
import yaml

from prompt_prix.promptfoo import parse_providers, parse_tests


class TestParseProviders:
    """Tests for parse_providers function."""

    def test_parses_string_provider_ids(self, tmp_path):
        """Parse simple string provider format."""
        config = {
            "providers": [
                "huggingface:meta-llama/Llama-3.1-8B-Instruct",
                "huggingface:mistralai/Mistral-7B-Instruct-v0.3",
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_providers(config_file)

        assert result == [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]

    def test_parses_dict_provider_format(self, tmp_path):
        """Parse provider with id key and config."""
        config = {
            "providers": [
                {
                    "id": "huggingface:Qwen/Qwen2.5-7B-Instruct",
                    "config": {"temperature": 0.7}
                },
                "huggingface:meta-llama/Llama-3.1-8B-Instruct",
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_providers(config_file)

        assert "Qwen/Qwen2.5-7B-Instruct" in result
        assert "meta-llama/Llama-3.1-8B-Instruct" in result

    def test_filters_only_huggingface_providers(self, tmp_path):
        """Only extract huggingface: prefixed providers."""
        config = {
            "providers": [
                "openai:gpt-4",
                "huggingface:meta-llama/Llama-3.1-8B-Instruct",
                "anthropic:claude-3-opus",
                "huggingface:mistralai/Mistral-7B-Instruct-v0.3",
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_providers(config_file)

        assert len(result) == 2
        assert "meta-llama/Llama-3.1-8B-Instruct" in result
        assert "mistralai/Mistral-7B-Instruct-v0.3" in result
        assert "gpt-4" not in result

    def test_handles_missing_providers_key(self, tmp_path):
        """Return empty list when providers key missing."""
        config = {"prompts": ["Hello"]}
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_providers(config_file)

        assert result == []

    def test_handles_empty_providers(self, tmp_path):
        """Return empty list when providers is empty."""
        config = {"providers": []}
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_providers(config_file)

        assert result == []

    def test_handles_malformed_yaml(self, tmp_path):
        """Raise error for invalid YAML."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("providers: [invalid yaml {")

        with pytest.raises(yaml.YAMLError):
            parse_providers(config_file)

    def test_handles_missing_file(self, tmp_path):
        """Raise error for missing file."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            parse_providers(config_file)

    def test_extracts_all_provider_types(self, tmp_path):
        """Return dict with all provider types when requested."""
        config = {
            "providers": [
                "openai:gpt-4",
                "huggingface:meta-llama/Llama-3.1-8B-Instruct",
                "ollama:llama2",
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_providers(config_file, filter_prefix=None)

        assert len(result) == 3


class TestParseTests:
    """Tests for parse_tests function."""

    def test_parses_simple_prompts(self, tmp_path):
        """Parse prompts array into test cases."""
        config = {
            "prompts": [
                "What is 2+2?",
                "What is the capital of France?",
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_tests(config_file)

        assert len(result) == 2
        assert result[0].user == "What is 2+2?"
        assert result[1].user == "What is the capital of France?"

    def test_parses_tests_with_vars(self, tmp_path):
        """Parse tests array with variable substitution."""
        config = {
            "prompts": ["What is the capital of {{country}}?"],
            "tests": [
                {"vars": {"country": "France"}},
                {"vars": {"country": "Germany"}},
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_tests(config_file)

        assert len(result) == 2
        assert "France" in result[0].user
        assert "Germany" in result[1].user

    def test_parses_tests_with_assertions(self, tmp_path):
        """Parse tests with assert conditions."""
        config = {
            "prompts": ["What is 2+2?"],
            "tests": [
                {
                    "assert": [
                        {"type": "contains", "value": "4"}
                    ]
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_tests(config_file)

        assert len(result) == 1
        # pass_criteria should capture the assertion
        assert result[0].pass_criteria is not None
        assert "4" in result[0].pass_criteria

    def test_generates_unique_ids(self, tmp_path):
        """Each test case should have a unique id."""
        config = {
            "prompts": ["Test 1", "Test 2", "Test 3"]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_tests(config_file)

        ids = [t.id for t in result]
        assert len(ids) == len(set(ids)), "IDs should be unique"

    def test_handles_missing_prompts(self, tmp_path):
        """Return empty list when prompts missing."""
        config = {"providers": ["openai:gpt-4"]}
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_tests(config_file)

        assert result == []

    def test_parses_description_as_name(self, tmp_path):
        """Use test description as name field."""
        config = {
            "prompts": ["What is 2+2?"],
            "tests": [
                {
                    "description": "Basic arithmetic test",
                    "assert": [{"type": "equals", "value": "4"}]
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = parse_tests(config_file)

        assert result[0].name == "Basic arithmetic test"
