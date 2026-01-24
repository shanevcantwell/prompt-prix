"""Tests for PromptfooLoader."""

import pytest
import yaml

from prompt_prix.benchmarks.promptfoo import PromptfooLoader


class TestPromptfooLoaderLoad:
    """Tests for PromptfooLoader.load()."""

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

        result = PromptfooLoader.load(config_file)

        assert len(result) == 2
        assert result[0].user == "What is 2+2?"
        assert result[1].user == "What is the capital of France?"

    def test_parses_tests_with_vars(self, tmp_path):
        """Parse tests array with user variable extraction."""
        config = {
            "prompts": ["{{user}}"],
            "tests": [
                {"vars": {"user": "What is the capital of France?"}},
                {"vars": {"user": "What is the capital of Germany?"}},
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 2
        assert result[0].user == "What is the capital of France?"
        assert result[1].user == "What is the capital of Germany?"

    def test_parses_system_from_vars(self, tmp_path):
        """Extract system prompt from vars."""
        config = {
            "prompts": ["{{system}}\n\n{{user}}"],
            "tests": [
                {
                    "vars": {
                        "system": "You are a math tutor.",
                        "user": "What is 2+2?"
                    }
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        assert result[0].system == "You are a math tutor."
        assert result[0].user == "What is 2+2?"

    def test_parses_tests_with_assertions(self, tmp_path):
        """Parse tests with assert conditions into pass_criteria."""
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

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        assert result[0].pass_criteria is not None
        assert "contains" in result[0].pass_criteria
        assert "4" in result[0].pass_criteria

    def test_parses_multiple_assertions(self, tmp_path):
        """Multiple assertions joined with semicolon."""
        config = {
            "prompts": ["Test prompt"],
            "tests": [
                {
                    "assert": [
                        {"type": "contains", "value": "hello"},
                        {"type": "not-contains", "value": "goodbye"},
                        {"type": "is-json"}
                    ]
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert result[0].pass_criteria is not None
        assert "contains: hello" in result[0].pass_criteria
        assert "not-contains: goodbye" in result[0].pass_criteria
        assert "is-json" in result[0].pass_criteria

    def test_generates_unique_ids(self, tmp_path):
        """Each test case should have a unique id."""
        config = {
            "prompts": ["Test 1", "Test 2", "Test 3"]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        ids = [t.id for t in result]
        assert len(ids) == len(set(ids)), "IDs should be unique"

    def test_uses_description_for_name_and_id(self, tmp_path):
        """Use test description as name field and slugified for id."""
        config = {
            "prompts": ["What is 2+2?"],
            "tests": [
                {
                    "description": "Basic Arithmetic Test",
                    "assert": [{"type": "equals", "value": "4"}]
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert result[0].name == "Basic Arithmetic Test"
        assert result[0].id == "basic-arithmetic-test"

    def test_uses_metadata_test_id(self, tmp_path):
        """Prefer metadata.test_id over slugified description."""
        config = {
            "prompts": ["Test prompt"],
            "tests": [
                {
                    "description": "Some Description",
                    "metadata": {"test_id": "custom-test-id"}
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert result[0].id == "custom-test-id"
        assert result[0].name == "Some Description"

    def test_handles_duplicate_descriptions(self, tmp_path):
        """Duplicate descriptions get unique IDs with suffixes."""
        config = {
            "prompts": ["Test"],
            "tests": [
                {"description": "Same Name"},
                {"description": "Same Name"},
                {"description": "Same Name"},
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        ids = [t.id for t in result]
        assert len(ids) == 3
        assert len(set(ids)) == 3  # All unique
        assert "same-name" in ids
        assert "same-name-2" in ids
        assert "same-name-3" in ids


class TestPromptfooLoaderValidation:
    """Tests for Pydantic schema validation."""

    def test_missing_prompts_key(self, tmp_path):
        """Fail-fast when required 'prompts' key is missing."""
        config = {"providers": ["openai:gpt-4"]}
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ValueError) as exc_info:
            PromptfooLoader.load(config_file)

        assert "prompts" in str(exc_info.value).lower()

    def test_prompts_wrong_type(self, tmp_path):
        """Fail when prompts is not a list."""
        config = {"prompts": "single string not a list"}
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        with pytest.raises(ValueError) as exc_info:
            PromptfooLoader.load(config_file)

        assert "prompts" in str(exc_info.value).lower()

    def test_malformed_yaml_syntax(self, tmp_path):
        """Fail on invalid YAML syntax."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("prompts: [invalid yaml {")

        with pytest.raises(ValueError) as exc_info:
            PromptfooLoader.load(config_file)

        assert "yaml" in str(exc_info.value).lower()

    def test_empty_yaml_file(self, tmp_path):
        """Fail on empty YAML file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError) as exc_info:
            PromptfooLoader.load(config_file)

        assert "empty" in str(exc_info.value).lower()

    def test_missing_file(self, tmp_path):
        """Fail on non-existent file."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            PromptfooLoader.load(config_file)

    def test_empty_prompts_array(self, tmp_path):
        """Empty prompts array is technically valid but produces no tests."""
        config = {"prompts": []}
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert result == []


class TestPromptfooLoaderValidate:
    """Tests for PromptfooLoader.validate()."""

    def test_valid_file_returns_success(self, tmp_path):
        """Valid file returns (True, success message)."""
        config = {
            "prompts": ["Test 1", "Test 2", "Test 3"]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        is_valid, message = PromptfooLoader.validate(config_file)

        assert is_valid is True
        assert "✅" in message
        assert "3 tests" in message

    def test_invalid_file_returns_failure(self, tmp_path):
        """Invalid file returns (False, error message)."""
        config = {"not_prompts": ["test"]}
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        is_valid, message = PromptfooLoader.validate(config_file)

        assert is_valid is False
        assert "❌" in message

    def test_missing_file_returns_failure(self, tmp_path):
        """Missing file returns (False, error message)."""
        config_file = tmp_path / "nonexistent.yaml"

        is_valid, message = PromptfooLoader.validate(config_file)

        assert is_valid is False
        assert "❌" in message
        assert "not found" in message.lower()
