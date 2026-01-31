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


class TestPromptfooLoaderPromptObjects:
    """Tests for prompt objects (id, label, raw) - not just strings."""

    def test_parses_prompt_objects(self, tmp_path):
        """Parse prompts array with object format."""
        config = {
            "prompts": [
                {"id": "prompt1", "label": "First Prompt", "raw": "What is 2+2?"},
                {"id": "prompt2", "label": "Second Prompt", "raw": "What is 3+3?"},
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 2
        assert result[0].user == "What is 2+2?"
        assert result[0].name == "First Prompt"
        assert result[1].user == "What is 3+3?"
        assert result[1].name == "Second Prompt"

    def test_parses_mixed_strings_and_objects(self, tmp_path):
        """Parse prompts array with mixed string and object formats."""
        config = {
            "prompts": [
                "Simple string prompt",
                {"id": "obj-prompt", "label": "Object Prompt", "raw": "Object prompt text"},
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 2
        assert result[0].user == "Simple string prompt"
        assert result[1].user == "Object prompt text"
        assert result[1].name == "Object Prompt"

    def test_prompt_object_with_variable_substitution(self, tmp_path):
        """Variable substitution works with prompt objects."""
        config = {
            "prompts": [
                {"id": "template", "label": "Template Prompt", "raw": "{{system}}\n\n{{user}}"}
            ],
            "tests": [
                {
                    "description": "Test with vars",
                    "vars": {
                        "system": "You are a helpful assistant.",
                        "user": "Hello!"
                    }
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        assert result[0].user == "Hello!"
        assert result[0].system == "You are a helpful assistant."

    def test_prompt_object_without_label_uses_id(self, tmp_path):
        """Prompt object uses id as fallback when label is missing."""
        config = {
            "prompts": [
                {"id": "my-prompt-id", "raw": "Test prompt"}
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert result[0].name == "my-prompt-id"

    def test_prompt_object_minimal(self, tmp_path):
        """Prompt object with only raw field (minimum required)."""
        config = {
            "prompts": [
                {"raw": "Just the raw text"}
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        assert result[0].user == "Just the raw text"

    def test_prompt_object_with_tests_and_assertions(self, tmp_path):
        """Full integration: prompt objects with tests and assertions."""
        config = {
            "prompts": [
                {
                    "id": "batch_processor",
                    "label": "BatchProcessor Prompt",
                    "raw": "Process these files: {{files}}"
                }
            ],
            "tests": [
                {
                    "description": "Sort files test",
                    "vars": {
                        "files": "a.txt, b.txt, c.txt"
                    },
                    "assert": [
                        {"type": "is-json"},
                        {"type": "contains", "value": "operations"}
                    ]
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        # When no 'user' var exists, the full substituted prompt becomes the user content
        assert result[0].user == "Process these files: a.txt, b.txt, c.txt"
        assert result[0].name == "Sort files test"
        assert "is-json" in result[0].pass_criteria
        assert "contains: operations" in result[0].pass_criteria


class TestPromptfooLoaderVarMappings:
    """Tests for special var mappings (expected_verdict, category)."""

    def test_expected_verdict_generates_pass_criteria(self, tmp_path):
        """expected_verdict var generates pass_criteria for judge competence tests."""
        config = {
            "prompts": ["Evaluate: {{actual_output}}"],
            "tests": [
                {
                    "description": "Should fail wrong function",
                    "vars": {
                        "expected_verdict": "FAIL",
                        "actual_output": '{"function": "wrong"}'
                    }
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        assert result[0].pass_criteria is not None
        assert "FAIL" in result[0].pass_criteria
        assert "verdict" in result[0].pass_criteria.lower()

    def test_category_var_maps_to_benchmark_category(self, tmp_path):
        """category var maps to BenchmarkCase.category field."""
        config = {
            "prompts": ["Test prompt"],
            "tests": [
                {
                    "description": "Test in category",
                    "vars": {
                        "category": "clear_discrimination"
                    }
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        assert result[0].category == "clear_discrimination"

    def test_assertions_take_precedence_over_expected_verdict(self, tmp_path):
        """When both assert and expected_verdict exist, assertions win."""
        config = {
            "prompts": ["Test prompt"],
            "tests": [
                {
                    "description": "Has both",
                    "vars": {
                        "expected_verdict": "FAIL"
                    },
                    "assert": [
                        {"type": "contains", "value": "specific text"}
                    ]
                }
            ]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        result = PromptfooLoader.load(config_file)

        assert len(result) == 1
        # Assertions take precedence
        assert "contains: specific text" in result[0].pass_criteria
        assert "FAIL" not in result[0].pass_criteria
