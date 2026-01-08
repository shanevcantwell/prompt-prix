"""Tests for text parsing utilities."""

import pytest
from prompt_prix.parsers import (
    parse_models_input,
    parse_servers_input,
    parse_prefixed_model,
)


class TestParseModelsInput:
    """Tests for parse_models_input."""

    def test_comma_separated(self):
        """Test parsing comma-separated models."""
        result = parse_models_input("model-a, model-b, model-c")
        assert result == ["model-a", "model-b", "model-c"]

    def test_newline_separated(self):
        """Test parsing newline-separated models."""
        result = parse_models_input("model-a\nmodel-b\nmodel-c")
        assert result == ["model-a", "model-b", "model-c"]

    def test_mixed_separators(self):
        """Test parsing with mixed separators."""
        result = parse_models_input("model-a, model-b\nmodel-c")
        assert result == ["model-a", "model-b", "model-c"]

    def test_empty_input(self):
        """Test empty input returns empty list."""
        result = parse_models_input("")
        assert result == []

    def test_whitespace_stripped(self):
        """Test whitespace is stripped from items."""
        result = parse_models_input("  model-a  ,  model-b  ")
        assert result == ["model-a", "model-b"]


class TestParseServersInput:
    """Tests for parse_servers_input."""

    def test_newline_separated(self):
        """Test parsing newline-separated servers."""
        result = parse_servers_input("http://localhost:1234\nhttp://localhost:5678")
        assert result == ["http://localhost:1234", "http://localhost:5678"]

    def test_comma_separated(self):
        """Test parsing comma-separated servers."""
        result = parse_servers_input("http://a:1234, http://b:1234")
        assert result == ["http://a:1234", "http://b:1234"]


class TestParsePrefixedModel:
    """Tests for parse_prefixed_model - GPU prefix parsing."""

    def test_basic_prefix(self):
        """Test parsing basic prefixed model."""
        idx, model_id = parse_prefixed_model("0: lfm2-1.2b-tool")
        assert idx == 0
        assert model_id == "lfm2-1.2b-tool"

    def test_different_index(self):
        """Test parsing with different server index."""
        idx, model_id = parse_prefixed_model("1: qwen-7b")
        assert idx == 1
        assert model_id == "qwen-7b"

    def test_model_with_slashes(self):
        """Test parsing model ID containing slashes (GGUF paths)."""
        idx, model_id = parse_prefixed_model("1: openai/gpt-oss-20b-gguf/gpt-oss-20b-router-mxfp4.gguf")
        assert idx == 1
        assert model_id == "openai/gpt-oss-20b-gguf/gpt-oss-20b-router-mxfp4.gguf"

    def test_model_with_colon_in_name(self):
        """Test model ID containing colons after the prefix."""
        idx, model_id = parse_prefixed_model("0: model:latest:v2")
        assert idx == 0
        assert model_id == "model:latest:v2"

    def test_double_digit_index(self):
        """Test parsing with double-digit server index."""
        idx, model_id = parse_prefixed_model("12: model-name")
        assert idx == 12
        assert model_id == "model-name"

    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_prefixed_model("model-without-prefix")

    def test_missing_space_after_colon_raises(self):
        """Test that missing space after colon raises ValueError."""
        with pytest.raises(ValueError):
            parse_prefixed_model("0:model-name")  # No space after colon
