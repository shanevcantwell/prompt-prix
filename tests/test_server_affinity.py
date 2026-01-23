"""Tests for server affinity prefix parsing.

Server affinity uses the format "idx:model_name" to route requests
to specific servers by index.
"""

import pytest
from prompt_prix.server_affinity import (
    parse_server_prefix,
    strip_server_prefix,
    extract_server_indices,
)


class TestParseServerPrefix:
    """Tests for parse_server_prefix function."""

    def test_valid_prefix_single_digit(self):
        """Test parsing "0:model" → (0, "model")."""
        server_idx, model_id = parse_server_prefix("0:model")
        assert server_idx == 0
        assert model_id == "model"

    def test_valid_prefix_double_digit(self):
        """Test parsing "10:model-v2" → (10, "model-v2")."""
        server_idx, model_id = parse_server_prefix("10:model-v2")
        assert server_idx == 10
        assert model_id == "model-v2"

    def test_no_prefix(self):
        """Test parsing "model" → (None, "model")."""
        server_idx, model_id = parse_server_prefix("model")
        assert server_idx is None
        assert model_id == "model"

    def test_invalid_prefix_no_digit(self):
        """Test parsing ":model" → (None, ":model") - invalid prefix."""
        server_idx, model_id = parse_server_prefix(":model")
        assert server_idx is None
        assert model_id == ":model"

    def test_empty_model_after_prefix(self):
        """Test parsing "0:" → (0, "") - edge case."""
        server_idx, model_id = parse_server_prefix("0:")
        assert server_idx == 0
        assert model_id == ""

    def test_prefix_with_letters(self):
        """Test parsing "abc:model" → (None, "abc:model") - not a valid prefix."""
        server_idx, model_id = parse_server_prefix("abc:model")
        assert server_idx is None
        assert model_id == "abc:model"

    def test_model_with_colon_in_name(self):
        """Test parsing "0:model:v2" → (0, "model:v2") - colon in model name."""
        server_idx, model_id = parse_server_prefix("0:model:v2")
        assert server_idx == 0
        assert model_id == "model:v2"

    def test_mixed_prefix(self):
        """Test parsing "1a:model" → (None, "1a:model") - not purely numeric."""
        server_idx, model_id = parse_server_prefix("1a:model")
        assert server_idx is None
        assert model_id == "1a:model"

    def test_large_server_index(self):
        """Test parsing "999:model" → (999, "model")."""
        server_idx, model_id = parse_server_prefix("999:model")
        assert server_idx == 999
        assert model_id == "model"

    def test_negative_prefix_invalid(self):
        """Test parsing "-1:model" → (None, "-1:model") - negative not valid."""
        server_idx, model_id = parse_server_prefix("-1:model")
        assert server_idx is None
        assert model_id == "-1:model"


class TestStripServerPrefix:
    """Tests for strip_server_prefix function."""

    def test_strips_valid_prefix(self):
        """Test stripping "0:model" → "model"."""
        assert strip_server_prefix("0:model") == "model"

    def test_no_prefix_unchanged(self):
        """Test stripping "model" → "model"."""
        assert strip_server_prefix("model") == "model"

    def test_strips_double_digit_prefix(self):
        """Test stripping "10:model-v2" → "model-v2"."""
        assert strip_server_prefix("10:model-v2") == "model-v2"

    def test_invalid_prefix_unchanged(self):
        """Test stripping ":model" → ":model"."""
        assert strip_server_prefix(":model") == ":model"

    def test_empty_after_strip(self):
        """Test stripping "0:" → ""."""
        assert strip_server_prefix("0:") == ""

    def test_preserves_colon_in_model_name(self):
        """Test stripping "1:llama:instruct" → "llama:instruct"."""
        assert strip_server_prefix("1:llama:instruct") == "llama:instruct"


class TestExtractServerIndices:
    """Tests for extract_server_indices function."""

    def test_extracts_multiple_indices(self):
        """Test extracting indices from ["0:a", "0:b", "1:c"] → {0, 1}."""
        indices = extract_server_indices(["0:a", "0:b", "1:c"])
        assert indices == {0, 1}

    def test_no_prefixes_returns_empty(self):
        """Test extracting from ["model1", "model2"] → set()."""
        indices = extract_server_indices(["model1", "model2"])
        assert indices == set()

    def test_mixed_prefixed_and_unprefixed(self):
        """Test extracting from ["0:a", "model", "1:b"] → {0, 1}."""
        indices = extract_server_indices(["0:a", "model", "1:b"])
        assert indices == {0, 1}

    def test_empty_list(self):
        """Test extracting from [] → set()."""
        indices = extract_server_indices([])
        assert indices == set()

    def test_single_model_with_prefix(self):
        """Test extracting from ["0:model"] → {0}."""
        indices = extract_server_indices(["0:model"])
        assert indices == {0}

    def test_duplicate_indices_deduplicated(self):
        """Test that duplicate indices are deduplicated."""
        indices = extract_server_indices(["0:a", "0:b", "0:c"])
        assert indices == {0}

    def test_large_index(self):
        """Test extracting large indices."""
        indices = extract_server_indices(["99:model"])
        assert indices == {99}
