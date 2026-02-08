"""Tests for prompt_prix.mcp.tools.geometry module.

analyze_variants: Mocks semantic-chunker (same pattern as test_drift.py).
generate_variants: Mocks adapter via registry (same pattern as test_mcp_judge.py).
"""

import json
import pytest
import sys
import types
from unittest.mock import AsyncMock, MagicMock

from prompt_prix.mcp.registry import register_adapter, clear_adapter


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _make_semantic_chunker_modules():
    """Create fake semantic_chunker module hierarchy for sys.modules patching."""
    sc = types.ModuleType("semantic_chunker")
    sc_mcp = types.ModuleType("semantic_chunker.mcp")
    sc_mcp_commands = types.ModuleType("semantic_chunker.mcp.commands")
    sc_mcp_commands_geometry = types.ModuleType("semantic_chunker.mcp.commands.geometry")
    sc_mcp_state = types.ModuleType("semantic_chunker.mcp.state_manager")

    sc_mcp_state.StateManager = MagicMock

    sc.mcp = sc_mcp
    sc_mcp.commands = sc_mcp_commands
    sc_mcp.state_manager = sc_mcp_state
    sc_mcp_commands.geometry = sc_mcp_commands_geometry

    modules_dict = {
        "semantic_chunker": sc,
        "semantic_chunker.mcp": sc_mcp,
        "semantic_chunker.mcp.commands": sc_mcp_commands,
        "semantic_chunker.mcp.commands.geometry": sc_mcp_commands_geometry,
        "semantic_chunker.mcp.state_manager": sc_mcp_state,
    }
    return modules_dict, sc_mcp_commands_geometry


def _reset_shared_module():
    """Reset cached state on the shared _semantic_chunker module."""
    import prompt_prix.mcp.tools._semantic_chunker as sc_mod
    sc_mod._manager = None
    sc_mod._available = None


def _make_variant_stream(variants_json: str):
    """Create an async generator that yields a JSON response for generate_variants."""
    async def stream(task):
        yield variants_json
    return stream


@pytest.fixture
def mock_adapter():
    """Register a mock adapter for generate_variants tests."""
    adapter = MagicMock()
    adapter.get_available_models = AsyncMock(return_value=["test-model"])
    adapter.get_models_by_server = MagicMock(return_value={})
    adapter.get_unreachable_servers = MagicMock(return_value=[])

    async def default_stream(task):
        yield '{"passive": "A bug should be filed."}'

    adapter.stream_completion = default_stream
    register_adapter(adapter)
    yield adapter
    clear_adapter()


# ─────────────────────────────────────────────────────────────────────
# ANALYZE VARIANTS TESTS
# ─────────────────────────────────────────────────────────────────────

class TestAnalyzeVariants:
    """Tests for analyze_variants MCP tool (delegates to semantic-chunker)."""

    @pytest.mark.asyncio
    async def test_returns_distances(self):
        """analyze_variants returns distance data from semantic-chunker."""
        from unittest.mock import patch

        mock_result = {
            "constraint_name": "test",
            "baseline_label": "imperative",
            "variants_count": 2,
            "from_baseline": {"passive": 0.084},
            "pairwise": {"imperative-passive": 0.084},
            "recommendations": [{"label": "passive", "distance": 0.084}],
        }

        modules_dict, geometry_mod = _make_semantic_chunker_modules()
        geometry_mod.analyze_variants = AsyncMock(return_value=mock_result)

        with patch.dict(sys.modules, modules_dict):
            _reset_shared_module()
            from prompt_prix.mcp.tools.geometry import analyze_variants
            result = await analyze_variants(
                variants={"imperative": "File a bug.", "passive": "A bug should be filed."},
                baseline_label="imperative",
            )

        assert result["from_baseline"]["passive"] == 0.084
        assert result["variants_count"] == 2

    @pytest.mark.asyncio
    async def test_forwards_args_correctly(self):
        """analyze_variants passes correct args dict to semantic-chunker."""
        from unittest.mock import patch

        captured_args = {}

        async def capture_call(manager, args):
            captured_args.update(args)
            return {"constraint_name": "c", "baseline_label": "b",
                    "variants_count": 1, "from_baseline": {}, "pairwise": {},
                    "recommendations": []}

        modules_dict, geometry_mod = _make_semantic_chunker_modules()
        geometry_mod.analyze_variants = capture_call

        with patch.dict(sys.modules, modules_dict):
            _reset_shared_module()
            from prompt_prix.mcp.tools.geometry import analyze_variants
            await analyze_variants(
                variants={"a": "text a", "b": "text b"},
                baseline_label="a",
                constraint_name="my_constraint",
            )

        assert captured_args["variants"] == {"a": "text a", "b": "text b"}
        assert captured_args["baseline_label"] == "a"
        assert captured_args["constraint_name"] == "my_constraint"

    @pytest.mark.asyncio
    async def test_raises_on_error_result(self):
        """analyze_variants raises RuntimeError when semantic-chunker returns error."""
        from unittest.mock import patch

        modules_dict, geometry_mod = _make_semantic_chunker_modules()
        geometry_mod.analyze_variants = AsyncMock(
            return_value={"error": "Embedding server unreachable"}
        )

        with patch.dict(sys.modules, modules_dict):
            _reset_shared_module()
            from prompt_prix.mcp.tools.geometry import analyze_variants
            with pytest.raises(RuntimeError, match="unreachable"):
                await analyze_variants(variants={"a": "text"})

    @pytest.mark.asyncio
    async def test_raises_import_error_when_unavailable(self):
        """analyze_variants raises ImportError when semantic-chunker not installed."""
        _reset_shared_module()
        # Don't patch sys.modules — semantic-chunker genuinely not available
        import prompt_prix.mcp.tools._semantic_chunker as sc_mod
        sc_mod._available = False

        from prompt_prix.mcp.tools.geometry import analyze_variants
        with pytest.raises(ImportError, match="not available"):
            await analyze_variants(variants={"a": "text"})


# ─────────────────────────────────────────────────────────────────────
# GENERATE VARIANTS TESTS
# ─────────────────────────────────────────────────────────────────────

class TestGenerateVariants:
    """Tests for generate_variants MCP tool (uses complete(), no semantic-chunker)."""

    @pytest.mark.asyncio
    async def test_returns_variants_with_baseline(self, mock_adapter):
        """generate_variants includes the imperative baseline in returned variants."""
        variants_json = json.dumps({
            "passive": "A bug should be filed before a feature is requested.",
            "interrogative": "Have you filed a bug report?",
        })
        mock_adapter.stream_completion = _make_variant_stream(variants_json)

        from prompt_prix.mcp.tools.geometry import generate_variants
        result = await generate_variants(
            baseline="File a bug report before requesting a feature.",
            model_id="test-model",
        )

        assert result["baseline"] == "File a bug report before requesting a feature."
        assert "imperative" in result["variants"]
        assert result["variants"]["imperative"] == result["baseline"]
        assert "passive" in result["variants"]
        assert result["variant_count"] == 3  # imperative + passive + interrogative

    @pytest.mark.asyncio
    async def test_uses_specified_dimensions(self, mock_adapter):
        """generate_variants includes requested dimensions in prompt."""
        captured_task = {}

        async def capture_stream(task):
            captured_task["messages"] = task.messages
            yield '{"tense_future": "A bug will be filed."}'

        mock_adapter.stream_completion = capture_stream

        from prompt_prix.mcp.tools.geometry import generate_variants
        await generate_variants(
            baseline="File a bug.",
            model_id="test-model",
            dimensions=["tense"],
        )

        prompt_text = captured_task["messages"][0]["content"]
        assert "tense" in prompt_text
        assert "present, past, future, perfect" in prompt_text

    @pytest.mark.asyncio
    async def test_parses_markdown_wrapped_json(self, mock_adapter):
        """generate_variants handles JSON wrapped in markdown code blocks."""
        response = '```json\n{"passive": "A bug should be filed."}\n```'
        mock_adapter.stream_completion = _make_variant_stream(response)

        from prompt_prix.mcp.tools.geometry import generate_variants
        result = await generate_variants(baseline="File a bug.", model_id="test-model")

        assert "passive" in result["variants"]

    @pytest.mark.asyncio
    async def test_raises_on_empty_baseline(self):
        """generate_variants raises ValueError for empty baseline."""
        from prompt_prix.mcp.tools.geometry import generate_variants
        with pytest.raises(ValueError, match="empty"):
            await generate_variants(baseline="", model_id="test-model")

    @pytest.mark.asyncio
    async def test_raises_on_unparseable_response(self, mock_adapter):
        """generate_variants raises ValueError when LLM returns non-JSON."""
        mock_adapter.stream_completion = _make_variant_stream(
            "I'd be happy to help! Here are some variants..."
        )

        from prompt_prix.mcp.tools.geometry import generate_variants
        with pytest.raises(ValueError, match="Could not parse"):
            await generate_variants(baseline="File a bug.", model_id="test-model")


# ─────────────────────────────────────────────────────────────────────
# PARSE HELPERS
# ─────────────────────────────────────────────────────────────────────

class TestParseVariantsResponse:
    """Tests for _parse_variants_response helper."""

    def test_bare_json(self):
        from prompt_prix.mcp.tools.geometry import _parse_variants_response
        result = _parse_variants_response('{"passive": "A bug should be filed."}')
        assert result == {"passive": "A bug should be filed."}

    def test_markdown_code_block(self):
        from prompt_prix.mcp.tools.geometry import _parse_variants_response
        result = _parse_variants_response(
            '```json\n{"passive": "A bug should be filed."}\n```'
        )
        assert result == {"passive": "A bug should be filed."}

    def test_raises_on_non_json(self):
        from prompt_prix.mcp.tools.geometry import _parse_variants_response
        with pytest.raises(ValueError):
            _parse_variants_response("Not JSON at all")
