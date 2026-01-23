"""Tests for prompt_prix.mcp.tools.judge module.

Per ADR-006: Mock at layer boundaries - MCP tests mock the adapter interface.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from prompt_prix.mcp.registry import register_adapter, clear_adapter
from prompt_prix.mcp.tools.judge import judge, _parse_judge_response, _parse_score


@pytest.fixture
def mock_adapter():
    """Create a mock adapter and register it with the MCP registry."""
    adapter = MagicMock()

    # Mock get_available_models
    adapter.get_available_models = AsyncMock(return_value=["judge-model"])

    # Default stream_completion - will be overridden in tests
    async def default_stream(*args, **kwargs):
        yield '{"pass": true, "reason": "Default", "score": null}'

    adapter.stream_completion = default_stream

    # For list_models compatibility
    adapter.get_models_by_server = MagicMock(return_value={"http://localhost:1234": ["judge-model"]})
    adapter.get_unreachable_servers = MagicMock(return_value=[])

    register_adapter(adapter)
    yield adapter
    clear_adapter()


def make_judge_stream(pass_value: bool, reason: str, score=None):
    """Create an async generator that yields a judge JSON response."""
    import json
    response = json.dumps({"pass": pass_value, "reason": reason, "score": score})

    async def stream(*args, **kwargs):
        yield response

    return stream


class TestJudge:
    """Tests for judge() MCP tool."""

    @pytest.mark.asyncio
    async def test_judge_pass(self, mock_adapter):
        """Test judge() returns pass when criteria are met."""
        mock_adapter.stream_completion = make_judge_stream(
            pass_value=True,
            reason="The response clearly indicates intent to delete the file.",
            score=8.5
        )

        result = await judge(
            response="I'll help you delete report.pdf right away.",
            criteria="Response must indicate intent to delete the file",
            judge_model="judge-model",
        )

        assert result["pass"] is True
        assert "delete" in result["reason"].lower() or len(result["reason"]) > 0
        assert result["score"] == 8.5

    @pytest.mark.asyncio
    async def test_judge_fail(self, mock_adapter):
        """Test judge() returns fail when criteria are not met."""
        mock_adapter.stream_completion = make_judge_stream(
            pass_value=False,
            reason="The response refuses to perform the requested action.",
            score=2.0
        )

        result = await judge(
            response="I'm sorry, but I can't help you delete files.",
            criteria="Response must indicate intent to delete the file",
            judge_model="judge-model",
        )

        assert result["pass"] is False
        assert len(result["reason"]) > 0
        assert result["score"] == 2.0

    @pytest.mark.asyncio
    async def test_judge_no_score(self, mock_adapter):
        """Test judge() handles null score."""
        mock_adapter.stream_completion = make_judge_stream(
            pass_value=True,
            reason="Response meets criteria.",
            score=None
        )

        result = await judge(
            response="Test response",
            criteria="Test criteria",
            judge_model="judge-model",
        )

        assert result["pass"] is True
        assert result["score"] is None

    @pytest.mark.asyncio
    async def test_judge_includes_raw_response(self, mock_adapter):
        """Test judge() includes raw_response in result."""
        mock_adapter.stream_completion = make_judge_stream(
            pass_value=True,
            reason="OK",
            score=None
        )

        result = await judge(
            response="Test",
            criteria="Test",
            judge_model="judge-model",
        )

        assert "raw_response" in result
        assert result["raw_response"] is not None

    @pytest.mark.asyncio
    async def test_judge_passes_correct_params(self, mock_adapter):
        """Test judge() passes model_id and other params to adapter."""
        call_args = {}

        async def capture_stream(*args, **kwargs):
            call_args.update(kwargs)
            yield '{"pass": true, "reason": "OK", "score": null}'

        mock_adapter.stream_completion = capture_stream

        await judge(
            response="Test response",
            criteria="Test criteria",
            judge_model="my-judge-model",
            temperature=0.2,
            max_tokens=128,
        )

        assert call_args["model_id"] == "my-judge-model"
        assert call_args["temperature"] == 0.2
        assert call_args["max_tokens"] == 128

    @pytest.mark.asyncio
    async def test_judge_no_adapter_registered(self):
        """Test judge() raises when no adapter registered."""
        clear_adapter()

        with pytest.raises(RuntimeError, match="No adapter registered"):
            await judge(
                response="Test",
                criteria="Test",
                judge_model="judge-model",
            )


class TestParseJudgeResponse:
    """Tests for _parse_judge_response helper."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"pass": true, "reason": "Looks good", "score": 9.5}'
        result = _parse_judge_response(response)

        assert result["pass"] is True
        assert result["reason"] == "Looks good"
        assert result["score"] == 9.5

    def test_parse_json_in_code_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = '''```json
{"pass": false, "reason": "Failed criteria", "score": 3.0}
```'''
        result = _parse_judge_response(response)

        assert result["pass"] is False
        assert result["reason"] == "Failed criteria"
        assert result["score"] == 3.0

    def test_parse_fallback_heuristics(self):
        """Test fallback heuristics for malformed response."""
        # Malformed but contains indicators
        response = 'The response passes. "pass": true, "reason": "Good job"'
        result = _parse_judge_response(response)

        assert result["pass"] is True
        assert result["reason"] == "Good job"

    def test_parse_score_clamping(self):
        """Test score is clamped to 0-10 range."""
        assert _parse_score(15) == 10.0
        assert _parse_score(-5) == 0.0
        assert _parse_score(5.5) == 5.5
        assert _parse_score(None) is None
        assert _parse_score("invalid") is None


class TestJudgeIntegration:
    """Integration tests requiring live LM Studio servers.

    These tests are skipped by default. Run with:
        pytest -m integration
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_judge_live(self):
        """Test judge() against real LM Studio server.

        Prerequisites:
        - LM Studio running on configured server
        - At least one model loaded
        """
        from dotenv import load_dotenv
        load_dotenv()

        from prompt_prix.mcp.tools.list_models import list_models
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        from prompt_prix.config import get_default_servers

        servers = get_default_servers()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        adapter = LMStudioAdapter(server_urls=servers)
        register_adapter(adapter)

        try:
            discovery = await list_models()
            if not discovery["models"]:
                pytest.skip("No models available on configured servers")

            model_id = discovery["models"][0]

            result = await judge(
                response="I'll delete report.pdf for you right away.",
                criteria="Response should indicate willingness to help with the task.",
                judge_model=model_id,
            )

            # Basic structure checks
            assert "pass" in result
            assert isinstance(result["pass"], bool)
            assert "reason" in result
            assert isinstance(result["reason"], str)
            assert len(result["reason"]) > 0
        finally:
            clear_adapter()
