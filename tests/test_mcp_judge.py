"""Tests for prompt_prix.mcp.tools.judge module."""

import pytest
import httpx
import respx

from tests.conftest import (
    MOCK_SERVER_1,
    MOCK_MODEL_1,
    MOCK_MANIFEST_RESPONSE,
)


def make_judge_streaming_body(pass_value: bool, reason: str, score=None) -> bytes:
    """Create SSE streaming response body for judge response."""
    import json
    response_json = json.dumps({
        "pass": pass_value,
        "reason": reason,
        "score": score
    })
    # Stream the JSON response in word-sized chunks
    chunks = []
    # Split into reasonable chunks (can't split by character due to JSON escaping issues)
    words = response_json.split()
    for i, word in enumerate(words):
        # Add space back except for first word
        content = word if i == 0 else " " + word
        # Escape the content for JSON embedding
        escaped = json.dumps(content)[1:-1]  # Remove outer quotes from json.dumps
        chunks.append(f'data: {{"choices":[{{"delta":{{"content":"{escaped}"}}}}]}}\n')
    chunks.append('data: [DONE]\n')
    return "".join(chunks).encode()


class TestJudge:
    """Tests for judge() MCP tool."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_judge_pass(self):
        """Test judge() returns pass when criteria are met."""
        from prompt_prix.mcp.tools.judge import judge

        # Mock manifest endpoint
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        # Mock completion endpoint with passing judgment
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=make_judge_streaming_body(
                    pass_value=True,
                    reason="The response clearly indicates intent to delete the file.",
                    score=8.5
                )
            )
        )

        result = await judge(
            response="I'll help you delete report.pdf right away.",
            criteria="Response must indicate intent to delete the file",
            judge_model=MOCK_MODEL_1,
            server_urls=[MOCK_SERVER_1],
        )

        assert result["pass"] is True
        assert "delete" in result["reason"].lower() or len(result["reason"]) > 0
        assert result["score"] == 8.5

    @respx.mock
    @pytest.mark.asyncio
    async def test_judge_fail(self):
        """Test judge() returns fail when criteria are not met."""
        from prompt_prix.mcp.tools.judge import judge

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=make_judge_streaming_body(
                    pass_value=False,
                    reason="The response refuses to perform the requested action.",
                    score=2.0
                )
            )
        )

        result = await judge(
            response="I'm sorry, but I can't help you delete files.",
            criteria="Response must indicate intent to delete the file",
            judge_model=MOCK_MODEL_1,
            server_urls=[MOCK_SERVER_1],
        )

        assert result["pass"] is False
        assert len(result["reason"]) > 0
        assert result["score"] == 2.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_judge_no_score(self):
        """Test judge() handles null score."""
        from prompt_prix.mcp.tools.judge import judge

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=make_judge_streaming_body(
                    pass_value=True,
                    reason="Response meets criteria.",
                    score=None
                )
            )
        )

        result = await judge(
            response="Test response",
            criteria="Test criteria",
            judge_model=MOCK_MODEL_1,
            server_urls=[MOCK_SERVER_1],
        )

        assert result["pass"] is True
        assert result["score"] is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_judge_includes_raw_response(self):
        """Test judge() includes raw_response in result."""
        from prompt_prix.mcp.tools.judge import judge

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=make_judge_streaming_body(
                    pass_value=True,
                    reason="OK",
                    score=None
                )
            )
        )

        result = await judge(
            response="Test",
            criteria="Test",
            judge_model=MOCK_MODEL_1,
            server_urls=[MOCK_SERVER_1],
        )

        assert "raw_response" in result
        assert result["raw_response"] is not None


class TestParseJudgeResponse:
    """Tests for _parse_judge_response helper."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        from prompt_prix.mcp.tools.judge import _parse_judge_response

        response = '{"pass": true, "reason": "Looks good", "score": 9.5}'
        result = _parse_judge_response(response)

        assert result["pass"] is True
        assert result["reason"] == "Looks good"
        assert result["score"] == 9.5

    def test_parse_json_in_code_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        from prompt_prix.mcp.tools.judge import _parse_judge_response

        response = '''```json
{"pass": false, "reason": "Failed criteria", "score": 3.0}
```'''
        result = _parse_judge_response(response)

        assert result["pass"] is False
        assert result["reason"] == "Failed criteria"
        assert result["score"] == 3.0

    def test_parse_fallback_heuristics(self):
        """Test fallback heuristics for malformed response."""
        from prompt_prix.mcp.tools.judge import _parse_judge_response

        # Malformed but contains indicators
        response = 'The response passes. "pass": true, "reason": "Good job"'
        result = _parse_judge_response(response)

        assert result["pass"] is True
        assert result["reason"] == "Good job"

    def test_parse_score_clamping(self):
        """Test score is clamped to 0-10 range."""
        from prompt_prix.mcp.tools.judge import _parse_score

        assert _parse_score(15) == 10.0
        assert _parse_score(-5) == 0.0
        assert _parse_score(5.5) == 5.5
        assert _parse_score(None) is None
        assert _parse_score("invalid") is None


class TestJudgeIntegration:
    """Integration tests requiring live LM Studio servers."""

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

        from prompt_prix.mcp.tools.judge import judge
        from prompt_prix.mcp.tools.list_models import list_models
        from prompt_prix.config import load_servers_from_env

        servers = load_servers_from_env()
        if not servers:
            pytest.skip("No LM Studio servers configured in .env")

        # First discover available models
        discovery = await list_models(servers)
        if not discovery["models"]:
            pytest.skip("No models available on configured servers")

        model_id = discovery["models"][0]

        result = await judge(
            response="I'll delete report.pdf for you right away.",
            criteria="Response should indicate willingness to help with the task.",
            judge_model=model_id,
            server_urls=servers,
        )

        # Basic structure checks
        assert "pass" in result
        assert isinstance(result["pass"], bool)
        assert "reason" in result
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0
