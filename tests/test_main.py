"""Tests for prompt_prix.main module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import httpx
import respx

from tests.conftest import (
    MOCK_SERVER_1, MOCK_SERVER_2, MOCK_SERVERS,
    MOCK_MODEL_1, MOCK_MODEL_2, MOCK_MODELS,
    MOCK_MANIFEST_RESPONSE, MOCK_COMPLETION_RESPONSE
)


class TestParseModelsInput:
    """Tests for parse_models_input function."""

    def test_parse_models_input_newline_separated(self):
        """Test parsing newline-separated models."""
        from prompt_prix.main import parse_models_input

        result = parse_models_input("model-a\nmodel-b\nmodel-c")

        assert result == ["model-a", "model-b", "model-c"]

    def test_parse_models_input_comma_separated(self):
        """Test parsing comma-separated models."""
        from prompt_prix.main import parse_models_input

        result = parse_models_input("model-a, model-b, model-c")

        assert result == ["model-a", "model-b", "model-c"]

    def test_parse_models_input_mixed(self):
        """Test parsing mixed newline and comma separated."""
        from prompt_prix.main import parse_models_input

        result = parse_models_input("model-a, model-b\nmodel-c")

        assert result == ["model-a", "model-b", "model-c"]

    def test_parse_models_input_strips_whitespace(self):
        """Test whitespace is stripped from model names."""
        from prompt_prix.main import parse_models_input

        result = parse_models_input("  model-a  \n  model-b  ")

        assert result == ["model-a", "model-b"]

    def test_parse_models_input_empty(self):
        """Test empty input returns empty list."""
        from prompt_prix.main import parse_models_input

        result = parse_models_input("")

        assert result == []

    def test_parse_models_input_skips_blank_lines(self):
        """Test blank lines are skipped."""
        from prompt_prix.main import parse_models_input

        result = parse_models_input("model-a\n\n\nmodel-b")

        assert result == ["model-a", "model-b"]


class TestParseServersInput:
    """Tests for parse_servers_input function."""

    def test_parse_servers_input_newline_separated(self):
        """Test parsing newline-separated servers."""
        from prompt_prix.main import parse_servers_input

        result = parse_servers_input("http://server1:1234\nhttp://server2:1234")

        assert result == ["http://server1:1234", "http://server2:1234"]

    def test_parse_servers_input_comma_separated(self):
        """Test parsing comma-separated servers."""
        from prompt_prix.main import parse_servers_input

        result = parse_servers_input("http://server1:1234, http://server2:1234")

        assert result == ["http://server1:1234", "http://server2:1234"]

    def test_parse_servers_input_empty(self):
        """Test empty input returns empty list."""
        from prompt_prix.main import parse_servers_input

        result = parse_servers_input("")

        assert result == []


class TestLoadSystemPrompt:
    """Tests for load_system_prompt function."""

    def test_load_system_prompt_from_file(self, tmp_system_prompt):
        """Test loading system prompt from provided file."""
        from prompt_prix.main import load_system_prompt

        result = load_system_prompt(str(tmp_system_prompt))

        assert result == "You are a test assistant."

    def test_load_system_prompt_default_when_no_file(self):
        """Test default prompt when no file provided."""
        from prompt_prix.main import load_system_prompt
        from prompt_prix.config import DEFAULT_SYSTEM_PROMPT

        result = load_system_prompt(None)

        assert result == DEFAULT_SYSTEM_PROMPT

    def test_load_system_prompt_nonexistent_file(self):
        """Test default prompt when file doesn't exist."""
        from prompt_prix.main import load_system_prompt
        from prompt_prix.config import DEFAULT_SYSTEM_PROMPT

        result = load_system_prompt("/nonexistent/path/file.txt")

        assert result == DEFAULT_SYSTEM_PROMPT


class TestParsePromptsFile:
    """Tests for parse_prompts_file function."""

    def test_parse_prompts_file_basic(self):
        """Test parsing prompts from file content."""
        from prompt_prix.main import parse_prompts_file

        content = "Question 1?\nQuestion 2?\nQuestion 3?"
        result = parse_prompts_file(content)

        assert result == ["Question 1?", "Question 2?", "Question 3?"]

    def test_parse_prompts_file_skips_empty_lines(self):
        """Test empty lines are skipped."""
        from prompt_prix.main import parse_prompts_file

        content = "Question 1?\n\n\nQuestion 2?"
        result = parse_prompts_file(content)

        assert result == ["Question 1?", "Question 2?"]

    def test_parse_prompts_file_strips_whitespace(self):
        """Test whitespace is stripped."""
        from prompt_prix.main import parse_prompts_file

        content = "  Question 1?  \n  Question 2?  "
        result = parse_prompts_file(content)

        assert result == ["Question 1?", "Question 2?"]


class TestInitializeSession:
    """Tests for initialize_session function."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_initialize_session_success(self):
        """Test successful session initialization."""
        from prompt_prix.main import initialize_session

        # Mock server responses
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.get(f"{MOCK_SERVER_2}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        servers_text = f"{MOCK_SERVER_1}\n{MOCK_SERVER_2}"
        models_text = f"{MOCK_MODEL_1}\n{MOCK_MODEL_2}"

        result = await initialize_session(
            servers_text=servers_text,
            models_text=models_text,
            system_prompt_file=None,
            temperature=0.7,
            timeout=300,
            max_tokens=2048
        )

        # Should return success status
        assert "initialized" in result[0].lower() or "✅" in result[0]

    @pytest.mark.asyncio
    async def test_initialize_session_missing_servers(self):
        """Test initialization fails without servers."""
        from prompt_prix.main import initialize_session

        result = await initialize_session(
            servers_text="",
            models_text=MOCK_MODEL_1,
            system_prompt_file=None,
            temperature=0.7,
            timeout=300,
            max_tokens=2048
        )

        assert "No servers" in result[0] or "❌" in result[0]

    @pytest.mark.asyncio
    async def test_initialize_session_missing_models(self):
        """Test initialization fails without models."""
        from prompt_prix.main import initialize_session

        result = await initialize_session(
            servers_text=MOCK_SERVER_1,
            models_text="",
            system_prompt_file=None,
            temperature=0.7,
            timeout=300,
            max_tokens=2048
        )

        assert "No models" in result[0] or "❌" in result[0]

    @respx.mock
    @pytest.mark.asyncio
    async def test_initialize_session_model_not_found(self):
        """Test initialization warns when model not on any server."""
        from prompt_prix.main import initialize_session

        # Servers respond but without the requested model
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )

        result = await initialize_session(
            servers_text=MOCK_SERVER_1,
            models_text="nonexistent-model",
            system_prompt_file=None,
            temperature=0.7,
            timeout=300,
            max_tokens=2048
        )

        assert "not found" in result[0].lower() or "⚠️" in result[0]


class TestSendSinglePrompt:
    """Tests for send_single_prompt function."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_single_prompt_success(self):
        """Test sending a single prompt."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        # Setup mocks
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_COMPLETION_RESPONSE)
        )

        # Initialize session
        pool = ServerPool([MOCK_SERVER_1])
        await pool.refresh_all_manifests()
        main.server_pool = pool
        main.session = ComparisonSession(
            models=[MOCK_MODEL_1],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        result = await main.send_single_prompt("Test prompt")

        # Should return success
        assert "✅" in result[0] or "sent" in result[0].lower()

    @pytest.mark.asyncio
    async def test_send_single_prompt_no_session(self):
        """Test sending prompt without initialized session."""
        from prompt_prix import main

        # Ensure no session
        main.session = None

        result = await main.send_single_prompt("Test prompt")

        assert "not initialized" in result[0].lower() or "❌" in result[0]

    @pytest.mark.asyncio
    async def test_send_single_prompt_empty(self):
        """Test sending empty prompt."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        # Setup minimal session
        pool = ServerPool([MOCK_SERVER_1])
        main.server_pool = pool
        main.session = ComparisonSession(
            models=[MOCK_MODEL_1],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        result = await main.send_single_prompt("   ")

        assert "Empty" in result[0] or "❌" in result[0]


class TestExportFunctions:
    """Tests for export_markdown and export_json functions."""

    def test_export_markdown_no_session(self):
        """Test export markdown without session."""
        from prompt_prix import main

        main.session = None

        status, content = main.export_markdown()

        assert "No session" in status or "❌" in status

    def test_export_json_no_session(self):
        """Test export JSON without session."""
        from prompt_prix import main

        main.session = None

        status, content = main.export_json()

        assert "No session" in status or "❌" in status

    def test_export_markdown_with_session(self, tmp_path):
        """Test export markdown with active session."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        # Change to tmp directory for file output
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            pool = ServerPool([MOCK_SERVER_1])
            main.server_pool = pool
            main.session = ComparisonSession(
                models=[MOCK_MODEL_1],
                server_pool=pool,
                system_prompt="Test",
                temperature=0.7,
                timeout_seconds=300,
                max_tokens=2048
            )

            status, content = main.export_markdown()

            assert "✅" in status or "Exported" in status
            assert "# LLM Comparison Report" in content
        finally:
            os.chdir(original_dir)

    def test_export_json_with_session(self, tmp_path):
        """Test export JSON with active session."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession
        import json

        # Change to tmp directory for file output
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            pool = ServerPool([MOCK_SERVER_1])
            main.server_pool = pool
            main.session = ComparisonSession(
                models=[MOCK_MODEL_1],
                server_pool=pool,
                system_prompt="Test",
                temperature=0.7,
                timeout_seconds=300,
                max_tokens=2048
            )

            status, content = main.export_json()

            assert "✅" in status or "Exported" in status
            # Verify valid JSON
            parsed = json.loads(content)
            assert "configuration" in parsed
        finally:
            os.chdir(original_dir)
