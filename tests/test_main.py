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
        """Test fallback prompt when no file provided."""
        from prompt_prix.main import load_system_prompt

        result = load_system_prompt(None)

        # Returns either DEFAULT_SYSTEM_PROMPT or content from system_prompt.txt if it exists
        assert len(result) > 0

    def test_load_system_prompt_nonexistent_file(self):
        """Test fallback prompt when file doesn't exist."""
        from prompt_prix.main import load_system_prompt

        result = load_system_prompt("/nonexistent/path/file.txt")

        # Returns either DEFAULT_SYSTEM_PROMPT or content from system_prompt.txt if it exists
        assert len(result) > 0


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
        models_selected = [MOCK_MODEL_1, MOCK_MODEL_2]

        result = await initialize_session(
            servers_text=servers_text,
            models_selected=models_selected,
            system_prompt_text="You are a test assistant.",
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
            models_selected=[MOCK_MODEL_1],
            system_prompt_text="",
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
            models_selected=[],
            system_prompt_text="",
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
            models_selected=["nonexistent-model"],
            system_prompt_text="",
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
        main.state.server_pool = pool
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # send_single_prompt is now an async generator for streaming
        result = None
        async for update in main.send_single_prompt("Test prompt"):
            result = update

        # Should return success
        assert "✅" in result[0] or "complete" in result[0].lower()

    @pytest.mark.asyncio
    async def test_send_single_prompt_no_session(self):
        """Test sending prompt without initialized session."""
        from prompt_prix import main

        # Ensure no session
        main.state.session = None

        # Consume the generator
        result = None
        async for update in main.send_single_prompt("Test prompt"):
            result = update

        assert "not initialized" in result[0].lower() or "❌" in result[0]

    @pytest.mark.asyncio
    async def test_send_single_prompt_empty(self):
        """Test sending empty prompt."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        # Setup minimal session
        pool = ServerPool([MOCK_SERVER_1])
        main.state.server_pool = pool
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # Consume the generator
        result = None
        async for update in main.send_single_prompt("   "):
            result = update

        assert "Empty" in result[0] or "❌" in result[0]


class TestExportFunctions:
    """Tests for export_markdown and export_json functions."""

    def test_export_markdown_no_session(self):
        """Test export markdown without session."""
        from prompt_prix import main

        main.state.session = None

        status, content = main.export_markdown()

        assert "No session" in status or "❌" in status

    def test_export_json_no_session(self):
        """Test export JSON without session."""
        from prompt_prix import main

        main.state.session = None

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
            main.state.server_pool = pool
            main.state.session = ComparisonSession(
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
            main.state.server_pool = pool
            main.state.session = ComparisonSession(
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


class TestStreamingOutputNoDuplication:
    """Tests to ensure streaming output doesn't duplicate messages."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_streaming_no_user_message_duplication(self):
        """Test that user message is not duplicated during streaming."""
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
        main.state.server_pool = pool
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # Collect all outputs during streaming
        outputs = []
        async for update in main.send_single_prompt("Hello world"):
            outputs.append(update)

        # Check final output - should have exactly one **User:** and one **Assistant:**
        # Output format: (status, tab_states, model1_output, model2_output, ...)
        final_output = outputs[-1][2]  # First model's output (index 2 after status and tab_states)
        user_count = final_output.count("**User:**")
        assistant_count = final_output.count("**Assistant:**")

        assert user_count == 1, f"Expected 1 **User:**, got {user_count}. Output: {final_output}"
        assert assistant_count == 1, f"Expected 1 **Assistant:**, got {assistant_count}. Output: {final_output}"

    @respx.mock
    @pytest.mark.asyncio
    async def test_streaming_intermediate_output_no_duplication(self):
        """Test intermediate streaming updates don't duplicate messages."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        # Setup mocks with streaming response
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        from tests.conftest import MOCK_STREAMING_CHUNKS
        streaming_content = "\n".join(MOCK_STREAMING_CHUNKS) + "\n"
        respx.post(f"{MOCK_SERVER_1}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=streaming_content)
        )

        # Initialize session
        pool = ServerPool([MOCK_SERVER_1])
        await pool.refresh_all_manifests()
        main.state.server_pool = pool
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # Collect all outputs during streaming
        outputs = []
        async for update in main.send_single_prompt("Test prompt"):
            outputs.append(update)

        # Check ALL intermediate outputs - none should have duplicate **User:**
        # Output format: (status, tab_states, model1_output, model2_output, ...)
        for i, output in enumerate(outputs):
            model_output = output[2]  # First model's output (index 2 after status and tab_states)
            if model_output and isinstance(model_output, str):  # Skip empty outputs
                user_count = model_output.count("**User:**")
                assert user_count <= 1, f"Output {i} has {user_count} **User:** tags. Output: {model_output}"


class TestLaunchBeyondCompare:
    """Tests for launch_beyond_compare function."""

    def test_launch_beyond_compare_no_session(self):
        """Test Beyond Compare fails without session."""
        from prompt_prix import main

        main.state.session = None

        result = main.launch_beyond_compare(MOCK_MODEL_1, MOCK_MODEL_2)

        assert "No session" in result or "❌" in result

    def test_launch_beyond_compare_missing_model_selection(self):
        """Test Beyond Compare fails when models not selected."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool([MOCK_SERVER_1])
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1, MOCK_MODEL_2],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        result = main.launch_beyond_compare("", MOCK_MODEL_2)
        assert "Select two models" in result or "❌" in result

        result = main.launch_beyond_compare(MOCK_MODEL_1, "")
        assert "Select two models" in result or "❌" in result

    def test_launch_beyond_compare_same_model(self):
        """Test Beyond Compare fails when same model selected twice."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool([MOCK_SERVER_1])
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1, MOCK_MODEL_2],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        result = main.launch_beyond_compare(MOCK_MODEL_1, MOCK_MODEL_1)

        assert "different models" in result or "❌" in result

    def test_launch_beyond_compare_model_not_in_session(self):
        """Test Beyond Compare fails when model not in session."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool([MOCK_SERVER_1])
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1],  # Only one model
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        result = main.launch_beyond_compare(MOCK_MODEL_1, "nonexistent-model")

        assert "not in session" in result or "❌" in result

    def test_launch_beyond_compare_no_content(self):
        """Test Beyond Compare fails when no conversation content."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool([MOCK_SERVER_1])
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1, MOCK_MODEL_2],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )
        # Contexts are empty by default

        result = main.launch_beyond_compare(MOCK_MODEL_1, MOCK_MODEL_2)

        assert "No conversation content" in result or "❌" in result

    def test_launch_beyond_compare_executable_not_found(self, monkeypatch):
        """Test Beyond Compare handles missing executable gracefully."""
        from prompt_prix import main
        from prompt_prix.core import ServerPool, ComparisonSession

        pool = ServerPool([MOCK_SERVER_1])
        main.state.session = ComparisonSession(
            models=[MOCK_MODEL_1, MOCK_MODEL_2],
            server_pool=pool,
            system_prompt="Test",
            temperature=0.7,
            timeout_seconds=300,
            max_tokens=2048
        )

        # Add some content to contexts
        main.state.session.state.contexts[MOCK_MODEL_1].add_user_message("Hello")
        main.state.session.state.contexts[MOCK_MODEL_1].add_assistant_message("Hi there!")
        main.state.session.state.contexts[MOCK_MODEL_2].add_user_message("Hello")
        main.state.session.state.contexts[MOCK_MODEL_2].add_assistant_message("Greetings!")

        # Mock get_beyond_compare_path to return non-existent path
        monkeypatch.setattr(
            "prompt_prix.config.get_beyond_compare_path",
            lambda: "/nonexistent/path/bcompare"
        )

        result = main.launch_beyond_compare(MOCK_MODEL_1, MOCK_MODEL_2)

        assert "Beyond Compare not found" in result or "❌" in result


class TestFetchAvailableModels:
    """Tests for fetch_available_models function."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_available_models_success(self):
        """Test fetching models from servers."""
        from prompt_prix.main import fetch_available_models

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        status, models_update = await fetch_available_models(MOCK_SERVER_1)

        assert "✅" in status
        # Returns gr.update dict with choices
        assert "choices" in models_update
        assert MOCK_MODEL_1 in models_update["choices"]
        assert MOCK_MODEL_2 in models_update["choices"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_available_models_no_servers(self):
        """Test fetch fails with no servers configured."""
        from prompt_prix.main import fetch_available_models

        status, models_update = await fetch_available_models("")

        assert "No servers" in status or "❌" in status
        # Returns gr.update dict with empty choices
        assert "choices" in models_update
        assert models_update["choices"] == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_available_models_server_down(self):
        """Test fetch handles unreachable servers."""
        from prompt_prix.main import fetch_available_models

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        status, models_update = await fetch_available_models(MOCK_SERVER_1)

        assert "No models found" in status or "⚠️" in status
        # Returns gr.update dict with empty choices
        assert "choices" in models_update
        assert models_update["choices"] == []


class TestOnlyLoadedFilter:
    """Tests for Only Loaded models filter functionality."""

    def _create_mock_lmstudio(self, models_list):
        """Create a mock lmstudio module with list_loaded_models."""
        mock_lms = MagicMock()
        mock_lms.list_loaded_models.return_value = models_list
        return mock_lms

    def test_get_loaded_models_with_model_key(self):
        """Test _get_loaded_models extracts model_key attribute."""
        from prompt_prix.handlers import _get_loaded_models

        # Create mock model objects with model_key attribute
        mock_model = MagicMock()
        mock_model.model_key = "test-model-1"

        mock_lms = self._create_mock_lmstudio([mock_model])

        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            result = _get_loaded_models()

        assert "test-model-1" in result
        mock_lms.list_loaded_models.assert_called_once_with("llm")

    def test_get_loaded_models_with_path(self):
        """Test _get_loaded_models falls back to path attribute."""
        from prompt_prix.handlers import _get_loaded_models

        # Create mock model without model_key but with path
        mock_model = MagicMock(spec=["path"])
        mock_model.path = "models/test-model-2"

        mock_lms = self._create_mock_lmstudio([mock_model])

        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            result = _get_loaded_models()

        assert "models/test-model-2" in result

    def test_get_loaded_models_with_identifier(self):
        """Test _get_loaded_models falls back to identifier attribute."""
        from prompt_prix.handlers import _get_loaded_models

        # Create mock model without model_key or path but with identifier
        mock_model = MagicMock(spec=["identifier"])
        mock_model.identifier = "test-model-id"

        mock_lms = self._create_mock_lmstudio([mock_model])

        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            result = _get_loaded_models()

        assert "test-model-id" in result

    def test_get_loaded_models_import_error(self):
        """Test _get_loaded_models returns empty set when lmstudio not installed."""
        from prompt_prix.handlers import _get_loaded_models
        import sys

        # Remove lmstudio from modules if it exists
        lmstudio_backup = sys.modules.get("lmstudio")
        sys.modules["lmstudio"] = None  # Simulate ImportError

        try:
            result = _get_loaded_models()
            # Should return empty set on import error
            assert result == set()
        finally:
            # Restore
            if lmstudio_backup:
                sys.modules["lmstudio"] = lmstudio_backup
            elif "lmstudio" in sys.modules:
                del sys.modules["lmstudio"]

    def test_get_loaded_models_sdk_exception(self):
        """Test _get_loaded_models returns empty set on SDK exception."""
        from prompt_prix.handlers import _get_loaded_models

        mock_lms = MagicMock()
        mock_lms.list_loaded_models.side_effect = RuntimeError("SDK error")

        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            result = _get_loaded_models()

        assert result == set()

    def test_get_loaded_models_multiple_models(self):
        """Test _get_loaded_models handles multiple loaded models."""
        from prompt_prix.handlers import _get_loaded_models

        mock_model_1 = MagicMock()
        mock_model_1.model_key = "model-a"
        mock_model_2 = MagicMock()
        mock_model_2.model_key = "model-b"
        mock_model_3 = MagicMock()
        mock_model_3.model_key = "model-c"

        mock_lms = self._create_mock_lmstudio([mock_model_1, mock_model_2, mock_model_3])

        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            result = _get_loaded_models()

        assert len(result) == 3
        assert "model-a" in result
        assert "model-b" in result
        assert "model-c" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_only_loaded_filters_models(self):
        """Test fetch with only_loaded=True filters to loaded models only."""
        from prompt_prix.handlers import fetch_available_models

        # Server has models A, B, C available
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"id": "model-a"},
                    {"id": "model-b"},
                    {"id": "model-c"}
                ]
            })
        )

        # Only model-a and model-c are loaded
        mock_model_a = MagicMock()
        mock_model_a.model_key = "model-a"
        mock_model_c = MagicMock()
        mock_model_c.model_key = "model-c"

        mock_lms = self._create_mock_lmstudio([mock_model_a, mock_model_c])

        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            status, models_update = await fetch_available_models(MOCK_SERVER_1, only_loaded=True)

        assert "✅" in status
        assert "(loaded only)" in status
        choices = models_update["choices"]
        assert "model-a" in choices
        assert "model-c" in choices
        assert "model-b" not in choices  # Not loaded, should be filtered out

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_only_loaded_no_match(self):
        """Test fetch with only_loaded=True when no loaded models match."""
        from prompt_prix.handlers import fetch_available_models

        # Server has models A, B available
        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"id": "model-a"},
                    {"id": "model-b"}
                ]
            })
        )

        # Only model-x is loaded (not on server)
        mock_model_x = MagicMock()
        mock_model_x.model_key = "model-x"

        mock_lms = self._create_mock_lmstudio([mock_model_x])

        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            status, models_update = await fetch_available_models(MOCK_SERVER_1, only_loaded=True)

        assert "⚠️" in status
        assert "No loaded models match" in status
        assert models_update["choices"] == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_only_loaded_sdk_unavailable(self):
        """Test fetch with only_loaded=True when SDK not available."""
        from prompt_prix.handlers import fetch_available_models

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json=MOCK_MANIFEST_RESPONSE)
        )

        with patch("prompt_prix.handlers._get_loaded_models", return_value=set()):
            status, models_update = await fetch_available_models(MOCK_SERVER_1, only_loaded=True)

        assert "⚠️" in status
        assert "Could not detect loaded models" in status
        # Should still return all models as fallback
        assert len(models_update["choices"]) > 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_without_only_loaded_returns_all(self):
        """Test fetch with only_loaded=False returns all available models."""
        from prompt_prix.handlers import fetch_available_models

        respx.get(f"{MOCK_SERVER_1}/v1/models").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"id": "model-a"},
                    {"id": "model-b"},
                    {"id": "model-c"}
                ]
            })
        )

        # Even with lmstudio available, should not call it when only_loaded=False
        status, models_update = await fetch_available_models(MOCK_SERVER_1, only_loaded=False)

        assert "✅" in status
        assert "(loaded only)" not in status
        choices = models_update["choices"]
        assert "model-a" in choices
        assert "model-b" in choices
        assert "model-c" in choices
