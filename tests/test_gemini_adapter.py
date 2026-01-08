"""
Tests for Gemini Web UI adapter.

Unit tests mock Playwright; integration tests require:
    pip install prompt-prix[gemini]
    playwright install chromium
    prompt-prix-gemini --on  (to set up session)

Run integration tests:
    pytest tests/test_gemini_adapter.py -m integration -v
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# Skip entire module if playwright not installed
pytest.importorskip("playwright")


class TestGeminiWebUIAdapter:
    """Unit tests for GeminiWebUIAdapter with mocked Playwright."""

    def test_default_state_dir(self):
        """State directory defaults to ~/.prompt-prix/gemini_state/"""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        adapter = GeminiWebUIAdapter()
        expected = Path.home() / ".prompt-prix" / "gemini_state"
        assert adapter.state_dir == expected

    def test_state_file_path(self):
        """State file is state.json in state directory."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        adapter = GeminiWebUIAdapter(state_dir="/tmp/test_gemini")
        assert adapter.state_file == Path("/tmp/test_gemini/state.json")

    def test_has_session_false_when_no_file(self, tmp_path):
        """has_session returns False when state file doesn't exist."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        adapter = GeminiWebUIAdapter(state_dir=str(tmp_path))
        assert adapter.has_session() is False

    def test_has_session_true_when_file_exists(self, tmp_path):
        """has_session returns True when state file exists."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        state_file = tmp_path / "state.json"
        state_file.write_text("{}")

        adapter = GeminiWebUIAdapter(state_dir=str(tmp_path))
        assert adapter.has_session() is True

    def test_headless_auto_false_without_session(self, tmp_path):
        """Auto headless is False when no session exists (need visible login)."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        adapter = GeminiWebUIAdapter(state_dir=str(tmp_path))
        assert adapter.headless is False

    def test_headless_auto_true_with_session(self, tmp_path):
        """Auto headless is True when session exists."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        state_file = tmp_path / "state.json"
        state_file.write_text("{}")

        adapter = GeminiWebUIAdapter(state_dir=str(tmp_path))
        assert adapter.headless is True

    def test_headless_override(self, tmp_path):
        """Explicit headless parameter overrides auto detection."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        # Has session but force visible
        state_file = tmp_path / "state.json"
        state_file.write_text("{}")

        adapter = GeminiWebUIAdapter(state_dir=str(tmp_path), headless=False)
        assert adapter.headless is False

    def test_clear_session(self, tmp_path):
        """clear_session removes state file."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        state_file = tmp_path / "state.json"
        state_file.write_text("{}")

        adapter = GeminiWebUIAdapter(state_dir=str(tmp_path))
        assert adapter.has_session() is True

        adapter.clear_session()
        assert adapter.has_session() is False


class TestGeminiSelectors:
    """Tests for DOM selector logic and timeout handling."""

    @pytest.fixture
    def mock_page(self):
        """Create a mock Playwright page."""
        page = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.query_selector_all = AsyncMock(return_value=[])
        page.evaluate = AsyncMock(return_value="Test response")
        return page

    @pytest.fixture
    def mock_adapter(self, tmp_path, mock_page):
        """Create adapter with mocked browser."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        # Create session file so it thinks we're logged in
        state_file = tmp_path / "state.json"
        state_file.write_text("{}")

        adapter = GeminiWebUIAdapter(state_dir=str(tmp_path))
        adapter._initialized = True
        adapter.page = mock_page
        return adapter

    async def test_send_prompt_textarea_timeout(self, mock_adapter):
        """send_prompt raises on textarea timeout."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        mock_adapter.page.wait_for_selector.side_effect = PlaywrightTimeout("Timeout 10000ms exceeded")

        with pytest.raises(PlaywrightTimeout):
            await mock_adapter.send_prompt("test prompt")

    async def test_send_prompt_send_button_selector(self, mock_adapter):
        """send_prompt looks for send button with correct selectors."""
        textarea_mock = AsyncMock()

        # First call returns textarea, second returns send button
        mock_adapter.page.wait_for_selector.side_effect = [
            textarea_mock,  # textarea
            AsyncMock(),    # send button
        ]

        # Mock _wait_for_response_complete and _extract_response
        mock_adapter._wait_for_response_complete = AsyncMock()
        mock_adapter._extract_response = AsyncMock(return_value={"response": "test"})

        await mock_adapter.send_prompt("test prompt")

        # Verify send button selector was used
        calls = mock_adapter.page.wait_for_selector.call_args_list
        assert len(calls) >= 2
        send_button_call = calls[1]
        selector = send_button_call[0][0]
        assert 'button[aria-label*="Send"]' in selector or 'button[type="submit"]' in selector

    async def test_regenerate_button_selector(self, mock_adapter):
        """regenerate looks for regenerate button with correct selectors."""
        regen_button_mock = AsyncMock()
        mock_adapter.page.wait_for_selector.return_value = regen_button_mock

        mock_adapter._wait_for_response_complete = AsyncMock()
        mock_adapter._extract_response = AsyncMock(return_value={"response": "regenerated"})

        await mock_adapter.regenerate()

        # Verify regenerate button selector
        calls = mock_adapter.page.wait_for_selector.call_args_list
        selector = calls[0][0][0]
        assert "Regenerate" in selector or "regenerate" in selector

    async def test_wait_for_response_timeout(self, mock_adapter):
        """_wait_for_response_complete uses appropriate timeouts."""
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        mock_adapter.page.wait_for_selector.side_effect = PlaywrightTimeout("Timeout 60000ms exceeded")

        with pytest.raises(PlaywrightTimeout):
            await mock_adapter._wait_for_response_complete()


class TestStabilityHandlerGeminiPath:
    """Tests for stability handler Gemini integration."""

    async def test_use_gemini_bypasses_model_validation(self):
        """When use_gemini=True, model dropdown is bypassed."""
        from prompt_prix.tabs.stability.handlers import run_regenerations

        # Patch the Gemini regeneration function
        with patch('prompt_prix.tabs.stability.handlers._run_gemini_regenerations') as mock_gemini:
            mock_gemini.return_value = async_generator([
                ("Completed",) + tuple(["response"] * 20)
            ])

            results = []
            async for result in run_regenerations(
                use_gemini=True,
                model_id=None,  # No model selected
                prompt="test prompt",
                regen_count=1,
                servers_text="",
                temperature=0.7,
                timeout=300,
                max_tokens=2048,
                system_prompt="",
                capture_thinking=True
            ):
                results.append(result)

            # Should have called Gemini handler
            mock_gemini.assert_called_once()

    async def test_empty_prompt_rejected(self):
        """Empty prompt returns validation error."""
        from prompt_prix.tabs.stability.handlers import run_regenerations

        results = []
        async for result in run_regenerations(
            use_gemini=True,
            model_id=None,
            prompt="",  # Empty
            regen_count=5,
            servers_text="",
            temperature=0.7,
            timeout=300,
            max_tokens=2048,
            system_prompt="",
            capture_thinking=True
        ):
            results.append(result)

        assert len(results) == 1
        assert "Enter a prompt" in results[0][0]

    async def test_gemini_model_id_set_when_checkbox_true(self):
        """When use_gemini=True, model_id is set to Gemini model name."""
        from prompt_prix.tabs.stability.handlers import run_regenerations, stability_run

        with patch('prompt_prix.tabs.stability.handlers._run_gemini_regenerations') as mock_gemini:
            mock_gemini.return_value = async_generator([
                ("Completed",) + tuple(["response"] * 20)
            ])

            async for _ in run_regenerations(
                use_gemini=True,
                model_id=None,
                prompt="test",
                regen_count=1,
                servers_text="",
                temperature=0.7,
                timeout=300,
                max_tokens=2048,
                system_prompt="",
                capture_thinking=True
            ):
                pass

            # Check stability_run was created with Gemini model
            from prompt_prix.tabs.stability import handlers
            assert handlers.stability_run is not None
            assert "gemini" in handlers.stability_run.model_id.lower()


# Helper for async generators in tests
async def async_generator(items):
    """Convert list to async generator."""
    for item in items:
        yield item


# NOTE: Integration tests moved to tests/deferred/test_gemini_integration.py
# They require external services (Gemini session, Fara-7B) and are out of scope
# for the current release. See tests/deferred/README.md for details.
