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


# Integration tests - require real browser session
@pytest.mark.integration
class TestGeminiIntegration:
    """
    Integration tests that run against real Gemini Web UI.

    Prerequisites:
        prompt-prix-gemini --on  (to establish session)

    Run with:
        pytest tests/test_gemini_adapter.py -m integration -v
    """

    @pytest.mark.asyncio
    async def test_send_prompt_theory_of_mind(self):
        """
        Test sending a theory-of-mind prompt to Gemini.

        This is the prompt from the user's screenshot that timed out.
        """
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        adapter = GeminiWebUIAdapter()

        if not adapter.has_session():
            pytest.skip("No Gemini session. Run: prompt-prix-gemini --on")

        try:
            result = await adapter.send_prompt(
                "Given the data in `user_context` that you are encouraged to remember, "
                "describe your theory of mind for the user"
            )

            assert "response" in result
            assert len(result["response"]) > 0
            print(f"\nResponse length: {len(result['response'])} chars")

            if "thinking_blocks" in result:
                print(f"Thinking blocks: {len(result['thinking_blocks'])}")
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_regenerate_produces_different_output(self):
        """Test that regenerate produces a response (may differ from original)."""
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        adapter = GeminiWebUIAdapter()

        if not adapter.has_session():
            pytest.skip("No Gemini session. Run: prompt-prix-gemini --on")

        try:
            # Send initial prompt
            result1 = await adapter.send_prompt("Generate a random 4-digit number")

            # Regenerate
            result2 = await adapter.regenerate()

            assert "response" in result1
            assert "response" in result2
            print(f"\nOriginal: {result1['response'][:100]}")
            print(f"Regenerated: {result2['response'][:100]}")
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_stability_regeneration_pattern(self):
        """
        End-to-end test: Multiple regenerations for stability analysis.

        This tests the exact pattern used by the Stability tab:
        1. Send initial prompt
        2. Regenerate N times
        3. Collect all responses for variance analysis

        Run with:
            pytest tests/test_gemini_adapter.py::TestGeminiIntegration::test_stability_regeneration_pattern -m integration -v
        """
        import time
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter

        adapter = GeminiWebUIAdapter()

        if not adapter.has_session():
            pytest.skip("No Gemini session. Run: prompt-prix-gemini --on")

        REGEN_COUNT = 3
        PROMPT = (
            "Given the data in `user_context` that you are encouraged to remember, "
            "describe your theory of mind for the user"
        )

        responses = []
        timings = []

        try:
            # Initial prompt
            print(f"\n{'='*60}")
            print(f"STABILITY TEST: {REGEN_COUNT} regenerations")
            print(f"{'='*60}")

            start = time.time()
            result = await adapter.send_prompt(PROMPT)
            elapsed = time.time() - start

            assert "response" in result, "Initial prompt failed"
            responses.append(result["response"])
            timings.append(elapsed)
            print(f"\n[1/{REGEN_COUNT+1}] Initial: {elapsed:.1f}s, {len(result['response'])} chars")

            # Regenerations
            for i in range(REGEN_COUNT):
                start = time.time()
                result = await adapter.regenerate()
                elapsed = time.time() - start

                assert "response" in result, f"Regeneration {i+1} failed"
                responses.append(result["response"])
                timings.append(elapsed)
                print(f"[{i+2}/{REGEN_COUNT+1}] Regen {i+1}: {elapsed:.1f}s, {len(result['response'])} chars")

            # Summary
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Total responses: {len(responses)}")
            print(f"Avg response length: {sum(len(r) for r in responses) / len(responses):.0f} chars")
            print(f"Avg latency: {sum(timings) / len(timings):.1f}s")
            print(f"Total time: {sum(timings):.1f}s")

            # Basic variance check
            unique_responses = set(responses)
            print(f"Unique responses: {len(unique_responses)} of {len(responses)}")

        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_full_stability_handler_e2e(self):
        """
        End-to-end test through the stability handler.

        This tests the actual UI handler path, not just the adapter.

        Run with:
            pytest tests/test_gemini_adapter.py::TestGeminiIntegration::test_full_stability_handler_e2e -m integration -v
        """
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter
        from prompt_prix.tabs.stability.handlers import run_regenerations

        # Check session exists
        adapter = GeminiWebUIAdapter()
        if not adapter.has_session():
            pytest.skip("No Gemini session. Run: prompt-prix-gemini --on")

        PROMPT = (
            "Given the data in `user_context` that you are encouraged to remember, "
            "describe your theory of mind for the user"
        )

        results = []
        print(f"\n{'='*60}")
        print("STABILITY HANDLER E2E TEST")
        print(f"{'='*60}")

        async for result in run_regenerations(
            use_gemini=True,
            model_id=None,
            prompt=PROMPT,
            regen_count=3,
            servers_text="",
            temperature=0.7,
            timeout=300,
            max_tokens=2048,
            system_prompt="",
            capture_thinking=True
        ):
            status = result[0]
            print(f"Status: {status}")
            results.append(result)

        # Should have multiple status updates
        assert len(results) > 0, "No results from handler"

        # Final status should indicate completion or provide responses
        final_status = results[-1][0]
        print(f"\nFinal status: {final_status}")

        # Check we got regeneration outputs (positions 1-20 in the tuple)
        final_outputs = results[-1][1:4]  # First 3 regen outputs
        non_waiting = [o for o in final_outputs if o != "*Waiting...*"]
        print(f"Completed regenerations: {len(non_waiting)}")

        assert len(non_waiting) > 0, "No regenerations completed"


@pytest.mark.integration
class TestGeminiVisualAdapter:
    """
    Integration tests for the visual adapter using Fara-7B.

    Prerequisites:
        - Fara-7B GGUF loaded in LM Studio
        - LM Studio server running at localhost:1234
        - prompt-prix-gemini --on (for browser session)

    Run with:
        pytest tests/test_gemini_adapter.py::TestGeminiVisualAdapter -m integration -v
    """

    @pytest.mark.asyncio
    async def test_fara_locate_element(self):
        """Test that Fara can locate a UI element from a screenshot."""
        import base64
        from prompt_prix.adapters.fara import FaraService

        # Create a simple test image (just to verify the service works)
        # In real use, this would be a browser screenshot
        fara = FaraService()

        # Skip if Fara server not available
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{fara.server_url}/v1/models")
                if response.status_code != 200:
                    pytest.skip("Fara server not available")
        except Exception:
            pytest.skip("Fara server not available at localhost:1234")

        # This is a placeholder - real test would use actual screenshot
        print("\nFara service initialized successfully")
        print(f"Server: {fara.server_url}")
        print(f"Model: {fara.model_id}")

    @pytest.mark.asyncio
    async def test_visual_adapter_send_prompt(self):
        """
        Test sending a prompt using the visual adapter.

        This is the key test - validates Fara-7B can locate UI elements.
        """
        from prompt_prix.adapters.gemini_visual import GeminiVisualAdapter
        from prompt_prix.adapters.fara import FaraService

        # Check Fara server available
        fara = FaraService()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{fara.server_url}/v1/models")
                if response.status_code != 200:
                    pytest.skip("Fara server not available")
        except Exception:
            pytest.skip("Fara server not available")

        adapter = GeminiVisualAdapter()

        if not adapter.has_session():
            pytest.skip("No Gemini session. Run: prompt-prix-gemini --on")

        try:
            print(f"\n{'='*60}")
            print("VISUAL ADAPTER TEST")
            print(f"{'='*60}")

            result = await adapter.send_prompt(
                "What is 2 + 2? Answer with just the number."
            )

            assert "response" in result
            print(f"\nResponse: {result['response'][:200]}")
            print(f"Method: {result.get('method', 'unknown')}")

        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_visual_adapter_regeneration_pattern(self):
        """
        Test multiple regenerations using visual adapter.

        This validates the full stability analysis workflow with Fara-7B.
        """
        import time
        from prompt_prix.adapters.gemini_visual import GeminiVisualAdapter
        from prompt_prix.adapters.fara import FaraService

        # Check Fara server
        fara = FaraService()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{fara.server_url}/v1/models")
                if response.status_code != 200:
                    pytest.skip("Fara server not available")
        except Exception:
            pytest.skip("Fara server not available")

        adapter = GeminiVisualAdapter()

        if not adapter.has_session():
            pytest.skip("No Gemini session. Run: prompt-prix-gemini --on")

        REGEN_COUNT = 2
        PROMPT = "Generate a random number between 1 and 100"

        responses = []
        timings = []

        try:
            print(f"\n{'='*60}")
            print(f"VISUAL ADAPTER STABILITY TEST: {REGEN_COUNT} regenerations")
            print(f"{'='*60}")

            # Initial prompt
            start = time.time()
            result = await adapter.send_prompt(PROMPT)
            elapsed = time.time() - start

            assert "response" in result
            responses.append(result["response"])
            timings.append(elapsed)
            print(f"\n[1/{REGEN_COUNT+1}] Initial: {elapsed:.1f}s")

            # Regenerations
            for i in range(REGEN_COUNT):
                start = time.time()
                result = await adapter.regenerate()
                elapsed = time.time() - start

                assert "response" in result
                responses.append(result["response"])
                timings.append(elapsed)
                print(f"[{i+2}/{REGEN_COUNT+1}] Regen: {elapsed:.1f}s")

            print(f"\nTotal responses: {len(responses)}")
            print(f"Avg latency: {sum(timings)/len(timings):.1f}s")

        finally:
            await adapter.close()


# Helper for async generators in tests
async def async_generator(items):
    """Convert list to async generator."""
    for item in items:
        yield item
