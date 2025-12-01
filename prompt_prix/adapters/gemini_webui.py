"""
Gemini Web UI Adapter using Playwright for browser automation.

This adapter enables interaction with Gemini's web interface to:
1. Submit prompts and receive responses
2. Trigger regeneration of responses
3. Extract thinking blocks (reasoning traces)

Uses Playwright for browser automation since the Gemini API
does not expose thinking blocks.

Reference: docs/adr/ADR-DISTILL-006_GeminiWebUI_Adapter.md
"""

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Check for Playwright availability
try:
    from playwright.async_api import async_playwright, Page, Browser, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "Playwright not installed. Install with: pip install playwright && "
        "playwright install chromium"
    )


class GeminiWebUIAdapter:
    """
    Adapter for Gemini web UI using browser automation.

    Extracts thinking blocks via DOM scraping.
    Supports both initial prompts and regeneration.
    """

    GEMINI_URL = "https://gemini.google.com/app"

    def __init__(self, cookies_path: Optional[str] = None, headless: bool = True):
        """
        Initialize the adapter.

        Args:
            cookies_path: Path to cookies JSON file for session auth.
                         Defaults to ~/.prompt-prix/gemini_cookies.json
            headless: Run browser in headless mode (default True)
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for Gemini Web UI adapter. "
                "Install with: pip install playwright && playwright install chromium"
            )

        self.cookies_path = cookies_path or self._default_cookies_path()
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._initialized = False

    def _default_cookies_path(self) -> str:
        """Get default cookies path."""
        home = Path.home()
        return str(home / ".prompt-prix" / "gemini_cookies.json")

    async def _ensure_initialized(self):
        """Initialize browser if not already done."""
        if self._initialized:
            return

        logger.info("Initializing Playwright browser for Gemini...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

        # Create context and restore cookies if available
        context = await self.browser.new_context()

        if os.path.exists(self.cookies_path):
            import json
            with open(self.cookies_path, 'r') as f:
                cookies = json.load(f)
            await context.add_cookies(cookies)
            logger.info(f"Restored session from {self.cookies_path}")
        else:
            logger.warning(
                f"No cookies file at {self.cookies_path}. "
                "You may need to log in manually."
            )

        self.page = await context.new_page()
        await self.page.goto(self.GEMINI_URL)
        await self.page.wait_for_load_state("networkidle")

        self._initialized = True
        logger.info("Gemini browser initialized")

    async def send_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Send a prompt to Gemini and get the response.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt (prepended to user prompt)

        Returns:
            Dict with 'response' and optionally 'thinking_blocks'
        """
        await self._ensure_initialized()

        # Combine system and user prompt if needed
        full_prompt = prompt
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt.strip()}\n\n{prompt}"

        # Find and fill textarea
        textarea = await self.page.wait_for_selector("textarea", timeout=10000)
        await textarea.fill(full_prompt)

        # Click send button
        send_button = await self.page.wait_for_selector(
            'button[aria-label*="Send"], button[type="submit"]',
            timeout=5000
        )
        await send_button.click()

        # Wait for response to complete
        await self._wait_for_response_complete()

        # Extract response and thinking blocks
        return await self._extract_response()

    async def regenerate(self) -> dict:
        """
        Trigger regeneration of the last response.

        Returns:
            Dict with 'response' and optionally 'thinking_blocks'
        """
        await self._ensure_initialized()

        # Find and click regenerate button
        # The regenerate button is typically in the response area
        regen_button = await self.page.wait_for_selector(
            'button[aria-label*="Regenerate"], button[aria-label*="regenerate"], '
            'button[data-test-id*="regenerate"]',
            timeout=10000
        )
        await regen_button.click()

        # Wait for new response
        await self._wait_for_response_complete()

        # Extract response and thinking blocks
        return await self._extract_response()

    async def _wait_for_response_complete(self):
        """Wait for Gemini to finish generating response."""
        # Wait for conversation container to appear
        await self.page.wait_for_selector(
            ".conversation-container:last-child",
            timeout=60000
        )

        # Wait for "Stop generating" button to disappear
        try:
            stop_button = await self.page.wait_for_selector(
                'button[aria-label*="Stop"]',
                timeout=5000
            )
            await stop_button.wait_for(state="detached", timeout=120000)
        except:
            # No stop button, response might already be complete
            pass

        # Wait for rendering to stabilize
        await asyncio.sleep(2)

    async def _expand_thinking_blocks(self):
        """Click 'Show thinking' buttons to expand thinking blocks."""
        max_passes = 10

        for pass_num in range(max_passes):
            buttons = await self.page.query_selector_all(
                'button[data-test-id="thoughts-header-button"]'
            )

            unexpanded = []
            for btn in buttons:
                text = await btn.inner_text()
                if "show thinking" in text.lower():
                    unexpanded.append(btn)

            if not unexpanded:
                logger.debug(f"Thinking blocks expanded after {pass_num + 1} passes")
                break

            for button in unexpanded:
                try:
                    await button.scroll_into_view_if_needed()
                    await button.click()
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Error expanding thinking button: {e}")

            await asyncio.sleep(1)

    async def _extract_response(self) -> dict:
        """Extract response text and thinking blocks from the page."""
        # Expand thinking blocks first
        await self._expand_thinking_blocks()

        # Get the most recent conversation container
        container = await self.page.query_selector(".conversation-container:last-child")
        if not container:
            raise RuntimeError("No conversation container found")

        # Extract thinking blocks
        thinking_blocks = await self._extract_thinking(container)

        # Extract response text (excluding thinking)
        response_text = await self._extract_response_text(container)

        return {
            "response": response_text,
            "thinking_blocks": thinking_blocks,
        }

    async def _extract_thinking(self, container) -> Optional[list[dict]]:
        """Extract thinking stages from container."""
        thinking_container = await container.query_selector(
            '[data-test-id="model-thoughts"]'
        )
        if not thinking_container:
            return None

        stages = []
        elements = await thinking_container.query_selector_all("p, div")
        current_stage = None

        for el in elements:
            text = (await el.inner_text()).strip()
            if not text:
                continue

            # Check if this is a stage header (bold text)
            bold_el = await el.query_selector("strong, b")
            if bold_el:
                bold_text = (await bold_el.inner_text()).strip()
                if text == bold_text:
                    # Save previous stage
                    if current_stage and current_stage["text"].strip():
                        stages.append(current_stage)
                    current_stage = {"stage_name": text, "text": ""}
                    continue

            if current_stage:
                if current_stage["text"]:
                    current_stage["text"] += "\n\n"
                current_stage["text"] += text

        # Don't forget the last stage
        if current_stage and current_stage["text"].strip():
            stages.append(current_stage)

        return stages if stages else None

    async def _extract_response_text(self, container) -> str:
        """Extract response text, excluding thinking blocks."""
        markdown_panel = await container.query_selector(".markdown-main-panel")
        if not markdown_panel:
            raise RuntimeError("No response markdown panel found")

        # Use JavaScript to clone and clean the content
        response_text = await markdown_panel.evaluate(
            """
            (element) => {
                const clone = element.cloneNode(true);

                // Remove thinking containers
                clone.querySelectorAll('[data-test-id="model-thoughts"]').forEach(el => el.remove());
                clone.querySelectorAll('button[data-test-id="thoughts-header-button"]').forEach(el => el.remove());

                // Remove UI buttons but keep code blocks
                clone.querySelectorAll('button:not(pre button)').forEach(el => {
                    if (!el.closest('pre, code')) {
                        el.remove();
                    }
                });

                return clone.innerText;
            }
            """
        )

        return response_text.strip()

    async def close(self):
        """Close browser and cleanup resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._initialized = False

    def __del__(self):
        """Cleanup on deletion."""
        # Can't use async in __del__, so just log a warning if not cleaned up
        if self._initialized:
            logger.warning(
                "GeminiWebUIAdapter was not properly closed. "
                "Call await adapter.close() before deletion."
            )
