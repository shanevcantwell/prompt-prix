"""
GeminiVisualAdapter - Gemini Web UI adapter using visual element location.

Uses Microsoft Fara-7B to locate UI elements instead of brittle DOM selectors.
This approach survives UI redesigns since it uses natural language descriptions.

Architecture:
    Playwright: Browser control (navigation, clicking, typing, screenshots)
    Fara-7B: Element location via vision model ("locate the send button")

Ref: https://github.com/microsoft/fara

Usage:
    adapter = GeminiVisualAdapter(
        fara_server="http://localhost:1234",
        fara_model="fara-7b"
    )
    await adapter.send_prompt("Hello, Gemini!")
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Optional

from .fara import FaraService

logger = logging.getLogger(__name__)

# Check Playwright availability
try:
    from playwright.async_api import async_playwright, Playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Playwright = None
    Browser = None
    Page = None


class GeminiVisualAdapter:
    """
    Gemini Web UI adapter using visual element location.

    Element descriptions (not selectors!):
        prompt_input: "The text input area at the bottom"
        send_button: "The send or submit button"
        response_area: "Gemini's response content"
        regenerate_button: "Regenerate response button"
        loading_indicator: "Loading or generating indicator"
    """

    GEMINI_URL = "https://gemini.google.com/app"

    # Visual element descriptions
    ELEMENTS = {
        "prompt_input": "The message input textarea at the bottom of the page",
        "send_button": "The send message button, usually an arrow icon",
        "response_area": "The most recent assistant response message",
        "regenerate_button": "The regenerate or retry button near the response",
        "loading_indicator": "Loading spinner or generating animation",
        "stop_button": "Stop generating button",
    }

    def __init__(
        self,
        fara_server: Optional[str] = None,
        fara_model: Optional[str] = None,
        state_dir: Optional[str] = None,
        headless: Optional[bool] = None,
    ):
        """
        Initialize Gemini visual adapter.

        Args:
            fara_server: Fara server URL (default from FARA_SERVER_URL env var)
            fara_model: Fara model ID (default from FARA_MODEL_ID env var)
            state_dir: Browser state directory (default: ~/.prompt-prix/gemini_state)
            headless: Run browser headless (default: True if session exists)
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright required. Install with: pip install playwright && playwright install chromium"
            )

        # FaraService resolves defaults from config if None
        self.fara = FaraService(server_url=fara_server, model_id=fara_model)
        self.state_dir = Path(state_dir) if state_dir else self._default_state_dir()
        self._headless_override = headless

        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._initialized = False

    def _default_state_dir(self) -> Path:
        return Path.home() / ".prompt-prix" / "gemini_state"

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

    def has_session(self) -> bool:
        return self.state_file.exists()

    @property
    def headless(self) -> bool:
        if self._headless_override is not None:
            return self._headless_override
        return self.has_session()

    async def _ensure_initialized(self):
        """Initialize browser if needed."""
        if self._initialized:
            return

        self.state_dir.mkdir(parents=True, exist_ok=True)
        has_session = self.has_session()

        logger.info(f"Initializing browser (headless={self.headless})...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )

        if has_session:
            context = await self.browser.new_context(storage_state=str(self.state_file))
            logger.info(f"Restored session from {self.state_file}")
        else:
            context = await self.browser.new_context()

        self.page = await context.new_page()
        await self.page.goto(self.GEMINI_URL)
        await self.page.wait_for_load_state("networkidle")

        if not has_session:
            await self._wait_for_login_visual()
            await context.storage_state(path=str(self.state_file))
            logger.info(f"Session saved to {self.state_file}")

        self._initialized = True

    async def _wait_for_login_visual(self):
        """Wait for login using visual detection."""
        logger.info("Waiting for login... (detecting input area)")

        for _ in range(60):  # 5 minutes max
            screenshot = await self._take_screenshot()
            result = await self.fara.verify(self.ELEMENTS["prompt_input"], screenshot)

            if result.get("exists"):
                logger.info("Login detected via visual check")
                await asyncio.sleep(2)  # Stabilize
                return

            await asyncio.sleep(5)

        raise RuntimeError("Login timeout - input area not detected after 5 minutes")

    async def _take_screenshot(self) -> str:
        """Take screenshot and return as base64."""
        screenshot_bytes = await self.page.screenshot()
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def _locate_and_click(self, element_key: str) -> bool:
        """Locate element visually and execute Fara's suggested action."""
        description = self.ELEMENTS.get(element_key, element_key)
        screenshot = await self._take_screenshot()

        logger.info(f"Locating element: {description}")
        result = await self.fara.locate(description, screenshot)
        logger.info(f"Fara result: {result}")

        if not result.get("found"):
            logger.warning(f"Could not locate: {description}")
            return False

        await self._execute_playwright_action(result)
        return True

    async def _execute_playwright_action(self, action: dict) -> None:
        """Execute a Playwright action from Fara's response.

        Fara outputs standard Playwright actions:
            left_click, right_click, double_click, type, scroll, key, etc.
        """
        action_type = action.get("action", "left_click")
        x = action.get("x")
        y = action.get("y")

        logger.info(f"Executing Playwright action: {action_type} at ({x}, {y})")

        if action_type == "left_click" and x is not None and y is not None:
            await self.page.mouse.click(x, y)

        elif action_type == "right_click" and x is not None and y is not None:
            await self.page.mouse.click(x, y, button="right")

        elif action_type == "double_click" and x is not None and y is not None:
            await self.page.mouse.dblclick(x, y)

        elif action_type == "type":
            text = action.get("text", "")
            if x is not None and y is not None:
                await self.page.mouse.click(x, y)
            await self.page.keyboard.type(text)

        elif action_type == "key":
            key = action.get("text", action.get("key", ""))
            await self.page.keyboard.press(key)

        elif action_type == "scroll":
            direction = action.get("direction", "down")
            delta = 300 if direction == "down" else -300
            if x is not None and y is not None:
                await self.page.mouse.move(x, y)
            await self.page.mouse.wheel(0, delta)

        elif action_type == "hover" and x is not None and y is not None:
            await self.page.mouse.move(x, y)

        else:
            # Default to click if we have coordinates
            if x is not None and y is not None:
                await self.page.mouse.click(x, y)
            else:
                logger.warning(f"Unknown action or missing coordinates: {action}")

    async def send_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Send a prompt to Gemini using visual navigation.

        Args:
            prompt: User prompt to send
            system_prompt: Optional system prompt (prepended)

        Returns:
            Dict with 'response' and optionally 'thinking_blocks'
        """
        await self._ensure_initialized()

        full_prompt = prompt
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt.strip()}\n\n{prompt}"

        # Locate and click input area
        if not await self._locate_and_click("prompt_input"):
            raise RuntimeError("Could not find prompt input area")

        # Type the prompt
        await self.page.keyboard.type(full_prompt)
        await asyncio.sleep(0.5)

        # Locate and click send button
        if not await self._locate_and_click("send_button"):
            # Fallback: try Enter key
            logger.info("Send button not found, trying Enter key")
            await self.page.keyboard.press("Enter")

        # Wait for response to complete
        await self._wait_for_response_visual()

        # Extract response
        return await self._extract_response_visual()

    async def regenerate(self) -> dict:
        """Trigger regeneration using visual navigation."""
        await self._ensure_initialized()

        if not await self._locate_and_click("regenerate_button"):
            raise RuntimeError("Could not find regenerate button")

        await self._wait_for_response_visual()
        return await self._extract_response_visual()

    async def _wait_for_response_visual(self, timeout: float = 120.0):
        """Wait for response using visual detection."""
        logger.info("Waiting for response...")

        start = asyncio.get_event_loop().time()

        # First, wait for loading indicator to appear
        for _ in range(10):
            screenshot = await self._take_screenshot()
            result = await self.fara.verify(self.ELEMENTS["loading_indicator"], screenshot)
            if result.get("exists"):
                logger.debug("Loading indicator detected")
                break
            await asyncio.sleep(0.5)

        # Then wait for it to disappear
        while (asyncio.get_event_loop().time() - start) < timeout:
            screenshot = await self._take_screenshot()
            result = await self.fara.verify(self.ELEMENTS["loading_indicator"], screenshot)

            if not result.get("exists"):
                logger.info("Response complete (loading indicator gone)")
                await asyncio.sleep(1)  # Brief settle time
                return

            await asyncio.sleep(1)

        logger.warning("Timeout waiting for response")

    async def _extract_response_visual(self) -> dict:
        """
        Extract response text.

        For now, uses DOM extraction since OCR from screenshots would be
        less reliable. This is the "hybrid" approach mentioned in the vision doc.
        """
        # Use JavaScript to extract the last response
        response_text = await self.page.evaluate("""
            () => {
                // Find all message containers
                const messages = document.querySelectorAll('[data-message-id]');
                if (!messages.length) return '';

                // Get the last one (most recent response)
                const lastMessage = messages[messages.length - 1];
                return lastMessage?.innerText || '';
            }
        """)

        return {
            "response": response_text.strip() if response_text else "",
            "method": "visual"
        }

    async def close(self, save_session: bool = True):
        """Close browser."""
        if self.page and save_session:
            try:
                context = self.page.context
                await context.storage_state(path=str(self.state_file))
            except Exception as e:
                logger.warning(f"Failed to save session: {e}")

        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._initialized = False

    def clear_session(self):
        """Clear saved session."""
        if self.state_file.exists():
            self.state_file.unlink()
