# ADR-DISTILL-006: GeminiWebUI Adapter for Thinking Block Extraction

**Status:** PROPOSED
**Date:** 2025-01-09
**Depends On:** ADR-DISTILL-001, ADR-DISTILL-004
**Context:** Gemini API does not expose thinking blocks. Web UI automation required to capture reasoning chains.
**Reference Implementation:** `/home/shane/github/shanevcantwell/gemini-exporter/content.js`

---

## Decision

Build a `GeminiWebUIAdapter` that uses Playwright for browser automation and Pydantic schemas for validated DOM extraction. The adapter integrates with the existing LLM adapter infrastructure as a first-class provider.

## Architecture

### Component Structure

```
app/src/llm/
├── gemini_webui_schemas.py    # Pydantic models for DOM validation
├── gemini_webui_adapter.py    # BaseAdapter implementation
└── factory.py                  # Register in ADAPTER_REGISTRY
```

### Integration Point

```python
# app/src/llm/factory.py
ADAPTER_REGISTRY = {
    "gemini": GeminiAdapter,
    "lmstudio": LMStudioAdapter,
    "gemini_webui": GeminiWebUIAdapter,  # New adapter
}
```

```yaml
# user_settings.yaml
llm_providers:
  gemini_webui_thinking:
    type: "gemini_webui"
    session_cookies: "path/to/cookies.json"
    rate_limit_delay: 2.0  # Seconds between requests

specialist_model_bindings:
  distillation_response_collector_specialist: "gemini_webui_thinking"
```

---

## Pydantic Schemas

### Schema Design Philosophy

**Fail-Fast Validation:** When Google changes the Gemini UI, Pydantic validation fails immediately with clear error messages showing exactly which field failed. This transforms fragile DOM scraping into a structured, maintainable system.

### Schema Hierarchy

```python
# app/src/llm/gemini_webui_schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class ThinkingStage(BaseModel):
    """
    Represents one stage of thinking (e.g., "Understanding the Request").

    Extracted from bold headers + following text in model-thoughts container.
    Maps to gemini-exporter SCHEMA.md lines 86-91.
    """
    stage_name: str = Field(
        description="Stage header from bold/strong element (e.g., 'Understanding the Request')"
    )
    text: str = Field(
        description="Stage reasoning content (paragraphs following the header)"
    )

    @validator('stage_name')
    def stage_name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("stage_name cannot be empty")
        return v.strip()

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("stage text cannot be empty")
        return v.strip()


class ThinkingBlock(BaseModel):
    """
    Container for all thinking stages in a response.

    Extracted from [data-test-id='model-thoughts'] DOM container.
    """
    stages: List[ThinkingStage] = Field(
        description="Ordered list of thinking stages",
        min_items=1  # If thinking exists, must have at least 1 stage
    )


class ConversationContainer(BaseModel):
    """
    Represents one .conversation-container DOM element (one complete turn).

    Maps to gemini-exporter's exchange structure.
    """
    container_id: Optional[str] = Field(
        None,
        description="DOM element ID (e.g., 'c61fbdc59e290cd9') for timestamp merging"
    )
    user_text: str = Field(
        description="User's message from .user-query-container"
    )
    thinking: Optional[ThinkingBlock] = Field(
        None,
        description="Thinking stages if present, None otherwise"
    )
    response_text: str = Field(
        description="Gemini's response from .markdown-main-panel (thinking excluded)"
    )

    @validator('user_text')
    def user_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("user_text cannot be empty")
        return v.strip()

    @validator('response_text')
    def response_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("response_text cannot be empty")
        return v.strip()


class GeminiUIResponse(BaseModel):
    """
    Complete validated response from Gemini web UI extraction.

    This is the top-level schema returned by _extract_and_validate().
    """
    conversation_id: str = Field(
        description="Gemini conversation ID from URL (e.g., '210cdaa5f25daa51')"
    )
    container: ConversationContainer = Field(
        description="The extracted conversation container"
    )

    @validator('conversation_id')
    def conversation_id_valid(cls, v):
        if not v or len(v) < 10:
            raise ValueError("conversation_id must be valid hex string")
        return v


# Helper for adapter return format (not used for validation, just documentation)
class AdapterReturnFormat:
    """
    Documents the format returned by GeminiWebUIAdapter.invoke()

    This is NOT a Pydantic model - just documentation.
    Actual return is a plain dict matching this structure.
    """
    text_response: str  # From GeminiUIResponse.container.response_text
    thinking_stages: Optional[List[dict]]  # From ThinkingBlock.stages (as dicts)
```

---

## Adapter Implementation

### Execution Flow

The adapter follows the gemini-exporter flow closely:

```
1. Browser Initialization
   ├─ Create Playwright browser (headless)
   ├─ Restore session from cookies
   └─ Navigate to gemini.google.com

2. Prompt Submission (content.js lines 888-960)
   ├─ Find textarea input field
   ├─ Enter prompt text
   ├─ Click send button
   └─ Wait for response streaming to complete

3. Thinking Block Expansion (content.js lines 700-805)
   ├─ Find all 'Show thinking' buttons
   ├─ Scroll each into view
   ├─ Click and wait for expansion
   └─ Verify content loaded (retry up to 10x)

4. DOM Extraction (content.js lines 807-1185)
   ├─ Get .conversation-container:last-child
   ├─ Extract user text from .user-query-container
   ├─ Extract thinking stages (bold headers + content)
   ├─ Extract response from .markdown-main-panel (thinking excluded)
   └─ Get container ID for timestamp merging

5. Pydantic Validation
   ├─ Build data dict from extracted fields
   ├─ Validate with GeminiUIResponse(**data)
   └─ Fail fast if structure changed

6. Return Standardized Format
   └─ Return dict with text_response + thinking_stages
```

### Core Adapter Implementation

```python
# app/src/llm/gemini_webui_adapter.py
import logging
import time
from typing import Dict, Any, Optional
from playwright.sync_api import sync_playwright, Page, Browser
from pydantic import ValidationError

from .adapter import BaseAdapter, StandardizedLLMRequest
from .gemini_webui_schemas import GeminiUIResponse, ThinkingStage
from ..utils.errors import LLMInvocationError

logger = logging.getLogger(__name__)


class GeminiWebUIAdapter(BaseAdapter):
    """
    Adapter for Gemini web UI using browser automation.
    Extracts thinking blocks via DOM scraping (like gemini-exporter).
    Uses Pydantic schemas for fail-fast validation when UI structure changes.

    Reference Implementation: /home/shane/github/shanevcantwell/gemini-exporter/content.js
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        credentials: Dict[str, Any],
        system_prompt: str
    ):
        super().__init__(model_config)
        self.session_cookies_path = credentials.get("session_cookies")
        self.rate_limit_delay = model_config.get("rate_limit_delay", 2.0)
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None

    @property
    def api_base(self) -> Optional[str]:
        """Web UI URL"""
        return "https://gemini.google.com"

    @property
    def api_key(self) -> Optional[str]:
        """No API key for web UI adapter"""
        return None

    @classmethod
    def from_config(
        cls,
        provider_config: Dict[str, Any],
        system_prompt: str
    ) -> "GeminiWebUIAdapter":
        """Factory method for adapter creation"""
        model_config = {
            "api_identifier": "gemini-2.0-flash-thinking-webui",
            "parameters": provider_config.get("parameters", {}),
            "rate_limit_delay": provider_config.get("rate_limit_delay", 2.0),
        }
        credentials = {
            "session_cookies": provider_config.get("session_cookies"),
        }
        return cls(model_config, credentials, system_prompt)

    def invoke(self, request: StandardizedLLMRequest) -> Dict[str, Any]:
        """
        Submit prompt to Gemini web UI and extract response + thinking.

        Args:
            request: StandardizedLLMRequest with messages

        Returns:
            Dict with text_response and thinking_stages

        Raises:
            LLMInvocationError: If DOM extraction fails or validation fails
        """
        prompt_text = request.messages[-1].content
        logger.info(f"GeminiWebUIAdapter: Submitting prompt ({len(prompt_text)} chars)")

        # Initialize browser if needed
        if not self.page:
            self._initialize_browser()

        try:
            # Submit prompt and wait for response
            self._submit_prompt(prompt_text)
            self._wait_for_response_complete()

            # Rate limiting (mimic human interaction)
            time.sleep(self.rate_limit_delay)

            # Extract and validate using Pydantic schema
            ui_response = self._extract_and_validate()

            # Return standardized format
            return {
                "text_response": ui_response.container.response_text,
                "thinking_stages": (
                    [stage.dict() for stage in ui_response.container.thinking.stages]
                    if ui_response.container.thinking
                    else None
                ),
            }

        except ValidationError as e:
            logger.error(f"Gemini UI structure changed, validation failed: {e}")
            logger.error("DOM selectors may need updating in gemini_webui_adapter.py")
            raise LLMInvocationError(f"DOM extraction failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in GeminiWebUIAdapter: {e}", exc_info=True)
            raise LLMInvocationError(f"Web UI automation failed: {e}") from e

    def _initialize_browser(self):
        """
        Initialize Playwright browser and restore session.

        Mirrors gemini-exporter's approach but uses server-side Playwright
        instead of Chrome extension.
        """
        logger.info("Initializing Playwright browser...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,  # Run without visible window
            args=[
                "--disable-blink-features=AutomationControlled",  # Avoid detection
            ],
        )

        # Create browser context with session cookies
        context = self.browser.new_context()
        if self.session_cookies_path:
            import json
            with open(self.session_cookies_path, 'r') as f:
                cookies = json.load(f)
            context.add_cookies(cookies)
            logger.info("Restored session from cookies")

        self.page = context.new_page()
        self.page.goto("https://gemini.google.com/app")
        logger.info("Browser initialized and navigated to Gemini")

        # Wait for page load
        self.page.wait_for_load_state("networkidle")

    def _submit_prompt(self, prompt_text: str):
        """
        Submit prompt to Gemini web UI.

        Mirrors content.js prompt submission logic.
        """
        # Find textarea input (may need selector adjustment)
        textarea = self.page.wait_for_selector("textarea", timeout=10000)
        textarea.fill(prompt_text)

        # Click send button (selector may need adjustment)
        send_button = self.page.wait_for_selector(
            'button[aria-label*="Send"], button[type="submit"]',
            timeout=5000
        )
        send_button.click()
        logger.debug("Prompt submitted")

    def _wait_for_response_complete(self):
        """
        Wait for Gemini to finish generating response.

        Mirrors content.js waiting logic - wait for streaming to stop.
        """
        # Wait for new conversation container to appear
        self.page.wait_for_selector(".conversation-container:last-child", timeout=60000)

        # Wait for response streaming to complete
        # Strategy: Wait for "Stop generating" button to disappear
        try:
            stop_button = self.page.wait_for_selector(
                'button[aria-label*="Stop"]',
                timeout=5000
            )
            # Wait for it to disappear (response complete)
            stop_button.wait_for(state="detached", timeout=120000)
        except:
            # No stop button found, response likely already complete
            pass

        # Additional wait for rendering to stabilize
        time.sleep(2)
        logger.debug("Response generation complete")

    def _expand_thinking_blocks(self):
        """
        Click 'Show thinking' buttons and wait for expansion.

        Mirrors content.js expandThinkingBlocks() function (lines 700-805).
        Multi-pass expansion with retry logic.
        """
        max_passes = 10
        expanded_count = 0

        for pass_num in range(1, max_passes + 1):
            # Find all 'Show thinking' buttons
            buttons = self.page.query_selector_all(
                'button[data-test-id="thoughts-header-button"]'
            )

            unexpanded = [
                btn for btn in buttons
                if "Show thinking" in btn.inner_text().lower()
            ]

            if not unexpanded:
                logger.debug(
                    f"Thinking block expansion complete after {pass_num} passes. "
                    f"Expanded {expanded_count} blocks."
                )
                break

            logger.debug(
                f"Pass {pass_num}: Found {len(unexpanded)} unexpanded thinking blocks"
            )

            for button in unexpanded:
                try:
                    # Scroll into view and click
                    button.scroll_into_view_if_needed()
                    button.click()
                    expanded_count += 1

                    # Verify expansion (retry up to 10 times, like gemini-exporter)
                    verified = False
                    for retry in range(10):
                        time.sleep(0.5)
                        # Check if content appeared
                        container = button.evaluate(
                            "btn => btn.closest('.conversation-container')"
                        )
                        thinking_content = self.page.query_selector(
                            '[class*="model-thoughts"], [class*="thinking"]'
                        )
                        if thinking_content and len(thinking_content.inner_text()) > 50:
                            verified = True
                            break

                    if not verified:
                        logger.warning(f"Failed to verify thinking block expansion")

                except Exception as e:
                    logger.warning(f"Error expanding thinking button: {e}")
                    continue

            # Wait before next pass
            time.sleep(1)

    def _extract_and_validate(self) -> GeminiUIResponse:
        """
        Extract DOM data and validate against Pydantic schema.

        This is the main extraction orchestrator. Mirrors content.js
        extractStructuredConversation() function (lines 888-1118).

        Returns:
            Validated GeminiUIResponse

        Raises:
            ValidationError: If DOM structure doesn't match schema
        """
        # Step 1: Expand thinking blocks
        self._expand_thinking_blocks()

        # Step 2: Get most recent conversation container
        container = self.page.query_selector(".conversation-container:last-child")
        if not container:
            raise LLMInvocationError("No conversation container found")

        # Step 3: Extract all fields
        data = {
            "conversation_id": self._extract_conversation_id(),
            "container": {
                "container_id": container.get_attribute("id"),
                "user_text": self._extract_user_text(container),
                "thinking": self._extract_thinking(container),
                "response_text": self._extract_response_text(container),
            },
        }

        # Step 4: Validate with Pydantic - fails fast if structure changed
        return GeminiUIResponse(**data)

    def _extract_conversation_id(self) -> str:
        """Extract conversation ID from URL"""
        url = self.page.url
        # URL format: https://gemini.google.com/app/<conversation_id>
        import re
        match = re.search(r"/app/([a-fA-F0-9]+)", url)
        if not match:
            raise LLMInvocationError(f"Could not extract conversation ID from URL: {url}")
        return match.group(1)

    def _extract_user_text(self, container) -> str:
        """
        Extract user's message from container.

        Mirrors content.js lines 989-1011.
        Note: gemini-exporter extracts ALL .user-query-container duplicates,
        but we only need one (the first complete one).
        """
        user_container = container.query_selector(".user-query-container")
        if not user_container:
            raise LLMInvocationError("No user message found in container")

        # Extract text content, excluding UI elements
        text = user_container.inner_text().strip()
        if not text:
            raise LLMInvocationError("User message is empty")

        return text

    def _extract_thinking(self, container) -> Optional[Dict]:
        """
        Extract thinking stages from DOM.

        Mirrors content.js extractThinkingStages() function (lines 807-846).

        Returns:
            Dict with {"stages": [{"stage_name": str, "text": str}, ...]} or None
        """
        thinking_container = container.query_selector('[data-test-id="model-thoughts"]')
        if not thinking_container:
            return None

        stages = []
        elements = thinking_container.query_selector_all("p, div")
        current_stage = None

        for el in elements:
            text = el.inner_text().strip()
            if not text:
                continue

            # Check if element is a stage header (contains bold/strong text)
            bold_el = el.query_selector("strong, b")
            if bold_el and text == bold_el.inner_text().strip():
                # This is a stage header - save previous stage if exists
                if current_stage and current_stage["text"].strip():
                    stages.append(current_stage)

                # Start new stage
                current_stage = {"stage_name": text, "text": ""}
            elif current_stage:
                # This is stage content - add to current stage
                current_stage["text"] += (current_stage["text"] and "\n\n") + text

        # Don't forget the last stage
        if current_stage and current_stage["text"].strip():
            stages.append(current_stage)

        return {"stages": stages} if stages else None

    def _extract_response_text(self, container) -> str:
        """
        Extract response text, excluding thinking blocks.

        Mirrors content.js extractResponseText() function (lines 848-872).
        """
        # Find markdown panel
        markdown_panel = container.query_selector(".markdown-main-panel")
        if not markdown_panel:
            raise LLMInvocationError("No response markdown panel found")

        # Clone to avoid modifying DOM
        clone = markdown_panel.evaluate(
            """
            (element) => {
                const clone = element.cloneNode(true);

                // Remove thinking containers
                clone.querySelectorAll('[data-test-id="model-thoughts"]').forEach(el => el.remove());
                clone.querySelectorAll('button[data-test-id="thoughts-header-button"]').forEach(el => el.remove());

                // Remove other UI elements (but keep code blocks)
                clone.querySelectorAll('button:not(pre button), [role="button"]:not(pre [role="button"])').forEach(el => {
                    if (!el.closest('pre, code')) {
                        el.remove();
                    }
                });

                return clone.innerText;
            }
            """
        )

        response_text = clone.strip()
        if not response_text:
            raise LLMInvocationError("Response text is empty")

        return response_text

    def __del__(self):
        """Cleanup browser resources"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
```

---

## Error Handling

### Validation Failures

When Google changes the UI, Pydantic validation will fail with clear messages:

```python
# Example validation error:
ValidationError: 1 validation error for GeminiUIResponse
container -> response_text
  field required (type=value_error.missing)
```

This tells you exactly what's missing. Update the selector in `_extract_response_text()` to fix.

### Retry Strategy

```python
# Retry logic for transient failures
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def invoke(self, request: StandardizedLLMRequest) -> Dict[str, Any]:
    # ... implementation
```

---

## Testing Strategy

### Unit Tests with Mock DOM

```python
# tests/llm/test_gemini_webui_schemas.py
import pytest
from app.src.llm.gemini_webui_schemas import GeminiUIResponse, ThinkingStage

def test_thinking_stage_validation():
    """Test ThinkingStage schema validation"""
    # Valid stage
    stage = ThinkingStage(
        stage_name="Understanding the Request",
        text="The user wants to know about..."
    )
    assert stage.stage_name == "Understanding the Request"

    # Invalid: empty stage name
    with pytest.raises(ValueError):
        ThinkingStage(stage_name="", text="content")

def test_gemini_ui_response_validation():
    """Test complete response validation"""
    data = {
        "conversation_id": "abc123def456",
        "container": {
            "container_id": "c61fbdc59e290cd9",
            "user_text": "How do I design a resilient system?",
            "thinking": {
                "stages": [
                    {"stage_name": "Understanding", "text": "First..."}
                ]
            },
            "response_text": "A resilient system requires..."
        }
    }

    response = GeminiUIResponse(**data)
    assert response.conversation_id == "abc123def456"
    assert len(response.container.thinking.stages) == 1
```

### Integration Tests

```python
# tests/llm/test_gemini_webui_adapter_integration.py
import pytest
from app.src.llm.gemini_webui_adapter import GeminiWebUIAdapter

@pytest.mark.integration
@pytest.mark.slow
def test_gemini_webui_adapter_real_request():
    """Integration test with real Gemini web UI (slow, requires session)"""
    # Requires valid session cookies
    adapter = GeminiWebUIAdapter.from_config({
        "session_cookies": "tests/fixtures/gemini_cookies.json",
        "rate_limit_delay": 3.0
    }, "")

    request = StandardizedLLMRequest(
        messages=[HumanMessage(content="What is 2+2?")]
    )

    response = adapter.invoke(request)
    assert "text_response" in response
    assert "4" in response["text_response"]
```

---

## Maintenance

### When Google Changes the UI

1. **Validation will fail** with clear Pydantic error showing missing field
2. **Inspect the new DOM structure** using browser dev tools
3. **Update the selector** in the corresponding `_extract_*()` method
4. **Update the schema** if field semantics changed
5. **Test** with mock data first, then integration test
6. **Document** the change in this ADR

### Selector Update Example

If `.markdown-main-panel` changes to `.response-content-panel`:

```python
# Before
markdown_panel = container.query_selector(".markdown-main-panel")

# After
markdown_panel = container.query_selector(".response-content-panel")
```

If validation still fails, the schema may need updating too.

---

## Performance Characteristics

- **Speed**: ~3-5 seconds per request (vs <1s for API)
- **Memory**: ~200-400 MB for headless Chrome
- **CPU**: Moderate (DOM rendering + JavaScript execution)
- **Throughput**: ~20-30 requests per minute (with rate limiting)

## Deployment Considerations

- **Docker**: Playwright requires `playwright install chromium` in container
- **Headless**: Must run in headless mode on server (no display)
- **Session Management**: Cookie file must be accessible, periodically refresh
- **Rate Limiting**: 2-3 second delays recommended to avoid detection

---

## Related ADRs

- **ADR-DISTILL-001**: Overall distillation architecture
- **ADR-DISTILL-004**: JSONL dataset format (thinking stages schema)
- **ADR-CORE-002**: BaseAdapter contract

---

## Consequences

### Positive
- Captures thinking blocks (only way to access this data)
- Pydantic validation provides fail-fast error detection
- Isolated implementation (single file, easily removable)
- Integrates seamlessly with existing adapter infrastructure

### Negative
- Fragile (breaks when Google changes UI)
- Slow (~3-5s per request)
- Resource intensive (headless browser)
- Maintenance burden (selector updates required)
- ToS violation risk (covert usage)

### Neutral
- Temporary infrastructure (disposable after data capture)
- Not intended for long-term production use
- Value is in the dataset, not the tool
