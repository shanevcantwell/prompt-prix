# VISION: Fara Visual Adapter for Frontier Model Interaction

**Status**: Exploratory
**Date**: 2024-12-04
**Context**: Considering Fara-7B as a durable alternative to DOM-based automation for frontier model UI interaction

---

## The Problem with DOM-Based Automation

Current approaches to automating frontier model web UIs rely on:
- CSS selectors that break on UI updates
- XPath queries tied to specific DOM structure
- Browser automation (Playwright/Puppeteer) with hardcoded element paths
- API access (when available, often rate-limited or feature-incomplete)

**Pain points observed:**
- Gemini's DOM structure changes frequently, breaking automation
- Claude's web UI has no public API; DOM automation is the only programmatic path
- Each model provider requires bespoke selector maintenance
- "Regenerate" buttons, error states, and loading indicators have inconsistent markup

---

## The Fara Proposition

**What if we could interact with frontier model UIs the way a human does - by looking at them?**

Fara-7B is a visual AI that can:
1. **Locate elements** by description → returns (x, y) coordinates
2. **Verify presence** of UI components → returns boolean + confidence
3. **Read text** from screenshots → extracts visible content
4. **Detect patterns** in UI state → identifies error states, loading, completion

### Durable by Design

| DOM Approach | Fara Approach |
|--------------|---------------|
| `button.regenerate-btn` | "The regenerate button" |
| `div.error-message.visible` | "Error message displayed" |
| `textarea#prompt-input` | "The text input area" |
| Breaks on redesign | Survives redesign |

Fara doesn't care about class names. It sees what's visually present.

---

## Regeneration Escalation Detection

A key use case: **detecting when a model is struggling**.

### The Pattern

When frontier models encounter difficulty:
1. Initial response is weak/incomplete
2. User clicks "Regenerate"
3. Second attempt may also fail
4. Repeated regeneration indicates systemic issue

### What Fara Can See

```python
class RegenerationEscalationDetector:
    """
    Uses Fara to detect regeneration patterns in frontier model UIs.
    """

    def detect_regenerate_button(self, screenshot: str) -> dict:
        """Locate the regenerate/retry button."""
        return fara.locate("Regenerate or retry button", screenshot)

    def detect_error_state(self, screenshot: str) -> dict:
        """Check for error messages or failure indicators."""
        return fara.verify("Error message or failure notification", screenshot)

    def detect_incomplete_response(self, screenshot: str) -> dict:
        """Check for truncation or incomplete output."""
        return fara.verify("Response that appears cut off or incomplete", screenshot)

    def assess_escalation_risk(self, history: List[Screenshot]) -> float:
        """
        Analyze screenshot history to detect escalation pattern.

        Returns escalation score 0.0 (stable) to 1.0 (critical).
        """
        regeneration_count = sum(
            1 for s in history if self.detect_regenerate_button(s)["clicked"]
        )
        error_count = sum(
            1 for s in history if self.detect_error_state(s)["exists"]
        )

        # Simple escalation heuristic
        return min(1.0, (regeneration_count * 0.3) + (error_count * 0.4))
```

---

## Multi-Provider Strategy Adapter

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    prompt-prix                          │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Claude    │  │   Gemini    │  │   GPT-4     │     │
│  │  Strategy   │  │  Strategy   │  │  Strategy   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│                ┌─────────────────┐                      │
│                │  Fara Adapter   │                      │
│                │                 │                      │
│                │  - locate()     │                      │
│                │  - verify()     │                      │
│                │  - read_text()  │                      │
│                │  - click()      │                      │
│                └────────┬────────┘                      │
│                         │                               │
│                         ▼                               │
│              ┌────────────────────┐                     │
│              │  Screenshot Feed   │                     │
│              │  (pyautogui/scrot) │                     │
│              └────────────────────┘                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Strategy Interface

```python
class FaraUIStrategy(Protocol):
    """Base protocol for Fara-driven UI strategies."""

    # Element descriptions (not selectors!)
    prompt_input: str        # "The message input textarea"
    send_button: str         # "Send or submit button"
    response_area: str       # "The assistant's response area"
    regenerate_button: str   # "Regenerate or retry button"
    stop_button: str         # "Stop generation button"

    # Provider-specific patterns
    loading_indicator: str   # "Loading spinner or typing indicator"
    error_message: str       # "Error notification or message"
    token_limit_warning: str # "Token or context limit warning"


class ClaudeUIStrategy(FaraUIStrategy):
    """Claude.ai visual element descriptions."""

    prompt_input = "The message input box at the bottom"
    send_button = "The send arrow button"
    response_area = "Claude's response text"
    regenerate_button = "Retry or regenerate option"
    stop_button = "Stop generating button"
    loading_indicator = "Pulsing dots or typing indicator"
    error_message = "Red error banner or message"
    token_limit_warning = "Context length or message limit warning"


class GeminiUIStrategy(FaraUIStrategy):
    """Gemini visual element descriptions."""

    prompt_input = "The text input area at the bottom"
    send_button = "The send or submit button"
    response_area = "Gemini's response content"
    regenerate_button = "Regenerate response button"
    stop_button = "Stop button during generation"
    loading_indicator = "Animated loading indicator"
    error_message = "Error or failure message"
    token_limit_warning = "Input too long warning"
```

---

## Integration with prompt-prix

### Evaluation Loop

```python
class FaraEvaluator:
    """
    Uses Fara to evaluate prompts across frontier model UIs.
    """

    def __init__(self, strategy: FaraUIStrategy):
        self.strategy = strategy
        self.fara = FaraService()
        self.escalation_detector = RegenerationEscalationDetector()

    async def evaluate_prompt(self, prompt: str) -> EvaluationResult:
        """
        Full evaluation cycle:
        1. Enter prompt
        2. Wait for response
        3. Capture result
        4. Detect quality issues
        """
        screenshot = capture_screen()

        # Find and click input
        input_loc = self.fara.locate(self.strategy.prompt_input, screenshot)
        click(input_loc["x"], input_loc["y"])
        type_text(prompt)

        # Find and click send
        screenshot = capture_screen()
        send_loc = self.fara.locate(self.strategy.send_button, screenshot)
        click(send_loc["x"], send_loc["y"])

        # Wait for completion (with escalation monitoring)
        history = []
        while True:
            screenshot = capture_screen()
            history.append(screenshot)

            # Check for completion
            loading = self.fara.verify(self.strategy.loading_indicator, screenshot)
            if not loading["exists"]:
                break

            # Check for escalation
            escalation = self.escalation_detector.assess_escalation_risk(history)
            if escalation > 0.7:
                return EvaluationResult(
                    status="escalation_detected",
                    escalation_score=escalation,
                    screenshots=history
                )

            await asyncio.sleep(1)

        # Extract response
        response_text = self.fara.read_text(
            self.strategy.response_area,
            screenshot
        )

        return EvaluationResult(
            status="complete",
            response=response_text,
            screenshots=history
        )
```

---

## Advantages Over DOM Automation

| Aspect | DOM-Based | Fara-Based |
|--------|-----------|------------|
| **Maintenance** | High - breaks on UI changes | Low - descriptions are stable |
| **Cross-provider** | Separate selectors per provider | Same Fara, different descriptions |
| **Error detection** | Complex selector chains | "Is there an error message?" |
| **New provider onboarding** | Full selector mapping | Write ~10 descriptions |
| **Debugging** | Inspect element, guess selectors | Look at screenshot, describe what you see |

---

## Limitations and Considerations

### Performance
- Fara inference adds latency (~1-2s per locate/verify)
- Screenshot capture overhead
- Not suitable for high-frequency automation

### Accuracy
- Vision models can mislocate elements
- Confidence thresholds needed
- May need fallback to DOM for precision clicks

### Model Requirements
- Fara-7B needs 2048+ context for screenshots
- VRAM requirement (~8GB for inference)
- JiT loading pattern recommended

### Ethical Considerations
- Respect provider ToS on automation
- Rate limiting and fair use
- Not intended for scraping or abuse

---

## Next Steps

1. **Prototype**: Single-provider (Claude) Fara adapter
2. **Baseline**: Compare accuracy vs. DOM automation
3. **Escalation**: Implement regeneration detection
4. **Multi-provider**: Add Gemini, GPT-4 strategies
5. **Integration**: Hook into prompt-prix evaluation pipeline

---

## Open Questions

1. **Coordinate precision**: Is Fara accurate enough for reliable clicking?
2. **Response extraction**: Can Fara read long responses accurately, or do we need OCR fallback?
3. **State detection**: How reliably can Fara distinguish loading vs. complete vs. error?
4. **Provider coverage**: Which frontier models have UIs amenable to visual automation?
5. **Hybrid approach**: Should we use Fara for detection but DOM for interaction?
