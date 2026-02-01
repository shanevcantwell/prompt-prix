# Extending prompt-prix

This guide explains how to extend prompt-prix with new features, following the established patterns and conventions.

## Table of Contents

1. [Adding a New UI Component](#adding-a-new-ui-component)
2. [Adding a New Handler](#adding-a-new-handler)
3. [Adding a New Export Format](#adding-a-new-export-format)
4. [Modifying the Session State](#modifying-the-session-state)
5. [Customizing Semantic Validation](#customizing-semantic-validation)
6. [Adding Tests](#adding-tests)
7. [Common Patterns](#common-patterns)
8. [Gotchas and Tips](#gotchas-and-tips)

---

## Adding a New UI Component

### Step 1: Define the Component in ui.py

Components are defined inside `create_app()`. Group related components with comments.

```python
# In ui.py, inside create_app()

with gr.Accordion("My New Feature", open=False):
    my_input = gr.Textbox(
        label="My Input",
        value="",
        elem_id="my_input"  # Required for localStorage persistence
    )
    my_button = gr.Button("Do Something", variant="primary")
```

### Step 2: Add Event Binding

Wire the component to a handler at the bottom of `create_app()`:

```python
# In the EVENT BINDINGS section

my_button.click(
    fn=my_handler_function,
    inputs=[my_input, other_input],
    outputs=[status_display, my_output]
)
```

### Step 3: Add to Persistence (Optional)

If the component's value should persist across page loads:

1. Add to the Save State JavaScript:
```javascript
// In save_state_btn.click js parameter
const myInputEl = document.querySelector('#my_input textarea');
if (myInputEl) {
    localStorage.setItem('promptprix_my_input', myInputEl.value);
}
```

2. Add to the Load State JavaScript:
```javascript
// In app.load js parameter
const myInput = localStorage.getItem('promptprix_my_input');
// Include in return array
return [
    // ...existing values...
    myInput ? myInput : undefined
];
```

3. Add to the outputs list:
```python
outputs=[
    servers_input, models_checkboxes, ..., my_input
]
```

---

## Adding a New Handler

Handlers go in `handlers.py`. They can be sync or async.

### Async Handler (Recommended for I/O)

```python
async def my_new_handler(input_value: str) -> tuple[str, str]:
    """
    Do something async.
    Returns (status_message, result).
    """
    if not input_value.strip():
        return "❌ Input is empty", ""

    try:
        # Do async work
        result = await some_async_operation(input_value)
        return f"✅ Success: {result}", result
    except Exception as e:
        return f"❌ Error: {e}", ""
```

### Streaming Handler (For Long Operations)

Use a generator to yield intermediate updates:

```python
async def my_streaming_handler(input_value: str):
    """
    Yields updates as work progresses.
    """
    yield ("⏳ Starting...", "")

    for i in range(5):
        await asyncio.sleep(1)
        yield (f"⏳ Step {i+1}/5...", f"Progress: {i+1}")

    yield ("✅ Complete!", "Final result")
```

### Accessing Session State

```python
from prompt_prix import state

async def my_handler():
    if state.session is None:
        return "❌ Session not initialized"

    # Access session data
    models = state.session.state.models
    contexts = state.session.state.contexts

    # Access server pool
    servers = state.server_pool.servers
```

---

## Adding a New Export Format

### Step 1: Add Generator Function in export.py

```python
def generate_csv_report(state: SessionState) -> str:
    """Generate a CSV report with model responses."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["Model", "Role", "Content"])

    # Data
    for model_id in state.models:
        context = state.contexts.get(model_id)
        if context:
            for msg in context.messages:
                writer.writerow([model_id, msg.role, msg.content])

    return output.getvalue()
```

### Step 2: Add Handler in handlers.py

```python
def export_csv() -> tuple[str, str]:
    """Export current session as CSV."""
    session = state.session

    if session is None:
        return "❌ No session to export", ""

    report = generate_csv_report(session.state)
    filename = f"prompt-prix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_report(report, filename)

    return f"✅ Exported to {filename}", report
```

### Step 3: Add UI Button in ui.py

```python
# In EXPORT PANEL section
with gr.Row():
    export_md_button = gr.Button("Export Markdown")
    export_json_button = gr.Button("Export JSON")
    export_csv_button = gr.Button("Export CSV")  # New

# In EVENT BINDINGS section
export_csv_button.click(
    fn=export_csv,
    inputs=[],
    outputs=[status_display, export_preview]
).then(
    fn=lambda: gr.update(visible=True),
    outputs=[export_preview]
)
```

### Step 4: Add Tests

```python
# In tests/test_export.py

class TestGenerateCsvReport:
    def test_generate_csv_report_basic(self):
        from prompt_prix.export import generate_csv_report

        state = SessionState(models=["model-a"])
        state.contexts["model-a"] = ModelContext(model_id="model-a")
        state.contexts["model-a"].add_user_message("Hello")
        state.contexts["model-a"].add_assistant_message("Hi there!")

        result = generate_csv_report(state)

        assert "model-a,user,Hello" in result
        assert "model-a,assistant,Hi there!" in result
```

---

## Modifying the Session State

### Step 1: Update Pydantic Model in config.py

```python
class SessionState(BaseModel):
    models: list[str]
    contexts: dict[str, ModelContext] = {}
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = DEFAULT_TEMPERATURE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    halted: bool = False
    halt_reason: Optional[str] = None
    my_new_field: str = ""  # Add new field with default
```

### Step 2: Update ComparisonSession Initialization

```python
# In core.py, ComparisonSession.__init__

def __init__(
    self,
    models: list[str],
    server_pool: ServerPool,
    system_prompt: str,
    temperature: float,
    timeout_seconds: int,
    max_tokens: int,
    my_new_field: str = ""  # Add parameter
):
    self.state = SessionState(
        models=models,
        system_prompt=system_prompt,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        my_new_field=my_new_field  # Pass to state
    )
```

### Step 3: Update initialize_session Handler

```python
# In handlers.py

async def initialize_session(
    servers_text: str,
    models_selected: list[str],
    system_prompt_text: str,
    temperature: float,
    timeout: int,
    max_tokens: int,
    my_new_value: str  # Add parameter
) -> tuple:
    # ...
    state.session = ComparisonSession(
        models=models,
        server_pool=state.server_pool,
        system_prompt=system_prompt,
        temperature=temperature,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        my_new_field=my_new_value  # Pass through
    )
```

### Step 4: Update UI

```python
# In ui.py

# Add component
my_new_input = gr.Textbox(label="My New Field", value="")

# Update init_button inputs
init_button.click(
    fn=initialize_session,
    inputs=[
        servers_input,
        models_checkboxes,
        system_prompt_input,
        temperature_slider,
        timeout_slider,
        max_tokens_slider,
        my_new_input  # Add to inputs
    ],
    outputs=[status_display] + model_outputs
)
```

---

## Customizing Semantic Validation

Battery tests validate model responses beyond HTTP success. The semantic validator (`prompt_prix/semantic_validator.py`) applies these checks in order:

1. **Empty response** - No content returned
2. **Model refusals** - "I'm sorry, but I can't help with that"
3. **Missing tool calls** - When `tool_choice: "required"` but no tool was called
4. **Verdict matching** - When `pass_criteria` specifies an expected verdict

### Understanding Test Status

| Status | Symbol | Meaning |
|--------|--------|---------|
| `COMPLETED` | ✓ | Completed and passed semantic validation |
| `ERROR` | ⚠ | Failure in response from adapter or semantic check |
| `SEMANTIC_FAILURE` | ❌ | Response received but did not meet expected criteria |

### Verdict Matching (LLM-as-Judge)

When a test has `pass_criteria` containing a verdict expectation, the validator:

1. Extracts the `"verdict"` field from JSON in the response
2. Compares it (case-insensitive) against the expected verdict
3. Handles JSON inside markdown code fences (`` ```json ... ``` ``)

**Example pass_criteria:**
```
The verdict in the JSON response must be 'PASS'
```

The validator extracts `PASS`, `FAIL`, or `PARTIAL` from the model's JSON response and compares against the expected value.

**Note:** The validator does NOT enforce strict JSON parsing. A response with valid JSON followed by extra prose ("reasoning bleed") will still pass if the verdict matches. Use the planned **Strict JSON** toggle (ADR-010) to enforce valid-only JSON.

### Adding New Refusal Patterns

Edit `prompt_prix/semantic_validator.py`:

```python
REFUSAL_PATTERNS = [
    r"i(?:'m| am) sorry,? but",
    r"i can(?:'t|not)",
    r"i(?:'m| am) (?:not )?(?:able|unable)",
    r"(?:cannot|can't) (?:execute|run|perform|help with)",
    r"i(?:'m| am) not (?:designed|programmed|able)",
    r"(?:as an ai|as a language model)",
    r"i don't have (?:the ability|access)",
    # Add your pattern here:
    r"(?:that's|this is) beyond my capabilities",
]
```

Then add a test in `tests/test_semantic_validator.py`:

```python
def test_detects_beyond_capabilities(self):
    response = "That's beyond my capabilities as an assistant."
    assert detect_refusal(response) is not None
```

Run the test:
```bash
pytest tests/test_semantic_validator.py -v
```

### Adding New Validation Types

The `validate_response_semantic()` function checks responses in order. Add new checks after existing ones:

```python
# In prompt_prix/semantic_validator.py

def validate_response_semantic(
    test: "TestCase",
    response: str
) -> Tuple[bool, Optional[str]]:
    # Existing refusal check
    refusal = detect_refusal(response)
    if refusal:
        return False, f"Model refused: '{refusal}'"

    # Existing tool call checks
    if test.tools and test.tool_choice == "required":
        if not has_tool_calls(response):
            return False, "Expected tool call but got text response"

    if test.tools and test.tool_choice == "none":
        if has_tool_calls(response):
            return False, "Tool call made when tool_choice='none'"

    # ADD YOUR NEW VALIDATION HERE:
    # Example: Check for hallucination markers
    if contains_hallucination_markers(response):
        return False, "Response contains hallucination markers"

    return True, None
```

### Tool Call Detection

Tool calls are detected by the `**Tool Call:**` marker in formatted responses. This marker is added by `stream_completion()` when the model returns tool calls.

The validation rules for `tool_choice`:

| `tool_choice` | Validation |
|---------------|------------|
| `"required"` | Fails if no `**Tool Call:**` in response |
| `"none"` | Fails if `**Tool Call:**` appears in response |
| `"auto"` or unset | Always passes (model decides) |

### Validation Order

Checks run in this order (first failure wins):
1. Empty response detection
2. Refusal detection
3. Tool call validation (if applicable)
4. Verdict matching (if `pass_criteria` specifies expected verdict)
5. Custom validations (if added)

A response containing both a refusal phrase AND a tool call will fail with "Model refused" because refusals are checked first.

### Promptfoo YAML Files

When loading promptfoo YAML files (`prompt_prix/benchmarks/promptfoo.py`):

**Supported:**
- `expected_verdict` in `vars` → converted to `pass_criteria` for verdict matching
- `category` in `vars` → stored for filtering/grouping
- `{{variable}}` substitution in prompts

**Not evaluated by prompt-prix:**
- `assert:` blocks (e.g., `type: contains`, `type: llm-rubric`)
- These are logged with a warning but not executed

**Example promptfoo test:**
```yaml
tests:
  - description: "Should pass tool call"
    vars:
      system: "You are a helpful assistant."
      user: "What's the weather in Tokyo?"
      expected_verdict: "PASS"
      category: "tool_calls"
```

This becomes a `BenchmarkCase` with:
- `pass_criteria`: `"The verdict in the JSON response must be 'PASS'"`
- `category`: `"tool_calls"`

### Testing Your Changes

Always test both positive and negative cases:

```python
class TestMyNewValidation:
    def test_detects_bad_response(self):
        test = TestCase(id="test", user="Do something")
        response = "This response should fail validation"
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is False
        assert "expected reason" in reason

    def test_passes_good_response(self):
        test = TestCase(id="test", user="Do something")
        response = "This response should pass validation"
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is True
```

---

## Adding Tests

### Test Location

Tests go in `tests/test_*.py` files:
- `test_config.py` - Pydantic model tests
- `test_core.py` - ServerPool and ComparisonSession tests
- `test_main.py` - Handler tests (uses the main.py re-exports)
- `test_export.py` - Export function tests

### Async Test Pattern

```python
import pytest
import respx
import httpx

class TestMyHandler:
    @respx.mock
    @pytest.mark.asyncio
    async def test_my_handler_success(self):
        from prompt_prix.main import my_handler

        # Mock HTTP calls
        respx.get("http://localhost:1234/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "model-a"}]})
        )

        result = await my_handler("test input")

        assert "✅" in result[0]
```

### Using Fixtures

Common fixtures are in `tests/conftest.py`:

```python
@pytest.fixture
def mock_session():
    """Create a mock ComparisonSession for testing."""
    from prompt_prix import state
    from prompt_prix.core import ServerPool, ComparisonSession

    pool = ServerPool(["http://localhost:1234"])
    pool.servers["http://localhost:1234"].available_models = ["model-a"]

    state.server_pool = pool
    state.session = ComparisonSession(
        models=["model-a"],
        server_pool=pool,
        system_prompt="Test prompt",
        temperature=0.7,
        timeout_seconds=300,
        max_tokens=2048
    )

    yield state.session

    state.session = None
    state.server_pool = None
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_main.py

# Specific test
pytest tests/test_main.py::TestMyHandler::test_my_handler_success

# With coverage
pytest --cov=prompt_prix --cov-report=html
```

---

## Common Patterns

### Pattern 1: Gradio Update Objects

When returning component updates that only change some properties:

```python
# Update only value
return gr.update(value="new value")

# Update only choices (for dropdowns/checkboxes)
return gr.update(choices=["a", "b", "c"])

# Update multiple properties
return gr.update(choices=["a", "b"], value=["a"])

# No change (use in lists where some outputs don't change)
return gr.update()
```

### Pattern 2: Status Messages

Use consistent status prefixes:

```python
return "✅ Success message"      # Success
return "⏳ Processing..."        # In progress
return "⚠️ Warning message"      # Warning (non-fatal)
return "❌ Error message"        # Error (fatal)
```

### Pattern 3: Generator Handlers

For streaming updates, yield tuples matching the outputs list:

```python
async def my_generator_handler(prompt: str):
    """Handler that yields updates."""
    # outputs=[status_display, tab_states] + model_outputs (12 items total)

    # Initial state
    yield ("⏳ Starting...", [], "", "", "", "", "", "", "", "", "", "")

    # Progress updates
    yield ("⏳ Working...", ["streaming"], "Partial result", "", "", "", "", "", "", "", "", "")

    # Final result
    yield ("✅ Done!", ["completed"], "Final result", "", "", "", "", "", "", "", "", "")
```

### Pattern 4: JavaScript for DOM Manipulation

Gradio's `js` parameter runs in browser:

```python
component.click(
    fn=python_handler,
    inputs=[],
    outputs=[status_display],
    js="""
    () => {
        // This runs in the browser
        console.log('Button clicked');
        localStorage.setItem('key', 'value');
        return [];  // Must return array matching outputs
    }
    """
)
```

---

## Gotchas and Tips

### 1. Circular Imports

The codebase uses `state.py` to break circular imports:
- `ui.py` imports from `handlers.py`
- `handlers.py` imports from `core.py`
- `core.py` does NOT import from `handlers.py` or `ui.py`
- All modules can import from `state.py`

### 2. Gradio Component elem_id

Always set `elem_id` for components you need to access via JavaScript:

```python
my_input = gr.Textbox(elem_id="my_input")

# Then in JS:
document.querySelector('#my_input textarea')
```

### 3. Async/Await in Handlers

Gradio automatically handles async handlers, but remember:
- Use `await` for I/O operations
- Use `asyncio.gather()` for parallel operations
- Yield for streaming updates

### 4. CheckboxGroup Values

CheckboxGroup returns a list of selected values:

```python
def handler(models_selected: list[str]):
    # models_selected is already a list, not comma-separated text
    for model in models_selected:
        print(model)
```

### 5. Testing Gradio Updates

When testing functions that return `gr.update()`:

```python
result = await my_handler()
# result is a dict like {"choices": [...], "__type__": "update"}
assert "choices" in result
assert result["choices"] == ["a", "b"]
```

### 6. LocalStorage Persistence

Values stored in localStorage are strings:
- Numbers: `parseFloat()` or `parseInt()`
- Arrays/Objects: `JSON.stringify()` / `JSON.parse()`
- Empty check: `localStorage.getItem()` returns `null` if not set

### 7. Tab Colors with Inline Styles

CSS classes don't work reliably due to Gradio theme overrides. Use inline styles:

```javascript
btn.style.background = 'linear-gradient(...)';
btn.style.borderLeft = '4px solid #color';
```

### 8. Server Manifest Caching

`refresh_all_manifests()` is expensive (HTTP calls). Avoid calling it in loops:

```python
# Good: Refresh once before loop
await session.server_pool.refresh_all_manifests()
for model in models:
    await process(model)

# Bad: Refresh on every iteration
for model in models:
    await session.server_pool.refresh_all_manifests()  # Don't do this
    await process(model)
```

---

## Example: Adding a "Copy to Clipboard" Button

Here's a complete example adding a button to copy model output:

### 1. Add UI Component (ui.py)

```python
# After model_outputs creation
copy_buttons = []
with gr.Tabs(elem_id="model-tabs"):
    for i in range(10):
        with gr.Tab(f"Model {i + 1}"):
            output = gr.Markdown(value="", label="Conversation")
            copy_btn = gr.Button("📋 Copy", size="sm")
            model_outputs.append(output)
            copy_buttons.append(copy_btn)
```

### 2. Add Event Binding (ui.py)

```python
# Copy button - uses JavaScript only, no Python handler needed
for i, copy_btn in enumerate(copy_buttons):
    copy_btn.click(
        fn=None,
        inputs=[],
        outputs=[status_display],
        js=f"""
        () => {{
            const tab = document.querySelector('#model-tabs');
            const outputs = tab.querySelectorAll('.markdown-text');
            if (outputs[{i}]) {{
                navigator.clipboard.writeText(outputs[{i}].innerText);
                return ["✅ Copied to clipboard"];
            }}
            return ["❌ Nothing to copy"];
        }}
        """
    )
```

No Python handler needed since clipboard is browser-only!
