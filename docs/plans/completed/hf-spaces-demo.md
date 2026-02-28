# HF Spaces Demo - Final Polish

## Completed ✓

- HuggingFaceAdapter wired to UI
- app.py + requirements.txt for Spaces
- UX labels improved (Available Models → Models to Test)
- Local test successful with HF_TOKEN

## Remaining Tasks

### 1. Simplify Model Selection
**File:** `prompt_prix/ui.py`

Remove the textbox, use hardcoded vetted models:
```python
# Replace textbox + sync button with just CheckboxGroup
models_selector = gr.CheckboxGroup(
    label="Models to Test",
    choices=[
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
    ],
    value=["meta-llama/Llama-3.2-3B-Instruct"],  # Default selection
    info="Select HuggingFace models to compare"
)
```

Remove:
- `models_input` textbox
- `sync_models_btn` button
- `on_sync_models()` function
- `app.load()` auto-sync binding

### 2. Add Example Test Loader
**File:** `prompt_prix/tabs/battery/ui.py`

Add example file buttons/dropdown near the file upload:
```python
gr.Markdown("**Quick Start:** Load an example test suite")
with gr.Row():
    example_btn = gr.Button("📋 Tool Competence Tests", size="sm")
```

**File:** `prompt_prix/ui.py` (event binding)
```python
def load_example(example_name):
    # Return path to example file
    return "examples/tool_competence_tests.json"

battery.example_btn.click(
    fn=load_example,
    outputs=[battery.file]
)
```

### 3. Files to Modify

| File | Changes |
|------|---------|
| `prompt_prix/ui.py` | Remove textbox, simplify model config |
| `prompt_prix/tabs/battery/ui.py` | Add example loader button |

## Test Cases

### Update existing tests
**File:** `tests/test_main.py`
- Remove/update tests that reference `models_input` textbox
- Remove tests for `on_sync_models` function

### New tests for example loader
**File:** `tests/test_battery.py`
```python
def test_example_loader_returns_valid_path():
    """Example loader should return path to existing file."""
    from prompt_prix.tabs.battery.handlers import load_example
    path = load_example()
    assert Path(path).exists()
    assert path.endswith('.json')

def test_example_file_is_valid_battery_format():
    """Example file should be loadable as battery tests."""
    from prompt_prix.benchmarks import CustomJSONLoader
    tests = CustomJSONLoader.load("examples/tool_competence_tests.json")
    assert len(tests) > 0
    assert all(hasattr(t, 'id') for t in tests)
```

### Smoke test for UI components
**File:** `tests/test_main.py`
```python
def test_models_selector_has_vetted_choices():
    """Model selector should have hardcoded HF model choices."""
    app = create_app()
    # Verify CheckboxGroup has expected models
    # (inspect app.blocks or mock check)
```

## Verification

```bash
# Run tests
.venv/bin/python -m pytest tests/ -v

# Manual verification
.venv/bin/python -m prompt_prix
# 1. Models should appear as checkboxes (no textbox)
# 2. Example button should load test file
# 3. Run battery with example file
```
