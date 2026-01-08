# Testing Workflow

**Checkpoint: TESTING-LOADED** — Confirm you've read this file before writing or modifying tests.

---

## Philosophy

For functionality that depends on LLM behavior (adapters, visual verification, tool use), **integration tests are the meaningful tests**. Unit tests with mocked LLM responses don't validate anything useful—they just test that your mock returns what you told it to return.

---

## When to Use What

### Unit Tests

- Data parsing (JSON/JSONL loaders)
- Configuration validation
- Pure utility functions
- State management logic

### Integration Tests

- Adapter implementations (LMStudio, Gemini, Fara)
- Visual UI verification (FaraService)
- ReAct tool loops
- End-to-end prompt execution

---

## Running Tests

```bash
# Unit tests only (default, skips integration)
pytest

# Integration tests (the ones that matter for model code)
pytest -m integration

# Specific test class
pytest tests/test_gemini_adapter.py::TestGeminiVisualAdapter -m integration -v

# With coverage
pytest --cov=prompt_prix
```

---

## Test Output Verification

**After every test run, check for:**

### Skipped Tests

"224 passed, 15 skipped" requires explanation before proceeding.

Common reasons for skips:
- `@pytest.mark.integration` for integration tests (expected if not running integration)
- Missing fixture or import (problem—investigate)
- Conditional skip that shouldn't apply (problem—investigate)

**Do not proceed until you understand why tests were skipped.**

### Warnings

May indicate silent failures. Read them.

### Coverage

Did the test actually exercise the changed code path? If you modified `scheduler.py` but coverage shows those lines weren't hit, your test isn't testing what you think.

---

## Integration Test Prerequisites

Before running integration tests:

1. **LM Studio running** with required models loaded
2. **Gemini session active** (if testing Gemini adapters):
   ```bash
   prompt-prix-gemini --on
   prompt-prix-gemini --status
   ```
3. **`.env` configured** with correct server URLs

---

## Writing New Tests

When adding model-dependent functionality:

1. Write the integration test first (TDD for real behavior)
2. Mark with `@pytest.mark.integration`
3. Document prerequisites in test docstring
4. Consider adding a "smoke test" for quick verification

---

## Adapting Code from Other Projects

When bringing code from other projects (e.g., langgraph-agentic-scaffold):

1. **Check dependencies first** — Does it import packages not in pyproject.toml?
2. **Adapt, don't copy** — Rewrite to use prompt-prix infrastructure:
   - Use `httpx` not `langchain` for HTTP
   - Use `pydantic` models that match existing patterns
   - Use existing adapters, not foreign abstractions
3. **Tests must run** — If tests fail due to missing imports, the code isn't ready
4. **No orphan files** — Don't commit code that can't be imported or tested

Code that "looks right" but breaks `pytest` is worse than no code—it creates false confidence and cleanup debt.

---

## Test Markers

- `@pytest.mark.integration` — Requires external services
- Default pytest config skips integration tests for CI convenience
