# Testing Workflow

**Checkpoint: TESTING-LOADED** — Confirm you've read this file before writing or modifying tests.

---

## Philosophy

**The distinction:** Mock to test *handling*, not to test *behavior*.

Mocked LLM/API responses don't validate real behavior—they just confirm the mock returns what you told it to return. This is reward hacking: optimizing for green tests rather than correct code.

For prompt-prix, anything that flows through `ServerPool` or hits LM Studio APIs requires **live servers** to be meaningful.

---

## When to Use What

### Unit Tests (Mocks Valid)

Mock to test handling of responses, not to simulate the external system:

- Data parsing (JSON/JSONL/BFCL loaders)
- String utilities (`parse_prefixed_model()`, `parse_servers_input()`)
- Export generation (given results, test markdown/JSON output)
- Semantic validation patterns (refusal detection, tool call parsing)
- Error paths (timeout handling, malformed response)
- State management (SessionState, ModelContext)

### Integration Tests (Live LM Studio Required)

These require real API calls—mocks cannot verify correctness:

- `ServerPool.refresh()` — real `/v1/models` and `/api/v0/models` responses
- `find_server()` routing — server hints, prefer-loaded logic
- `only_loaded` filtering — actual VRAM state queries
- Battery/Compare end-to-end execution
- Adapter implementations (LMStudioAdapter)

**LM Studio calls are free and fast (<250ms).** There is no cost barrier to integration tests.

---

## xfail and Lowering Expectations - CRITICAL

**ALWAYS ask before marking tests as `xfail` or changing assertions to be more permissive.**

LLMs optimize toward "green"—we instinctively want tests to pass. This creates bias toward marking failures as "expected" rather than fixing underlying issues.

When a test fails:
1. **First**: Understand why it's failing
2. **Second**: Propose options (fix the issue, adjust test, mark xfail)
3. **Third**: Ask the user which approach they prefer
4. **Never**: Unilaterally mark something xfail or weaken assertions

This applies to any change that lowers the bar: removing expected values, broadening assertions, etc.

---

## Running Tests

```bash
# Unit tests only (default, skips integration)
pytest

# Integration tests (the ones that matter for model code)
pytest -m integration

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
2. **`.env` configured** with correct server URLs

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
