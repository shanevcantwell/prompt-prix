# Deferred Tests

Tests in this directory are out of scope for the current release.

## Why Deferred

These tests cover Gemini web UI automation functionality that was de-scoped
when narrowing focus to LM Studio core features. The underlying adapters
(`gemini_webui.py`, `gemini_visual.py`, `fara.py`) remain in the codebase
for future development.

## Contents

- `test_gemini_integration.py` - Integration tests requiring:
  - Active Gemini browser session (`prompt-prix-gemini --on`)
  - Fara-7B vision model running in LM Studio
  - External network access to Gemini web UI

## Running Deferred Tests

When Gemini functionality returns to scope:

```bash
pytest tests/deferred/ -m integration -v
```

## Re-integrating

To bring these tests back into the main suite:
1. Move files from `tests/deferred/` to `tests/`
2. Ensure prerequisites documented above are met
3. Update `pytest.ini` if integration marker handling changes
