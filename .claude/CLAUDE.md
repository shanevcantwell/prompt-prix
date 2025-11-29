# prompt-prix Development Guide

**Purpose:** Find your optimal open-weights model by running benchmarks across LM Studio servers.

---

## Project Architecture

```
prompt_prix/
├── main.py          # Entry point, Gradio launch
├── ui.py            # Gradio UI definition (Battery-first design)
├── ui_helpers.py    # CSS, JS constants
├── handlers.py      # Event handlers for UI
├── core.py          # ServerPool, ComparisonSession, streaming
├── adapters.py      # LMStudioAdapter (provider abstraction)
├── battery.py       # BatteryRunner, test execution engine
├── benchmarks/      # Test loaders (CustomJSONLoader)
├── config.py        # Configuration, .env loading
├── parsers.py       # Input parsing utilities
└── state.py         # Global mutable state
```

### Key Design Patterns

**1. Provider-Agnostic Adapters**
```python
# adapters.py - Abstract interface for LLM providers
class LMStudioAdapter:
    """Wraps ServerPool for battery execution."""
    async def complete(self, model_id, messages, ...) -> str
```

**2. Fail-Fast Validation**
```python
# Validate before expensive operations
def battery_validate_file(file_obj) -> str:
    """Returns ✅ message if valid, ❌ if not."""
```

**3. Explicit State Management**
```python
# state.py - Clear ownership of mutable state
server_pool: Optional[ServerPool] = None
session: Optional[ComparisonSession] = None
battery_run: Optional[BatteryRun] = None
```

---

## Configuration

### Environment Variables (.env)
```bash
# Server URLs (LM_STUDIO_SERVER_1, LM_STUDIO_SERVER_2, ...)
LM_STUDIO_SERVER_1=http://192.168.1.10:1234
LM_STUDIO_SERVER_2=http://192.168.1.11:1234

# Optional
GRADIO_PORT=7860
BEYOND_COMPARE_PATH=/path/to/bcompare
```

### Gradio Compatibility
- **Gradio 5.x**: `theme`/`css` on `gr.Blocks()`
- **Gradio 6.x**: `theme`/`css` on `.launch()`
- Code auto-detects via `inspect.signature()`

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_battery.py -v
```

### Test Patterns
- Use `respx` for mocking HTTP requests
- Use `pytest-asyncio` for async tests
- Fixtures in `conftest.py` for shared setup

---

## Git Operations Safety

### NEVER Use These Commands
```bash
# FORBIDDEN - wipes working directory, breaks VS Code workspace
git rm -rf .
git clean -fdx  # without explicit user confirmation

# FORBIDDEN - orphan + cleanup combo is dangerous
git checkout --orphan <branch> && git rm -rf .
```

### Critical Files to Preserve
- `*.code-workspace` - VS Code workspace (deleting detaches Claude)
- `.vscode/` - Editor settings
- `.claude/` - Claude Code configuration

### Safe Branch Operations
```bash
# SAFE - create branch, preserve working directory
git branch <new-branch>
git checkout <new-branch>

# SAFE - use GitHub UI for orphan branches

# SAFE - selective file operations
git checkout <branch> -- specific/file.txt
```

### Pre-Flight Checks
Before any destructive git operation:
1. Verify `development/testing` branch has all work
2. Confirm working directory can be restored from `.git/`
3. Ask user before `rm`, `clean`, or `reset --hard`

### Recovery
```bash
git checkout HEAD -- .    # restore all tracked files
git stash list            # check stashed changes
git reflog                # find lost commits
```

---

## Development Workflow

1. **Run locally**: `python -m prompt_prix.main`
2. **Test changes**: `pytest tests/ -v`
3. **Commit**: Use descriptive messages, include Co-Authored-By for Claude

### Branch Structure
- `main` - Release branch
- `development/functional` - Feature preview
- `development/testing` - Active development
