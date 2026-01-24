# prompt-prix

**Purpose:** Visual fan-out MCP service for passing identical data across multiple LLMs simultaneously. Contains the ability to utilize another LLM to act as judge for semantic pass/fail. (TBD: cherry pick implementation from branch v2-simplified)

---

## Communication Style

- Level, professional tone
- Avoid premature confidence: "Ready for production!" before testing is overconfident
- Follow the user's pace: "4 tests passed, let's run a full batch!" is unnecessarily urgent

---

## Working Model: Co-Architects

This is a pair programming relationship, not a code generation vending machine. There is much value in the conversation that precedes implementation.

**What good looks like:**
- Understanding *why* before proposing *how*.
- Exploring options for solutions with detailed scope and risk: "Option A affects files D, E and functionality Q, R, with a level of risk of Y."
- `explore` first documentation, then existing code to understand patterns before writing any new code.
- Treating the codebase as a long-term asset we're stewarding together: it is not just this tool, it also facilitates agentic behavior.

**The goal is working software that remains maintainable** - not code that appears to work, not impressive-looking output, not maximum tokens of plausible implementation.

When in doubt: discuss the approach first. Plan mode is opportunity for the ideal solution, not a penalty box.

---

## Core Concept: Fan-Out Pattern

prompt-prix is NOT an eval framework. It's a visual comparison layer.

```
Input:  One prompt (or benchmark test case)
        ↓
Fan-Out: Dispatch to N models in parallel
        ↓
Output: Side-by-side visual comparison
```

---

## Architecture (Quick Reference)

> **Full details:** See `docs/ARCHITECTURE.md`

### Layer Import Rules

| Layer | MAY Import | MUST NOT Import |
|-------|------------|-----------------|
| **Orchestration** (BatteryRunner, ComparisonSession) | `mcp.tools.*`, `mcp.registry` | `adapters/*`, ServerPool |
| **MCP Primitives** | `adapters.base.HostAdapter` (protocol) | Individual adapter class implementations |
| **Adapters** | httpx, internal utilities | Nothing from orchestration or MCP |

> ServerPool and ConcurrentDispatcher are exclusively internal to LMStudioAdapter.
> No file outside `adapters/lmstudio.py` may import or reference them.

### Key Modules

| Module | Layer | Purpose |
|--------|-------|---------|
| `battery.py` | Orchestration | BatteryRunner - calls MCP tools |
| `core.py` | Orchestration | ComparisonSession |
| `mcp/tools/` | MCP | Primitives (complete, fan_out, judge) |
| `adapters/lmstudio.py` | One Example of an Adapter | LMStudioAdapter (owns ServerPool) |

---

## Battery File Formats

> **Full details:** See `docs/ARCHITECTURE.md` or `benchmarks/base.py`

**Required fields:** `id`, `user`

**Optional fields:** `name`, `category`, `severity`, `system`, `tools`, `tool_choice`, `expected`, `pass_criteria`, `fail_criteria`

**Formats supported:** JSON (with `prompts` array), JSONL (one per line), yaml (TBD: promptfoo - need to cherry pick from branch v2-simplified)

---

## Semantic Validation

> **Full details:** See `prompt_prix/semantic_validator.py`

Battery tests validate responses beyond HTTP success:
1. **Refusal Detection** - Matches common refusal phrases
2. **Tool Call Validation** - For `tool_choice: "required"`, verifies tool calls exist

| Status | Symbol | Meaning |
|--------|--------|---------|
| `COMPLETED` | ✓ | Completed, and passed semantic validation |
| `ERROR` | ⚠ | Failure in response from the adapter or from semantic check |
| `SEMANTIC_FAILURE` | ❌ | Response was successfully received and optionally judged, but did not meet expected criteria |

---

## Testing

> **Full details:** See `docs/ARCHITECTURE.md` "Testing" section

```bash
.venv/                    # virtual environment (necessary for pytest)
pytest                    # Unit tests (default)
pytest -m integration     # Integration tests (requires LM Studio)
```

**Unit test mocking strategy (ADR-006):**
- MCP tool tests → mock adapter via registry
- Orchestration tests → mock MCP tools
- Adapter tests → mock httpx

---

## Environment Configuration

```bash
# .env file
LM_STUDIO_SERVER_1=http://127.0.0.1:1234
LM_STUDIO_SERVER_2=http://192.168.137.2:1234
GRADIO_PORT=7860
```

---

## Git Operations Safety

### NEVER Use These Commands, or otherwise cause the local workspace file to be deleted:
```bash
git rm -rf .
git clean -fdx  # without explicit confirmation
```

### Critical Files to Preserve
- `*.code-workspace`, `.vscode/`, `.claude/`
- `pyproject.toml`, `.env`

---

## Bug Spotters' Guide

1. **File the bug** - `gh issue create --title "..." --body "..."`
2. **Write tests before fixing** - Demonstrate bad behavior, define expected behavior
3. **Implement the fix** - Minimal changes
4. **Run tests** - `pytest -v --tb=short`
5. **Commit with bug reference** - `git commit -m "Fix #N: ..."`
6. **Close the bug with commit reference** - `gh issue close N --comment "Fixed in <commit>"`

---

## Development Workflow

1. Edit code in `prompt_prix/`
2. Run `pytest` to verify
3. Launch with `prompt-prix` command
4. Test against LM Studio servers
5. Commit with descriptive messages

### Adding a New Adapter

1. Create `prompt_prix/adapters/new_adapter.py`
2. Implement `HostAdapter` protocol: `get_available_models()`, `stream_completion()`
3. Encapsulate backend internals (pools, sessions, rate limiters) INSIDE the adapter
4. Add integration tests marked with `@pytest.mark.integration`

**Critical:** Orchestration NEVER imports adapters - it calls MCP tools via registry.
