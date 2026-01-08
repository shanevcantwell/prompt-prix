# prompt-prix

**Purpose:** Visual fan-out UI for running evaluation prompts across multiple LLMs simultaneously and comparing results side-by-side.

---

## Communication Style

- Calm and level, professional tone
- Avoid premature confidence: "Ready for production!" before testing is overconfident
- Follow the user's pace: don't escalate urgency unnecessarily

---

## Working Model: Co-Architects

Pair programming relationship, not code generation service. The value is in the conversation that precedes implementation.

- Understand *why* before proposing *how*
- Explore tradeoffs out loud
- Read existing code before writing new code
- Every line of code is measured investment—make it count

When in doubt: discuss the approach first.

---

## Change Management

**Primary workflow. Not optional.**

```
IDENTIFY → FILE → TEST-DRIVEN LOOP → COMMIT → CLOSE
```

Core rules:
- File the issue before writing code, even for "quick fixes"
- Write failing test before implementing fix
- One issue = one atomic commit
- Close with commit reference

**Before any commit or significant code change:** Read `.claude/CHANGE_MANAGEMENT.md` and confirm: "Reviewed CHANGE_MANAGEMENT.md, proceeding with [step]."

---

## Git Safety

**Before any branch operation** (checkout, delete, rebase): Read `.claude/GIT_SAFETY.md`

Core rules:
- Never delete a branch containing the only copy of work
- Never checkout in a way that invalidates the workspace file
- Confirm current branch and uncommitted changes before operations

---

## Repository Boundaries

This project is: `prompt-prix`

Do not commit to or push other repositories without explicit instruction. Symlinks may cross repo boundaries—this does not grant write permission.

---

## Testing

**Before writing or modifying tests:** Read `.claude/TESTING.md`

Core rules:
- Integration tests for model-dependent code (adapters, tool use)
- Unit tests for parsing, config, pure utilities
- Check for skipped tests—"224 passed, 15 skipped" requires explanation

---

## Session Start Ritual

```bash
gh issue list --state open   # What's in flight
git status                   # Uncommitted work
git log --oneline -5         # Recent context
```

---

## Core Concept: Fan-Out Pattern

prompt-prix is a visual comparison layer, not an eval framework.

```
Input:  One prompt (or benchmark test case)
        ↓
Fan-Out: Dispatch to N models in parallel
        ↓
Output: Side-by-side visual comparison
```

| Tool | Purpose |
|------|---------|
| BFCL | Function-calling benchmark |
| Inspect AI | Evaluation framework |
| prompt-prix | Visual fan-out comparison |

---

## Architecture

### UI Layout (v2)

```
┌─────────────────────────────────────────────────────────┐
│  [Servers]              [Models]                        │
│  http://localhost:1234  ☑ model-a  ☑ model-b            │
│  Temperature: 0.7    Timeout: 120s    Max Tokens: 2048  │
├─────────────────────────────────────────────────────────┤
│  [Battery]  [Compare]                                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Tab-specific content                             │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `core.py` | ComparisonSession, streaming |
| `scheduler.py` | ServerPool, BatchRunner |
| `battery.py` | BatteryRunner orchestrator |
| `semantic_validator.py` | Refusal detection, tool call validation |
| `adapters/` | LMStudio, Gemini, Fara providers |
| `tabs/` | Battery and Compare tab UI/handlers |

For full file structure and module details, see `docs/ARCHITECTURE.md`.

---

## Adapters

- **LMStudioAdapter**: OpenAI-compatible, local models
- **GeminiVisualAdapter**: Fara-7B vision for Gemini web UI (preferred)
- **FaraService**: Visual element location
- **GeminiWebUIAdapter**: DOM-based, deprecated

---

## Tabs

### Battery
Run benchmark test suites. Model × Test grid with ✓/⚠/❌ status.

### Compare  
Multi-turn context engineering workshop. Build scenarios, export as Battery test cases.

See `docs/adr/004-compare-to-battery-export.md` for export workflow.

---

## Battery File Formats

Supports JSON, JSONL, and BFCL formats. Required fields: `id`, `user`.

See `docs/BATTERY_FORMATS.md` for full schema and examples.

---

## Semantic Validation

Validates beyond HTTP success:
- **Refusal detection**: "I'm sorry, but..." patterns
- **Tool call validation**: Enforces `tool_choice` constraints

| Status | Symbol | Meaning |
|--------|--------|---------|
| COMPLETED | ✓ | Passed validation |
| SEMANTIC_FAILURE | ⚠ | Response failed semantic check |
| ERROR | ❌ | Infrastructure error |

---

## Environment

```bash
LM_STUDIO_SERVER_1=http://localhost:1234
FARA_SERVER_URL=http://localhost:1234
FARA_MODEL_ID=microsoft_fara-7b
GRADIO_PORT=7860
```

---

## Future Direction

Fara adapter evolving toward MCP service architecture. See `docs/MCP_ROADMAP.md`.
