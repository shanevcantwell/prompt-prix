# ADR-007: CLI Interface Layer

**Status**: Accepted
**Date**: 2025-01-23
**Extends**: ADR-006 (adds interface layer above orchestration)

## Context

prompt-prix started as a Gradio UI for visual model comparison. However, manual UI interaction is a bottleneck for:

- **Automated testing** — CI/CD pipelines need headless execution
- **Iteration speed** — Loading browser, uploading files, clicking buttons
- **Scripting** — Batch runs across different benchmark files
- **Integration** — Other tools want to invoke battery runs programmatically

The existing architecture (ADR-006) cleanly separates orchestration from adapters:

```
ORCHESTRATION (BatteryRunner, ComparisonSession)
    ↓
MCP PRIMITIVES (complete, list_models)
    ↓
ADAPTER (LMStudioAdapter)
```

But there's an implicit assumption: Gradio handlers are the only entry point.

## Decision

**The interface layer is separate from orchestration.**

prompt-prix has multiple interface options that share the same orchestration and adapter stack:

```
┌─────────────────────────────────────────────────────────────────┐
│                       INTERFACE LAYER                           │
│                                                                 │
│  Gradio UI (ui.py, tabs/*)     │  CLI (scripts/run_battery.py) │
│  • Interactive visual          │  • Headless automation        │
│  • Browser-based               │  • Terminal output            │
│  • Human-in-the-loop           │  • CI/CD friendly             │
│                                │  • Scriptable                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION                            │
│  BatteryRunner │ ComparisonSession                              │
│  (same code for both interfaces)                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ↓
                    [MCP → Adapter per ADR-006]
```

### Interface Implementations

| Interface | Location | Use Case |
|-----------|----------|----------|
| Gradio UI | `ui.py`, `tabs/*/handlers.py` | Interactive exploration, visual comparison |
| CLI | `scripts/run_battery.py` | Automation, CI/CD, scripting |
| (future) Python API | `prompt_prix.api` | Library usage, custom tooling |

### CLI Design

The CLI is a thin layer that:

1. Parses arguments and environment
2. Creates adapter and registers it
3. Loads benchmark file
4. Calls `BatteryRunner` (same as UI)
5. Formats output for terminal

```bash
# Basic usage
python scripts/run_battery.py examples/tool_competence_tests.json

# Options
python scripts/run_battery.py file.json --random 2  # 2 random models per GPU
python scripts/run_battery.py file.json -v          # Verbose per-test output
python scripts/run_battery.py file.json -e out.json # Export results
```

### Rules

#### MUST

1. Interface code lives outside core package (`scripts/` for CLI, `ui.py` for Gradio)
2. All interfaces use the same orchestration classes (BatteryRunner, etc.)
3. Adapter registration happens in interface layer (not buried in orchestration)
4. CLI reads configuration from same sources as UI (`.env`, environment variables)

#### MUST NOT

1. Orchestration MUST NOT import interface code
2. Orchestration MUST NOT assume Gradio is present
3. CLI MUST NOT duplicate orchestration logic (reuse BatteryRunner)

## Rationale

### Why Not Just Gradio?

Gradio is excellent for interactive exploration but:
- Requires browser and server running
- Manual clicks for each test run
- Can't be automated in CI/CD
- Human bottleneck for iteration

### Why `scripts/` Not `prompt_prix.cli`?

1. **Simplicity** — One-file script is easier to understand and modify
2. **No entry point overhead** — Run directly with `python scripts/...`
3. **Separate from package** — Clear that this is a tool, not library code
4. **Easy to add more** — `scripts/compare.py`, `scripts/export.py`, etc.

If CLI grows complex, can refactor to `prompt_prix.cli` module with proper entry points.

### Precedent

This mirrors common patterns:
- Django: `manage.py` scripts alongside web interface
- pytest: CLI runner over same core library
- alembic: CLI for migrations, programmatic API for same operations

## Consequences

### Positive

- Unblocks automated testing and CI/CD
- Faster iteration (no browser needed)
- Same behavior guaranteed between UI and CLI
- Foundation for future API/library usage

### Negative

- Two places to update if BatteryRunner interface changes
- CLI output formatting is separate from Gradio grid display
- Need to keep CLI in sync with new features

### Future Work

- `scripts/compare.py` — Interactive comparison CLI
- Entry point in `pyproject.toml` for `prompt-prix battery ...`
- Python API (`from prompt_prix import run_battery`)

## References

- ADR-006: Adapter Resource Ownership (the layer below interfaces)
- Issue #99: Per-server queues for parallelism (affects both interfaces)
- Script: `scripts/run_battery.py`
