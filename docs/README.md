# prompt-prix Documentation

Welcome to the prompt-prix codebase documentation. This guide is designed to help humans and AI assistants quickly understand the project's goals, architecture, and patterns.

## What is prompt-prix?

**prompt-prix** is a visual fan-out UI for running the same evaluation prompts across multiple LLMs simultaneously and comparing results side-by-side. Rather than creating yet another eval framework, it serves as a **visual comparison layer** on top of existing benchmark ecosystems like [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) (Berkeley Function Calling Leaderboard) and [Inspect AI](https://inspect.ai-safety-institute.org.uk/).

### Core Concept: Fan-Out Pattern

The **fan-out pattern** is the central abstraction:
- **Input**: One prompt (or benchmark test case)
- **Output**: N model responses in parallel
- **Value**: Visual comparison that eval frameworks don't provide

This is not a replacement for proper evaluation frameworks—it's a way to quickly "audition" models using prompts from those frameworks.

### Primary Use Cases

1. **Model Selection** - Compare candidate models for agentic workflows using standardized benchmarks
2. **Quick Comparison** - Fan out a single prompt to see how different models (or quantizations) respond
3. **Benchmark Exploration** - Import test cases from BFCL, Inspect AI, or custom JSON and see results visually
4. **Multi-GPU Utilization** - Dispatch tests across multiple inference servers in parallel

### Positioning in the Ecosystem

| Tool | Purpose |
|------|---------|
| **BFCL** | Function-calling benchmark with leaderboard |
| **Inspect AI** | Evaluation framework for safety testing |
| **NESTful** | Academic benchmark for nested API calls |
| **prompt-prix** | Visual fan-out UI for comparing model responses |

prompt-prix complements these tools by providing a visual layer for side-by-side comparison during model selection.

### Key Features

1. **Fan-Out Dispatch** - Same prompt to N models simultaneously
2. **Visual Comparison** - Real-time streaming with status indicators per model
3. **Parallel Dispatch** - Concurrent execution across multiple GPUs
4. **Semantic Validation** - Detect model refusals, missing tool calls, and verdict mismatches (not just HTTP success)
5. **Promptfoo Support** - Load test suites from promptfoo YAML files with `expected_verdict` validation
6. **LLM-as-Judge** - Select a judge model for semantic pass/fail evaluation
7. **ReAct Loop Evaluation** - Execute multi-step tool-use loops with cycle detection and trace recording
8. **Embedding Drift Validation** - Compare model responses against expected exemplars via semantic distance
9. **Consistency Testing** - Run each test N times with different seeds to identify variance
10. **Session Persistence** - Save and restore UI state
11. **Export** - Generate Markdown/JSON/CSV reports for analysis
12. **Image Attachment** - Send images to vision models in Compare tab
13. **Reproducible Outputs** - Optional seed parameter for deterministic results
14. **Repeat Penalty** - Configurable penalty to reduce repetitive outputs

## Documentation Index

| Document | Description |
|----------|-------------|
| [DESIGN.md](DESIGN.md) | Original design specification and requirements |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, module breakdown, and data flow |
| [EXTENDING.md](EXTENDING.md) | Guide for adding features and understanding patterns |
| [adr/](adr/) | Architecture Decision Records |

### Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [ADR-001](adr/001-use-existing-benchmarks.md) | Use existing benchmarks instead of custom eval schema |
| [ADR-002](adr/002-fan-out-pattern-as-core.md) | Fan-out pattern as core architectural abstraction |
| [ADR-003](adr/003-openai-compatible-api.md) | OpenAI-compatible API as integration layer |
| [ADR-006](adr/ADR-006-adapter-resource-ownership.md) | Adapters own their resource management |
| [ADR-008](adr/ADR-008-judge-scheduling.md) | Pipelined judge scheduling for multi-GPU efficiency |
| [ADR-009](adr/ADR-009-interactive-battery-grid.md) | Dismissible dialog for battery grid cell detail |
| [ADR-010](adr/ADR-010-consistency-runner.md) | Multi-run consistency analysis |
| [ADR-011](adr/ADR-011-embedding-based-validation.md) | Embedding-based semantic validation |

## Quick Start for Developers

### Prerequisites

- Python 3.10+
- LM Studio running on one or more servers with models loaded
- Virtual environment recommended

### Setup

```bash
# Clone and install
git clone https://github.com/shanevcantwell/prompt-prix.git
cd prompt-prix
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest

# Start the app
prompt-prix
```

### Environment Configuration

Create a `.env` file with your LM Studio servers:

```bash
LM_STUDIO_SERVER_1=http://192.168.1.10:1234
LM_STUDIO_SERVER_2=http://192.168.1.11:1234
GRADIO_PORT=7860
BEYOND_COMPARE_PATH=/usr/bin/bcompare  # Optional
```

## Core Concepts

### Server Pool
Multiple LM Studio servers can host different (or overlapping) sets of models. The ServerPool manages:
- Manifest refresh (discovering available models)
- Server busy state tracking
- Work distribution across servers

### Comparison Session
A session maintains:
- Selected models to compare
- Separate conversation context per model
- Configuration (temperature, max tokens, system prompt)
- Halt state (if any model fails)

### Concurrent Dispatcher
For efficient GPU utilization, tests are dispatched in parallel:
- Queue all work items (model + test pairs)
- Find idle servers that have the required model loaded
- Dispatch work and stream responses concurrently

### Battery File Formats

The Battery tab accepts test files in multiple formats:
- **JSON** - Array of test cases with `prompts` key
- **JSONL** - One test case per line
- **Promptfoo YAML** - Promptfoo config files with `prompts` and `tests` sections

For promptfoo files, `expected_verdict` in `vars` enables verdict matching validation. See [ARCHITECTURE.md](ARCHITECTURE.md) for format details.

### Semantic Validation

Battery tests validate responses beyond HTTP success:

| Check | Description |
|-------|-------------|
| Empty response | No content returned |
| Refusal detection | Common refusal phrases |
| Tool call validation | When `tool_choice: "required"` |
| Verdict matching | When `pass_criteria` specifies expected verdict |
| Drift validation | When `expected_response` set and `drift_threshold > 0` |

See [EXTENDING.md](EXTENDING.md) for customizing validation.

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| UI Framework | Gradio 4.x | Web interface with reactive components |
| HTTP Client | httpx | Async HTTP for LM Studio API calls |
| Data Models | Pydantic 2.x | Configuration and state validation |
| Streaming | SSE (Server-Sent Events) | Real-time response streaming |
| Persistence | localStorage (browser) | Save/restore UI state |
| Testing | pytest + pytest-asyncio | Async test support |

## Design Principles

The codebase follows patterns documented in `.claude/CLAUDE.md`:

1. **Fail-Fast Validation** - Validate servers and models before starting sessions
2. **Explicit State Management** - Separate session state from UI state
3. **Separation of Concerns** - UI (ui.py), handlers (handlers.py), and core logic (core.py)
4. **Progressive Error Handling** - Show human-readable errors, halt on model failures

## File Structure Overview

```
prompt_prix/
├── __init__.py           # Package version
├── main.py               # Entry point, adapter registration
├── ui.py                 # Gradio UI definition
├── handlers.py           # Shared event handlers (fetch, stop)
├── state.py              # Global mutable state
├── core.py               # ComparisonSession (orchestration)
├── battery.py            # BatteryRunner (orchestration)
├── consistency.py        # ConsistencyRunner - multi-run variance testing
├── config.py             # Pydantic models, constants, env loading
├── parsers.py            # Text parsing utilities
├── export.py             # Markdown/JSON report generation
├── semantic_validator.py # Refusal detection, tool call validation
├── react/                # ReAct loop execution
│   ├── dispatch.py       # execute_test_case() — single dispatch
│   ├── schemas.py        # ReActIteration, ToolCall data models
│   └── cycle_detection.py
├── mcp/
│   ├── registry.py       # Adapter registry
│   └── tools/            # MCP primitives (complete, react_step, judge, drift)
├── tabs/
│   ├── battery/handlers.py
│   └── compare/handlers.py
├── adapters/
│   ├── base.py           # HostAdapter protocol
│   └── lmstudio.py       # LMStudioAdapter
└── benchmarks/           # Test case loaders (JSON, JSONL, Promptfoo YAML)
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for detailed module documentation
- Read [EXTENDING.md](EXTENDING.md) to understand how to add features
- Run `pytest -v` to see the test suite structure
