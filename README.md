---
title: prompt-prix
emoji: 🏎️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
---

<img width="1024" height="506" alt="LLM function calling benchmark grid - prompt-prix visual comparison dashboard" src="https://github.com/user-attachments/assets/2b251520-ed77-40e1-8fef-2a48a1156f2a" />

# prompt-prix

**MCP toolkit for multi-model testing and agentic self-improvement.**

9 stateless tools over MCP stdio for completion, judging, semantic drift, ReAct execution, prompt geometry, and trajectory analysis. Agents call these tools to audition specialist LLMs, measure reliability across quantizations, and drive multi-step tool-use loops against real services.

Includes a Gradio UI for human visual comparison.

| Audience | Entry Point | Transport |
|----------|-------------|-----------|
| Agents (LAS, Claude Desktop) | `prompt-prix-mcp` | MCP stdio (JSON-RPC) |
| Scripts / CI | `prompt-prix-cli` | CLI, structured JSON output |
| Humans | `prompt-prix` | Gradio web UI at localhost:7860 |

## Why Function Calling Benchmarks Matter

Agentic frameworks like LangGraph, AutoGPT, and CrewAI assume models will:
- Call the right tool from a set of options
- Respect `tool_choice: "none"` when told not to use tools
- Emit valid JSON schemas, not hallucinated parameters
- Follow system prompt constraints instead of refusing

**Most benchmarks don't test this.** They measure next-token prediction (perplexity) or multiple-choice accuracy (MMLU). Neither tells you if a model will stay inside the guardrails when you deploy it.

prompt-prix provides the MCP tools an agentic system needs to answer this question for itself — testing candidate models on your hardware, against your actual tool-use patterns, and feeding the results back into model selection.

## Quantization Testing: Is 4-bit Good Enough?

The local LLM community has repeated "Q4 quantization is fine" for years. That claim is based on perplexity scores and vibes — not structured output reliability.

Is it actually true for function calling? Run the same tests against:
- `llama-3-8b-instruct` (FP16)
- `llama-3-8b-instruct-q4_k_m`
- `llama-3-8b-instruct-iq4_xs`

If FP16 passes 15/15 and Q4 passes 8/15, you have actionable data. An agentic system can use this to select its own specialist models — no human in the loop required.

## Core Features

| Feature | What It Does |
|---------|--------------|
| **9 MCP Tools** | complete, judge, react_step, drift, geometry, trajectory — all stateless, all over stdio |
| **Fan-Out Dispatch** | Same test dispatched to N models in parallel |
| **Tool-Forwarding Mode** | `react_step(mock_tools=None)` returns parsed tool calls for caller dispatch against real services |
| **Multi-Adapter Routing** | LM Studio, Together AI, HuggingFace — CompositeAdapter routes by model_id |
| **Semantic Validation** | Detects refusals and missing tool calls (not just HTTP success) |
| **LLM-as-Judge** | Semantic pass/fail evaluation with pipelined GPU scheduling |
| **Consistency Testing** | Run tests N times with different seeds to find unreliable models |
| **Multi-GPU Dispatch** | Model-drain guard prevents VRAM swap mid-stream via [local-inference-pool](https://github.com/shanevcantwell/local-inference-pool) |
| **Latency Capture** | Per-test timing on YOUR hardware |
| **Visual Grid** | Model x Test results at a glance (Gradio UI) |

<img width="1024" height="506" alt="LLM tool use test results - model comparison grid" src="https://github.com/user-attachments/assets/1bc2b4df-90fb-4212-8789-338b84e77ed4" />

## Quick Start

```bash
git clone https://github.com/shanevcantwell/prompt-prix.git
cd prompt-prix
pip install -e .

# Configure your LM Studio server(s)
cp .env.example .env
# Edit .env: LM_STUDIO_SERVER_1=http://localhost:1234

prompt-prix
```

Opens at `http://localhost:7860`. Requires [LM Studio](https://lmstudio.ai/) with models loaded.

**Docker:**
```bash
docker compose up
```

**MCP server (for agents):**
```bash
prompt-prix-mcp
```

Launches the MCP stdio server. Agents connect via JSON-RPC and call tools directly.

## LLM Tool-Use Test Suites

prompt-prix ships with `examples/tool_competence_tests.json` — 15 tests covering:

| Category | Tests |
|----------|-------|
| Basic tool invocation | Does it call the tool at all? |
| Tool selection | Does it pick the right tool from 3 options? |
| Constraint compliance | Does it respect "don't use this tool"? |
| Schema compliance | Does it emit valid enum values, nested objects, required params? |
| Tool judgment | Does it know when NOT to use tools? |

Load your own tests in JSON/JSONL format, or import directly from [BFCL](https://github.com/ShishirPatil/gorilla).

## From Observability to Model Improvement

The real power of BFCL-compatible formats: **your production traces become your test suite**.

```
Agentic system in production
    |
Observability captures tool calls (LangSmith, Arize, custom)
    |
Export traces as BFCL/JSON test cases
    |
prompt-prix MCP tools audition:
    - Base models: which handles YOUR patterns?
    - SFT checkpoints: is fine-tuning improving?
    - Quantizations: what precision do you need?
    |
JSON export -> informed RL/SFT decisions
```

An agentic system can close this loop autonomously — capturing its own tool-call traces, converting them to test cases, and using prompt-prix to evaluate candidate models without human intervention.

## Semantic Validation

HTTP 200 doesn't mean success. A model that returns:

> "I'm sorry, but I can't execute scripts or run code..."

...has **semantically failed** the task, even though the API call succeeded.

prompt-prix detects:
- **Refusals**: Common patterns like "I can't", "As an AI", "I'm not able to"
- **Missing tool calls**: When `tool_choice: "required"` but response is plain text
- **Forbidden tool calls**: When `tool_choice: "none"` but model calls anyway

Results show COMPLETED, SEMANTIC_FAILURE, or ERROR.

## LLM-as-Judge Evaluation

Select a judge model to evaluate whether responses meet semantic criteria defined in your test cases (`pass_criteria` / `fail_criteria`). Judge tasks are **pipelined** with inference — as each inference result completes, its judge task is submitted to the dispatcher. Idle GPUs pick up judge work while busy GPUs continue inference.

```
GPU0: inference -> inference -> judge -> judge -> judge    (GPU0 finishes inference early, starts judging)
GPU1: inference -> inference -> inference -> inference     (GPU1 still doing heavy models)
```

No priority queues, no server affinity — the existing `current_model` drain guard routes judge tasks to whichever GPU is idle.

## Consistency Testing

Run each (test, model) cell N times with different random seeds to identify models that produce inconsistent results:

| Symbol | Meaning |
|--------|---------|
| COMPLETED | N/N passed (consistent pass) |
| SEMANTIC_FAILURE | 0/N passed (consistent fail) |
| INCONSISTENT 3/5 | Inconsistent — passed some runs but not all |

## Related Projects

- **[langgraph-agentic-scaffold](https://github.com/shanevcantwell/langgraph-agentic-scaffold)** — Agentic framework built on the principle that safety comes from structure, not trust. LAS uses prompt-prix MCP tools for specialist model audition and ReAct loop execution.
- **[local-inference-pool](https://github.com/shanevcantwell/local-inference-pool)** — Multi-GPU dispatch with model-drain guards. Extracted from prompt-prix, shared by both prompt-prix and LAS.

## Status

Alpha release. Core functionality works. Expect rough edges.

## Documentation

- [docs/README.md](docs/README.md) — User guide (UI tabs + MCP tools)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture and layer model
- [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md) — MCP tool API reference (9 tools)
- [CLAUDE.md](.claude/CLAUDE.md) — AI assistant context

## License

MIT

---

(C) 2025-2026 Reflective Attention
