---
title: prompt-prix
emoji: 🏎️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
---

<img width="1024" height="506" alt="LLM function calling benchmark grid - prompt-prix visual comparison dashboard" src="https://github.com/user-attachments/assets/2b251520-ed77-40e1-8fef-2a48a1156f2a" />

# prompt-prix

**Audit local LLM function calling and agentic reliability.**

You have a 24GB GPU. Should you run `gpt-oss-20b` or `lfm2.5-1.2b-instruct` for tool calling? BFCL gives you leaderboard scores for full-precision models on datacenter hardware. That doesn't answer your question.

prompt-prix answers the question that matters: **Which model follows tool-use constraints reliably on YOUR hardware, TODAY?**

## Why Function Calling Benchmarks Matter

Agentic AI frameworks like LangGraph, AutoGPT, and CrewAI assume models will:
- Call the right tool from a set of options
- Respect `tool_choice: "none"` when told not to use tools
- Emit valid JSON schemas, not hallucinated parameters
- Follow system prompt constraints instead of refusing

**Most benchmarks don't test this.** They measure next-token prediction (perplexity) or multiple-choice accuracy (MMLU). Neither tells you if a model will stay inside the guardrails when you deploy it.

prompt-prix runs **tool-use compliance tests** against your candidate models, on your hardware, and shows you which ones pass.

## Quantization Testing: Is 4-bit Good Enough?

The local LLM community has repeated "Q4 quantization is fine" for years. That claim is based on perplexity scores and vibes—not structured output reliability.

Is it actually true for function calling? Run the same tests against:
- `llama-3-8b-instruct` (FP16)
- `llama-3-8b-instruct-q4_k_m`
- `llama-3-8b-instruct-iq4_xs`

If FP16 passes 15/15 and Q4 passes 8/15, you have actionable data. If they both pass, you've validated the quantization for your use case.

## Prompt-Prix Core Features

| Feature | What It Does |
|---------|--------------|
| **Fan-Out Dispatch** | Same test → N models in parallel |
| **Semantic Validation** | Detects refusals and missing tool calls (not just HTTP success) |
| **Model-Family Parsing** | Recognizes tool calls from LiquidAI, Hermes, OpenAI formats |
| **Parallel Dispatch** | Concurrent multi-GPU execution |
| **Latency Capture** | Per-test timing on YOUR hardware |
| **Visual Grid** | Model × Test results at a glance |

<img width="1024" height="506" alt="LLM tool use test results - model comparison grid" src="https://github.com/user-attachments/assets/1bc2b4df-90fb-4212-8789-338b84e77ed4" />

## Tested Models

Works with any model served via OpenAI-compatible API. Tested on:

- **Llama 3 / 3.1 / 3.2** — Instruct variants, various quantizations
- **Qwen 2.5** — 7B, 14B, 72B instruct
- **Mistral / Mixtral** — 7B instruct, 8x7B
- **Phi-3 / Phi-3.5** — Mini, Medium
- **DeepSeek** — V2, V2.5, Coder
- **LiquidAI LFM** — 1.2B, 3B tool-use variants

*Using [LM Studio](https://lmstudio.ai/) as the inference backend. Ollama support planned.*

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

## LLM Tool-Use Test Suites

prompt-prix ships with `examples/tool_competence_tests.json`—15 tests covering:

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
    ↓
Observability captures tool calls (LangSmith, Arize, custom)
    ↓
Export traces as BFCL/JSON test cases
    ↓
prompt-prix auditions:
    • Base models → which handles YOUR patterns?
    • SFT checkpoints → is fine-tuning improving?
    • Quantizations → what precision do you need?
    ↓
Visual grid + JSON export
    ↓
Informed RL/SFT decisions
```

This isn't about running someone else's benchmarks. It's about testing models against **your actual usage patterns**—the tool calls your system makes in production—and rapidly iterating on fine-tuning with immediate visual feedback.

## Semantic Validation

HTTP 200 doesn't mean success. A model that returns:

> "I'm sorry, but I can't execute scripts or run code..."

...has **semantically failed** the task, even though the API call succeeded.

prompt-prix detects:
- **Refusals**: Common patterns like "I can't", "As an AI", "I'm not able to"
- **Missing tool calls**: When `tool_choice: "required"` but response is plain text
- **Forbidden tool calls**: When `tool_choice: "none"` but model calls anyway

Results show ✓ (pass), ⚠ (semantic failure), or ❌ (error).

## Ecosystem Position

| Tool | Purpose |
|------|---------|
| [BFCL](https://github.com/ShishirPatil/gorilla) | Function-calling leaderboard (datacenter benchmarks) |
| [Inspect AI](https://inspect.ai-safety-institute.org.uk/) | Safety evaluation framework |
| **prompt-prix** | Visual function-calling audit on YOUR hardware |

## Related Projects

- **[langgraph-agentic-scaffold](https://github.com/shanevcantwell/langgraph-agentic-scaffold)** — Agentic framework built on the principle that safety comes from structure, not trust. prompt-prix auditions models for deployment in LAS workflows.

## Status

Alpha release. Core functionality works. Expect rough edges.

## Documentation

- [docs/README.md](docs/README.md) — Architecture overview
- [docs/EXTENDING.md](docs/EXTENDING.md) — Adding features
- [CLAUDE.md](.claude/CLAUDE.md) — AI assistant context

## License

MIT

---

(C) 2025 Reflective Attention
