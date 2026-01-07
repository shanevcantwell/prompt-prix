<img width="1024" height="506" alt="image" src="https://github.com/user-attachments/assets/2b251520-ed77-40e1-8fef-2a48a1156f2a" />


# prompt-prix

**Find your optimal open-weights model.**

prompt-prix is a visual tool for running benchmark test suites across multiple LLMs simultaneously, helping you discover which model and quantization best fits your VRAM constraints and task requirements.

## The Problem

You have a 24GB GPU. Should you run `gpt-oss-20b` or `lfm2.5-1.2b-instruct` for tool calling? BFCL gives you leaderboard scores for full-precision models. That doesn't answer your question. This is a different kind of metric.

## The Solution

Run existing benchmarks against *your* candidate models, on *your* hardware, and see results side-by-side.

- **Fan-out dispatch**: Same test case → N models in parallel
- **Work-stealing scheduler**: Efficient multi-GPU utilization across heterogeneous workstations
- **Visual comparison**: Real-time streaming with Model × Test result grid
- **Benchmark-native**: Consumes BFCL and Inspect AI test formats directly

<img width="1024" height="506" alt="image" src="https://github.com/user-attachments/assets/1bc2b4df-90fb-4212-8789-338b84e77ed4" />

## Status

🚧 **Alpha Release Dec 30, 2025**

## Quick Start

```bash
# Clone and install
git clone https://github.com/shanevcantwell/prompt-prix.git
cd prompt-prix
pip install -e .

# Configure LM Studio servers
cp .env.example .env
# Edit .env: LM_STUDIO_SERVER_1=http://localhost:1234

# Run
prompt-prix
```

Opens at `http://localhost:7860`. Requires [LM Studio](https://lmstudio.ai/) running with models loaded.

**Or with Docker:**
```bash
docker compose up
```

## Ecosystem Position

| Tool | Purpose |
|------|---------|
| [BFCL](https://github.com/ShishirPatil/gorilla) | Function-calling benchmark with leaderboard |
| [Inspect AI](https://inspect.ai-safety-institute.org.uk/) | Evaluation framework (UK AISI) |
| **prompt-prix** | Visual fan-out for model selection |

prompt-prix complements these tools—it's a visual layer for comparing models during selection, not a replacement for rigorous evaluation.

## Architecture Highlights

- **Adapter pattern**: OpenAI-compatible API now (LM Studio), extensible to Ollama/vLLM
- **Fail-fast validation**: Invalid benchmark files rejected immediately
- **Pydantic state management**: Explicit, typed, observable
- **Work-stealing dispatcher**: Asymmetric GPU setups handled automatically

## License

MIT

## Links

- [Development branch](https://github.com/shanevcantwell/prompt-prix/tree/development/testing) — working code
- [BFCL](https://github.com/ShishirPatil/gorilla) — upstream benchmark source
- [Inspect AI](https://inspect.ai-safety-institute.org.uk/) — UK AISI evaluation framework

(C) 2025 Reflective Attention
