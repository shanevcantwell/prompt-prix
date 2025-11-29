# prompt-prix

Find your optimal open-weights model. Run benchmarks across LM Studio servers.

## Overview

prompt-prix is a Gradio-based tool for benchmarking and comparing responses from multiple LLMs (open-weights models, quantizations, or any combination) served via LM Studio's OpenAI-compatible API. Designed to "audition" models for use as backing LLMs in agentic workflows.

## Features

**Battery Mode (Primary)**
- Run benchmark test suites across multiple models
- Model x Test results grid with real-time updates
- JSON and JSONL test formats (including BFCL compatibility)
- Export results to JSON or CSV

**Compare Mode (Secondary)**
- Interactive multi-turn conversations
- Side-by-side response comparison
- Streaming responses per model tab
- Export reports in Markdown or JSON

**Shared**
- Support for 2+ LM Studio servers with work-stealing dispatch
- Configurable temperature, timeout, and max tokens
- Quick prompt testing against selected models

## Installation

```bash
# Clone the repository
git clone https://github.com/shanevcantwell/prompt-prix.git
cd prompt-prix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your LM Studio server addresses:
   ```
   LM_STUDIO_SERVER_1=http://192.168.1.10:1234
   LM_STUDIO_SERVER_2=http://192.168.1.11:1234
   ```

3. Optionally configure the Gradio port (default 7860):
   ```
   GRADIO_PORT=7865
   ```

4. Ensure LM Studio is running on your servers with models loaded.

## Usage

### Start the application

```bash
prompt-prix
```

Or run directly:

```bash
python -m prompt_prix.main
```

Open `http://localhost:7860` in your browser.

### Using the interface

**Battery Mode**
1. **Configure Servers**: Expand the Servers panel and enter your LM Studio URLs
2. **Fetch Models**: Click "Fetch Models" to discover available models
3. **Upload Test Suite**: Upload a JSON benchmark file (see format below)
4. **Select Models**: Check the models you want to test
5. **Run Battery**: Click "Run Battery" to execute all tests
6. **View Results**: See the Model x Test grid update in real-time
7. **Export**: Download results as JSON or CSV

**Compare Mode**
1. **Select Models**: Check models to compare
2. **Send Prompt**: Enter a prompt and click "Send to All"
3. **View Responses**: See streaming responses in model tabs
4. **Continue Conversation**: Send follow-up prompts for multi-turn
5. **Export**: Save as Markdown or JSON

### Test Suite Formats

**JSON format:**
```json
{
  "prompts": [
    {
      "id": "test-1",
      "name": "Basic greeting",
      "system": "You are a helpful assistant.",
      "user": "Say hello!"
    },
    {
      "id": "test-2",
      "user": "What is 2+2?",
      "tools": [{"type": "function", "function": {...}}]
    }
  ]
}
```

**JSONL format** (one test per line):
```jsonl
{"id": "test-1", "user": "Hello", "system": "Be helpful"}
{"id": "test-2", "user": "What is 2+2?"}
```

**BFCL format** (auto-detected and normalized):
```jsonl
{"id": "...", "question": [{"role": "user", "content": "..."}], "function": [...]}
```

## Development

### Running tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=prompt_prix --cov-report=html
```

### Project structure

```
prompt-prix/
├── prompt_prix/
│   ├── main.py          # Entry point, Gradio launch
│   ├── ui.py            # Gradio UI components and event bindings
│   ├── ui_helpers.py    # CSS, JS constants
│   ├── handlers.py      # Async event handlers
│   ├── core.py          # ServerPool, ComparisonSession, streaming
│   ├── config.py        # Pydantic models, constants, .env loading
│   ├── parsers.py       # Input parsing utilities
│   ├── export.py        # Markdown/JSON report generation
│   ├── state.py         # Global mutable state
│   ├── battery.py       # BatteryRunner, BatteryRun state
│   ├── adapters/
│   │   ├── base.py      # LLMAdapter protocol
│   │   └── lmstudio.py  # LMStudioAdapter implementation
│   └── benchmarks/
│       ├── base.py      # TestCase model
│       └── custom.py    # CustomJSONLoader (JSON/JSONL/BFCL)
└── tests/
    ├── conftest.py      # Shared fixtures
    ├── test_battery.py  # Battery runner tests
    ├── test_config.py   # Pydantic model tests
    ├── test_core.py     # ServerPool, streaming tests
    ├── test_export.py   # Report generation tests
    └── test_main.py     # Handler tests
```

## License

MIT

(C) 2025 Reflective Attention
