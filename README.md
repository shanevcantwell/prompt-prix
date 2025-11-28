# prompt-prix

Multi-LLM head-to-head comparison tool for LM Studio servers.

## Overview

prompt-prix is a Gradio-based tool for comparing responses from multiple LLMs (open-weights models, quantizations, or any combination) served via LM Studio's OpenAI-compatible API. Designed to "audition" models for use as backing LLMs in agentic workflows.

## Features

- Compare responses from multiple models side-by-side
- Support for 2+ LM Studio servers
- Multi-turn conversations with isolated context per model
- Interactive prompt input or batch file processing
- Streaming responses per model tab
- Export reports in Markdown or JSON format
- Configurable temperature, timeout, and system prompts

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

1. **Configure**: Enter your LM Studio server URLs and model names in the Configuration panel
2. **Initialize**: Click "Initialize Session" to connect to servers and verify models
3. **Prompt**: Enter prompts interactively or upload a batch file (one prompt per line)
4. **Compare**: View responses in model-specific tabs
5. **Export**: Save comparison reports as Markdown or JSON

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
│   ├── __init__.py
│   ├── config.py      # Configuration and data models
│   ├── core.py        # Server pool and session management
│   ├── export.py      # Report generation
│   └── main.py        # Gradio UI
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_core.py
    ├── test_export.py
    └── test_main.py
```

## License

MIT

(C) 2025 Reflective Attention
