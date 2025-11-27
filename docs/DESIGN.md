# LLM Compare Tool - Design Document

## Purpose

A Gradio-based tool for comparing responses from multiple LLMs (open-weights models, quantizations, or any combination) served via LM Studio's OpenAI-compatible API. Designed to "audition" models for use as backing LLMs in LAS (langgraph-agentic-scaffold) specialists.

---

## Requirements Summary

| Requirement | Decision |
|-------------|----------|
| Model configuration | Arbitrary-length list in config |
| Context isolation | Separate conversation history per model |
| Multi-turn support | Yes, contexts persist across prompts within session |
| Server topology | 2 static LM Studio servers |
| Model-to-server routing | First available server with model in manifest |
| Execution model | Synchronized rounds (all models complete prompt N before prompt N+1) |
| Parallelism | Async across servers when same model available on multiple |
| Input modes | Interactive prompt box + file upload (newline-separated prompts) |
| Output display | Tabs per model with full `[User]/[Assistant]` conversation |
| Streaming | Per-tab streaming into each model's conversation display |
| Report format | Markdown (default) or JSON export |
| Persistence | In-memory only; export on demand |
| Failure handling | Any error → auto-export all contexts and halt |
| Configuration UI | Gradio controls for temperature, timeout, system prompt file |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Gradio UI                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Config Panel    │  │ Input Panel     │  │ Export Panel        │  │
│  │ - Servers       │  │ - Prompt box    │  │ - Export MD button  │  │
│  │ - Models list   │  │ - Upload file   │  │ - Export JSON btn   │  │
│  │ - Temperature   │  │ - Submit button │  │ - Status display    │  │
│  │ - Timeout       │  │ - Progress bar  │  │                     │  │
│  │ - System prompt │  │                 │  │                     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        Model Output Tabs                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                    │
│  │ Model A │ │ Model B │ │ Model C │ │ Model D │ ...                │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ [User]: What is the capital of France?                         ││
│  │                                                                 ││
│  │ [Assistant]: The capital of France is Paris...                 ││
│  │                                                                 ││
│  │ [User]: What about Germany?                                    ││
│  │                                                                 ││
│  │ [Assistant]: The capital of Germany is Berlin...               ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
llm_compare/
├── main.py              # Entry point, Gradio app definition
├── core.py              # ServerPool, ComparisonSession, async logic
├── config.py            # Defaults, type definitions, Pydantic models
├── export.py            # Markdown and JSON report generation
├── system_prompt.txt    # Default system prompt (optional)
└── requirements.txt     # Dependencies
```

---

## Module Specifications

### `config.py`

```python
"""
Configuration constants and Pydantic models.
"""

from pydantic import BaseModel
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# DEFAULTS - User-configurable via Gradio UI
# ─────────────────────────────────────────────────────────────────────

DEFAULT_SERVERS: list[str] = [
    "http://192.168.1.10:1234",  # Update with actual server addresses
    "http://192.168.1.11:1234",
]

DEFAULT_MODELS: list[str] = [
    # Populate with model identifiers as they appear in LM Studio
    # Examples:
    # "llama-3.2-3b-instruct",
    # "qwen2.5-7b-instruct-q4_k_m",
    # "mistral-7b-instruct-v0.3",
]

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TIMEOUT_SECONDS: int = 300  # 5 minutes
DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."

# ─────────────────────────────────────────────────────────────────────
# INTERNAL CONSTANTS - Not exposed to UI
# ─────────────────────────────────────────────────────────────────────

MANIFEST_REFRESH_INTERVAL_SECONDS: int = 30
REPORT_DIVIDER: str = "\n\n" + "=" * 80 + "\n\n"
CONVERSATION_SEPARATOR: str = "\n\n---\n\n"

# ─────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────

class ServerConfig(BaseModel):
    """Configuration for a single LM Studio server."""
    url: str
    available_models: list[str] = []
    is_busy: bool = False


class ModelConfig(BaseModel):
    """Configuration for a model to be tested."""
    model_id: str  # As it appears in LM Studio manifest
    display_name: Optional[str] = None  # Friendly name for UI tabs
    
    @property
    def tab_name(self) -> str:
        return self.display_name or self.model_id


class Message(BaseModel):
    """A single message in a conversation."""
    role: str  # "user", "assistant", or "system"
    content: str


class ModelContext(BaseModel):
    """Complete conversation context for a single model."""
    model_id: str
    messages: list[Message] = []
    error: Optional[str] = None  # Set if model encountered an error
    
    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))
    
    def to_openai_messages(self, system_prompt: str) -> list[dict]:
        """Convert to OpenAI API message format."""
        result = [{"role": "system", "content": system_prompt}]
        result.extend([{"role": m.role, "content": m.content} for m in self.messages])
        return result
    
    def to_display_format(self) -> str:
        """Convert to human-readable format for UI display."""
        lines = []
        for msg in self.messages:
            if msg.role == "user":
                lines.append(f"[User]: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"[Assistant]: {msg.content}")
        if self.error:
            lines.append(f"\n[ERROR]: {self.error}")
        return "\n\n".join(lines)


class SessionState(BaseModel):
    """Complete state for a comparison session."""
    models: list[str]
    contexts: dict[str, ModelContext] = {}
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = DEFAULT_TEMPERATURE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    halted: bool = False
    halt_reason: Optional[str] = None
```

---

### `core.py`

```python
"""
Core logic: server pool management, manifest fetching, prompt execution.
"""

import asyncio
import httpx
from typing import AsyncGenerator, Optional
from config import (
    ServerConfig, ModelContext, SessionState,
    DEFAULT_SERVERS, MANIFEST_REFRESH_INTERVAL_SECONDS
)

# ─────────────────────────────────────────────────────────────────────
# SERVER POOL
# ─────────────────────────────────────────────────────────────────────

class ServerPool:
    """
    Manages multiple LM Studio servers.
    Tracks which models are available on each server and server busy state.
    """
    
    def __init__(self, server_urls: list[str]):
        self.servers: dict[str, ServerConfig] = {
            url: ServerConfig(url=url) for url in server_urls
        }
        self._locks: dict[str, asyncio.Lock] = {
            url: asyncio.Lock() for url in server_urls
        }
    
    async def refresh_all_manifests(self) -> None:
        """Fetch model lists from all servers."""
        tasks = [self._refresh_manifest(url) for url in self.servers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _refresh_manifest(self, server_url: str) -> None:
        """Fetch model list from a single server."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{server_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                # LM Studio returns {"data": [{"id": "model-name", ...}, ...]}
                model_ids = [m["id"] for m in data.get("data", [])]
                self.servers[server_url].available_models = model_ids
        except Exception as e:
            # Server unreachable or error - clear its model list
            self.servers[server_url].available_models = []
    
    def find_available_server(self, model_id: str) -> Optional[str]:
        """
        Find a server that has the requested model and is not busy.
        Returns server URL or None if no server available.
        """
        for url, server in self.servers.items():
            if model_id in server.available_models and not server.is_busy:
                return url
        return None
    
    def get_all_available_models(self) -> set[str]:
        """Return union of all models across all servers."""
        result = set()
        for server in self.servers.values():
            result.update(server.available_models)
        return result
    
    async def acquire_server(self, server_url: str) -> None:
        """Mark server as busy."""
        await self._locks[server_url].acquire()
        self.servers[server_url].is_busy = True
    
    def release_server(self, server_url: str) -> None:
        """Mark server as available."""
        self.servers[server_url].is_busy = False
        try:
            self._locks[server_url].release()
        except RuntimeError:
            pass  # Lock wasn't held


# ─────────────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────

async def stream_completion(
    server_url: str,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int
) -> AsyncGenerator[str, None]:
    """
    Stream a completion from an LM Studio server.
    Yields text chunks as they arrive.
    Raises exceptions on error (context limit, timeout, etc.).
    """
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        async with client.stream(
            "POST",
            f"{server_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


async def get_completion(
    server_url: str,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int
) -> str:
    """
    Get a complete (non-streaming) response from an LM Studio server.
    Returns full response text.
    Raises exceptions on error.
    """
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────
# COMPARISON SESSION
# ─────────────────────────────────────────────────────────────────────

class ComparisonSession:
    """
    Manages a multi-model comparison session.
    Handles prompt dispatch, context tracking, and failure handling.
    """
    
    def __init__(
        self,
        models: list[str],
        server_pool: ServerPool,
        system_prompt: str,
        temperature: float,
        timeout_seconds: int,
        max_tokens: int
    ):
        self.server_pool = server_pool
        self.state = SessionState(
            models=models,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens
        )
        # Initialize empty context for each model
        for model_id in models:
            self.state.contexts[model_id] = ModelContext(model_id=model_id)
    
    async def send_prompt_to_model(
        self,
        model_id: str,
        prompt: str,
        on_chunk: callable = None  # async callback(model_id, chunk)
    ) -> str:
        """
        Send a prompt to a single model and update its context.
        Returns the complete response.
        Raises exception on failure.
        """
        context = self.state.contexts[model_id]
        context.add_user_message(prompt)
        
        # Find available server
        server_url = None
        while server_url is None:
            await self.server_pool.refresh_all_manifests()
            server_url = self.server_pool.find_available_server(model_id)
            if server_url is None:
                await asyncio.sleep(1.0)  # Wait and retry
        
        # Acquire server
        await self.server_pool.acquire_server(server_url)
        
        try:
            messages = context.to_openai_messages(self.state.system_prompt)
            
            if on_chunk:
                # Streaming mode
                full_response = ""
                async for chunk in stream_completion(
                    server_url=server_url,
                    model_id=model_id,
                    messages=messages,
                    temperature=self.state.temperature,
                    max_tokens=self.state.max_tokens,
                    timeout_seconds=self.state.timeout_seconds
                ):
                    full_response += chunk
                    await on_chunk(model_id, chunk)
            else:
                # Non-streaming mode
                full_response = await get_completion(
                    server_url=server_url,
                    model_id=model_id,
                    messages=messages,
                    temperature=self.state.temperature,
                    max_tokens=self.state.max_tokens,
                    timeout_seconds=self.state.timeout_seconds
                )
            
            context.add_assistant_message(full_response)
            return full_response
            
        finally:
            self.server_pool.release_server(server_url)
    
    async def send_prompt_to_all(
        self,
        prompt: str,
        on_chunk: callable = None  # async callback(model_id, chunk)
    ) -> dict[str, str]:
        """
        Send a prompt to all models in parallel (limited by server availability).
        Returns dict of model_id -> response.
        On any failure, sets halted=True and records error.
        """
        if self.state.halted:
            raise RuntimeError(f"Session halted: {self.state.halt_reason}")
        
        results = {}
        tasks = []
        
        async def run_model(model_id: str):
            try:
                response = await self.send_prompt_to_model(model_id, prompt, on_chunk)
                results[model_id] = response
            except Exception as e:
                self.state.contexts[model_id].error = str(e)
                self.state.halted = True
                self.state.halt_reason = f"Model {model_id} failed: {e}"
                raise
        
        for model_id in self.state.models:
            tasks.append(run_model(model_id))
        
        # Run all, but stop on first failure
        try:
            await asyncio.gather(*tasks)
        except Exception:
            pass  # Error already recorded in state
        
        return results
    
    def get_context_display(self, model_id: str) -> str:
        """Get displayable conversation for a model."""
        return self.state.contexts[model_id].to_display_format()
    
    def get_all_contexts(self) -> dict[str, str]:
        """Get displayable conversations for all models."""
        return {
            model_id: self.get_context_display(model_id)
            for model_id in self.state.models
        }
```

---

### `export.py`

```python
"""
Report generation in Markdown and JSON formats.
"""

import json
from datetime import datetime
from config import SessionState, REPORT_DIVIDER, CONVERSATION_SEPARATOR


def generate_markdown_report(state: SessionState) -> str:
    """
    Generate a Markdown report with all model contexts.
    """
    lines = []
    
    # Header
    lines.append("# LLM Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Models:** {', '.join(state.models)}")
    lines.append(f"**Temperature:** {state.temperature}")
    lines.append(f"**Max Tokens:** {state.max_tokens}")
    lines.append("")
    
    if state.halted:
        lines.append(f"⚠️ **Session Halted:** {state.halt_reason}")
        lines.append("")
    
    # System prompt
    lines.append("## System Prompt")
    lines.append("")
    lines.append("```")
    lines.append(state.system_prompt)
    lines.append("```")
    lines.append("")
    
    # Model sections
    for model_id in state.models:
        context = state.contexts.get(model_id)
        if not context:
            continue
        
        lines.append(REPORT_DIVIDER.strip())
        lines.append("")
        lines.append(f"## Model: {model_id}")
        lines.append("")
        
        if context.error:
            lines.append(f"**Error:** {context.error}")
            lines.append("")
        
        for msg in context.messages:
            if msg.role == "user":
                lines.append("### User")
                lines.append("")
                lines.append(msg.content)
                lines.append("")
            elif msg.role == "assistant":
                lines.append("### Assistant")
                lines.append("")
                lines.append(msg.content)
                lines.append("")
    
    return "\n".join(lines)


def generate_json_report(state: SessionState) -> str:
    """
    Generate a JSON report with all model contexts.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "configuration": {
            "models": state.models,
            "system_prompt": state.system_prompt,
            "temperature": state.temperature,
            "max_tokens": state.max_tokens,
            "timeout_seconds": state.timeout_seconds
        },
        "halted": state.halted,
        "halt_reason": state.halt_reason,
        "conversations": {}
    }
    
    for model_id in state.models:
        context = state.contexts.get(model_id)
        if not context:
            continue
        
        report["conversations"][model_id] = {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in context.messages
            ],
            "error": context.error
        }
    
    return json.dumps(report, indent=2)


def save_report(content: str, filepath: str) -> None:
    """Save report content to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
```

---

### `main.py`

```python
"""
Gradio application entry point.
"""

import gradio as gr
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import (
    DEFAULT_SERVERS, DEFAULT_MODELS, DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS, DEFAULT_MAX_TOKENS, DEFAULT_SYSTEM_PROMPT
)
from core import ServerPool, ComparisonSession
from export import generate_markdown_report, generate_json_report, save_report


# ─────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────

# These will be initialized when the app starts
server_pool: Optional[ServerPool] = None
session: Optional[ComparisonSession] = None


# ─────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def parse_models_input(models_text: str) -> list[str]:
    """Parse newline or comma-separated model list."""
    models = []
    for line in models_text.strip().split("\n"):
        for item in line.split(","):
            item = item.strip()
            if item:
                models.append(item)
    return models


def parse_servers_input(servers_text: str) -> list[str]:
    """Parse newline or comma-separated server list."""
    servers = []
    for line in servers_text.strip().split("\n"):
        for item in line.split(","):
            item = item.strip()
            if item:
                servers.append(item)
    return servers


def load_system_prompt(file_path: Optional[str]) -> str:
    """Load system prompt from file or return default."""
    if file_path:
        path = Path(file_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
    # Try default file
    default_path = Path("system_prompt.txt")
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")
    return DEFAULT_SYSTEM_PROMPT


def parse_prompts_file(file_content: str) -> list[str]:
    """Parse uploaded file into list of prompts (newline-separated)."""
    prompts = []
    for line in file_content.strip().split("\n"):
        line = line.strip()
        if line:
            prompts.append(line)
    return prompts


# ─────────────────────────────────────────────────────────────────────
# GRADIO EVENT HANDLERS
# ─────────────────────────────────────────────────────────────────────

async def initialize_session(
    servers_text: str,
    models_text: str,
    system_prompt_file: Optional[str],
    temperature: float,
    timeout: int,
    max_tokens: int
) -> tuple:
    """
    Initialize or reinitialize the comparison session.
    Returns tuple of (status_message, *model_tab_contents)
    """
    global server_pool, session
    
    servers = parse_servers_input(servers_text)
    models = parse_models_input(models_text)
    system_prompt = load_system_prompt(system_prompt_file)
    
    if not servers:
        return ("❌ No servers configured",) + tuple("" for _ in range(10))
    if not models:
        return ("❌ No models configured",) + tuple("" for _ in range(10))
    
    # Initialize server pool and refresh manifests
    server_pool = ServerPool(servers)
    await server_pool.refresh_all_manifests()
    
    # Check which models are actually available
    available = server_pool.get_all_available_models()
    missing = [m for m in models if m not in available]
    
    if missing:
        return (
            f"⚠️ Models not found on any server: {', '.join(missing)}",
        ) + tuple("" for _ in range(10))
    
    # Create session
    session = ComparisonSession(
        models=models,
        server_pool=server_pool,
        system_prompt=system_prompt,
        temperature=temperature,
        timeout_seconds=timeout,
        max_tokens=max_tokens
    )
    
    return (f"✅ Session initialized with {len(models)} models",) + tuple(
        "" for _ in range(10)
    )


async def send_single_prompt(prompt: str) -> tuple:
    """
    Send a single prompt to all models.
    Returns tuple of (status_message, *model_tab_contents)
    """
    global session
    
    if session is None:
        return ("❌ Session not initialized",) + tuple("" for _ in range(10))
    
    if session.state.halted:
        return (
            f"❌ Session halted: {session.state.halt_reason}",
        ) + tuple(
            session.get_context_display(m) if m in session.state.contexts else ""
            for m in session.state.models[:10]
        ) + tuple("" for _ in range(10 - len(session.state.models)))
    
    if not prompt.strip():
        return ("❌ Empty prompt",) + tuple("" for _ in range(10))
    
    # Send prompt to all models
    await session.send_prompt_to_all(prompt.strip())
    
    # Build result tuple
    status = "✅ Prompt sent to all models"
    if session.state.halted:
        status = f"⚠️ Session halted: {session.state.halt_reason}"
    
    contexts = []
    for i in range(10):
        if i < len(session.state.models):
            model_id = session.state.models[i]
            contexts.append(session.get_context_display(model_id))
        else:
            contexts.append("")
    
    return (status,) + tuple(contexts)


async def run_batch_prompts(file_obj) -> tuple:
    """
    Run a batch of prompts from uploaded file.
    Returns tuple of (status_message, *model_tab_contents)
    """
    global session
    
    if session is None:
        return ("❌ Session not initialized",) + tuple("" for _ in range(10))
    
    if file_obj is None:
        return ("❌ No file uploaded",) + tuple("" for _ in range(10))
    
    # Read file content
    content = Path(file_obj.name).read_text(encoding="utf-8")
    prompts = parse_prompts_file(content)
    
    if not prompts:
        return ("❌ No prompts found in file",) + tuple("" for _ in range(10))
    
    # Run each prompt in sequence
    for i, prompt in enumerate(prompts):
        if session.state.halted:
            break
        await session.send_prompt_to_all(prompt)
    
    # Build result
    status = f"✅ Completed {i + 1}/{len(prompts)} prompts"
    if session.state.halted:
        status = f"⚠️ Halted after {i + 1}/{len(prompts)} prompts: {session.state.halt_reason}"
    
    contexts = []
    for j in range(10):
        if j < len(session.state.models):
            model_id = session.state.models[j]
            contexts.append(session.get_context_display(model_id))
        else:
            contexts.append("")
    
    return (status,) + tuple(contexts)


def export_markdown() -> tuple[str, str]:
    """Export current session as Markdown."""
    global session
    
    if session is None:
        return "❌ No session to export", ""
    
    report = generate_markdown_report(session.state)
    filename = f"llm_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    save_report(report, filename)
    
    return f"✅ Exported to {filename}", report


def export_json() -> tuple[str, str]:
    """Export current session as JSON."""
    global session
    
    if session is None:
        return "❌ No session to export", ""
    
    report = generate_json_report(session.state)
    filename = f"llm_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_report(report, filename)
    
    return f"✅ Exported to {filename}", report


# ─────────────────────────────────────────────────────────────────────
# GRADIO UI DEFINITION
# ─────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    
    with gr.Blocks(title="LLM Compare", theme=gr.themes.Soft()) as app:
        gr.Markdown("# LLM Compare Tool")
        gr.Markdown("Compare responses from multiple LLMs served via LM Studio.")
        
        # ─────────────────────────────────────────────────────────────
        # CONFIGURATION PANEL
        # ─────────────────────────────────────────────────────────────
        
        with gr.Accordion("Configuration", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    servers_input = gr.Textbox(
                        label="LM Studio Servers (one per line)",
                        value="\n".join(DEFAULT_SERVERS),
                        lines=3,
                        placeholder="http://192.168.1.10:1234\nhttp://192.168.1.11:1234"
                    )
                with gr.Column(scale=1):
                    models_input = gr.Textbox(
                        label="Models to Compare (one per line)",
                        value="\n".join(DEFAULT_MODELS),
                        lines=5,
                        placeholder="llama-3.2-3b-instruct\nqwen2.5-7b-instruct"
                    )
            
            with gr.Row():
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=DEFAULT_TEMPERATURE
                )
                timeout_slider = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=DEFAULT_TIMEOUT_SECONDS
                )
                max_tokens_slider = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=DEFAULT_MAX_TOKENS
                )
            
            with gr.Row():
                system_prompt_file = gr.File(
                    label="System Prompt File (optional)",
                    file_types=[".txt"],
                    type="filepath"
                )
                init_button = gr.Button("Initialize Session", variant="primary")
        
        # ─────────────────────────────────────────────────────────────
        # STATUS DISPLAY
        # ─────────────────────────────────────────────────────────────
        
        status_display = gr.Textbox(
            label="Status",
            value="Session not initialized",
            interactive=False
        )
        
        # ─────────────────────────────────────────────────────────────
        # INPUT PANEL
        # ─────────────────────────────────────────────────────────────
        
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                send_button = gr.Button("Send Prompt", variant="primary")
            
            with gr.Column(scale=1):
                batch_file = gr.File(
                    label="Batch Prompts File",
                    file_types=[".txt"],
                    type="filepath"
                )
                batch_button = gr.Button("Run Batch")
        
        # ─────────────────────────────────────────────────────────────
        # MODEL OUTPUT TABS
        # ─────────────────────────────────────────────────────────────
        
        # Create tabs for up to 10 models (can be extended)
        model_outputs = []
        with gr.Tabs():
            for i in range(10):
                with gr.Tab(f"Model {i + 1}"):
                    output = gr.Textbox(
                        label=f"Conversation",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )
                    model_outputs.append(output)
        
        # ─────────────────────────────────────────────────────────────
        # EXPORT PANEL
        # ─────────────────────────────────────────────────────────────
        
        with gr.Row():
            export_md_button = gr.Button("Export Markdown")
            export_json_button = gr.Button("Export JSON")
        
        export_preview = gr.Textbox(
            label="Export Preview",
            lines=10,
            interactive=False,
            visible=False
        )
        
        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS
        # ─────────────────────────────────────────────────────────────
        
        init_button.click(
            fn=initialize_session,
            inputs=[
                servers_input,
                models_input,
                system_prompt_file,
                temperature_slider,
                timeout_slider,
                max_tokens_slider
            ],
            outputs=[status_display] + model_outputs
        )
        
        send_button.click(
            fn=send_single_prompt,
            inputs=[prompt_input],
            outputs=[status_display] + model_outputs
        )
        
        # Also send on Enter in prompt box
        prompt_input.submit(
            fn=send_single_prompt,
            inputs=[prompt_input],
            outputs=[status_display] + model_outputs
        )
        
        batch_button.click(
            fn=run_batch_prompts,
            inputs=[batch_file],
            outputs=[status_display] + model_outputs
        )
        
        export_md_button.click(
            fn=export_markdown,
            inputs=[],
            outputs=[status_display, export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[export_preview]
        )
        
        export_json_button.click(
            fn=export_json,
            inputs=[],
            outputs=[status_display, export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[export_preview]
        )
    
    return app


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False
    )
```

---

### `requirements.txt`

```
gradio>=4.0.0
httpx>=0.25.0
pydantic>=2.0.0
```

---

## Implementation Notes for Claude Code

### Priority Order

1. **`config.py`** — Implement first, no dependencies
2. **`export.py`** — Implement second, depends only on config
3. **`core.py`** — Implement third, depends on config
4. **`main.py`** — Implement last, integrates everything

### Key Implementation Details

1. **Async/Await Pattern**: The Gradio handlers are async. Gradio 4.x supports async natively. Ensure `asyncio.run()` is not called inside handlers—Gradio manages the event loop.

2. **Server Locking**: The `ServerPool._locks` dict ensures a server isn't double-booked. The lock is acquired before sending a request and released in a `finally` block.

3. **Streaming vs Non-Streaming**: The design includes both `stream_completion()` and `get_completion()`. Initial implementation can use non-streaming for simplicity. Streaming can be added later by wiring up Gradio's streaming text output.

4. **Tab Limitation**: The current design hardcodes 10 model tabs. To support dynamic tab creation, use Gradio's `@gr.render` decorator pattern. This is a future enhancement.

5. **Error Detection**: LM Studio returns HTTP 400 with specific error messages for context limit exceeded. Parse the response body to detect this:
   ```python
   if response.status_code == 400:
       error_data = response.json()
       if "context" in error_data.get("error", {}).get("message", "").lower():
           raise ContextLimitError(...)
   ```

6. **Model Display Names**: The `ModelConfig.display_name` field allows friendly names in tabs (e.g., "Llama 3.2 3B Q4" instead of "lmstudio-community/Meta-Llama-3.2-3B-Instruct-GGUF/Meta-Llama-3.2-3B-Instruct-Q4_K_M.gguf"). Consider adding a UI field or config file for this mapping.

### Testing Checklist

- [ ] Single server, single model
- [ ] Single server, multiple models (sequential, model swapping)
- [ ] Multiple servers, single model (should use first available)
- [ ] Multiple servers, multiple models (parallel execution)
- [ ] Server goes offline mid-session
- [ ] Model returns context limit error
- [ ] Timeout during generation
- [ ] Batch file with 10+ prompts
- [ ] Export with partial completion (halted session)

### Future Enhancements (Out of Scope for Initial Implementation)

1. **Dynamic Tabs**: Use `@gr.render` to create tabs matching the actual model list
2. **Per-Tab Streaming**: Wire up `stream_completion()` to Gradio's streaming output
3. **Model Hot-Reload**: Refresh manifest button without reinitializing session
4. **Session Persistence**: Save/load session to JSON file
5. **Diff View**: Side-by-side response comparison mode
6. **Response Metrics**: Token count, generation time per model

---

## Usage Example

1. Start LM Studio on both servers with desired models loaded
2. Run `python main.py`
3. Open `http://localhost:7860` in browser
4. Configure servers and models in the Configuration panel
5. Click "Initialize Session"
6. Enter prompts interactively or upload a batch file
7. Review responses in model tabs
8. Export report when finished
