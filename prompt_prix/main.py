"""
Gradio application entry point.
"""

import gradio as gr
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

from prompt_prix.config import (
    get_default_servers, get_gradio_port, get_beyond_compare_path,
    DEFAULT_MODELS, DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS, DEFAULT_MAX_TOKENS, DEFAULT_SYSTEM_PROMPT
)
from prompt_prix.core import ServerPool, ComparisonSession
from prompt_prix.export import generate_markdown_report, generate_json_report, save_report


# Load environment variables from .env file
load_dotenv()


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
    # Try default file in package directory
    default_path = Path(__file__).parent / "system_prompt.txt"
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
    completed = 0
    for i, prompt in enumerate(prompts):
        if session.state.halted:
            break
        await session.send_prompt_to_all(prompt)
        completed = i + 1

    # Build result
    status = f"✅ Completed {completed}/{len(prompts)} prompts"
    if session.state.halted:
        status = f"⚠️ Halted after {completed}/{len(prompts)} prompts: {session.state.halt_reason}"

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
    filename = f"prompt-prix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    save_report(report, filename)

    return f"✅ Exported to {filename}", report


def export_json() -> tuple[str, str]:
    """Export current session as JSON."""
    global session

    if session is None:
        return "❌ No session to export", ""

    report = generate_json_report(session.state)
    filename = f"prompt-prix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_report(report, filename)

    return f"✅ Exported to {filename}", report


def launch_beyond_compare(model_a: str, model_b: str) -> str:
    """
    Launch Beyond Compare with two model outputs for side-by-side diff.
    """
    global session

    if session is None:
        return "❌ No session - initialize first"

    if not model_a or not model_b:
        return "❌ Select two models to compare"

    if model_a == model_b:
        return "❌ Select two different models"

    if model_a not in session.state.contexts:
        return f"❌ Model '{model_a}' not in session"
    if model_b not in session.state.contexts:
        return f"❌ Model '{model_b}' not in session"

    # Get the conversation content
    content_a = session.get_context_display(model_a)
    content_b = session.get_context_display(model_b)

    if not content_a.strip() and not content_b.strip():
        return "❌ No conversation content to compare"

    # Write to temp files with model names in filename for clarity
    try:
        # Create temp files that persist (delete=False)
        safe_name_a = model_a.replace("/", "_").replace("\\", "_")[:50]
        safe_name_b = model_b.replace("/", "_").replace("\\", "_")[:50]

        file_a = tempfile.NamedTemporaryFile(
            mode='w', suffix=f'_{safe_name_a}.txt',
            delete=False, encoding='utf-8'
        )
        file_a.write(f"# Model: {model_a}\n\n{content_a}")
        file_a.close()

        file_b = tempfile.NamedTemporaryFile(
            mode='w', suffix=f'_{safe_name_b}.txt',
            delete=False, encoding='utf-8'
        )
        file_b.write(f"# Model: {model_b}\n\n{content_b}")
        file_b.close()

        # Launch Beyond Compare
        bc_path = get_beyond_compare_path()
        subprocess.Popen([bc_path, file_a.name, file_b.name])

        return f"✅ Launched Beyond Compare: {model_a} vs {model_b}"

    except FileNotFoundError:
        return f"❌ Beyond Compare not found. Set BEYOND_COMPARE_PATH in .env"
    except Exception as e:
        return f"❌ Failed to launch Beyond Compare: {e}"


# ─────────────────────────────────────────────────────────────────────
# GRADIO UI DEFINITION
# ─────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="prompt-prix", theme=gr.themes.Soft()) as app:
        gr.Markdown("# prompt-prix")
        gr.Markdown("Compare responses from multiple LLMs served via LM Studio.")

        # ─────────────────────────────────────────────────────────────
        # CONFIGURATION PANEL
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("Configuration", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    servers_input = gr.Textbox(
                        label="LM Studio Servers (one per line)",
                        value="\n".join(get_default_servers()),
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
        # BEYOND COMPARE PANEL
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("Compare Models (Beyond Compare)", open=False):
            gr.Markdown("Select two models to open their outputs in Beyond Compare for side-by-side diff.")
            with gr.Row():
                compare_model_a = gr.Dropdown(
                    label="Model A",
                    choices=[],
                    interactive=True
                )
                compare_model_b = gr.Dropdown(
                    label="Model B",
                    choices=[],
                    interactive=True
                )
                compare_button = gr.Button("Open in Beyond Compare", variant="secondary")

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

        # Beyond Compare - update dropdowns after init
        def get_model_choices():
            if session is None:
                return gr.update(choices=[]), gr.update(choices=[])
            models = session.state.models
            return gr.update(choices=models), gr.update(choices=models)

        init_button.click(
            fn=get_model_choices,
            inputs=[],
            outputs=[compare_model_a, compare_model_b]
        )

        compare_button.click(
            fn=launch_beyond_compare,
            inputs=[compare_model_a, compare_model_b],
            outputs=[status_display]
        )

    return app


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def run():
    """Entry point for the application."""
    app = create_app()
    port = get_gradio_port()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=port,
        share=False
    )


if __name__ == "__main__":
    run()
