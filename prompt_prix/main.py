"""
Gradio application entry point for prompt-prix.

CLI commands:
    prompt-prix          - Launch the Gradio UI
"""

import gradio as gr

try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv not needed on HF Spaces (env vars set via secrets UI)
    load_dotenv = lambda: None

from prompt_prix.config import get_gradio_port
from prompt_prix.ui import create_app
from prompt_prix.ui_helpers import CUSTOM_CSS

# Re-export for backwards compatibility with tests
from prompt_prix.parsers import (
    parse_models_input,
    parse_servers_input,
    parse_prompts_file,
    load_system_prompt
)
from prompt_prix.handlers import fetch_available_models
from prompt_prix.tabs.compare.handlers import (
    initialize_session,
    send_single_prompt,
    clear_session,
    export_markdown,
    export_json,
    launch_beyond_compare
)
from prompt_prix import state

# Expose global state for tests that access main.session
session = state.session

# Load environment variables from .env file
load_dotenv()


def _register_default_adapter() -> None:
    """
    Register the appropriate adapter based on environment.

    HF mode (HF_TOKEN set, no LM_STUDIO_SERVER_*):
        Registers HuggingFaceAdapter with vetted models

    LM Studio mode (default):
        Registers LMStudioAdapter with servers from environment
    """
    from prompt_prix.config import is_huggingface_mode, get_hf_models, get_hf_token
    from prompt_prix.config import get_default_servers
    from prompt_prix.mcp.registry import register_adapter

    if is_huggingface_mode():
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter
        models = get_hf_models()
        token = get_hf_token()
        print(f"  HuggingFace mode: {len(models)} models")
        adapter = HuggingFaceAdapter(models=models, token=token)
        register_adapter(adapter)
    else:
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        servers = get_default_servers()
        if servers:
            print(f"  LM Studio mode: {len(servers)} server(s)")
            adapter = LMStudioAdapter(server_urls=servers)
            register_adapter(adapter)


def run():
    """Entry point for the application."""
    print("prompt-prix starting...")

    # Register adapter with default servers from env
    _register_default_adapter()

    app = create_app()
    port = get_gradio_port()

    print(f"Launching on http://0.0.0.0:{port}")

    # Gradio 6.x moved theme/css from Blocks() to launch()
    # Gradio 5.x had them on Blocks() - detect and adapt
    import inspect
    launch_params = inspect.signature(gr.Blocks.launch).parameters

    launch_kwargs = {
        "server_name": "0.0.0.0",  # Allow external connections
        "server_port": port,
        "share": False,
    }

    # Add theme/css only if launch() accepts them (Gradio 6+)
    if "theme" in launch_params:
        launch_kwargs["theme"] = gr.themes.Soft()
        launch_kwargs["css"] = CUSTOM_CSS

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    run()
