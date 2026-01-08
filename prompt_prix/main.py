"""
Gradio application entry point for prompt-prix.

CLI commands:
    prompt-prix          - Launch the Gradio UI
"""

import gradio as gr
from dotenv import load_dotenv

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

# Expose global state for tests that access main.session, main.server_pool
server_pool = state.server_pool
session = state.session

# Load environment variables from .env file
load_dotenv()


def run():
    """Entry point for the application."""
    print("prompt-prix starting...")

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
