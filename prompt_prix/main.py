"""
Gradio application entry point for prompt-prix.
"""

from dotenv import load_dotenv

from prompt_prix.config import get_gradio_port
from prompt_prix.ui import create_app

# Re-export for backwards compatibility with tests
from prompt_prix.parsers import (
    parse_models_input,
    parse_servers_input,
    parse_prompts_file,
    load_system_prompt
)
from prompt_prix.handlers import (
    fetch_available_models,
    initialize_session,
    send_single_prompt,
    run_batch_prompts,
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
    app = create_app()
    port = get_gradio_port()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=port,
        share=False
    )


if __name__ == "__main__":
    run()
