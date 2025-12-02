"""
Gradio application entry point for prompt-prix.

CLI commands:
    prompt-prix          - Launch the Gradio UI
    prompt-prix-gemini   - Set up Gemini Web UI session (login)
"""

import asyncio
import sys

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


def gemini_setup():
    """
    Set up Gemini Web UI session.

    Opens a browser window for you to log into Google/Gemini.
    Session is saved for future headless use.

    Usage:
        prompt-prix-gemini          # Login to Gemini
        prompt-prix-gemini --clear  # Clear saved session
        prompt-prix-gemini --check  # Check if session exists
    """
    try:
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter, PLAYWRIGHT_AVAILABLE
    except ImportError:
        print("Error: Playwright not installed.")
        print("Install with: pip install prompt-prix[gemini] && playwright install chromium")
        sys.exit(1)

    if not PLAYWRIGHT_AVAILABLE:
        print("Error: Playwright not installed.")
        print("Install with: pip install playwright && playwright install chromium")
        sys.exit(1)

    # Parse simple args
    args = sys.argv[1:]

    if "--clear" in args:
        adapter = GeminiWebUIAdapter()
        adapter.clear_session()
        print(f"Session cleared from {adapter.state_file}")
        return

    if "--check" in args:
        adapter = GeminiWebUIAdapter()
        if adapter.has_session():
            print(f"✅ Session exists at {adapter.state_file}")
        else:
            print(f"❌ No session found. Run 'prompt-prix-gemini' to log in.")
        return

    # Default: login flow
    async def do_login():
        print("Starting Gemini login flow...")
        print("A browser window will open. Please log into your Google account.")
        print()

        adapter = GeminiWebUIAdapter(headless=False)  # Force visible for login

        try:
            await adapter._ensure_initialized()
            print()
            print(f"✅ Session saved to {adapter.state_file}")
            print("You can now use Gemini in prompt-prix!")
        finally:
            await adapter.close()

    asyncio.run(do_login())


if __name__ == "__main__":
    run()
