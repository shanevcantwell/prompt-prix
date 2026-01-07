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


def gemini_setup():
    """
    Manage Gemini Web UI session for prompt-prix.

    This tool handles browser session persistence for the Gemini adapter.
    Sessions are stored at ~/.prompt-prix/gemini_state/state.json
    """
    HELP = """
prompt-prix-gemini - Manage Gemini Web UI session

USAGE:
    prompt-prix-gemini [--on | --off | --status | --help]

COMMANDS:
    (no args)   Same as --on: start login flow if no session exists
    --on        Open browser, log into Google, save session for headless use
    --off       Clear saved session (forces re-login next time)
    --status    Check if a valid session exists
    --help      Show this help message

WORKFLOW:
    1. First time:  prompt-prix-gemini --on
                    → Browser opens, you log into Google
                    → Session saved to ~/.prompt-prix/gemini_state/

    2. Normal use:  prompt-prix
                    → Check "Gemini" box, click Fetch
                    → Gemini runs headless using saved session

    3. Session expired or need to switch accounts:
                    prompt-prix-gemini --off
                    prompt-prix-gemini --on

PREREQUISITES:
    pip install prompt-prix[gemini]
    playwright install chromium
"""

    try:
        from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter, PLAYWRIGHT_AVAILABLE
    except ImportError:
        print("Error: Playwright not installed.")
        print()
        print("Install with:")
        print("    pip install prompt-prix[gemini]")
        print("    playwright install chromium")
        sys.exit(1)

    if not PLAYWRIGHT_AVAILABLE:
        print("Error: Playwright not installed.")
        print()
        print("Install with:")
        print("    pip install playwright")
        print("    playwright install chromium")
        sys.exit(1)

    args = sys.argv[1:]

    # --help
    if "--help" in args or "-h" in args:
        print(HELP)
        return

    # --off (clear session)
    if "--off" in args or "--clear" in args or "--reset" in args:
        adapter = GeminiWebUIAdapter()
        if adapter.has_session():
            adapter.clear_session()
            print(f"✅ Session cleared: {adapter.state_file}")
        else:
            print("ℹ️  No session to clear")
        return

    # --status (check session)
    if "--status" in args or "--check" in args:
        adapter = GeminiWebUIAdapter()
        if adapter.has_session():
            print(f"✅ Session ON: {adapter.state_file}")
        else:
            print("❌ Session OFF: No saved session")
            print("   Run: prompt-prix-gemini --on")
        return

    # --on or default: login flow
    async def do_login():
        adapter = GeminiWebUIAdapter()

        if adapter.has_session():
            print(f"ℹ️  Session already exists: {adapter.state_file}")
            print("   Use --off to clear it first, or --status to check")
            return

        print("=" * 60)
        print("GEMINI LOGIN")
        print("=" * 60)
        print()
        print("A browser window will open to gemini.google.com")
        print()
        print("STEPS:")
        print("  1. Log into your Google account in the browser")
        print("  2. Wait for the Gemini chat page to fully load")
        print("  3. This script will detect login and save the session")
        print()
        print("TIMEOUT: 5 minutes")
        print()
        print("Opening browser...")
        print()

        # Force visible for login
        adapter._headless_override = False

        try:
            await adapter._ensure_initialized()
            print()
            print("=" * 60)
            print(f"✅ SESSION SAVED: {adapter.state_file}")
            print("=" * 60)
            print()
            print("You can now use Gemini in prompt-prix:")
            print("  1. Run: prompt-prix")
            print("  2. Check the 'Gemini' checkbox")
            print("  3. Click 'Fetch' to load models")
            print()
        except Exception as e:
            print()
            print(f"❌ Login failed: {e}")
            print()
            print("Try again with: prompt-prix-gemini --on")
            sys.exit(1)
        finally:
            await adapter.close()

    asyncio.run(do_login())


if __name__ == "__main__":
    run()
