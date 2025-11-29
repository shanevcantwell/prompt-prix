"""
Shared event handlers and helpers for prompt-prix.

Tab-specific handlers are in prompt_prix.tabs.{battery,compare}.handlers
"""

from typing import Optional

import gradio as gr

from prompt_prix import state
from prompt_prix.core import ServerPool
from prompt_prix.parsers import parse_servers_input


async def _init_pool_and_validate(
    servers_text: str,
    models_selected: list[str]
) -> tuple[Optional[ServerPool], Optional[str]]:
    """
    Initialize server pool and validate models are available.

    Returns:
        (pool, None) on success
        (None, error_message) on failure
    """
    servers = parse_servers_input(servers_text)
    if not servers:
        return None, "❌ No servers configured"

    pool = ServerPool(servers)
    await pool.refresh_all_manifests()

    available = pool.get_all_available_models()
    missing = [m for m in models_selected if m not in available]
    if missing:
        return None, f"❌ Models not available: {', '.join(missing)}"

    return pool, None


def handle_stop():
    """Handle Stop button click - signal cancellation."""
    state.request_stop()
    return "🛑 Stop requested..."


async def fetch_available_models(servers_text: str) -> tuple[str, dict]:
    """
    Query all configured servers and return available models.
    Returns (status_message, gr.update for CheckboxGroup choices).
    """
    servers = parse_servers_input(servers_text)

    if not servers:
        return "❌ No servers configured", gr.update(choices=[])

    pool = ServerPool(servers)
    await pool.refresh_all_manifests()

    models_by_server: dict[str, list[str]] = {}
    for url, server in pool.servers.items():
        if server.available_models:
            models_by_server[url] = server.available_models

    if not models_by_server:
        return "⚠️ No models found on any server. Are models loaded in LM Studio?", gr.update(choices=[])

    all_models = set()
    for models in models_by_server.values():
        all_models.update(models)

    sorted_models = sorted(all_models)

    status_parts = [f"✅ Found {len(all_models)} model(s):"]
    for url, models in models_by_server.items():
        status_parts.append(f"  {url}: {len(models)} model(s)")

    return " | ".join(status_parts), gr.update(choices=sorted_models)
