"""
Shared event handlers and helpers for prompt-prix.

Tab-specific handlers are in prompt_prix.tabs.{battery,compare}.handlers
"""

import logging
from typing import Optional

import gradio as gr

from prompt_prix import state
from prompt_prix.scheduler import ServerPool
from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.parsers import parse_servers_input

logger = logging.getLogger(__name__)


async def _init_pool_and_validate(
    servers_text: str,
    models_selected: list[str],
    only_loaded: bool = False
) -> tuple[Optional[ServerPool], Optional[str]]:
    """
    Initialize server pool and validate models are available.

    Args:
        servers_text: Newline-separated server URLs
        models_selected: Models that must be available
        only_loaded: If True, only consider loaded models as available

    Returns:
        (pool, None) on success
        (None, error_message) on failure
    """
    servers = parse_servers_input(servers_text)
    if not servers:
        return None, "❌ No servers configured"

    pool = ServerPool(servers)
    await pool.refresh()

    available = pool.get_available_models(only_loaded=only_loaded)
    missing = [m for m in models_selected if m not in available]
    if missing:
        return None, f"❌ Models not available: {', '.join(missing)}"

    return pool, None


def handle_stop():
    """Handle Stop button click - signal cancellation."""
    state.request_stop()
    return "🛑 Stop requested..."


async def fetch_available_models(
    servers_text: str,
    only_loaded: bool = False
) -> tuple[str, dict]:
    """
    Query all configured servers and return available models with GPU prefix.

    Uses ServerPool which queries both manifest and load state per-server,
    ensuring only_loaded correctly filters to models loaded on specific servers.

    Models are returned with server prefix: "0: model-name", "1: model-name"
    to indicate which GPU/server they're on.

    Args:
        servers_text: Newline-separated server URLs
        only_loaded: If True, filter to only models currently loaded in LM Studio

    Returns (status_message, gr.update for CheckboxGroup choices).
    """
    logger.info(f"fetch_available_models called: servers_text={repr(servers_text)}, only_loaded={only_loaded}")
    servers = parse_servers_input(servers_text)
    logger.info(f"Parsed {len(servers)} servers: {servers}")

    if not servers:
        return "❌ No servers configured", gr.update(choices=[])

    pool = ServerPool(servers)
    adapter = LMStudioAdapter(pool)
    await adapter.refresh()

    # Check for connection errors via adapter (Issue #92)
    failed = adapter.get_connection_errors()
    if len(failed) == len(pool.servers):
        # All servers failed - surface the error
        error_msg = failed[0][1]  # Get first error message
        logger.warning(f"All servers failed to connect: {error_msg}")
        return f"❌ Failed to connect: {error_msg}", gr.update(choices=[])

    # Build server index map (idx → URL)
    server_urls = list(pool.servers.keys())
    state.set_server_map({i: url for i, url in enumerate(server_urls)})

    # Build prefixed model list (each server×model combo)
    prefixed_models = []
    for idx, (url, server) in enumerate(pool.servers.items()):
        models = server.loaded_models if only_loaded else server.manifest_models
        for model_id in models:
            prefixed_models.append(f"{idx}: {model_id}")

    if not prefixed_models:
        if only_loaded:
            return "⚠️ No models currently loaded. Load models in LM Studio first.", gr.update(choices=[])
        else:
            return "⚠️ No models found on any server. Are models loaded in LM Studio?", gr.update(choices=[])

    # Sort by model name (the part after prefix) for grouped display
    sorted_models = sorted(prefixed_models, key=lambda x: x.split(": ", 1)[1])

    # Build status message
    status_parts = [f"✅ Found {len(sorted_models)} model(s)"]
    if only_loaded:
        status_parts[0] += " (loaded only)"

    # Per-server breakdown (show errors for failed servers)
    for idx, (url, server) in enumerate(pool.servers.items()):
        if server.error:
            status_parts.append(f"  [{idx}] ❌ {server.error}")
        elif only_loaded:
            if server.loaded_models:
                status_parts.append(f"  [{idx}] {len(server.loaded_models)} loaded")
        else:
            count = len(server.manifest_models)
            if count > 0:
                status_parts.append(f"  [{idx}] {count} model(s)")

    status = " | ".join(status_parts)
    logger.info(f"fetch_available_models returning: {len(sorted_models)} models")
    return status, gr.update(choices=sorted_models)
