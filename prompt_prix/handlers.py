"""
Shared event handlers and helpers for prompt-prix.

Tab-specific handlers are in prompt_prix.tabs.{battery,compare}.handlers
"""

import logging
from typing import Optional

import gradio as gr

from prompt_prix import state
from prompt_prix.core import ServerPool
from prompt_prix.parsers import parse_servers_input

logger = logging.getLogger(__name__)


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


def _get_loaded_models() -> set[str]:
    """
    Get currently loaded models from LM Studio using the lmstudio SDK.

    Returns set of model identifiers that are currently loaded in memory.

    Note: This uses SDK auto-discovery which fails in Docker.
    Use _get_loaded_models_via_http() instead when server URLs are available.
    """
    try:
        import lmstudio as lms
        loaded = lms.list_loaded_models("llm")
        # Extract model identifiers from loaded model objects
        loaded_ids = set()
        for model in loaded:
            # The model object has a .model_key or similar attribute
            if hasattr(model, 'model_key'):
                loaded_ids.add(model.model_key)
            elif hasattr(model, 'path'):
                loaded_ids.add(model.path)
            elif hasattr(model, 'identifier'):
                loaded_ids.add(model.identifier)
            else:
                # Fallback: try string representation
                loaded_ids.add(str(model))
        return loaded_ids
    except ImportError:
        logger.warning("lmstudio SDK not installed - cannot filter by loaded models")
        return set()
    except Exception as e:
        logger.warning(f"Failed to get loaded models: {e}")
        return set()


async def _get_loaded_models_via_http(servers: list[str]) -> set[str]:
    """
    Query servers directly for loaded models via HTTP.

    This is the Docker-compatible alternative to _get_loaded_models().
    Uses the LM Studio REST API which returns model state (loaded/not-loaded).

    Args:
        servers: List of server URLs (e.g., ["http://localhost:1234"])

    Returns:
        Set of model IDs that are currently loaded in memory.
    """
    import httpx

    loaded_ids = set()
    async with httpx.AsyncClient(timeout=5.0) as client:
        for server_url in servers:
            try:
                # Use LM Studio's native REST API (not OpenAI-compat) for state info
                url = f"{server_url.rstrip('/')}/api/v0/models"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("data", []):
                        model_id = model.get("id")
                        state = model.get("state")
                        # Debug: log what we're seeing
                        logger.debug(f"Model {model_id}: state={state}, keys={list(model.keys())}")
                        # LM Studio REST API returns "state": "loaded" or "not-loaded"
                        if state == "loaded":
                            if model_id:
                                loaded_ids.add(model_id)
            except Exception as e:
                logger.debug(f"Could not query {server_url} for loaded models: {e}")
    return loaded_ids


async def fetch_available_models(
    servers_text: str,
    only_loaded: bool = False
) -> tuple[str, dict]:
    """
    Query all configured servers and return available models.

    Args:
        servers_text: Newline-separated server URLs
        only_loaded: If True, filter to only models currently loaded in LM Studio

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

    # Filter by loaded models if requested
    if only_loaded:
        # Try HTTP-based approach first (works in Docker)
        loaded_models = await _get_loaded_models_via_http(servers)
        if not loaded_models:
            # Fall back to SDK approach (works locally when SDK can auto-discover)
            loaded_models = _get_loaded_models()

        if loaded_models:
            all_models = all_models & loaded_models
            if not all_models:
                return "⚠️ No loaded models match available models", gr.update(choices=[])
        else:
            # Couldn't get loaded models via either method
            return "⚠️ Could not detect loaded models (server may not report load state)", gr.update(choices=sorted(all_models))

    sorted_models = sorted(all_models)

    status_parts = [f"✅ Found {len(sorted_models)} model(s)"]
    if only_loaded:
        status_parts[0] += " (loaded only)"

    for url, models in models_by_server.items():
        count = len([m for m in models if m in all_models]) if only_loaded else len(models)
        status_parts.append(f"  {url}: {count} model(s)")

    return " | ".join(status_parts), gr.update(choices=sorted_models)
