"""
Shared event handlers and helpers for prompt-prix.

Tab-specific handlers are in prompt_prix.tabs.{battery,compare}.handlers
"""

import logging

import gradio as gr

from prompt_prix import state
from prompt_prix.parsers import parse_servers_input

logger = logging.getLogger(__name__)


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


async def _get_loaded_models_via_http(servers: list[str]) -> set[str] | None:
    """
    Query servers directly for loaded models via HTTP.

    This is the Docker-compatible alternative to _get_loaded_models().
    Uses the LM Studio REST API which returns model state (loaded/not-loaded).

    Args:
        servers: List of server URLs (e.g., ["http://localhost:1234"])

    Returns:
        Set of model IDs that are currently loaded in memory.
        Returns None if all servers failed to respond (can't determine load state).
        Returns empty set if servers responded but no models are loaded.
    """
    import httpx

    loaded_ids = set()
    any_server_responded = False
    async with httpx.AsyncClient(timeout=5.0) as client:
        for server_url in servers:
            try:
                # Use LM Studio's native REST API (not OpenAI-compat) for state info
                url = f"{server_url.rstrip('/')}/api/v0/models"
                resp = await client.get(url)
                if resp.status_code == 200:
                    any_server_responded = True
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

    # Return None if no servers responded (can't determine state)
    # Return empty set if servers responded but nothing is loaded
    return loaded_ids if any_server_responded else None


def _ensure_adapter_registered(servers: list[str]) -> None:
    """
    Ensure an adapter is registered with the given servers.

    This re-registers the adapter if the server list changes, allowing
    the UI to dynamically update server configuration.
    """
    from prompt_prix.adapters.lmstudio import LMStudioAdapter
    from prompt_prix.mcp.registry import register_adapter

    # Always re-register with current servers from UI
    adapter = LMStudioAdapter(server_urls=servers)
    register_adapter(adapter)


async def fetch_available_models(
    servers_text: str,
    only_loaded: bool = False
) -> dict:
    """
    Query all configured servers and return available models per-server.

    Uses the list_models MCP tool for model discovery.
    Registers/re-registers the adapter with the provided servers.

    Args:
        servers_text: Newline-separated server URLs
        only_loaded: If True, filter to only models currently loaded in LM Studio

    Returns dict with:
        - status: Status message string
        - servers: {url: [models]} mapping
        - all_models: Combined list of all models
    """
    from prompt_prix.mcp.tools.list_models import list_models

    servers = parse_servers_input(servers_text)

    if not servers:
        return {"status": "❌ No servers configured", "servers": {}, "all_models": []}

    # Register adapter with current servers before using MCP tool
    _ensure_adapter_registered(servers)

    # Use MCP tool for model discovery (uses registry internally)
    result = await list_models()

    models_by_server = {
        url: models for url, models in result["servers"].items()
        if models  # Only include servers with models
    }

    if not models_by_server:
        return {
            "status": "⚠️ No models found on any server. Are models loaded in LM Studio?",
            "servers": {},
            "all_models": []
        }

    all_models = set(result["models"])

    # Filter by loaded models if requested
    if only_loaded:
        # Try HTTP-based approach first (works in Docker)
        loaded_models = await _get_loaded_models_via_http(servers)
        if loaded_models is None:
            # HTTP approach failed, fall back to SDK approach
            loaded_models = _get_loaded_models()
            if not loaded_models:
                # SDK also failed - can't determine load state
                return {
                    "status": "⚠️ Could not detect loaded models (server may not report load state)",
                    "servers": models_by_server,
                    "all_models": sorted(all_models)
                }

        # Filter to only loaded models (may be empty set if nothing loaded)
        all_models = all_models & loaded_models
        models_by_server = {
            url: [m for m in models if m in all_models]
            for url, models in models_by_server.items()
        }
        if not all_models:
            return {
                "status": "⚠️ No models currently loaded in VRAM",
                "servers": {},
                "all_models": []
            }

    sorted_models = sorted(all_models)

    status_parts = [f"✅ Found {len(sorted_models)} model(s)"]
    if only_loaded:
        status_parts[0] += " (loaded only)"

    for url, models in models_by_server.items():
        count = len([m for m in models if m in all_models]) if only_loaded else len(models)
        status_parts.append(f"  {url}: {count} model(s)")

    return {
        "status": " | ".join(status_parts),
        "servers": models_by_server,
        "all_models": sorted_models
    }
