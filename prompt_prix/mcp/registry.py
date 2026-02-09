"""
MCP Adapter Registry - Central point for adapter dependency injection.

Supports named multi-adapter registration. When multiple adapters are
registered, get_adapter() returns a CompositeAdapter that routes by model_id.
When only one is registered, it returns that adapter directly (passthrough).

Usage:
    # At startup (main.py) — additive registration
    register_adapter(LMStudioAdapter(servers), name="lmstudio")
    register_adapter(TogetherAdapter(models, key), name="together")

    # In MCP tools — unchanged
    adapter = get_adapter()
    async for chunk in adapter.stream_completion(...):
        yield chunk
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_prix.adapters.base import HostAdapter

_adapters: dict[str, "HostAdapter"] = {}


def register_adapter(adapter: "HostAdapter", name: str = "default") -> None:
    """
    Register an adapter instance by name.

    Multiple adapters can be registered. get_adapter() will return a
    CompositeAdapter when more than one is registered.

    Args:
        adapter: A configured HostAdapter implementation
        name: Identifier for this adapter (e.g., "lmstudio", "together")
    """
    _adapters[name] = adapter


def get_adapter() -> "HostAdapter":
    """
    Get the registered adapter(s).

    Returns a single adapter if only one is registered,
    or a CompositeAdapter wrapping all registered adapters.

    Raises:
        RuntimeError: If no adapter has been registered
    """
    if not _adapters:
        raise RuntimeError(
            "No adapter registered. Call register_adapter() at startup."
        )

    if len(_adapters) == 1:
        return next(iter(_adapters.values()))

    from prompt_prix.adapters.composite import CompositeAdapter
    return CompositeAdapter(_adapters)


def clear_adapter() -> None:
    """
    Clear all registered adapters.

    Primarily useful for testing to reset state between tests.
    """
    _adapters.clear()


def register_default_adapter() -> None:
    """
    Register adapters based on environment variables.

    Additive: registers all configured adapters, not just one.
    - LM Studio: if LM_STUDIO_SERVER_* vars are set
    - Together: if TOGETHER_API_KEY is set
    - HuggingFace: if HF_TOKEN is set (and no LM Studio servers)
    """
    from prompt_prix.config import get_default_servers, load_servers_from_env
    from prompt_prix.config import is_huggingface_mode, get_hf_models, get_hf_token
    from prompt_prix.config import get_together_api_key, get_together_models

    # LM Studio — register if servers configured
    servers = load_servers_from_env()
    if servers:
        from prompt_prix.adapters.lmstudio import LMStudioAdapter
        register_adapter(LMStudioAdapter(server_urls=servers), name="lmstudio")

    # Together — register if API key present
    together_key = get_together_api_key()
    if together_key:
        from prompt_prix.adapters.together import TogetherAdapter
        register_adapter(
            TogetherAdapter(models=get_together_models(), api_key=together_key),
            name="together",
        )

    # HuggingFace — register if HF_TOKEN set and no LM Studio servers
    if is_huggingface_mode():
        from prompt_prix.adapters.huggingface import HuggingFaceAdapter
        models = get_hf_models()
        token = get_hf_token()
        register_adapter(
            HuggingFaceAdapter(models=models, token=token),
            name="huggingface",
        )

    # Fallback: if nothing configured, register LM Studio with defaults
    if not _adapters:
        default_servers = get_default_servers()
        if default_servers:
            from prompt_prix.adapters.lmstudio import LMStudioAdapter
            register_adapter(
                LMStudioAdapter(server_urls=default_servers), name="lmstudio"
            )
