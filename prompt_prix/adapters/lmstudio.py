"""
LMStudioAdapter - LM Studio-specific subclass of PooledLocalInferenceAdapter.

Thin wrapper that defaults to LMSTUDIO_API_KEY for authentication.
Extension point for future LM Studio-specific quirks (Harmony token stripping,
$ref inlining, etc.). Currently behaviour-identical to the generic adapter.

All real implementation lives in pooled_local.py.
"""

from prompt_prix.adapters.pooled_local import (
    PooledLocalInferenceAdapter,
    LocalInferenceError,
    stream_completion,            # re-export for existing callers
    _normalize_tools_for_openai,  # re-export for existing callers
)

# Backwards-compatible alias
LMStudioError = LocalInferenceError


class LMStudioAdapter(PooledLocalInferenceAdapter):
    """
    LM Studio adapter — thin subclass of PooledLocalInferenceAdapter.

    Currently identical to the generic adapter. Provides an extension point
    for LM Studio-specific quirks if needed in the future.
    """

    def __init__(self, server_urls, api_key=None):
        super().__init__(
            server_urls,
            api_key=api_key,
            fallback_api_key_env="LMSTUDIO_API_KEY",
        )


# Re-export for backwards compatibility
__all__ = [
    "LMStudioAdapter",
    "LMStudioError",
    "stream_completion",
    "_normalize_tools_for_openai",
]
