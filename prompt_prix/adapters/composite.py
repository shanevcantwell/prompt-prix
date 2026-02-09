"""
CompositeAdapter — routes model_id to the correct child adapter.

Implements HostAdapter protocol. When multiple adapters are registered,
merges their model lists and routes stream_completion() to the adapter
that owns the requested model.

Collision handling: when two adapters serve the same model_id, the model
is prefixed with the adapter name (e.g., "lmstudio/qwen3-8b" vs
"together/qwen3-8b"). Bare names are used when unique (the common case).
"""

import logging
from typing import AsyncGenerator

from prompt_prix.adapters.schema import InferenceTask

logger = logging.getLogger(__name__)


class CompositeAdapter:
    """Routes model_id to the correct child adapter.

    Implements HostAdapter protocol by delegating to child adapters.
    """

    def __init__(self, adapters: dict):
        """
        Args:
            adapters: dict of name -> HostAdapter instance
        """
        self._adapters: dict = dict(adapters)
        self._model_index: dict[str, str] = {}  # model_id -> adapter name

    async def get_available_models(self) -> list[str]:
        """Fetch models from all children and build routing index."""
        # Gather raw models per adapter
        raw: dict[str, list[str]] = {}
        for name, adapter in self._adapters.items():
            raw[name] = await adapter.get_available_models()

        # Detect collisions
        model_to_adapters: dict[str, list[str]] = {}
        for name, models in raw.items():
            for model_id in models:
                model_to_adapters.setdefault(model_id, []).append(name)

        # Build index with collision prefixing
        self._model_index = {}
        all_models = []

        for model_id, adapter_names in model_to_adapters.items():
            if len(adapter_names) == 1:
                # Unique — bare name
                self._model_index[model_id] = adapter_names[0]
                all_models.append(model_id)
            else:
                # Collision — prefix each
                for name in adapter_names:
                    prefixed = f"{name}/{model_id}"
                    self._model_index[prefixed] = name
                    all_models.append(prefixed)

        return all_models

    def get_models_by_server(self) -> dict[str, list[str]]:
        """Merge models_by_server from all children."""
        result = {}
        for adapter in self._adapters.values():
            result.update(adapter.get_models_by_server())
        return result

    def get_unreachable_servers(self) -> list[str]:
        """Merge unreachable servers from all children."""
        result = []
        for adapter in self._adapters.values():
            result.extend(adapter.get_unreachable_servers())
        return result

    async def stream_completion(self, task: InferenceTask) -> AsyncGenerator[str, None]:
        """Route to the correct child adapter based on model_id."""
        adapter_name = self._model_index.get(task.model_id)

        if adapter_name is None:
            raise ValueError(
                f"Unknown model '{task.model_id}'. "
                f"Available: {sorted(self._model_index.keys())}"
            )

        adapter = self._adapters[adapter_name]

        # Strip prefix if present (adapter expects bare model_id)
        actual_model_id = task.model_id
        prefix = f"{adapter_name}/"
        if actual_model_id.startswith(prefix):
            actual_model_id = actual_model_id[len(prefix):]

        # Create task with resolved model_id
        routed_task = task.model_copy(update={"model_id": actual_model_id})

        async for chunk in adapter.stream_completion(routed_task):
            yield chunk
