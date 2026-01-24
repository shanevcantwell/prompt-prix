"""
HostAdapter Protocol - defines the contract for LLM inference backends.

This is the WHAT (interface), not the HOW (implementation).
See lmstudio.py for concrete implementation.
"""

from typing import Protocol, AsyncGenerator, Optional
from prompt_prix.adapters.schema import InferenceTask

class HostAdapter(Protocol):
    """
    Contract for LLM inference backends.

    Implementations must provide:
    - Model discovery (get_available_models)
    - Streaming completion (stream_completion)

    Design rationale (from CLAUDE.md):
    - Separation of Concerns: Protocol defines capability, adapters define implementation
    - Provider-Agnostic: Same interface works for LM Studio, Ollama, vLLM, etc.
    """

    async def get_available_models(self) -> list[str]:
        """
        Return list of model IDs available on this host.

        Returns:
            List of model identifiers (e.g., ["llama-3.2-3b", "qwen2.5-7b"])
        """
        ...

    def get_models_by_server(self) -> dict[str, list[str]]:
        """Return models grouped by server URL."""
        ...

    def get_unreachable_servers(self) -> list[str]:
        """Return list of servers that returned no models."""
        ...

    async def stream_completion(self, task: InferenceTask) -> AsyncGenerator[str, None]:
        """
        Stream completion chunks from the model based on the task.

        Args:
            task: Strongly-typed InferenceTask containing model_id, messages, etc.

        Yields:
            Text chunks as they arrive from the model

        Raises:
            Exception on model error (fail loudly per CLAUDE.md)
        """
        ...
