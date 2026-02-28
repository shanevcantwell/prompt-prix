"""
HuggingFace Inference API adapter.

Implements HostAdapter protocol for HuggingFace serverless inference.
Uses huggingface_hub.AsyncInferenceClient for native async support.

Key differences from LMStudioAdapter:
- No discovery endpoint: Models are user-configured, not discovered
- No server pool: HF handles rate limiting server-side
- Cloud API: Always "reachable" - errors occur at call time
"""

import os
import time
from typing import AsyncGenerator, Optional

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import InferenceTimeoutError, HfHubHTTPError

from prompt_prix.adapters.schema import InferenceTask


class HuggingFaceError(Exception):
    """HuggingFace adapter-specific error."""
    pass


class HuggingFaceAdapter:
    """
    HuggingFace implementation of HostAdapter protocol.

    Design decisions:
    - No discovery: Models are user-provided (HF has no list endpoint)
    - Token from env: HF_TOKEN env var (Spaces-compatible)
    - Async native: Uses AsyncInferenceClient directly
    - No resource pool: HF manages rate limiting server-side

    Usage:
        adapter = HuggingFaceAdapter(
            models=["meta-llama/Llama-3.2-3B-Instruct"],
            token="hf_xxx"  # or set HF_TOKEN env var
        )
    """

    def __init__(
        self,
        models: list[str],
        token: Optional[str] = None,
    ):
        """
        Initialize adapter with model list.

        Args:
            models: List of HuggingFace model IDs
            token: HuggingFace API token. Falls back to HF_TOKEN env var.

        Raises:
            ValueError: If no token provided and HF_TOKEN not set
        """
        self._token = token or os.environ.get("HF_TOKEN")
        if not self._token:
            raise ValueError(
                "HuggingFace token required. "
                "Provide token parameter or set HF_TOKEN environment variable."
            )

        self._models = list(models)
        self._client = AsyncInferenceClient(token=self._token)

    async def get_available_models(self) -> list[str]:
        """
        Return list of user-configured model IDs.

        Note: HuggingFace has no discovery endpoint.
        This returns the models configured at initialization.
        """
        return list(self._models)

    def get_models_by_server(self) -> dict[str, list[str]]:
        """
        Return models grouped by "server".

        For HF, there's conceptually one server (the inference API).
        """
        return {"huggingface-inference": list(self._models)}

    def get_unreachable_servers(self) -> list[str]:
        """
        Return list of unreachable servers.

        HF is a cloud API - always reachable. Errors occur at call time.
        """
        return []

    def add_model(self, model_id: str) -> None:
        """Add a model to the available list."""
        if model_id not in self._models:
            self._models.append(model_id)

    def remove_model(self, model_id: str) -> None:
        """Remove a model from the available list."""
        if model_id in self._models:
            self._models.remove(model_id)

    async def stream_completion(
        self,
        task: InferenceTask
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from HuggingFace Inference API.

        Args:
            task: InferenceTask with model_id, messages, temperature, etc.

        Yields:
            Text chunks as they arrive, then "__LATENCY_MS__:{ms}" sentinel

        Raises:
            HuggingFaceError: On API errors
        """
        start_time = time.time()
        has_content = False

        try:
            # Build kwargs for chat_completion
            kwargs = {
                "model": task.model_id,
                "messages": task.messages,
                "temperature": task.temperature,
                "stream": True,
            }

            # HF uses max_tokens (not -1 for unlimited like LM Studio)
            if task.max_tokens > 0:
                kwargs["max_tokens"] = task.max_tokens

            # Seed for reproducibility
            if task.seed is not None:
                kwargs["seed"] = task.seed

            if task.response_format is not None:
                kwargs["response_format"] = task.response_format

            # Note: tools support deferred to Phase 4
            # if task.tools:
            #     kwargs["tools"] = task.tools

            # Stream from HF API
            stream = await self._client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        has_content = True
                        yield content

                    # Check for finish_reason
                    if chunk.choices[0].finish_reason:
                        break

        except InferenceTimeoutError as e:
            raise HuggingFaceError(
                f"HuggingFace timeout for {task.model_id}: {e}"
            ) from e
        except HfHubHTTPError as e:
            raise HuggingFaceError(
                f"HuggingFace API error for {task.model_id}: {e}"
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            raise HuggingFaceError(
                f"HuggingFace error for {task.model_id}: {e}"
            ) from e

        # Empty response handling: model returned nothing
        # This is valid (semantic validator will catch it)
        if not has_content:
            pass  # Let semantic validator handle empty responses

        # Yield latency sentinel
        latency_ms = (time.time() - start_time) * 1000
        yield f"__LATENCY_MS__:{latency_ms}"
