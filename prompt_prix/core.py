"""
Core logic: comparison session management.

Per ADR-006, server pool management is INTERNAL to LMStudioAdapter.
This module contains orchestration-layer code only.
"""

import asyncio
from typing import Optional, Callable

from prompt_prix.config import ModelContext, SessionState


class LMStudioError(Exception):
    """Human-readable error from LM Studio API."""
    pass


# ─────────────────────────────────────────────────────────────────────
# COMPARISON SESSION
# ─────────────────────────────────────────────────────────────────────

class ComparisonSession:
    """
    Manages a multi-model comparison session.

    Per ADR-006 (Orchestration Layer):
    - Tracks conversation context per model
    - Calls MCP primitives ONLY — never adapters directly
    - DOES NOT know about servers, ServerPool, or ConcurrentDispatcher
    """

    def __init__(
        self,
        models: list[str],
        system_prompt: str,
        temperature: float,
        timeout_seconds: int,
        max_tokens: int
    ):
        self.state = SessionState(
            models=models,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens
        )
        # Initialize empty context for each model
        for model_id in models:
            self.state.contexts[model_id] = ModelContext(model_id=model_id)

    async def send_prompt_to_model(
        self,
        model_id: str,
        prompt: str,
        on_chunk: Optional[Callable] = None  # async callback(model_id, chunk)
    ) -> str:
        """
        Send a prompt to a single model and update its context.
        Returns the complete response.
        Raises exception on failure.
        """
        from prompt_prix.mcp.tools.complete import complete_stream

        context = self.state.contexts[model_id]
        context.add_user_message(prompt)

        messages = context.to_openai_messages(self.state.system_prompt)

        full_response = ""
        async for chunk in complete_stream(
            model_id=model_id,
            messages=messages,
            temperature=self.state.temperature,
            max_tokens=self.state.max_tokens,
            timeout_seconds=self.state.timeout_seconds,
        ):
            full_response += chunk
            if on_chunk:
                await on_chunk(model_id, chunk)

        context.add_assistant_message(full_response)
        return full_response

    async def send_prompt_to_all(
        self,
        prompt: str,
        on_chunk: Optional[Callable] = None  # async callback(model_id, chunk)
    ) -> dict[str, str]:
        """
        Send a prompt to all models in parallel (limited by server availability).
        Returns dict of model_id -> response.
        On any failure, sets halted=True and records error.
        """
        if self.state.halted:
            raise RuntimeError(f"Session halted: {self.state.halt_reason}")

        results = {}
        tasks = []

        async def run_model(model_id: str):
            try:
                response = await self.send_prompt_to_model(model_id, prompt, on_chunk)
                results[model_id] = response
            except Exception as e:
                self.state.contexts[model_id].error = str(e)
                self.state.halted = True
                self.state.halt_reason = f"Model {model_id} failed: {e}"
                raise

        for model_id in self.state.models:
            tasks.append(run_model(model_id))

        # Run all, but stop on first failure
        try:
            await asyncio.gather(*tasks)
        except Exception:
            pass  # Error already recorded in state

        return results

    def get_context_display(self, model_id: str) -> str:
        """Get displayable conversation for a model."""
        return self.state.contexts[model_id].to_display_format()

    def get_all_contexts(self) -> dict[str, str]:
        """Get displayable conversations for all models."""
        return {
            model_id: self.get_context_display(model_id)
            for model_id in self.state.models
        }
