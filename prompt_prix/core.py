"""
Core logic: server pool management, manifest fetching, prompt execution.
"""

import asyncio
import httpx
from typing import Optional, Callable

from prompt_prix.config import (
    ServerConfig, ModelContext, SessionState,
    MANIFEST_REFRESH_INTERVAL_SECONDS
)


class LMStudioError(Exception):
    """Human-readable error from LM Studio API."""
    pass


def parse_lm_studio_error(response: httpx.Response) -> str:
    """Extract a user-friendly error message from LM Studio response."""
    try:
        data = response.json()
        # LM Studio typically returns {"error": {"message": "..."}}
        if isinstance(data, dict):
            error = data.get("error", {})
            if isinstance(error, dict):
                message = error.get("message", "")
                if message:
                    return message
            elif isinstance(error, str):
                return error
        return f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:200]}"


# ─────────────────────────────────────────────────────────────────────
# SERVER POOL
# ─────────────────────────────────────────────────────────────────────

class ServerPool:
    """
    Manages multiple LM Studio servers.
    Tracks which models are available on each server and server busy state.
    """

    def __init__(self, server_urls: list[str]):
        self.servers: dict[str, ServerConfig] = {
            url: ServerConfig(url=url) for url in server_urls
        }
        self._locks: dict[str, asyncio.Lock] = {
            url: asyncio.Lock() for url in server_urls
        }

    async def refresh_all_manifests(self) -> None:
        """Fetch model lists from all servers."""
        tasks = [self._refresh_manifest(url) for url in self.servers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh_manifest(self, server_url: str) -> None:
        """Fetch model list from a single server."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{server_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                # LM Studio returns {"data": [{"id": "model-name", ...}, ...]}
                model_ids = [m["id"] for m in data.get("data", [])]
                self.servers[server_url].available_models = model_ids
        except Exception as e:
            # Server unreachable or error - clear its model list
            self.servers[server_url].available_models = []

    def find_available_server(self, model_id: str) -> Optional[str]:
        """
        Find a server that has the requested model and is not busy.
        Returns server URL or None if no server available.
        """
        for url, server in self.servers.items():
            if model_id in server.available_models and not server.is_busy:
                return url
        return None

    def get_all_available_models(self) -> set[str]:
        """Return union of all models across all servers."""
        result = set()
        for server in self.servers.values():
            result.update(server.available_models)
        return result

    async def acquire_server(self, server_url: str) -> None:
        """Mark server as busy."""
        await self._locks[server_url].acquire()
        self.servers[server_url].is_busy = True

    def release_server(self, server_url: str) -> None:
        """Mark server as available."""
        self.servers[server_url].is_busy = False
        try:
            self._locks[server_url].release()
        except RuntimeError:
            pass  # Lock wasn't held


# ─────────────────────────────────────────────────────────────────────
# COMPARISON SESSION
# ─────────────────────────────────────────────────────────────────────

class ComparisonSession:
    """
    Manages a multi-model comparison session.
    Handles prompt dispatch, context tracking, and failure handling.
    """

    def __init__(
        self,
        models: list[str],
        server_pool: ServerPool,
        system_prompt: str,
        temperature: float,
        timeout_seconds: int,
        max_tokens: int
    ):
        self.server_pool = server_pool
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
        server_urls = list(self.server_pool.servers.keys())

        full_response = ""
        async for chunk in complete_stream(
            server_urls=server_urls,
            model_id=model_id,
            messages=messages,
            temperature=self.state.temperature,
            max_tokens=self.state.max_tokens,
            timeout_seconds=self.state.timeout_seconds,
            pool=self.server_pool,
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
