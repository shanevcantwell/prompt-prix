"""
LMStudioAdapter - LM Studio implementation of HostAdapter.

Owns the httpx streaming logic for OpenAI-compatible LM Studio servers.
"""

import asyncio
import json
import httpx
from typing import AsyncGenerator, Optional

from prompt_prix.core import ServerPool, LMStudioError


def _normalize_tools_for_openai(tools: list[dict]) -> list[dict]:
    """
    Normalize tool definitions to OpenAI format.

    BFCL flat format:
        {"name": "...", "description": "...", "parameters": {...}}

    OpenAI nested format:
        {"type": "function", "function": {"name": "...", ...}}
    """
    normalized = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            normalized.append(tool)
        else:
            normalized.append({"type": "function", "function": tool})
    return normalized


class LMStudioAdapter:
    """
    LM Studio implementation of HostAdapter.

    Owns httpx streaming logic. Uses ServerPool for server discovery.
    Uses asyncio.Lock for thread-safety (state hygiene per CLAUDE.md).
    """

    def __init__(self, server_pool: ServerPool):
        """
        Initialize adapter with existing ServerPool.

        Args:
            server_pool: Pre-configured ServerPool instance
        """
        self._pool = server_pool
        self._lock = asyncio.Lock()

    @property
    def pool(self) -> ServerPool:
        """Expose ServerPool for concurrent dispatcher access."""
        return self._pool

    async def get_available_models(self) -> list[str]:
        """Return list of all models available across all servers."""
        async with self._lock:
            await self._pool.refresh_all_manifests()
            return list(self._pool.get_all_available_models())

    def get_models_by_server(self) -> dict[str, list[str]]:
        """
        Return models grouped by server URL.

        Must be called after get_available_models() or refresh.
        """
        return {
            url: list(server.available_models)
            for url, server in self._pool.servers.items()
        }

    def get_unreachable_servers(self) -> list[str]:
        """
        Return list of servers that returned no models.

        This is a proxy for connection failures - if a server was reachable
        but has no models loaded, it still appears here.
        """
        return [
            url for url, server in self._pool.servers.items()
            if not server.available_models
        ]

    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tools: Optional[list[dict]] = None,
        seed: Optional[int] = None,
        repeat_penalty: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from LM Studio server.

        Finds an available server with the model, acquires it,
        streams the completion, then releases the server.
        """
        # Find available server
        server_url = None
        async with self._lock:
            await self._pool.refresh_all_manifests()
            server_url = self._pool.find_available_server(model_id)

        if server_url is None:
            raise RuntimeError(f"No server available for model: {model_id}")

        # Acquire and stream
        await self._pool.acquire_server(server_url)
        try:
            # Build payload
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            if tools:
                payload["tools"] = _normalize_tools_for_openai(tools)
            if seed is not None:
                payload["seed"] = int(seed)
            if repeat_penalty is not None and repeat_penalty != 1.0:
                payload["repeat_penalty"] = float(repeat_penalty)

            # Stream from server
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{server_url}/v1/chat/completions",
                    json=payload
                ) as response:
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        try:
                            error_data = json.loads(error_body)
                            error = error_data.get("error", {})
                            if isinstance(error, dict):
                                msg = error.get("message", str(error_body[:200]))
                            else:
                                msg = str(error)
                        except Exception:
                            msg = error_body.decode()[:200]
                        raise LMStudioError(f"LM Studio error for '{model_id}': {msg}")

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                                # Handle tool calls
                                tool_calls = delta.get("tool_calls", [])
                                for tc in tool_calls:
                                    func = tc.get("function", {})
                                    name = func.get("name", "")
                                    args = func.get("arguments", "")
                                    if name:
                                        yield f"\n**Tool Call:** `{name}`\n"
                                    if args:
                                        yield f"```json\n{args}\n```\n"
                            except json.JSONDecodeError:
                                continue
        finally:
            self._pool.release_server(server_url)
