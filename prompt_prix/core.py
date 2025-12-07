"""
Core logic: server pool management, manifest fetching, prompt execution.
"""

import asyncio
import json
import httpx
from typing import AsyncGenerator, Optional, Callable

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
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────

def _normalize_tools_for_openai(tools: list[dict]) -> list[dict]:
    """
    Normalize tool definitions to OpenAI format.

    BFCL flat format:
        {"name": "...", "description": "...", "parameters": {...}}

    OpenAI nested format:
        {"type": "function", "function": {"name": "...", ...}}

    This function wraps flat tools in the OpenAI structure.
    Already-wrapped tools are returned as-is.
    """
    normalized = []
    for tool in tools:
        # Already in OpenAI format (has "type": "function" wrapper)
        if tool.get("type") == "function" and "function" in tool:
            normalized.append(tool)
        # Flat format - wrap it
        else:
            normalized.append({
                "type": "function",
                "function": tool
            })
    return normalized


async def stream_completion(
    server_url: str,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    tools: Optional[list[dict]] = None,
    seed: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """
    Stream a completion from an LM Studio server.
    Yields text chunks as they arrive.
    Raises LMStudioError with user-friendly message on error.
    """
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

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        async with client.stream(
            "POST",
            f"{server_url}/v1/chat/completions",
            json=payload
        ) as response:
            if response.status_code >= 400:
                # Read the error body for streaming responses
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

                        # Handle regular content
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


async def get_completion(
    server_url: str,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int
) -> str:
    """
    Get a complete (non-streaming) response from an LM Studio server.
    Returns full response text.
    Raises LMStudioError with user-friendly message on error.
    """
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
        )
        if response.status_code >= 400:
            error_msg = parse_lm_studio_error(response)
            raise LMStudioError(f"LM Studio error for '{model_id}': {error_msg}")
        data = response.json()
        return data["choices"][0]["message"]["content"]


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
        context = self.state.contexts[model_id]
        context.add_user_message(prompt)

        # Find available server
        server_url = None
        while server_url is None:
            await self.server_pool.refresh_all_manifests()
            server_url = self.server_pool.find_available_server(model_id)
            if server_url is None:
                await asyncio.sleep(1.0)  # Wait and retry

        # Acquire server
        await self.server_pool.acquire_server(server_url)

        try:
            messages = context.to_openai_messages(self.state.system_prompt)

            if on_chunk:
                # Streaming mode
                full_response = ""
                async for chunk in stream_completion(
                    server_url=server_url,
                    model_id=model_id,
                    messages=messages,
                    temperature=self.state.temperature,
                    max_tokens=self.state.max_tokens,
                    timeout_seconds=self.state.timeout_seconds
                ):
                    full_response += chunk
                    await on_chunk(model_id, chunk)
            else:
                # Non-streaming mode
                full_response = await get_completion(
                    server_url=server_url,
                    model_id=model_id,
                    messages=messages,
                    temperature=self.state.temperature,
                    max_tokens=self.state.max_tokens,
                    timeout_seconds=self.state.timeout_seconds
                )

            context.add_assistant_message(full_response)
            return full_response

        finally:
            self.server_pool.release_server(server_url)

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
