"""
LMStudioAdapter - LM Studio implementation of HostAdapter.

Per ADR-006, this adapter OWNS:
- ServerPool (multi-server management)
- ConcurrentDispatcher (parallel execution across GPUs)
- httpx streaming logic

These are INTERNAL implementation details. No other module may import them.
Orchestration and MCP layers only see the HostAdapter protocol interface.
"""

import asyncio
import json
import logging
import httpx
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Coroutine, Optional, Protocol, TypeVar

from prompt_prix.config import ServerConfig

logger = logging.getLogger(__name__)


class LMStudioError(Exception):
    """Human-readable error from LM Studio API."""
    pass


# ─────────────────────────────────────────────────────────────────────
# INTERNAL: ServerPool (not exported)
# ─────────────────────────────────────────────────────────────────────

class _ServerPool:
    """
    INTERNAL: Manages multiple LM Studio servers.

    This class is internal to LMStudioAdapter. Per ADR-006, no code outside
    this module should reference ServerPool.
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
                model_ids = [m["id"] for m in data.get("data", [])]
                self.servers[server_url].available_models = model_ids
        except Exception:
            self.servers[server_url].available_models = []

    def find_available_server(self, model_id: str) -> Optional[str]:
        """Find a server that has the model and is not busy."""
        for url, server in self.servers.items():
            if model_id in server.available_models and not server.is_busy:
                return url
        return None

    def get_server_url_by_index(self, idx: int) -> Optional[str]:
        """Get server URL by index (0-based)."""
        server_urls = list(self.servers.keys())
        if 0 <= idx < len(server_urls):
            return server_urls[idx]
        return None

    def find_specific_server(self, server_idx: int, model_id: str) -> Optional[str]:
        """Find a specific server by index if it has the model and is available."""
        server_url = self.get_server_url_by_index(server_idx)
        if server_url is None:
            return None
        server = self.servers[server_url]
        if model_id in server.available_models and not server.is_busy:
            return server_url
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
            pass


# ─────────────────────────────────────────────────────────────────────
# INTERNAL: ConcurrentDispatcher (not exported)
# ─────────────────────────────────────────────────────────────────────

class _WorkItem(Protocol):
    """Protocol for dispatchable work items."""
    @property
    def model_id(self) -> str: ...


_T = TypeVar("_T", bound=_WorkItem)
_CancellationCheck = Optional[Callable[[], bool]]


class _ConcurrentDispatcher:
    """
    INTERNAL: Parallel dispatch across multiple LM Studio servers.

    This class is internal to LMStudioAdapter. Per ADR-006, no code outside
    this module should reference ConcurrentDispatcher.
    """

    def __init__(self, pool: _ServerPool):
        self.pool = pool

    async def dispatch(
        self,
        work_items: list[_T],
        execute_fn: Callable[[_T, str], Coroutine[None, None, None]],
        should_cancel: _CancellationCheck = None,
    ) -> AsyncGenerator[int, None]:
        """Execute work items across available servers."""
        work_queue = list(work_items)
        active_tasks: dict[str, asyncio.Task] = {}
        completed = 0

        yield completed

        while work_queue or active_tasks:
            if should_cancel and should_cancel():
                logger.info("Cancellation requested, clearing work queue")
                work_queue.clear()

            for server_url, server in self.pool.servers.items():
                if server.is_busy:
                    continue

                matched_item = None
                for item in work_queue:
                    if item.model_id in server.available_models:
                        matched_item = item
                        break

                if matched_item:
                    work_queue.remove(matched_item)
                    await self.pool.acquire_server(server_url)
                    task = asyncio.create_task(
                        self._run_and_release(matched_item, server_url, execute_fn)
                    )
                    active_tasks[server_url] = task

            await asyncio.sleep(0.1)
            yield completed

            for url in list(active_tasks.keys()):
                if active_tasks[url].done():
                    try:
                        active_tasks[url].result()
                    except Exception:
                        pass
                    completed += 1
                    del active_tasks[url]

            if work_queue and not active_tasks:
                await self.pool.refresh_all_manifests()

        yield completed

    async def _run_and_release(
        self,
        item: _T,
        server_url: str,
        execute_fn: Callable[[_T, str], Coroutine[None, None, None]],
    ) -> None:
        """Execute work item and release server when done."""
        logger.debug(f"Acquired server {server_url} for model {item.model_id}")
        try:
            await execute_fn(item, server_url)
        except Exception as e:
            logger.warning(f"Task failed on {server_url}: {e}")
            raise
        finally:
            logger.debug(f"Releasing server {server_url}")
            self.pool.release_server(server_url)


# ─────────────────────────────────────────────────────────────────────
# INTERNAL: Utility functions
# ─────────────────────────────────────────────────────────────────────

def _normalize_tools_for_openai(tools: list[dict]) -> list[dict]:
    """Normalize tool definitions to OpenAI format."""
    normalized = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            normalized.append(tool)
        else:
            normalized.append({"type": "function", "function": tool})
    return normalized


# ─────────────────────────────────────────────────────────────────────
# PUBLIC: LMStudioAdapter
# ─────────────────────────────────────────────────────────────────────

class LMStudioAdapter:
    """
    LM Studio implementation of HostAdapter.

    Takes server_urls and creates ServerPool internally.
    Orchestration layers should not know about the internal pooling.
    """

    def __init__(self, server_urls: list[str]):
        """
        Initialize adapter with server URLs.

        Args:
            server_urls: List of LM Studio server URLs
        """
        self._pool = _ServerPool(server_urls)
        self._dispatcher = _ConcurrentDispatcher(self._pool)
        self._lock = asyncio.Lock()

    async def get_available_models(self) -> list[str]:
        """Return list of all models available across all servers."""
        async with self._lock:
            await self._pool.refresh_all_manifests()
            return list(self._pool.get_all_available_models())

    def get_models_by_server(self) -> dict[str, list[str]]:
        """Return models grouped by server URL."""
        return {
            url: list(server.available_models)
            for url, server in self._pool.servers.items()
        }

    def get_unreachable_servers(self) -> list[str]:
        """Return list of servers that returned no models."""
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

        Supports server affinity via prefix: "0:model_name" routes to server 0.
        Without prefix, finds any available server with the model.

        Waits for a server to become available if all are busy.
        Find + acquire is atomic (inside same lock) to prevent race
        condition where multiple tasks find the same server "available".
        """
        import time
        from prompt_prix.server_affinity import parse_server_prefix

        start_time = time.time()
        server_url = None

        # Parse server affinity prefix (e.g., "0:model_name" -> server_idx=0)
        server_idx, actual_model_id = parse_server_prefix(model_id)
        if server_idx is not None:
            logger.debug(f"Server affinity: model={actual_model_id} -> server {server_idx}")

        # Refresh manifests once at start
        async with self._lock:
            await self._pool.refresh_all_manifests()

        # Wait for a server to become available
        while True:
            # Check timeout (use completion timeout as max wait)
            if time.time() - start_time > timeout_seconds:
                raise RuntimeError(f"Timeout waiting for server for model: {actual_model_id}")

            async with self._lock:
                if server_idx is not None:
                    # Server affinity: only use the specified server
                    server_url = self._pool.find_specific_server(server_idx, actual_model_id)
                else:
                    # No affinity: use any available server
                    server_url = self._pool.find_available_server(actual_model_id)

                if server_url is not None:
                    # Acquire inside lock to prevent TOCTOU race
                    await self._pool.acquire_server(server_url)
                    break

            # Server not available, wait briefly and retry
            # Releasing lock allows other tasks to release their servers
            await asyncio.sleep(0.2)
        try:
            # Delegate to module-level function for actual streaming
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=actual_model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                tools=tools,
                seed=seed,
                repeat_penalty=repeat_penalty,
            ):
                yield chunk
        finally:
            self._pool.release_server(server_url)


# ─────────────────────────────────────────────────────────────────────
# Module-level convenience function for MCP tools
# ─────────────────────────────────────────────────────────────────────

async def stream_completion(
    server_url: str,
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    tools: Optional[list[dict]] = None,
    seed: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream completion from a specific LM Studio server.

    This is a raw function for cases where the caller manages
    server selection directly. For most use cases, use the
    adapter's stream_completion method via the MCP registry.
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
    if repeat_penalty is not None and repeat_penalty != 1.0:
        payload["repeat_penalty"] = float(repeat_penalty)

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

            # Accumulate tool calls until stream completes
            tool_call_accumulator: dict[int, dict] = {}

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
                        # Accumulate tool calls
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            idx = tc.get("index", 0)
                            func = tc.get("function", {})
                            if idx not in tool_call_accumulator:
                                tool_call_accumulator[idx] = {"name": "", "arguments": ""}
                            if func.get("name"):
                                tool_call_accumulator[idx]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_call_accumulator[idx]["arguments"] += func["arguments"]
                    except json.JSONDecodeError:
                        continue

            # Yield accumulated tool calls after stream completes
            for tc_data in tool_call_accumulator.values():
                if tc_data["name"]:
                    yield f"\n**Tool Call:** `{tc_data['name']}`\n"
                if tc_data["arguments"]:
                    yield f"```json\n{tc_data['arguments']}\n```\n"
