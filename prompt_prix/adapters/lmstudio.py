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
import collections
import json
import logging
import httpx
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Coroutine, Optional, Protocol, TypeVar

from prompt_prix.config import ServerConfig

# INTERNAL SCHEMA
@dataclass
class _InferenceTask:
    """Represents a pending inference request."""
    model_id: str
    server_idx: Optional[int]
    # Future resolves to the server_url when acquired
    future: asyncio.Future[str] = field(default_factory=asyncio.Future)

logger = logging.getLogger(__name__)


class LMStudioError(Exception):
    """Human-readable error from LM Studio API."""
    pass


# ─────────────────────────────────────────────────────────────────────
# INTERNAL: ServerPool (not exported)
# ─────────────────────────────────────────────────────────────────────

class _ServerPool:
    """
    INTERNAL: Manages multiple LM Studio servers with atomic acquisition.
    
    Provides atomic find_and_acquire operations and strictly synchronous
    is_busy state management to prevent race conditions.
    """

    def __init__(self, server_urls: list[str]):
        self.servers: dict[str, ServerConfig] = {
            url: ServerConfig(url=url) for url in server_urls
        }
        # Event triggered whenever a server becomes available (released)
        self.resource_available = asyncio.Event()

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

    def get_all_available_models(self) -> set[str]:
        """Return union of all models across all servers."""
        result = set()
        for server in self.servers.values():
            result.update(server.available_models)
        return result

    def get_server_url_by_index(self, idx: int) -> Optional[str]:
        """Get server URL by index (0-based)."""
        server_urls = list(self.servers.keys())
        if 0 <= idx < len(server_urls):
            return server_urls[idx]
        return None

    def find_and_acquire(self, model_id: str) -> Optional[str]:
        """
        Atomically find a free server with the model and mark it busy.
        Returns server_url if successful, None otherwise.
        """
        for url, server in self.servers.items():
            # Check availability AND busy status in the same synchronous block
            if model_id in server.available_models:
                if not server.is_busy:
                    server.is_busy = True
                    logger.info(f"Acquired server {url} for {model_id}")
                    return url
                else:
                    logger.debug(f"Server {url} has model {model_id} but IS BUSY")
        return None

    def find_and_acquire_specific(self, server_idx: int, model_id: str) -> Optional[str]:
        """
        Atomically acquire a specific server index if available.
        """
        server_url = self.get_server_url_by_index(server_idx)
        if server_url is None:
            logger.warning(f"Invalid server index {server_idx} requested")
            return None
        
        server = self.servers[server_url]
        if model_id in server.available_models:
            if not server.is_busy:
                server.is_busy = True
                logger.info(f"Acquired specific server {server_url} (idx {server_idx}) for {model_id}")
                return server_url
            else:
                logger.debug(f"Specific server {server_url} (idx {server_idx}) IS BUSY")
        else:
            logger.warning(f"Specific server {server_url} (idx {server_idx}) does not have model {model_id}")
            
        return None

    def release_server(self, server_url: str) -> None:
        """Mark server as available and notify listeners."""
        if server_url in self.servers:
            self.servers[server_url].is_busy = False
            logger.info(f"Released server {server_url}")
            self.resource_available.set()


# ─────────────────────────────────────────────────────────────────────
# INTERNAL: ConcurrentDispatcher (not exported)
# ─────────────────────────────────────────────────────────────────────

class _ConcurrentDispatcher:
    """
    INTERNAL: Queue-based dispatcher for inference tasks.
    
    Maintains a queue of pending requests and manages the lifecycle of
    assigning them to available servers in the pool.
    """

    def __init__(self, pool: _ServerPool):
        self.pool = pool
        self._queue: collections.deque[_InferenceTask] = collections.deque()
        self._state_changed = asyncio.Event()
        self._dispatcher_task = None # Lazily started

    async def submit(self, model_id: str, server_idx: Optional[int]) -> str:
        """
        Submit a request and wait for a server to be acquired.
        Returns the acquired server_url.
        """
        # Ensure the dispatcher loop is running
        if self._dispatcher_task is None or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(self._process_queue_loop())

        future = asyncio.Future()
        task = _InferenceTask(model_id=model_id, server_idx=server_idx, future=future)
        
        self._queue.append(task)
        self._state_changed.set()
        
        try:
            # Wait for the dispatcher to assign a server
            return await future
        except asyncio.CancelledError:
            # If we were cancelled, but we actually got a server (race condition), we must release it!
            if future.done() and not future.cancelled():
                try:
                    server_url = future.result()
                    self.pool.release_server(server_url)
                except Exception:
                    pass
            
            # If not done, cancel the future so dispatcher knows to skip/remove it
            if not future.done():
                future.cancel()
            raise

    async def _process_queue_loop(self) -> None:
        """Background loop creating matches between Tasks and Servers."""
        try:
            while True:
                # Wait for either a new task or a server release
                # We clear the events before processing to ensure we catch updates during processing
                self._state_changed.clear()
                self.pool.resource_available.clear()

                if not self._queue:
                    # If nothing in queue, wait for a new task
                    await self._state_changed.wait()
                
                # Processing cycle
                # We rotate the queue to handle head-of-line blocking if a specific 
                # model/server combo is unavailable but others are not.
                rotated_tasks = 0
                queue_len = len(self._queue)
                
                for _ in range(queue_len):
                    if not self._queue:
                        break
                    
                    task = self._queue[0] # Peek

                    # Cleanup cancelled tasks
                    if task.future.done():
                        self._queue.popleft()
                        continue

                    # Try to acquire
                    url: Optional[str] = None
                    if task.server_idx is not None:
                        url = self.pool.find_and_acquire_specific(task.server_idx, task.model_id)
                    else:
                        url = self.pool.find_and_acquire(task.model_id)

                    if url:
                        # Success
                        self._queue.popleft()
                        try:
                            task.future.set_result(url)
                        except asyncio.InvalidStateError:
                            # Future was cancelled/completed concurrently. Release server.
                            self.pool.release_server(url)
                    else:
                        # Failed to acquire, move to back to let others try
                        # (Simple round-robin for availability)
                        self._queue.rotate(-1)
                        rotated_tasks += 1

                # If we still have tasks pending, we wait for a resource to free up
                # OR for a new task to come in.
                if self._queue:
                    # We wait for either event
                    wait_objs = [
                        asyncio.create_task(self.pool.resource_available.wait()),
                        asyncio.create_task(self._state_changed.wait())
                    ]
                    done, pending = await asyncio.wait(
                        wait_objs, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for p in pending:
                        p.cancel()
        except Exception as e:
            logger.error(f"Dispatcher loop crashed: {e}", exc_info=True)


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


from prompt_prix.adapters.schema import InferenceTask

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
        self._manifests_loaded = False

    async def get_available_models(self) -> list[str]:
        """Return list of all models available across all servers."""
        async with self._lock:
            await self._pool.refresh_all_manifests()
            self._manifests_loaded = True
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

    async def stream_completion(self, task: InferenceTask) -> AsyncGenerator[str, None]:
        """
        Stream completion from LM Studio server using InferenceTask.
        """
        import time

        # Task already contains parsed server affinity if caller set it
        # If not, we might need to parse it from model_id just in case caller didn't
        if task.preferred_server_idx is None or task.api_model_id is None:
             # Just to be safe, though caller should handle this
             from prompt_prix.server_affinity import parse_server_prefix
             server_idx, actual_model_id = parse_server_prefix(task.model_id)
             task.preferred_server_idx = server_idx
             task.api_model_id = actual_model_id

        if task.preferred_server_idx is not None:
            logger.debug(f"Server affinity: model={task.api_model_id} -> server {task.preferred_server_idx}")

        # Lazy manifest initialization: only refresh once if never loaded
        # This avoids serializing all requests while still ensuring manifests exist
        if not self._manifests_loaded:
            async with self._lock:
                if not self._manifests_loaded:  # Double-check after acquiring lock
                    await self._pool.refresh_all_manifests()
                    self._manifests_loaded = True

        server_url = None
        try:
            # 1. Acquire via Dispatcher (Queue-based)
            # This replaces the previous polling loop for atomic acquisition
            server_url = await asyncio.wait_for(
                self._dispatcher.submit(task.api_model_id, task.preferred_server_idx),
                timeout=task.timeout_seconds
            )

            # 2. Stream
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=task.api_model_id,
                messages=task.messages,
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                timeout_seconds=int(task.timeout_seconds),
                tools=task.tools,
                seed=task.seed,
                repeat_penalty=task.repeat_penalty,
            ):
                yield chunk

        except asyncio.TimeoutError:
             raise RuntimeError(f"Timeout waiting for server/completion for model: {task.api_model_id}")
        finally:
            if server_url:
                # 3. Release (Notifies Dispatcher)
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
    adapter stream_completion method via the MCP registry.
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
                raise LMStudioError(f"LM Studio error for {model_id}: {msg}")

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
                    yield f"\n**Tool Call:** `{tc_data["name"]}`\n"
                if tc_data["arguments"]:
                    yield f"```json\n{tc_data["arguments"]}\n```\n"
