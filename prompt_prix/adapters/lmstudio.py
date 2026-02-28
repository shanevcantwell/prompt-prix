"""
LMStudioAdapter - LM Studio implementation of HostAdapter.

Per ADR-006, this adapter OWNS its pool and dispatcher infrastructure.
Pool classes are imported from local-inference-pool (extracted per ADR-068).
Orchestration and MCP layers only see the HostAdapter protocol interface.
"""

import asyncio
import json
import logging
import httpx
from typing import AsyncGenerator, Optional

from local_inference_pool import ServerPool, ConcurrentDispatcher

from prompt_prix.adapters.schema import InferenceTask

logger = logging.getLogger(__name__)


class LMStudioError(Exception):
    """Human-readable error from LM Studio API."""
    pass


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
        self._pool = ServerPool(server_urls)
        self._dispatcher = ConcurrentDispatcher(self._pool)
        self._lock = asyncio.Lock()
        self._manifests_loaded = False

    def set_parallel_slots(self, slots: int) -> None:
        """Set max concurrent requests per server."""
        self._pool.set_max_concurrent(slots)

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
        # Lazy manifest initialization: only refresh once if never loaded
        # This avoids serializing all requests while still ensuring manifests exist
        if not self._manifests_loaded:
            async with self._lock:
                if not self._manifests_loaded:  # Double-check after acquiring lock
                    await self._pool.refresh_all_manifests()
                    self._manifests_loaded = True
                    # Log server state for debugging
                    for idx, (url, server) in enumerate(self._pool.servers.items()):
                        logger.info(f"Server[{idx}] {url}: models={server.available_models}")

        server_url = None
        try:
            # 1. Acquire via Dispatcher (Queue-based, no timeout)
            # Queue wait time is excluded from timeout, matching how latency is measured.
            # Only the actual HTTP call (in stream_completion) has a timeout.
            server_url = await self._dispatcher.submit(task.model_id)

            # 2. Stream (timeout applies here via httpx)
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=task.model_id,
                messages=task.messages,
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                timeout_seconds=int(task.timeout_seconds),
                tools=task.tools,
                seed=task.seed,
                repeat_penalty=task.repeat_penalty,
                response_format=task.response_format,
            ):
                yield chunk

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
    response_format: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream completion from a specific LM Studio server.

    This is a raw function for cases where the caller manages
    server selection directly. For most use cases, use the
    adapter stream_completion method via the MCP registry.

    Yields content chunks during streaming, then a final sentinel
    with inference latency: "__LATENCY_MS__:{milliseconds}"
    """
    import time

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
    if response_format is not None:
        payload["response_format"] = response_format

    start_time = time.time()
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
            stream_done = False
            has_content = False

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        stream_done = True
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            has_content = True
                            yield content
                        # Accumulate tool calls
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            has_content = True
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

            # Detect aborted stream: no [DONE] and no content = likely model load abort
            if not stream_done and not has_content:
                raise LMStudioError(f"Stream aborted for {model_id} (no [DONE], no content)")

            # Yield structured tool call sentinel (for react_step and similar)
            if tool_call_accumulator:
                structured = [
                    {"name": tc["name"], "arguments": tc["arguments"]}
                    for tc in tool_call_accumulator.values()
                    if tc["name"]
                ]
                if structured:
                    yield f"__TOOL_CALLS__:{json.dumps(structured)}"

            # Yield accumulated tool calls as markdown (for UI display)
            for tc_data in tool_call_accumulator.values():
                if tc_data["name"]:
                    yield f"\n**Tool Call:** `{tc_data["name"]}`\n"
                if tc_data["arguments"]:
                    yield f"```json\n{tc_data["arguments"]}\n```\n"

    # Yield latency sentinel (measured from httpx request to stream complete)
    latency_ms = (time.time() - start_time) * 1000
    yield f"__LATENCY_MS__:{latency_ms}"
