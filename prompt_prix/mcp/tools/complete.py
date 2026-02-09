"""MCP tool: complete - model completion primitive.

This is the core building block for orchestration layers.
Adapter is retrieved via registry - callers don't manage servers.

Usage:
    # Non-streaming (for batch processing)
    response = await complete(
        "qwen2.5-7b",
        [{"role": "user", "content": "Hello"}]
    )

    # Streaming (for UI responsiveness)
    async for chunk in complete_stream(
        "qwen2.5-7b",
        [{"role": "user", "content": "Hello"}]
    ):
        print(chunk, end="")
"""

from typing import AsyncGenerator, Optional

from prompt_prix.mcp.registry import get_adapter
from prompt_prix.adapters.schema import InferenceTask

LATENCY_SENTINEL = "__LATENCY_MS__:"


def parse_latency_sentinel(chunk: str) -> Optional[float]:
    """Extract latency_ms from adapter sentinel chunk, or None if not a sentinel."""
    if not chunk.startswith(LATENCY_SENTINEL):
        return None
    try:
        return float(chunk[len(LATENCY_SENTINEL):])
    except ValueError:
        return None


async def complete(
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout_seconds: int = 300,
    tools: Optional[list[dict]] = None,
    seed: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> str:
    """
    Get a completion from an LLM.

    Non-streaming variant - returns complete response.
    Adapter handles server selection internally.
    """
    adapter = get_adapter()

    task = InferenceTask(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=float(timeout_seconds),
        tools=tools,
        seed=seed,
        repeat_penalty=repeat_penalty,
    )

    # Collect streaming response into final string, filtering out latency sentinel
    response_parts = []
    async for chunk in adapter.stream_completion(task):
        if parse_latency_sentinel(chunk) is None:
            response_parts.append(chunk)

    return "".join(response_parts)


async def complete_stream(
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout_seconds: int = 300,
    tools: Optional[list[dict]] = None,
    seed: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a completion from an LLM.

    Streaming variant - yields chunks as they arrive.
    Adapter handles server selection internally.
    """
    adapter = get_adapter()

    task = InferenceTask(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=float(timeout_seconds),
        tools=tools,
        seed=seed,
        repeat_penalty=repeat_penalty,
    )

    async for chunk in adapter.stream_completion(task):
        yield chunk
