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
from prompt_prix.server_affinity import parse_server_prefix


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
    
    # Parse affinity
    server_idx, actual_model_id = parse_server_prefix(model_id)

    # Create strongly-typed task
    task = InferenceTask(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=float(timeout_seconds),
        tools=tools,
        seed=seed,
        repeat_penalty=repeat_penalty,
        preferred_server_idx=server_idx,
        api_model_id=actual_model_id
    )

    # Collect streaming response into final string
    response_parts = []
    async for chunk in adapter.stream_completion(task):
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
    
    # Parse affinity
    server_idx, actual_model_id = parse_server_prefix(model_id)

    # Create strongly-typed task
    task = InferenceTask(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=float(timeout_seconds),
        tools=tools,
        seed=seed,
        repeat_penalty=repeat_penalty,
        preferred_server_idx=server_idx,
        api_model_id=actual_model_id
    )

    async for chunk in adapter.stream_completion(task):
        yield chunk
