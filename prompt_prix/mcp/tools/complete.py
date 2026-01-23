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

    This is an MCP primitive - a self-contained operation that can be called
    by orchestration layers (BatteryRunner), Gradio UI, CLI,
    or agentic systems.

    Args:
        model_id: Model identifier (must be available on at least one server)
        messages: OpenAI-format messages array
            e.g., [{"role": "user", "content": "Hello"}]
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum response tokens (default 2048)
        timeout_seconds: Request timeout (default 300)
        tools: Optional tool definitions in OpenAI format
        seed: Optional seed for reproducibility
        repeat_penalty: Optional penalty for repeated tokens

    Returns:
        Complete response text

    Raises:
        RuntimeError: If no adapter registered or no server available
        LMStudioError: On API errors
    """
    adapter = get_adapter()

    # Collect streaming response into final string
    response_parts = []
    async for chunk in adapter.stream_completion(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        tools=tools,
        seed=seed,
        repeat_penalty=repeat_penalty,
    ):
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

    This is an MCP primitive - a self-contained operation that can be called
    by UI layers that need responsive streaming output.

    Args:
        model_id: Model identifier (must be available on at least one server)
        messages: OpenAI-format messages array
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum response tokens (default 2048)
        timeout_seconds: Request timeout (default 300)
        tools: Optional tool definitions in OpenAI format
        seed: Optional seed for reproducibility
        repeat_penalty: Optional penalty for repeated tokens

    Yields:
        Text chunks as they arrive from the model

    Raises:
        RuntimeError: If no adapter registered or no server available
        LMStudioError: On API errors
    """
    adapter = get_adapter()

    async for chunk in adapter.stream_completion(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        tools=tools,
        seed=seed,
        repeat_penalty=repeat_penalty,
    ):
        yield chunk
