"""MCP tool: complete - model completion primitive.

This is the core building block for orchestration layers. Handles server
finding, acquiring, and releasing internally.

Usage:
    # Non-streaming (for batch processing)
    response = await complete(
        ["http://localhost:1234"],
        "qwen2.5-7b",
        [{"role": "user", "content": "Hello"}]
    )

    # Streaming (for UI responsiveness)
    async for chunk in complete_stream(
        ["http://localhost:1234"],
        "qwen2.5-7b",
        [{"role": "user", "content": "Hello"}]
    ):
        print(chunk, end="")
"""

from typing import AsyncGenerator, Optional

from prompt_prix.adapters.lmstudio import LMStudioAdapter
from prompt_prix.core import ServerPool


async def complete(
    server_urls: list[str],
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout_seconds: int = 300,
    tools: Optional[list[dict]] = None,
) -> str:
    """
    Get a completion from an LLM.

    Non-streaming variant - returns complete response.
    Handles server finding, acquiring, and releasing internally.

    This is an MCP primitive - a self-contained operation that can be called
    by orchestration layers (ConcurrentDispatcher), Gradio UI, CLI,
    or agentic systems.

    Args:
        server_urls: List of OpenAI-compatible server URLs
            e.g., ["http://192.168.1.10:1234", "http://localhost:1234"]
        model_id: Model identifier (must be available on at least one server)
        messages: OpenAI-format messages array
            e.g., [{"role": "user", "content": "Hello"}]
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum response tokens (default 2048)
        timeout_seconds: Request timeout (default 300)
        tools: Optional tool definitions in OpenAI format

    Returns:
        Complete response text

    Raises:
        RuntimeError: If no server available for model
        LMStudioError: On API errors
    """
    pool = ServerPool(server_urls)
    adapter = LMStudioAdapter(pool)

    # Collect streaming response into final string
    response_parts = []
    async for chunk in adapter.stream_completion(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        tools=tools,
    ):
        response_parts.append(chunk)

    return "".join(response_parts)


async def complete_stream(
    server_urls: list[str],
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout_seconds: int = 300,
    tools: Optional[list[dict]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a completion from an LLM.

    Streaming variant - yields chunks as they arrive.
    Handles server finding, acquiring, and releasing internally.

    This is an MCP primitive - a self-contained operation that can be called
    by UI layers that need responsive streaming output.

    Args:
        server_urls: List of OpenAI-compatible server URLs
        model_id: Model identifier (must be available on at least one server)
        messages: OpenAI-format messages array
        temperature: Sampling temperature (default 0.7)
        max_tokens: Maximum response tokens (default 2048)
        timeout_seconds: Request timeout (default 300)
        tools: Optional tool definitions in OpenAI format

    Yields:
        Text chunks as they arrive from the model

    Raises:
        RuntimeError: If no server available for model
        LMStudioError: On API errors
    """
    pool = ServerPool(server_urls)
    adapter = LMStudioAdapter(pool)

    async for chunk in adapter.stream_completion(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        tools=tools,
    ):
        yield chunk
