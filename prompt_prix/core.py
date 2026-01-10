"""
Core logic: streaming completion, comparison session management.

Server pool management is in scheduler.py.
"""

import asyncio
import json
import httpx
from typing import AsyncGenerator, Optional, Callable

from prompt_prix.config import ModelContext, SessionState
from prompt_prix.scheduler import ServerPool


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
# LLM CLIENT
# ─────────────────────────────────────────────────────────────────────

def _normalize_tools_for_openai(tools: list[dict]) -> list[dict]:
    """
    Normalize tool definitions to OpenAI format.

    Flat format:
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
    max_tokens: int,
    timeout_seconds: int,
    temperature: Optional[float] = None,
    tools: Optional[list[dict]] = None,
    seed: Optional[int] = None,
    repeat_penalty: Optional[float] = None
) -> AsyncGenerator[str, None]:
    """
    Stream a completion from an LM Studio server.
    Yields text chunks as they arrive.
    Raises LMStudioError with user-friendly message on error.

    Args:
        temperature: If None, omitted from payload so LM Studio uses per-model defaults.
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True
    }
    if temperature is not None:
        payload["temperature"] = temperature
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

            # Accumulate tool calls until complete (Bug #36 fix)
            # OpenAI streaming sends tool call arguments token-by-token.
            # We must collect all chunks before formatting to avoid fragmented JSON.
            tool_call_accumulator: dict[int, dict] = {}  # index → {name, arguments}

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

                        # Handle tool calls - accumulate, don't yield yet
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

            # After streaming completes, yield accumulated tool calls with proper formatting
            for tc_data in tool_call_accumulator.values():
                if tc_data["name"]:
                    yield f"\n**Tool Call:** `{tc_data['name']}`\n"
                if tc_data["arguments"]:
                    yield f"```json\n{tc_data['arguments']}\n```\n"


async def get_completion(
    server_url: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    timeout_seconds: int,
    temperature: Optional[float] = None
) -> str:
    """
    Get a complete (non-streaming) response from an LM Studio server.
    Returns full response text.
    Raises LMStudioError with user-friendly message on error.

    Args:
        temperature: If None, omitted from payload so LM Studio uses per-model defaults.
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False
    }
    if temperature is not None:
        payload["temperature"] = temperature

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{server_url}/v1/chat/completions",
            json=payload
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

    Uses HostAdapter protocol for backend-agnostic operation.
    Concurrency is managed externally via semaphores based on adapter.get_concurrency_limit().

    Note: temperature is optional and defaults to None. When None, temperature
    is omitted from API requests so backend uses per-model defaults.
    """

    def __init__(
        self,
        models: list[str],
        adapter,  # HostAdapter protocol
        system_prompt: str,
        timeout_seconds: int,
        max_tokens: int,
        temperature: Optional[float] = None
    ):
        self.adapter = adapter
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

        messages = context.to_openai_messages(self.state.system_prompt)

        # Stream completion through adapter
        full_response = ""
        async for chunk in self.adapter.stream_completion(
            model_id=model_id,
            messages=messages,
            temperature=self.state.temperature,
            max_tokens=self.state.max_tokens,
            timeout_seconds=self.state.timeout_seconds
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
