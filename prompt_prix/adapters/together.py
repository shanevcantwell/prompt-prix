"""
TogetherAdapter - Together AI implementation of HostAdapter.

Cloud inference adapter — no server pool needed.
Together handles load balancing server-side.
"""

import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

import httpx

from prompt_prix.adapters.schema import InferenceTask

logger = logging.getLogger(__name__)

TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class TogetherError(Exception):
    """Together AI API error."""
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


class TogetherAdapter:
    """
    Together AI implementation of HostAdapter protocol.

    Simple adapter for cloud inference — no server pool needed.
    Together handles load balancing server-side.
    """

    def __init__(
        self,
        models: list[str],
        api_key: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Together API key required. "
                "Provide api_key parameter or set TOGETHER_API_KEY environment variable."
            )
        self._models = list(models)

    async def get_available_models(self) -> list[str]:
        """Return configured models (no discovery endpoint used)."""
        return list(self._models)

    def get_models_by_server(self) -> dict[str, list[str]]:
        """Return models grouped by 'server' (single endpoint for Together)."""
        return {"together-ai": list(self._models)}

    def get_unreachable_servers(self) -> list[str]:
        """Together is cloud — always reachable."""
        return []

    async def stream_completion(self, task: InferenceTask) -> AsyncGenerator[str, None]:
        """Stream completion from Together AI."""

        payload = {
            "model": task.model_id,
            "messages": task.messages,
            "temperature": task.temperature,
            "stream": True,
        }

        if task.max_tokens > 0:
            payload["max_tokens"] = task.max_tokens
        if task.tools:
            payload["tools"] = _normalize_tools_for_openai(task.tools)
        if task.seed is not None:
            payload["seed"] = int(task.seed)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        tool_call_accumulator: dict[int, dict] = {}

        try:
            async with httpx.AsyncClient(timeout=task.timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{TOGETHER_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        try:
                            error_data = json.loads(error_body)
                            msg = error_data.get("error", {}).get(
                                "message", str(error_body[:500])
                            )
                        except Exception:
                            msg = error_body.decode()[:500]
                        raise TogetherError(
                            f"Together API error for {task.model_id}: {msg}"
                        )

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})

                                content = delta.get("content", "")
                                if content:
                                    yield content

                                tool_calls = delta.get("tool_calls", [])
                                for tc in tool_calls:
                                    idx = tc.get("index", 0)
                                    func = tc.get("function", {})
                                    if idx not in tool_call_accumulator:
                                        tool_call_accumulator[idx] = {
                                            "name": "",
                                            "arguments": "",
                                        }
                                    if func.get("name"):
                                        tool_call_accumulator[idx]["name"] = func[
                                            "name"
                                        ]
                                    if func.get("arguments"):
                                        tool_call_accumulator[idx][
                                            "arguments"
                                        ] += func["arguments"]
                            except json.JSONDecodeError:
                                continue

            # Yield structured tool call sentinel
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
                    yield f"\n**Tool Call:** `{tc_data['name']}`\n"
                if tc_data["arguments"]:
                    yield f"```json\n{tc_data['arguments']}\n```\n"

        except httpx.TimeoutException as e:
            raise TogetherError(
                f"Together timeout for {task.model_id}: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise TogetherError(
                f"Together HTTP error for {task.model_id}: {e}"
            ) from e

        # Latency sentinel
        latency_ms = (time.time() - start_time) * 1000
        yield f"__LATENCY_MS__:{latency_ms}"
