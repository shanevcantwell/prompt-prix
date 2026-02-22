"""MCP tool: react_step — single ReAct iteration primitive.

Stateless: takes the current trace, executes one model call,
parses tool calls, dispatches mocks, returns the new iteration(s)
or signals completion if no tool calls in the response.

Adapted from react_execute.py per MCP statelessness mandate:
loop/orchestration belongs in ReactRunner, not in an MCP tool.
"""

import json
import logging
from typing import Optional

from prompt_prix.mcp.tools.complete import complete_stream, parse_latency_sentinel
from prompt_prix.react.schemas import (
    ToolCall,
    ReActIteration,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# PUBLIC UTILITIES (used by react_step and available to callers)
# ─────────────────────────────────────────────────────────────────────


def build_react_messages(
    system_prompt: str,
    goal: str,
    trace: list[ReActIteration],
) -> list[dict]:
    """Rebuild OpenAI messages from canonical trace.

    Fresh rebuild each call — trace is the canonical record,
    messages are ephemeral (ADR-CORE-055).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]
    for step in trace:
        messages.append({
            "role": "assistant",
            "content": step.thought or "",
            "tool_calls": [{
                "id": step.tool_call.id,
                "type": "function",
                "function": {
                    "name": step.tool_call.name,
                    "arguments": json.dumps(step.tool_call.args),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "content": step.observation,
            "tool_call_id": step.tool_call.id,
        })
    return messages


def dispatch_mock(
    tool_name: str,
    tool_args: dict,
    mock_tools: dict[str, dict[str, str]],
) -> str:
    """Look up mock response for a tool call.

    Resolution order:
    1. Exact args match (JSON-serialized, sorted keys)
    2. First arg value match (e.g., path for read_file)
    3. _default fallback
    4. Error message
    """
    tool_mocks = mock_tools.get(tool_name, {})

    # Exact args match
    args_key = json.dumps(tool_args, sort_keys=True)
    if args_key in tool_mocks:
        return tool_mocks[args_key]

    # First arg value match
    for arg_val in tool_args.values():
        if str(arg_val) in tool_mocks:
            return tool_mocks[str(arg_val)]

    # Default
    if "_default" in tool_mocks:
        return tool_mocks["_default"]

    return f"Error: No mock response for {tool_name}({tool_args})"


def parse_tool_calls_from_stream(chunks: list[str]) -> tuple[str, list[dict], float]:
    """Extract text, structured tool calls, and latency from stream chunks.

    Returns:
        (text_content, tool_calls, latency_ms)
        tool_calls is a list of {"name": str, "arguments": str} dicts
    """
    text_parts = []
    tool_calls = []
    latency_ms = 0.0

    for chunk in chunks:
        if chunk.startswith("__TOOL_CALLS__:"):
            try:
                tool_calls = json.loads(chunk[len("__TOOL_CALLS__:"):])
            except json.JSONDecodeError:
                logger.warning("Failed to parse __TOOL_CALLS__ sentinel")
        elif (lat := parse_latency_sentinel(chunk)) is not None:
            latency_ms = lat
        else:
            text_parts.append(chunk)

    return "".join(text_parts), tool_calls, latency_ms


# ─────────────────────────────────────────────────────────────────────
# MCP PRIMITIVE
# ─────────────────────────────────────────────────────────────────────


async def react_step(
    model_id: str,
    system_prompt: str,
    initial_message: str,
    trace: list[ReActIteration],
    mock_tools: dict[str, dict[str, str]] | None,
    tools: list[dict],
    call_counter: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: int = 300,
) -> dict:
    """Execute one ReAct iteration against a model.

    Stateless MCP primitive: takes trace in, returns one step out.
    The caller (ReactRunner) manages the loop, stagnation detection,
    and progress reporting.

    Args:
        model_id: Model to call
        system_prompt: System message
        initial_message: User's goal/task
        trace: Previous iterations (used to rebuild messages)
        mock_tools: {tool_name: {args_key: response}}, or None for
            tool-forwarding mode (returns pending calls without dispatch)
        tools: OpenAI tool definitions
        call_counter: Running tool call counter (for unique IDs)
        temperature: Model temperature
        max_tokens: Max tokens per response
        timeout_seconds: Timeout per call

    Returns:
        {
            "completed": bool,          # True if model responded with text only
            "final_response": str|None, # Text response when completed
            "new_iterations": list,     # New ReActIteration objects from this step
            "call_counter": int,        # Updated counter for next step
            "latency_ms": float,
        }
    """
    messages = build_react_messages(system_prompt, initial_message, trace)

    chunks = []
    async for chunk in complete_stream(
        model_id=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        tools=tools if tools else None,
    ):
        chunks.append(chunk)

    text_content, tool_calls_raw, latency_ms = parse_tool_calls_from_stream(chunks)

    # No tool calls → model is done
    if not tool_calls_raw:
        return {
            "completed": True,
            "final_response": text_content,
            "new_iterations": [],
            "pending_tool_calls": [],
            "call_counter": call_counter,
            "latency_ms": latency_ms,
        }

    # ── DONE interception: model signals completion via tool call ──
    done_raw = next(
        (tc for tc in tool_calls_raw if tc.get("name", "") == "DONE"), None
    )
    if done_raw is not None:
        call_counter += 1
        tc_args_raw = done_raw.get("arguments", "{}")
        try:
            done_args = json.loads(tc_args_raw) if isinstance(tc_args_raw, str) else tc_args_raw
            if not isinstance(done_args, dict):
                done_args = {}
        except (json.JSONDecodeError, ValueError):
            done_args = {}
        return {
            "completed": True,
            "final_response": done_args.get("response", text_content or ""),
            "done_args": done_args,
            "done_trace_entry": {
                "tool_call": {
                    "id": f"call_{call_counter}",
                    "name": "DONE",
                    "args": done_args,
                },
                "thought": text_content if text_content else None,
            },
            "new_iterations": [],
            "pending_tool_calls": [],
            "call_counter": call_counter,
            "latency_ms": latency_ms,
        }

    # ── Tool-forwarding mode: return parsed calls for caller dispatch ──
    if mock_tools is None:
        pending = []
        for tc_raw in tool_calls_raw:
            call_counter += 1
            tc_name = tc_raw.get("name", "")
            tc_args_raw = tc_raw.get("arguments", "{}")
            try:
                tc_args = json.loads(tc_args_raw) if isinstance(tc_args_raw, str) else tc_args_raw
                if not isinstance(tc_args, dict):
                    tc_args = {}
            except (json.JSONDecodeError, ValueError):
                tc_args = {}
            pending.append({
                "id": f"call_{call_counter}",
                "name": tc_name,
                "args": tc_args,
            })
        return {
            "completed": False,
            "final_response": None,
            "new_iterations": [],
            "pending_tool_calls": pending,
            "call_counter": call_counter,
            "thought": text_content if text_content else None,
            "latency_ms": latency_ms,
        }

    # Process tool calls (mock dispatch mode)
    new_iterations = []
    iteration_num = len(trace) + 1

    for tc_raw in tool_calls_raw:
        call_counter += 1
        tc_name = tc_raw.get("name", "")
        tc_args_raw = tc_raw.get("arguments", "{}")

        # Parse arguments
        try:
            tc_args = json.loads(tc_args_raw) if isinstance(tc_args_raw, str) else tc_args_raw
            if not isinstance(tc_args, dict):
                raise ValueError(f"Expected dict, got {type(tc_args).__name__}")
        except (json.JSONDecodeError, ValueError) as e:
            new_iterations.append(ReActIteration(
                iteration=iteration_num,
                tool_call=ToolCall(
                    id=f"call_{call_counter}",
                    name=tc_name or "unknown",
                    args={},
                ),
                observation=f"Error: Could not parse tool arguments: {e}",
                success=False,
                thought=text_content if text_content else None,
                latency_ms=latency_ms,
            ))
            continue

        # Dispatch mock response
        observation = dispatch_mock(tc_name, tc_args, mock_tools)
        success = not observation.startswith("Error:")

        new_iterations.append(ReActIteration(
            iteration=iteration_num,
            tool_call=ToolCall(
                id=f"call_{call_counter}",
                name=tc_name,
                args=tc_args,
            ),
            observation=observation,
            success=success,
            thought=text_content if text_content else None,
            latency_ms=latency_ms,
        ))

    return {
        "completed": False,
        "final_response": None,
        "new_iterations": new_iterations,
        "call_counter": call_counter,
        "latency_ms": latency_ms,
    }
