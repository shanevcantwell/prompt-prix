"""MCP tool: react_execute — ReAct loop evaluation primitive.

Higher-order tool that executes a ReAct loop against one model
with deterministic mock tools. Used to evaluate how models behave
under iterative tool-calling execution.

Adapted from langgraph-agentic-scaffold react_mixin.py.

Key design decisions:
- Rebuilds messages from trace each iteration (ADR-CORE-055)
- Completion is implicit: no tool calls in response = done
- Mock tool dispatch is internal (test infrastructure, not reusable primitive)
- Stagnation detection via cycle_detection from react package
"""

import json
import logging
import time
from typing import Optional

from prompt_prix.mcp.tools.complete import complete_stream
from prompt_prix.react.cycle_detection import detect_cycle_with_pattern
from prompt_prix.react.schemas import (
    ToolCall,
    ReActIteration,
)

logger = logging.getLogger(__name__)


def _build_messages(
    system_prompt: str,
    goal: str,
    trace: list[ReActIteration],
) -> list[dict]:
    """Rebuild OpenAI messages from canonical trace.

    Fresh rebuild each iteration — trace is the canonical record,
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


def _dispatch_mock(
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


def _parse_tool_calls_from_stream(chunks: list[str]) -> tuple[str, list[dict], float]:
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
        elif chunk.startswith("__LATENCY_MS__:"):
            try:
                latency_ms = float(chunk[len("__LATENCY_MS__:"):])
            except ValueError:
                pass
        else:
            text_parts.append(chunk)

    return "".join(text_parts), tool_calls, latency_ms


async def react_execute(
    model_id: str,
    system_prompt: str,
    initial_message: str,
    mock_tools: dict[str, dict[str, str]],
    tools: list[dict],
    max_iterations: int = 15,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: int = 300,
    cycle_min_repetitions: int = 3,
) -> dict:
    """Execute a ReAct loop against one model with mock tools.

    Args:
        model_id: Model to evaluate
        system_prompt: System message for the model
        initial_message: User's task/goal
        mock_tools: {tool_name: {args_key: response, "_default": response}}
        tools: OpenAI tool definitions passed to the model
        max_iterations: Max tool call iterations before termination
        temperature: Model temperature (0.0 for deterministic eval)
        max_tokens: Max tokens per model response
        timeout_seconds: Timeout per model call
        cycle_min_repetitions: Min repetitions for stagnation detection

    Returns:
        Dict with trajectory, completion status, and quality metrics.
    """
    trace: list[ReActIteration] = []
    call_counter = 0
    valid_iterations = 0
    invalid_iterations = 0
    total_latency_ms = 0.0
    final_response: Optional[str] = None
    termination_reason: Optional[str] = None
    cycle_detected = False
    cycle_pattern = None

    for iteration in range(max_iterations):
        # Rebuild messages fresh from trace
        messages = _build_messages(system_prompt, initial_message, trace)

        # Call model via MCP complete_stream
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

        text_content, tool_calls_raw, latency_ms = _parse_tool_calls_from_stream(chunks)
        total_latency_ms += latency_ms

        # No tool calls → model is done
        if not tool_calls_raw:
            final_response = text_content
            break

        # Process tool calls
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
                # Garbled arguments — record as invalid, send error back to model
                invalid_iterations += 1
                trace.append(ReActIteration(
                    iteration=iteration + 1,
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
            observation = _dispatch_mock(tc_name, tc_args, mock_tools)
            success = not observation.startswith("Error:")

            if success:
                valid_iterations += 1
            else:
                invalid_iterations += 1

            trace.append(ReActIteration(
                iteration=iteration + 1,
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

        # Check stagnation after each iteration
        if len(trace) >= cycle_min_repetitions * 2:
            signatures = [
                (step.tool_call.name, tuple(sorted(step.tool_call.args.items())))
                for step in trace
            ]
            period, pattern = detect_cycle_with_pattern(
                signatures, min_repetitions=cycle_min_repetitions
            )
            if period is not None:
                cycle_detected = True
                cycle_pattern = [
                    {"name": name, "args": dict(args)}
                    for name, args in pattern
                ] if pattern else None
                termination_reason = "cycle_detected"
                logger.info(
                    "Stagnation detected: period=%d, pattern=%s",
                    period, cycle_pattern,
                )
                break
    else:
        # Loop exhausted without break → max iterations
        termination_reason = "max_iterations"

    total_iterations = len(trace)
    completed = final_response is not None

    return {
        "model_id": model_id,
        "iterations": [step.model_dump() for step in trace],
        "completed": completed,
        "final_response": final_response,
        "total_iterations": total_iterations,
        "total_latency_ms": total_latency_ms,
        "cycle_detected": cycle_detected,
        "cycle_pattern": cycle_pattern,
        "termination_reason": termination_reason,
        "valid_iterations": valid_iterations,
        "invalid_iterations": invalid_iterations,
        "completion_rate": valid_iterations / total_iterations if total_iterations > 0 else 0.0,
    }
