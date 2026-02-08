"""MCP tool: judge - semantic evaluation primitive.

Evaluates a model response against criteria using a judge model.
Self-contained: calls complete() internally.
Adapter is retrieved via registry - callers don't manage servers.

Usage:
    result = await judge(
        response="I'll help you delete report.pdf",
        criteria="Response must indicate intent to delete the file",
        judge_model="qwen2.5-7b",
    )
    # result = {"pass": True, "reason": "Response clearly states...", "score": None}
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from prompt_prix.mcp.tools.complete import complete

logger = logging.getLogger(__name__)

# Prompt template for the judge model
JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge. Your task is to determine if a response meets the given criteria.

## Criteria
{criteria}

## Response to Evaluate
{response}

## Instructions
Evaluate whether the response meets the criteria. Respond with a JSON object containing:
- "pass": true or false
- "reason": a brief explanation of your judgment (1-2 sentences)
- "score": null (or a number 0-10 if you can quantify how well criteria were met)

Respond ONLY with the JSON object, no other text.

## Your Judgment
"""


@dataclass
class JudgeResult:
    """Result of judge evaluation."""

    passed: bool
    reason: str
    score: Optional[float] = None
    raw_response: Optional[str] = None


async def judge(
    response: str,
    criteria: str,
    judge_model: str,
    temperature: float = 0.1,
    max_tokens: int = 256,
    timeout_seconds: int = 60,
) -> dict:
    """
    Evaluate a response against criteria using a judge model.

    This is an MCP primitive - a self-contained operation that can be called
    by orchestration layers, CLI, or agentic systems.
    Adapter is retrieved via registry - callers don't manage servers.

    Args:
        response: The model response to evaluate
        criteria: Natural language description of pass/fail criteria
            e.g., "Response must call the delete_file tool"
            e.g., "Response should be helpful and not refuse the task"
        judge_model: Model identifier to use as judge
        temperature: Low temperature (0.1) for consistent judging
        max_tokens: Max tokens for judge response (256 is usually enough)
        timeout_seconds: Request timeout (default 60)

    Returns:
        dict with keys:
            - "pass": bool - whether response meets criteria
            - "reason": str - explanation of judgment
            - "score": float | None - optional quantified score (0-10)

    Raises:
        RuntimeError: If no adapter registered or no server available
        LMStudioError: On API errors
        ValueError: If judge response cannot be parsed
    """
    # Build the judging prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        criteria=criteria,
        response=response,
    )

    messages = [{"role": "user", "content": prompt}]

    # Get judge's evaluation
    judge_response = await complete(
        model_id=judge_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )

    # Parse the response
    result = _parse_judge_response(judge_response)
    result["raw_response"] = judge_response

    return result


def _parse_judge_response(response: str) -> dict:
    """
    Parse judge model's response into structured result.

    Attempts to extract JSON from the response, with fallback heuristics.
    """
    # Strip <think>...</think> blocks (Qwen, DeepSeek reasoning models)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Try to find JSON in markdown code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        response = json_match.group(1)
    else:
        # Try to find a JSON object anywhere in the response
        json_match = re.search(r"\{[^{}]*\"pass\"[^{}]*\}", response, re.DOTALL)
        if json_match:
            response = json_match.group(0)

    # Try direct JSON parse
    try:
        data = json.loads(response.strip())
        return {
            "pass": bool(data.get("pass", False)),
            "reason": str(data.get("reason", "No reason provided")),
            "score": _parse_score(data.get("score")),
        }
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract from partial/malformed response
    logger.warning(f"Could not parse judge response as JSON, using heuristics: {response[:100]}")

    # Look for pass/fail indicators
    response_lower = response.lower()
    passed = False
    if '"pass": true' in response_lower or '"pass":true' in response_lower:
        passed = True
    elif "pass" in response_lower and "true" in response_lower:
        passed = True
    elif '"pass": false' in response_lower or '"pass":false' in response_lower:
        passed = False

    # Extract reason if possible
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', response)
    reason = reason_match.group(1) if reason_match else "Could not parse judge response"

    return {
        "pass": passed,
        "reason": reason,
        "score": None,
    }


def _parse_score(value) -> Optional[float]:
    """Parse score value, returning None for null/invalid."""
    if value is None:
        return None
    try:
        score = float(value)
        # Clamp to 0-10 range
        return max(0.0, min(10.0, score))
    except (TypeError, ValueError):
        return None
