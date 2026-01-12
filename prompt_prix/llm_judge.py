"""
LLM-as-judge for semantic validation of battery test responses.

Uses a separate LLM to evaluate whether responses meet pass_criteria,
enabling natural language criteria that can't be expressed as regex patterns.
"""

import re
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_prix.adapters.lmstudio import LMStudioAdapter

JUDGE_PROMPT = """Evaluate if this response meets the criteria.

CRITERIA: {criteria}

RESPONSE:
{response}

Does the response satisfy the criteria? Answer only YES or NO, followed by a brief reason (one sentence)."""


async def judge_response(
    response: str,
    criteria: str,
    adapter: "LMStudioAdapter",
    judge_model: str,
    timeout_seconds: int = 30,
    max_tokens: int = 100
) -> Tuple[bool, str]:
    """
    Use LLM to evaluate if response meets criteria.

    Args:
        response: The model response to evaluate
        criteria: Natural language pass criteria (e.g., "Response starts with 'shpadoinkle'")
        adapter: LMStudioAdapter instance for making completions
        judge_model: Model ID to use as judge
        timeout_seconds: Timeout for judge call
        max_tokens: Max tokens for judge response

    Returns:
        (passed, reason) tuple where:
        - passed: True if response satisfies criteria
        - reason: Judge's explanation

    Note:
        Server routing is handled via adapter.set_server_hint() before calling.
    """
    prompt = JUDGE_PROMPT.format(criteria=criteria, response=response)

    messages = [
        {"role": "system", "content": "You are a precise evaluator. Answer only YES or NO followed by a brief reason."},
        {"role": "user", "content": prompt}
    ]

    result = ""
    async for chunk in adapter.stream_completion(
        model_id=judge_model,
        messages=messages,
        temperature=0.0,  # Deterministic for consistent judging
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    ):
        result += chunk

    result = result.strip()

    # Handle thinking models that wrap response in <think>...</think>
    # Extract the actual answer after the thinking block
    clean_result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL)
    clean_result = clean_result.strip()

    # Check if answer starts with YES (case-insensitive)
    passed = clean_result.upper().startswith("YES")

    # Return clean result for display (without thinking block)
    return passed, clean_result if clean_result else result


async def judge_with_context(
    response: str,
    criteria: str,
    system_prompt: str,
    user_message: str,
    adapter: "LMStudioAdapter",
    judge_model: str,
    timeout_seconds: int = 30,
    max_tokens: int = 150
) -> Tuple[bool, str]:
    """
    Use LLM to evaluate response with full test context.

    Provides the judge with the original system prompt and user message
    for more informed evaluation of complex criteria.

    Args:
        response: The model response to evaluate
        criteria: Natural language pass criteria
        system_prompt: Original system prompt given to the model
        user_message: Original user message
        adapter: LMStudioAdapter instance
        judge_model: Model ID to use as judge
        timeout_seconds: Timeout for judge call
        max_tokens: Max tokens for judge response

    Returns:
        (passed, reason) tuple

    Note:
        Server routing is handled via adapter.set_server_hint() before calling.
    """
    prompt = f"""Evaluate if this response meets the criteria, given the context.

CONTEXT:
System Prompt: {system_prompt}
User Message: {user_message}

CRITERIA: {criteria}

RESPONSE:
{response}

Does the response satisfy the criteria? Answer only YES or NO, followed by a brief reason (one sentence)."""

    messages = [
        {"role": "system", "content": "You are a precise evaluator. Answer only YES or NO followed by a brief reason."},
        {"role": "user", "content": prompt}
    ]

    result = ""
    async for chunk in adapter.stream_completion(
        model_id=judge_model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    ):
        result += chunk

    result = result.strip()

    # Handle thinking models that wrap response in <think>...</think>
    clean_result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL)
    clean_result = clean_result.strip()

    passed = clean_result.upper().startswith("YES")

    return passed, clean_result if clean_result else result
