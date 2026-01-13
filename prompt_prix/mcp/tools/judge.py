"""
LLM-as-judge tool for MCP.

Exposes semantic evaluation capability for agentic systems.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from prompt_prix.mcp.server import mcp
from prompt_prix.llm_judge import judge_response, judge_with_context
from prompt_prix.mcp.tools._adapter import get_adapter, get_available_models

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    """Result of judge evaluation."""

    passed: bool = Field(description="Whether response met criteria")
    reason: str = Field(description="Judge's explanation")


@mcp.tool()
async def evaluate_response(
    response: str,
    criteria: str,
    judge_model: str,
    system_prompt: Optional[str] = None,
    user_message: Optional[str] = None,
) -> JudgeResult:
    """
    Evaluate if a response meets specified criteria using LLM judgment.

    Use this for semantic validation that can't be expressed as regex patterns.
    Examples: "Response should be helpful", "Answer must include a code example",
    "Response begins with 'shpadoinkle'".

    Args:
        response: The model response to evaluate
        criteria: Natural language pass criteria
        judge_model: Model ID to use as judge (e.g., "gemma-3-12b")
        system_prompt: Optional original system prompt for context
        user_message: Optional original user message for context

    Returns:
        JudgeResult with passed boolean and reason string
    """
    adapter = await get_adapter()

    logger.info(f"Evaluating response with judge_model={judge_model}")

    if system_prompt or user_message:
        passed, reason = await judge_with_context(
            response=response,
            criteria=criteria,
            system_prompt=system_prompt or "",
            user_message=user_message or "",
            adapter=adapter,
            judge_model=judge_model,
        )
    else:
        passed, reason = await judge_response(
            response=response,
            criteria=criteria,
            adapter=adapter,
            judge_model=judge_model,
        )

    return JudgeResult(passed=passed, reason=reason)


@mcp.tool()
async def list_judge_models() -> list[str]:
    """
    List available models that can be used as judges.

    Returns:
        List of model IDs from configured LM Studio servers
    """
    return await get_available_models()
