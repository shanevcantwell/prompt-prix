"""
Semantic validation for battery test responses.

Validates that model responses meet semantic expectations beyond
just HTTP success. Handles:
- Refusal detection (common patterns like "I can't", "I'm sorry, but")
- Tool call validation (when test expects tool calls)

Uses model-aware tool call parsing to support different model families
(OpenAI, LiquidAI/LFM, Hermes, etc.)
"""

import re
from typing import Optional, Tuple, TYPE_CHECKING

from prompt_prix.tool_parsers import has_tool_calls as _has_tool_calls

if TYPE_CHECKING:
    from prompt_prix.benchmarks.base import BenchmarkCase

# Common refusal patterns - models often use these phrases to decline tasks
REFUSAL_PATTERNS = [
    r"i(?:'m| am) sorry,? but",
    r"i can(?:'t|not)",
    r"i(?:'m| am) (?:not )?(?:able|unable)",
    r"(?:cannot|can't) (?:execute|run|perform|help with)",
    r"i(?:'m| am) not (?:designed|programmed|able)",
    r"(?:as an ai|as a language model)",
    r"i don't have (?:the ability|access)",
]


def detect_refusal(response: str) -> Optional[str]:
    """
    Detect if response is a refusal rather than task completion.

    Returns the matched refusal phrase if found, None otherwise.
    """
    response_lower = response.lower()
    for pattern in REFUSAL_PATTERNS:
        match = re.search(pattern, response_lower)
        if match:
            return match.group(0)
    return None


def has_tool_calls(response: str, model_id: str | None = None) -> bool:
    """
    Check if response contains tool calls.

    Uses model-aware parsing to detect tool calls in various formats:
    - OpenAI: **Tool Call:** markers
    - LiquidAI/LFM: <|tool_call_start|>...<|tool_call_end|>
    - Hermes: <tool_call>...</tool_call>

    Args:
        response: The model's response text
        model_id: Optional model identifier for format-specific parsing
    """
    return _has_tool_calls(response, model_id)


def validate_response_semantic(
    test: "BenchmarkCase",
    response: str,
    model_id: str | None = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate response against test case semantic expectations.

    Args:
        test: The test case with expected behavior
        response: The model's response text
        model_id: Optional model identifier for format-specific parsing

    Returns:
        (is_valid, failure_reason) tuple.
        - (True, None) if response passes validation
        - (False, "reason") if response fails validation
    """
    # Empty response = model didn't answer
    if not response or not response.strip():
        return False, "Empty response"

    # Check for refusals - applies to all test types
    refusal = detect_refusal(response)
    if refusal:
        return False, f"Model refused: '{refusal}'"

    # Tool-calling validation: required means we MUST see tool calls
    if test.tools and test.tool_choice == "required":
        if not has_tool_calls(response, model_id):
            return False, "Expected tool call but got text response"

    # Tool-calling validation: none means we should NOT see tool calls
    if test.tools and test.tool_choice == "none":
        if has_tool_calls(response, model_id):
            return False, "Tool call made when tool_choice='none'"

    return True, None
