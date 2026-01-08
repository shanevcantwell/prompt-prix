"""
Tool call parsers for different model families.

Different model families output tool calls in different formats:
- OpenAI API: Returns in `tool_calls` field, formatted as `**Tool Call:**`
- LiquidAI/LFM: `<|tool_call_start|>[function(args)]<|tool_call_end|>`
- Llama/Hermes: `<tool_call>...</tool_call>` or similar
- Qwen: Various formats

This module provides a strategy pattern for parsing tool calls from
different model families.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedToolCall:
    """Represents a parsed tool call."""
    name: str
    arguments: str  # Raw argument string (may be JSON or other format)


class ToolCallParser:
    """
    Base parser for OpenAI-style tool calls.

    Detects tool calls formatted as:
    **Tool Call:** `function_name`
    ```json
    {args}
    ```
    """

    def parse(self, response: str) -> list[ParsedToolCall] | None:
        """
        Parse tool calls from response.

        Returns list of ParsedToolCall if found, None if no tool calls detected.
        """
        if "**Tool Call:**" not in response:
            return None

        # Pattern: **Tool Call:** `name` followed by json block
        pattern = r"\*\*Tool Call:\*\*\s*`(\w+)`\s*```(?:json)?\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            return None

        return [ParsedToolCall(name=m[0], arguments=m[1].strip()) for m in matches]


class LiquidAIParser(ToolCallParser):
    """
    Parser for LiquidAI/LFM models.

    Detects tool calls formatted as:
    <|tool_call_start|>[function_name(arg1="value", arg2=123)]<|tool_call_end|>
    """

    # Pattern for LFM tool call tokens
    PATTERN = r"<\|tool_call_start\|>\[(\w+)\((.*?)\)\]<\|tool_call_end\|>"

    def parse(self, response: str) -> list[ParsedToolCall] | None:
        # First check for LFM-style tokens
        matches = re.findall(self.PATTERN, response, re.DOTALL)

        if matches:
            return [ParsedToolCall(name=m[0], arguments=m[1]) for m in matches]

        # Fall back to base parser (OpenAI style)
        return super().parse(response)


class HermesParser(ToolCallParser):
    """
    Parser for Llama/Hermes models.

    Detects tool calls formatted as:
    <tool_call>{"name": "function", "arguments": {...}}</tool_call>
    or
    <function_call>...</function_call>
    """

    TOOL_CALL_PATTERN = r"<tool_call>\s*(.*?)\s*</tool_call>"
    FUNCTION_CALL_PATTERN = r"<function_call>\s*(.*?)\s*</function_call>"

    def parse(self, response: str) -> list[ParsedToolCall] | None:
        import json

        results = []

        # Check for <tool_call> tags
        for pattern in [self.TOOL_CALL_PATTERN, self.FUNCTION_CALL_PATTERN]:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    name = data.get("name") or data.get("function")
                    args = data.get("arguments") or data.get("parameters") or {}
                    if name:
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                        results.append(ParsedToolCall(name=name, arguments=args_str))
                except json.JSONDecodeError:
                    # Not valid JSON, try to extract name at least
                    name_match = re.search(r'"name"\s*:\s*"(\w+)"', match)
                    if name_match:
                        results.append(ParsedToolCall(name=name_match.group(1), arguments=match))

        if results:
            return results

        # Fall back to base parser
        return super().parse(response)


# Registry maps model ID patterns to parser classes
# Order matters - first match wins
PARSER_REGISTRY: list[tuple[str, type[ToolCallParser]]] = [
    (r"lfm.*", LiquidAIParser),
    (r".*liquid.*", LiquidAIParser),
    (r".*hermes.*", HermesParser),
    (r".*nous.*hermes.*", HermesParser),
    # Default (OpenAI style) is used if no pattern matches
]


def get_parser_for_model(model_id: str | None) -> ToolCallParser:
    """
    Get the appropriate parser for a model ID.

    Args:
        model_id: The model identifier (e.g., "lfm2-1.2b-tool")

    Returns:
        Parser instance appropriate for the model family
    """
    if model_id:
        model_lower = model_id.lower()
        for pattern, parser_cls in PARSER_REGISTRY:
            if re.search(pattern, model_lower):
                return parser_cls()

    return ToolCallParser()


def has_tool_calls(response: str, model_id: str | None = None) -> bool:
    """
    Check if response contains tool calls.

    Uses model-aware parsing to detect tool calls in various formats.

    Args:
        response: The model's response text
        model_id: Optional model identifier for format-specific parsing

    Returns:
        True if tool calls detected, False otherwise
    """
    parser = get_parser_for_model(model_id)
    result = parser.parse(response)
    return result is not None and len(result) > 0


def parse_tool_calls(response: str, model_id: str | None = None) -> list[ParsedToolCall] | None:
    """
    Parse tool calls from response.

    Args:
        response: The model's response text
        model_id: Optional model identifier for format-specific parsing

    Returns:
        List of ParsedToolCall if found, None otherwise
    """
    parser = get_parser_for_model(model_id)
    return parser.parse(response)
