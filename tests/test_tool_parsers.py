"""Tests for model-family-aware tool call parsing."""

import pytest
from prompt_prix.tool_parsers import (
    ToolCallParser,
    LiquidAIParser,
    HermesParser,
    ParsedToolCall,
    get_parser_for_model,
    has_tool_calls,
    parse_tool_calls,
    PARSER_REGISTRY,
)


class TestToolCallParser:
    """Tests for base OpenAI-style tool call parser."""

    def test_detects_openai_style_tool_call(self):
        response = '**Tool Call:** `get_weather`\n```json\n{"city": "Tokyo"}\n```'
        parser = ToolCallParser()
        result = parser.parse(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert "Tokyo" in result[0].arguments

    def test_detects_multiple_tool_calls(self):
        response = (
            '**Tool Call:** `search`\n```json\n{"query": "foo"}\n```\n\n'
            '**Tool Call:** `fetch`\n```json\n{"url": "bar"}\n```'
        )
        parser = ToolCallParser()
        result = parser.parse(response)
        assert result is not None
        assert len(result) == 2
        assert result[0].name == "search"
        assert result[1].name == "fetch"

    def test_returns_none_for_no_tool_calls(self):
        response = "The weather is sunny today."
        parser = ToolCallParser()
        result = parser.parse(response)
        assert result is None

    def test_handles_json_without_language_tag(self):
        response = '**Tool Call:** `calculate`\n```\n{"expression": "2+2"}\n```'
        parser = ToolCallParser()
        result = parser.parse(response)
        assert result is not None
        assert result[0].name == "calculate"


class TestLiquidAIParser:
    """Tests for LiquidAI/LFM tool call parser."""

    def test_detects_lfm_style_tool_call(self):
        response = '<|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>'
        parser = LiquidAIParser()
        result = parser.parse(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert 'city="Tokyo"' in result[0].arguments

    def test_detects_multiple_lfm_tool_calls(self):
        response = (
            '<|tool_call_start|>[search(query="foo")]<|tool_call_end|>'
            '<|tool_call_start|>[fetch(url="bar")]<|tool_call_end|>'
        )
        parser = LiquidAIParser()
        result = parser.parse(response)
        assert result is not None
        assert len(result) == 2
        assert result[0].name == "search"
        assert result[1].name == "fetch"

    def test_falls_back_to_openai_style(self):
        response = '**Tool Call:** `get_weather`\n```json\n{"city": "Tokyo"}\n```'
        parser = LiquidAIParser()
        result = parser.parse(response)
        assert result is not None
        assert result[0].name == "get_weather"

    def test_returns_none_for_no_tool_calls(self):
        response = "I cannot help with that."
        parser = LiquidAIParser()
        result = parser.parse(response)
        assert result is None


class TestHermesParser:
    """Tests for Hermes-style tool call parser."""

    def test_detects_tool_call_tag(self):
        response = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
        parser = HermesParser()
        result = parser.parse(response)
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "get_weather"

    def test_detects_function_call_tag(self):
        response = '<function_call>{"name": "search", "arguments": {"query": "test"}}</function_call>'
        parser = HermesParser()
        result = parser.parse(response)
        assert result is not None
        assert result[0].name == "search"

    def test_falls_back_to_openai_style(self):
        response = '**Tool Call:** `calculate`\n```json\n{}\n```'
        parser = HermesParser()
        result = parser.parse(response)
        assert result is not None
        assert result[0].name == "calculate"


class TestGetParserForModel:
    """Tests for model-to-parser mapping."""

    def test_returns_liquidai_parser_for_lfm(self):
        parser = get_parser_for_model("lfm2-1.2b-tool")
        assert isinstance(parser, LiquidAIParser)

    def test_returns_liquidai_parser_for_lfm_uppercase(self):
        parser = get_parser_for_model("LFM2.5-1.2b-instruct")
        assert isinstance(parser, LiquidAIParser)

    def test_returns_liquidai_parser_for_liquid(self):
        parser = get_parser_for_model("some-liquid-model")
        assert isinstance(parser, LiquidAIParser)

    def test_returns_hermes_parser_for_hermes(self):
        parser = get_parser_for_model("llama-3-hermes-8b")
        assert isinstance(parser, HermesParser)

    def test_returns_hermes_parser_for_nous_hermes(self):
        parser = get_parser_for_model("nous-hermes-2-pro")
        assert isinstance(parser, HermesParser)

    def test_returns_base_parser_for_unknown(self):
        parser = get_parser_for_model("gpt-4")
        assert type(parser) is ToolCallParser

    def test_returns_base_parser_for_none(self):
        parser = get_parser_for_model(None)
        assert type(parser) is ToolCallParser


class TestHasToolCalls:
    """Tests for the has_tool_calls convenience function."""

    def test_detects_openai_style_without_model(self):
        response = '**Tool Call:** `test`\n```json\n{}\n```'
        assert has_tool_calls(response) is True

    def test_detects_lfm_style_with_model(self):
        response = '<|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>'
        assert has_tool_calls(response, model_id="lfm2-1.2b-tool") is True

    def test_lfm_style_not_detected_without_model(self):
        response = '<|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>'
        # Base parser won't recognize LFM tokens
        assert has_tool_calls(response, model_id=None) is False

    def test_no_tool_calls_returns_false(self):
        response = "Just a text response."
        assert has_tool_calls(response) is False
        assert has_tool_calls(response, model_id="lfm2-1.2b-tool") is False


class TestParseToolCalls:
    """Tests for the parse_tool_calls convenience function."""

    def test_parses_and_returns_structured_data(self):
        response = '**Tool Call:** `get_weather`\n```json\n{"city": "Tokyo"}\n```'
        result = parse_tool_calls(response)
        assert result is not None
        assert isinstance(result[0], ParsedToolCall)
        assert result[0].name == "get_weather"

    def test_parses_lfm_with_model_id(self):
        response = '<|tool_call_start|>[delete_file(path="test.txt")]<|tool_call_end|>'
        result = parse_tool_calls(response, model_id="lfm2-1.2b-tool")
        assert result is not None
        assert result[0].name == "delete_file"
        assert 'path="test.txt"' in result[0].arguments
