"""Tests for semantic validation of model responses."""

import pytest
from prompt_prix.semantic_validator import (
    detect_refusal,
    has_tool_calls,
    validate_response_semantic,
    validate_verdict,
    REFUSAL_PATTERNS,
)
from prompt_prix.benchmarks.base import BenchmarkCase


class TestRefusalDetection:
    """Tests for refusal pattern detection."""

    def test_detects_sorry_but(self):
        response = "I'm sorry, but I can't execute scripts."
        assert detect_refusal(response) is not None

    def test_detects_cannot(self):
        response = "I cannot run code on your system."
        assert detect_refusal(response) is not None

    def test_detects_im_not_able(self):
        response = "I'm not able to perform that action."
        assert detect_refusal(response) is not None

    def test_detects_as_an_ai(self):
        response = "As an AI language model, I don't have access to files."
        assert detect_refusal(response) is not None

    def test_detects_cannot_execute(self):
        response = "I cannot execute scripts or run programs."
        assert detect_refusal(response) is not None

    def test_no_refusal_in_normal_response(self):
        response = "The weather in Tokyo is sunny, 25°C."
        assert detect_refusal(response) is None

    def test_no_refusal_in_tool_call_response(self):
        response = '**Tool Call:** `get_weather`\n```json\n{"city": "Tokyo"}\n```'
        assert detect_refusal(response) is None

    def test_no_refusal_in_code_response(self):
        response = "Here's the Python code:\n```python\nprint('hello')\n```"
        assert detect_refusal(response) is None

    def test_case_insensitive(self):
        response = "I'M SORRY, BUT I CAN'T DO THAT."
        assert detect_refusal(response) is not None


class TestToolCallDetection:
    """Tests for tool call detection in responses."""

    def test_detects_tool_call_marker(self):
        response = '**Tool Call:** `get_weather`\n```json\n{"city": "Tokyo"}\n```'
        assert has_tool_calls(response) is True

    def test_detects_multiple_tool_calls(self):
        response = (
            '**Tool Call:** `search`\n```json\n{"query": "foo"}\n```\n\n'
            '**Tool Call:** `fetch`\n```json\n{"url": "bar"}\n```'
        )
        assert has_tool_calls(response) is True

    def test_no_tool_call_in_text(self):
        response = "The answer is 4."
        assert has_tool_calls(response) is False

    def test_no_tool_call_in_refusal(self):
        response = "I'm sorry, but I can't help with that."
        assert has_tool_calls(response) is False


class TestSemanticValidation:
    """Tests for full semantic validation."""

    def test_simple_text_response_passes(self):
        """Text response to non-tool test should pass."""
        test = BenchmarkCase(id="test", user="What is 2+2?")
        response = "The answer is 4."
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is True
        assert reason is None

    def test_refusal_fails_any_test(self):
        """Refusal should fail regardless of test type."""
        test = BenchmarkCase(id="test", user="Calculate 2+2")
        response = "I'm sorry, but I can't help with that."
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is False
        assert "refused" in reason.lower()

    def test_tool_required_passes_with_call(self):
        """Tool test with tool_choice=required should pass when tool is called."""
        test = BenchmarkCase(
            id="test",
            user="What's the weather?",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            tool_choice="required"
        )
        response = '**Tool Call:** `get_weather`\n```json\n{"city": "Tokyo"}\n```'
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is True

    def test_tool_required_fails_without_call(self):
        """Tool test with tool_choice=required should fail when no tool called."""
        test = BenchmarkCase(
            id="test",
            user="What's the weather?",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            tool_choice="required"
        )
        # Use a response that doesn't match refusal patterns but still lacks tool call
        response = "The weather in Tokyo is currently sunny with temperatures around 25°C."
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is False
        assert "Expected tool call" in reason

    def test_tool_none_passes_without_call(self):
        """Tool test with tool_choice=none should pass when no tool called."""
        test = BenchmarkCase(
            id="test",
            user="What's 2+2? Don't use tools.",
            tools=[{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            tool_choice="none"
        )
        response = "The answer is 4."
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is True

    def test_tool_none_fails_with_call(self):
        """Tool test with tool_choice=none should fail when tool is called."""
        test = BenchmarkCase(
            id="test",
            user="What's 2+2? Don't use tools.",
            tools=[{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            tool_choice="none"
        )
        response = '**Tool Call:** `calculator`\n```json\n{"expr": "2+2"}\n```'
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is False
        assert "tool_choice='none'" in reason

    def test_tool_auto_passes_either_way(self):
        """Tool test with tool_choice=auto should pass with or without tool call."""
        test = BenchmarkCase(
            id="test",
            user="What's the weather?",
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            tool_choice="auto"
        )

        # With tool call
        response1 = '**Tool Call:** `get_weather`\n```json\n{}\n```'
        is_valid, _ = validate_response_semantic(test, response1)
        assert is_valid is True

        # Without tool call
        response2 = "I estimate it's sunny based on the season."
        is_valid, _ = validate_response_semantic(test, response2)
        assert is_valid is True

    def test_refusal_trumps_tool_presence(self):
        """Even with a tool call, refusal patterns should fail the test."""
        test = BenchmarkCase(
            id="test",
            user="Delete the file",
            tools=[{
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            tool_choice="required"
        )
        # Model says it can't but then makes a tool call anyway (unusual but possible)
        response = "I'm sorry, but I can't delete files. **Tool Call:** `delete_file`\n```json\n{}\n```"
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is False
        assert "refused" in reason.lower()


class TestVerdictValidation:
    """Tests for judge competence verdict validation."""

    def test_pass_verdict_matches_expected_pass(self):
        """Response with PASS verdict should pass when expected_verdict is PASS."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'PASS'"
        )
        response = '{"verdict": "PASS", "score": 1.0, "reasoning": "Correct"}'
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is True
        assert reason is None

    def test_fail_verdict_matches_expected_fail(self):
        """Response with FAIL verdict should pass when expected_verdict is FAIL."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'FAIL'"
        )
        response = '{"verdict": "FAIL", "score": 0.0, "reasoning": "Wrong function"}'
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is True
        assert reason is None

    def test_pass_verdict_fails_when_expected_fail(self):
        """Response with PASS verdict should fail when expected_verdict is FAIL."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'FAIL'"
        )
        response = '{"verdict": "PASS", "score": 0.95, "reasoning": "Looks good"}'
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is False
        assert "expected FAIL" in reason
        assert "got PASS" in reason

    def test_fail_verdict_fails_when_expected_pass(self):
        """Response with FAIL verdict should fail when expected_verdict is PASS."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'PASS'"
        )
        response = '{"verdict": "FAIL", "score": 0.0, "reasoning": "Wrong"}'
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is False
        assert "expected PASS" in reason
        assert "got FAIL" in reason

    def test_partial_verdict_supported(self):
        """PARTIAL verdict should be validated correctly."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'PARTIAL'"
        )
        response = '{"verdict": "PARTIAL", "score": 0.5, "reasoning": "Half right"}'
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is True

    def test_no_verdict_in_response_fails(self):
        """Response without verdict should fail validation."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'PASS'"
        )
        response = "I think this looks correct. Score: 1.0"
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is False
        assert "No verdict found" in reason

    def test_verdict_case_insensitive(self):
        """Verdict matching should be case insensitive."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'PASS'"
        )
        response = '{"verdict": "pass", "score": 1.0}'
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is True

    def test_no_pass_criteria_skips_validation(self):
        """Test without pass_criteria should skip verdict validation."""
        test = BenchmarkCase(id="test", user="What is 2+2?")
        response = '{"verdict": "FAIL"}'  # Would fail if checked
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is True
        assert reason is None

    def test_non_verdict_pass_criteria_skips_validation(self):
        """Pass criteria without verdict pattern should skip validation."""
        test = BenchmarkCase(
            id="test",
            user="Test something",
            pass_criteria="contains: hello"
        )
        response = "Some response without hello"
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is True  # Skipped, not validated

    def test_verdict_in_markdown_json_block(self):
        """Verdict should be found in markdown JSON blocks."""
        test = BenchmarkCase(
            id="test",
            user="Judge this",
            pass_criteria="The verdict in the JSON response must be 'FAIL'"
        )
        response = '''```json
{
  "verdict": "FAIL",
  "score": 0.0,
  "reasoning": "Wrong function called"
}
```'''
        is_valid, reason = validate_verdict(test, response)
        assert is_valid is True

    def test_full_semantic_validation_includes_verdict(self):
        """Full validation should include verdict check."""
        test = BenchmarkCase(
            id="test",
            user="Judge this output",
            pass_criteria="The verdict in the JSON response must be 'FAIL'"
        )
        # Model incorrectly says PASS when it should say FAIL
        response = '{"verdict": "PASS", "score": 0.95, "reasoning": "Looks fine"}'
        is_valid, reason = validate_response_semantic(test, response)
        assert is_valid is False
        assert "Verdict mismatch" in reason
