"""Tests for prompt_prix.export module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report function."""

    def test_generate_markdown_report_basic(self, sample_session_state):
        """Test generating a basic markdown report."""
        from prompt_prix.export import generate_markdown_report

        report = generate_markdown_report(sample_session_state)

        # Check header
        assert "# LLM Comparison Report" in report
        assert "**Generated:**" in report
        assert "**Models:**" in report
        assert "**Temperature:** 0.7" in report

        # Check model sections
        assert "## Model: llama-3.2-3b-instruct" in report
        assert "## Model: qwen2.5-7b-instruct" in report

        # Check conversation content
        assert "### User" in report
        assert "What is the capital of France?" in report
        assert "### Assistant" in report
        assert "The capital of France is Paris." in report

    def test_generate_markdown_report_with_error(self, sample_session_state):
        """Test markdown report includes model errors."""
        from prompt_prix.export import generate_markdown_report

        # Add error to one model
        sample_session_state.contexts["llama-3.2-3b-instruct"].error = "Connection timeout"

        report = generate_markdown_report(sample_session_state)

        assert "**Error:** Connection timeout" in report

    def test_generate_markdown_report_halted_session(self, halted_session_state):
        """Test markdown report shows halted status."""
        from prompt_prix.export import generate_markdown_report

        report = generate_markdown_report(halted_session_state)

        assert "**Session Halted:**" in report
        assert "Connection timeout" in report

    def test_generate_markdown_report_includes_system_prompt(self, sample_session_state):
        """Test markdown report includes system prompt."""
        from prompt_prix.export import generate_markdown_report

        report = generate_markdown_report(sample_session_state)

        assert "## System Prompt" in report
        assert "You are a helpful assistant." in report

    def test_generate_markdown_report_empty_contexts(self):
        """Test markdown report with no messages."""
        from prompt_prix.config import SessionState
        from prompt_prix.export import generate_markdown_report

        state = SessionState(models=["model-a"])
        report = generate_markdown_report(state)

        assert "# LLM Comparison Report" in report
        assert "## Model: model-a" not in report  # No context added


class TestGenerateJsonReport:
    """Tests for generate_json_report function."""

    def test_generate_json_report_structure(self, sample_session_state):
        """Test JSON report has correct structure."""
        from prompt_prix.export import generate_json_report

        report_str = generate_json_report(sample_session_state)
        report = json.loads(report_str)

        assert "generated_at" in report
        assert "configuration" in report
        assert "halted" in report
        assert "halt_reason" in report
        assert "conversations" in report

    def test_generate_json_report_configuration(self, sample_session_state):
        """Test JSON report configuration section."""
        from prompt_prix.export import generate_json_report

        report = json.loads(generate_json_report(sample_session_state))
        config = report["configuration"]

        assert config["models"] == ["llama-3.2-3b-instruct", "qwen2.5-7b-instruct"]
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 2048
        assert config["timeout_seconds"] == 300
        assert config["system_prompt"] == "You are a helpful assistant."

    def test_generate_json_report_conversations(self, sample_session_state):
        """Test JSON report conversations section."""
        from prompt_prix.export import generate_json_report

        report = json.loads(generate_json_report(sample_session_state))
        convos = report["conversations"]

        assert "llama-3.2-3b-instruct" in convos
        assert "qwen2.5-7b-instruct" in convos

        llama_convo = convos["llama-3.2-3b-instruct"]
        assert len(llama_convo["messages"]) == 2
        assert llama_convo["messages"][0]["role"] == "user"
        assert llama_convo["messages"][0]["content"] == "What is the capital of France?"
        assert llama_convo["error"] is None

    def test_generate_json_report_with_error(self, sample_session_state):
        """Test JSON report includes errors."""
        from prompt_prix.export import generate_json_report

        sample_session_state.contexts["llama-3.2-3b-instruct"].error = "Timeout"
        report = json.loads(generate_json_report(sample_session_state))

        assert report["conversations"]["llama-3.2-3b-instruct"]["error"] == "Timeout"

    def test_generate_json_report_halted(self, halted_session_state):
        """Test JSON report halted status."""
        from prompt_prix.export import generate_json_report

        report = json.loads(generate_json_report(halted_session_state))

        assert report["halted"] is True
        assert "Connection timeout" in report["halt_reason"]

    def test_generate_json_report_is_valid_json(self, sample_session_state):
        """Test JSON report is valid parseable JSON."""
        from prompt_prix.export import generate_json_report

        report_str = generate_json_report(sample_session_state)

        # Should not raise
        parsed = json.loads(report_str)
        assert isinstance(parsed, dict)


class TestSaveReport:
    """Tests for save_report function."""

    def test_save_report_writes_file(self, tmp_path, sample_session_state):
        """Test save_report writes content to file."""
        from prompt_prix.export import save_report

        filepath = tmp_path / "report.md"
        content = "# Test Report\n\nThis is a test."

        save_report(content, str(filepath))

        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == content

    def test_save_report_creates_utf8_file(self, tmp_path):
        """Test save_report creates UTF-8 encoded file."""
        from prompt_prix.export import save_report

        filepath = tmp_path / "report.md"
        content = "# Test Report\n\nUnicode: \u2713 \u2718 \u00e9\u00e8\u00e0"

        save_report(content, str(filepath))

        # Read back with explicit encoding
        result = filepath.read_text(encoding="utf-8")
        assert "\u2713" in result
        assert "\u00e9" in result

    def test_save_report_overwrites_existing(self, tmp_path):
        """Test save_report overwrites existing file."""
        from prompt_prix.export import save_report

        filepath = tmp_path / "report.md"
        filepath.write_text("Old content")

        save_report("New content", str(filepath))

        assert filepath.read_text() == "New content"


class TestReportTimestamp:
    """Tests for report timestamp handling."""

    def test_markdown_report_timestamp_format(self, sample_session_state):
        """Test markdown report has ISO format timestamp."""
        from prompt_prix.export import generate_markdown_report

        report = generate_markdown_report(sample_session_state)

        # Should contain ISO-like timestamp (YYYY-MM-DD)
        assert "**Generated:**" in report
        # The timestamp should be parseable

    def test_json_report_timestamp_format(self, sample_session_state):
        """Test JSON report has ISO format timestamp."""
        from prompt_prix.export import generate_json_report

        report = json.loads(generate_json_report(sample_session_state))

        # Should be ISO format
        timestamp = report["generated_at"]
        # Verify it's parseable as ISO format
        datetime.fromisoformat(timestamp)
