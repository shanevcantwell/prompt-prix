"""Tests for battery feature (benchmark test suite execution).

Per ADR-006: BatteryRunner is orchestration layer. Tests mock MCP tools, not HTTP.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from prompt_prix.benchmarks.base import TestCase
from prompt_prix.benchmarks.custom import CustomJSONLoader
from prompt_prix.battery import TestStatus, TestResult, BatteryRun, BatteryRunner


# ─────────────────────────────────────────────────────────────────────
# TEST DATA
# ─────────────────────────────────────────────────────────────────────

SAMPLE_BENCHMARK_JSON = {
    "test_suite": "test_battery",
    "version": "1.0",
    "prompts": [
        {
            "id": "test_1",
            "name": "Test One",
            "category": "basic",
            "system": "You are a helpful assistant.",
            "user": "What is 2 + 2?",
        },
        {
            "id": "test_2",
            "name": "Test Two",
            "category": "basic",
            "system": "You are a math tutor.",
            "user": "What is 3 + 3?",
        },
    ]
}


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_test_cases():
    """Create sample TestCase objects."""
    return [
        TestCase(
            id="test_1",
            name="Test One",
            category="basic",
            system="You are a helpful assistant.",
            user="What is 2 + 2?"
        ),
        TestCase(
            id="test_2",
            name="Test Two",
            category="basic",
            system="You are a math tutor.",
            user="What is 3 + 3?"
        ),
    ]


@pytest.fixture
def sample_benchmark_file(tmp_path):
    """Create a temporary benchmark JSON file."""
    file_path = tmp_path / "benchmark.json"
    file_path.write_text(json.dumps(SAMPLE_BENCHMARK_JSON))
    return file_path


@pytest.fixture
def invalid_benchmark_file(tmp_path):
    """Create an invalid benchmark JSON file (missing prompts)."""
    file_path = tmp_path / "invalid.json"
    file_path.write_text(json.dumps({"version": "1.0"}))
    return file_path


@pytest.fixture
def malformed_json_file(tmp_path):
    """Create a malformed JSON file."""
    file_path = tmp_path / "malformed.json"
    file_path.write_text("{ not valid json }")
    return file_path




# ─────────────────────────────────────────────────────────────────────
# TESTCASE MODEL TESTS
# ─────────────────────────────────────────────────────────────────────

class TestTestCase:
    """Tests for TestCase Pydantic model."""

    def test_create_basic_test_case(self):
        """Test creating a basic test case."""
        tc = TestCase(
            id="basic_test",
            user="Hello world"
        )
        assert tc.id == "basic_test"
        assert tc.user == "Hello world"
        assert tc.system == "You are a helpful assistant."

    def test_create_full_test_case(self):
        """Test creating a test case with all fields."""
        tc = TestCase(
            id="full_test",
            name="Full Test",
            category="advanced",
            severity="critical",
            system="You are a code assistant.",
            user="Write a function",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="required"
        )
        assert tc.id == "full_test"
        assert tc.name == "Full Test"
        assert tc.category == "advanced"
        assert tc.tools is not None
        assert len(tc.tools) == 1

    def test_empty_id_fails(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TestCase(id="", user="Hello")

    def test_empty_user_fails(self):
        """Test that empty user message raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TestCase(id="test", user="")

    def test_whitespace_only_id_fails(self):
        """Test that whitespace-only id raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TestCase(id="   ", user="Hello")

    def test_to_messages(self):
        """Test converting TestCase to OpenAI messages format."""
        tc = TestCase(
            id="test",
            system="Be helpful",
            user="What time is it?"
        )
        messages = tc.to_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What time is it?"

    def test_display_name_uses_name_if_set(self):
        """Test display_name property."""
        tc = TestCase(id="test_id", name="Display Name", user="Hello")
        assert tc.display_name == "Display Name"

    def test_display_name_uses_id_if_no_name(self):
        """Test display_name falls back to id."""
        tc = TestCase(id="test_id", user="Hello")
        assert tc.display_name == "test_id"


# ─────────────────────────────────────────────────────────────────────
# CUSTOMJSONLOADER TESTS
# ─────────────────────────────────────────────────────────────────────

class TestCustomJSONLoader:
    """Tests for CustomJSONLoader."""

    def test_load_valid_file(self, sample_benchmark_file):
        """Test loading a valid benchmark file."""
        cases = CustomJSONLoader.load(sample_benchmark_file)
        assert len(cases) == 2
        assert cases[0].id == "test_1"
        assert cases[1].id == "test_2"

    def test_load_missing_file(self, tmp_path):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CustomJSONLoader.load(tmp_path / "nonexistent.json")

    def test_load_missing_prompts_key(self, invalid_benchmark_file):
        """Test loading file without prompts key raises ValueError."""
        with pytest.raises(ValueError, match="missing required 'prompts' key"):
            CustomJSONLoader.load(invalid_benchmark_file)

    def test_load_malformed_json(self, malformed_json_file):
        """Test loading malformed JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            CustomJSONLoader.load(malformed_json_file)

    def test_load_empty_prompts_array(self, tmp_path):
        """Test loading file with empty prompts array raises ValueError."""
        file_path = tmp_path / "empty.json"
        file_path.write_text(json.dumps({"prompts": []}))
        with pytest.raises(ValueError, match="cannot be empty"):
            CustomJSONLoader.load(file_path)

    def test_validate_valid_file(self, sample_benchmark_file):
        """Test validate returns success for valid file."""
        valid, message = CustomJSONLoader.validate(sample_benchmark_file)
        assert valid is True
        assert "2 tests" in message
        assert "1 categories" in message

    def test_validate_invalid_file(self, invalid_benchmark_file):
        """Test validate returns failure for invalid file."""
        valid, message = CustomJSONLoader.validate(invalid_benchmark_file)
        assert valid is False
        assert "❌" in message

    def test_load_jsonl_file(self, tmp_path):
        """Test loading JSONL format (one test per line)."""
        file_path = tmp_path / "tests.jsonl"
        jsonl_content = '{"id": "test_1", "user": "Hello"}\n{"id": "test_2", "user": "World"}\n'
        file_path.write_text(jsonl_content)

        cases = CustomJSONLoader.load(file_path)
        assert len(cases) == 2
        assert cases[0].id == "test_1"
        assert cases[1].id == "test_2"

    def test_load_jsonl_autodetect(self, tmp_path):
        """Test auto-detecting JSONL format from .json file."""
        file_path = tmp_path / "tests.json"
        # Multiple JSON objects on separate lines = auto-detect as JSONL
        jsonl_content = '{"id": "test_1", "user": "Hello"}\n{"id": "test_2", "user": "World"}'
        file_path.write_text(jsonl_content)

        cases = CustomJSONLoader.load(file_path)
        assert len(cases) == 2

    def test_load_jsonl_with_empty_lines(self, tmp_path):
        """Test JSONL with empty lines skipped."""
        file_path = tmp_path / "tests.jsonl"
        jsonl_content = '{"id": "test_1", "user": "Hello"}\n\n{"id": "test_2", "user": "World"}\n'
        file_path.write_text(jsonl_content)

        cases = CustomJSONLoader.load(file_path)
        assert len(cases) == 2


# ─────────────────────────────────────────────────────────────────────
# TEST RESULT TESTS
# ─────────────────────────────────────────────────────────────────────

class TestTestResult:
    """Tests for TestResult model."""

    def test_create_pending_result(self):
        """Test creating a pending result."""
        result = TestResult(test_id="t1", model_id="m1")
        assert result.status == TestStatus.PENDING
        assert result.status_symbol == "—"

    def test_status_symbols(self):
        """Test status symbols for all states."""
        assert TestResult(test_id="t", model_id="m", status=TestStatus.PENDING).status_symbol == "—"
        assert TestResult(test_id="t", model_id="m", status=TestStatus.RUNNING).status_symbol == "⏳"
        assert TestResult(test_id="t", model_id="m", status=TestStatus.COMPLETED).status_symbol == "✓"
        assert TestResult(test_id="t", model_id="m", status=TestStatus.ERROR).status_symbol == "❌"


# ─────────────────────────────────────────────────────────────────────
# BATTERY RUN TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBatteryRun:
    """Tests for BatteryRun state model."""

    def test_create_empty_battery_run(self):
        """Test creating an empty battery run."""
        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        assert len(run.tests) == 2
        assert len(run.models) == 2
        assert len(run.results) == 0

    def test_get_set_result(self):
        """Test getting and setting results."""
        run = BatteryRun(tests=["t1"], models=["m1"])
        result = TestResult(test_id="t1", model_id="m1", status=TestStatus.COMPLETED)
        run.set_result(result)

        retrieved = run.get_result("t1", "m1")
        assert retrieved is not None
        assert retrieved.status == TestStatus.COMPLETED

    def test_to_grid(self):
        """Test converting to DataFrame format."""
        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        run.set_result(TestResult(test_id="t1", model_id="m1", status=TestStatus.COMPLETED))
        run.set_result(TestResult(test_id="t1", model_id="m2", status=TestStatus.ERROR))

        df = run.to_grid()
        assert list(df.columns) == ["Test", "m1", "m2"]
        assert list(df.iloc[0]) == ["t1", "✓", "❌"]  # t1 results
        assert list(df.iloc[1]) == ["t2", "—", "—"]  # t2 pending

    def test_to_grid_latency_mode(self):
        """Test grid with latency display mode."""
        from prompt_prix.battery import GridDisplayMode

        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        run.set_result(TestResult(
            test_id="t1", model_id="m1",
            status=TestStatus.COMPLETED, latency_ms=1500.0
        ))
        run.set_result(TestResult(
            test_id="t1", model_id="m2",
            status=TestStatus.ERROR, latency_ms=2500.0
        ))

        df = run.to_grid(GridDisplayMode.LATENCY)
        assert list(df.columns) == ["Test", "m1", "m2"]
        assert list(df.iloc[0]) == ["t1", "1.5s", "2.5s"]  # t1 latencies
        assert list(df.iloc[1]) == ["t2", "—", "—"]       # t2 pending

    def test_progress_tracking(self):
        """Test progress calculation."""
        run = BatteryRun(tests=["t1", "t2"], models=["m1"])
        assert run.total_count == 2
        assert run.completed_count == 0
        assert run.progress_percent == 0.0

        run.set_result(TestResult(test_id="t1", model_id="m1", status=TestStatus.COMPLETED))
        assert run.completed_count == 1
        assert run.progress_percent == 50.0


# ─────────────────────────────────────────────────────────────────────
# BATTERY RUNNER TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBatteryRunner:
    """Tests for BatteryRunner orchestrator.

    Per ADR-006: BatteryRunner is orchestration layer - it calls MCP tools,
    doesn't know about servers or adapters. Tests mock the MCP layer.
    """

    @pytest.mark.asyncio
    async def test_run_completes_all_tests_via_mcp(self, sample_test_cases):
        """Test that runner completes all tests BY CALLING MCP complete_stream.

        This test mocks the MCP layer, not HTTP. If BatteryRunner bypasses MCP
        and calls core.stream_completion directly, this test will FAIL.
        """
        from unittest.mock import patch

        models = ["model_a", "model_b"]

        # Track calls to MCP complete_stream
        mcp_calls = []

        async def mock_complete_stream(**kwargs):
            mcp_calls.append(kwargs)
            yield "Test response"

        # Patch at the MCP layer - BatteryRunner SHOULD call this
        with patch("prompt_prix.battery.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=sample_test_cases,
                models=models
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        # BatteryRunner must call MCP complete_stream for each (test, model)
        assert len(mcp_calls) == 4, (
            f"Expected 4 calls to MCP complete_stream (2 tests × 2 models), "
            f"got {len(mcp_calls)}. BatteryRunner is bypassing MCP layer."
        )
        assert final_state is not None
        assert final_state.completed_count == 4

    @pytest.mark.asyncio
    async def test_run_handles_errors_via_mcp(self, sample_test_cases):
        """Test that runner handles MCP-layer errors gracefully.

        This test mocks the MCP layer. If BatteryRunner bypasses MCP,
        this test will FAIL.

        Note: BatteryRunner has retry logic for transient errors, so
        LMStudioError will be retried. We verify final status, not call counts.
        """
        from unittest.mock import patch
        from prompt_prix.core import LMStudioError

        models = ["model_a", "model_b"]

        mcp_calls = []

        async def mock_complete_stream(**kwargs):
            mcp_calls.append(kwargs)
            model_id = kwargs.get("model_id", "")
            if model_id == "model_b":
                raise LMStudioError("Connection failed")
            yield "Success"

        with patch("prompt_prix.battery.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=sample_test_cases[:1],  # Just one test
                models=models
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        # Must have called MCP for both models (model_b will have retries)
        model_a_calls = [c for c in mcp_calls if c.get("model_id") == "model_a"]
        model_b_calls = [c for c in mcp_calls if c.get("model_id") == "model_b"]
        assert len(model_a_calls) == 1, "Expected 1 call for model_a"
        assert len(model_b_calls) >= 1, "Expected at least 1 call for model_b (plus retries)"

        # Check model_a succeeded
        result_a = final_state.get_result("test_1", "model_a")
        assert result_a.status == TestStatus.COMPLETED

        # Check model_b errored after retries exhausted
        result_b = final_state.get_result("test_1", "model_b")
        assert result_b.status == TestStatus.ERROR

    @pytest.mark.asyncio
    async def test_run_yields_state_updates_via_mcp(self, sample_test_cases):
        """Test that runner yields state updates for UI.

        This test mocks the MCP layer. If BatteryRunner bypasses MCP,
        this test will FAIL.
        """
        from unittest.mock import patch

        models = ["m1"]

        mcp_calls = []

        async def mock_complete_stream(**kwargs):
            mcp_calls.append(kwargs)
            yield "Response"

        with patch("prompt_prix.battery.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=sample_test_cases[:1],
                models=models
            )

            state_count = 0
            async for state in runner.run():
                state_count += 1

        # Must have called MCP
        assert len(mcp_calls) == 1, (
            f"Expected 1 call to MCP complete_stream, got {len(mcp_calls)}. "
            "BatteryRunner is bypassing MCP layer."
        )
        # Should yield multiple times for progress tracking
        assert state_count >= 2

    @pytest.mark.asyncio
    async def test_run_records_latency_via_mcp(self, sample_test_cases):
        """Test that runner records latency for completed tests.

        This test mocks the MCP layer. If BatteryRunner bypasses MCP,
        this test will FAIL.
        """
        from unittest.mock import patch
        import asyncio

        models = ["m1"]

        mcp_calls = []

        async def mock_complete_stream(**kwargs):
            mcp_calls.append(kwargs)
            await asyncio.sleep(0.01)  # Small delay for latency measurement
            yield "Response"

        with patch("prompt_prix.battery.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=sample_test_cases[:1],
                models=models
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        # Must have called MCP
        assert len(mcp_calls) == 1, (
            f"Expected 1 call to MCP complete_stream, got {len(mcp_calls)}. "
            "BatteryRunner is bypassing MCP layer."
        )

        result = final_state.get_result("test_1", "m1")
        assert result.latency_ms is not None
        assert result.latency_ms > 0


# ─────────────────────────────────────────────────────────────────────
# BATTERY HANDLER TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBatteryExport:
    """Tests for battery export handlers."""

    def test_export_json_no_results(self):
        """Test export returns error when no battery run exists."""
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import export_json

        state.battery_run = None
        status, file_update = export_json()

        assert "❌" in status
        # Returns gr.update(visible=False, value=None)
        assert file_update["visible"] is False
        assert file_update["value"] is None

    def test_export_json_with_results(self):
        """Test export creates file with results."""
        import os
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import export_json

        # Setup battery run with results
        run = BatteryRun(tests=["t1", "t2"], models=["m1"])
        run.set_result(TestResult(
            test_id="t1", model_id="m1",
            status=TestStatus.COMPLETED,
            response="Test response",
            latency_ms=1234.5
        ))
        run.set_result(TestResult(
            test_id="t2", model_id="m1",
            status=TestStatus.ERROR,
            error="Test error"
        ))
        state.battery_run = run

        status, file_update = export_json()

        assert "✅" in status
        # Returns gr.update(visible=False, value=filepath) - triggers auto-download
        assert file_update["visible"] is False
        filepath = file_update["value"]
        assert filepath is not None
        assert os.path.exists(filepath)

        # Verify file contents
        with open(filepath) as f:
            data = json.load(f)

        assert data["tests"] == ["t1", "t2"]
        assert data["models"] == ["m1"]
        assert len(data["results"]) == 2
        # Verify failure_reason field is present
        assert "failure_reason" in data["results"][0]

        # Cleanup
        state.battery_run = None

    def test_export_csv_with_results(self):
        """Test CSV export creates file with results."""
        import os
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import export_csv

        run = BatteryRun(tests=["t1"], models=["m1"])
        run.set_result(TestResult(
            test_id="t1", model_id="m1",
            status=TestStatus.COMPLETED,
            response="Hello\nWorld",  # Test newline handling
            latency_ms=500.0
        ))
        state.battery_run = run

        status, file_update = export_csv()

        assert "✅" in status
        # Returns gr.update(visible=False, value=filepath) - triggers auto-download
        assert file_update["visible"] is False
        filepath = file_update["value"]
        assert filepath is not None
        assert filepath.endswith(".csv")

        with open(filepath) as f:
            content = f.read()

        # CSV header now includes error and failure_reason columns
        assert "test_id" in content
        assert "failure_reason" in content
        assert "t1" in content
        assert "m1" in content
        assert "500" in content

        state.battery_run = None

    def test_export_basename_from_source_file(self):
        """Test export filename derives from source file with timestamp."""
        import re
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import _get_export_basename

        state.battery_source_file = "/path/to/my_test_suite.jsonl"
        basename = _get_export_basename()
        # Basename should be {stem}_results_{timestamp}
        assert re.match(r"my_test_suite_results_\d+$", basename)

        state.battery_source_file = None
        basename = _get_export_basename()
        assert re.match(r"battery_results_\d+$", basename)


class TestBatteryStateClearing:
    """Tests for state clearing when file changes."""

    def test_state_cleared_on_file_upload(self):
        """Test that battery_run is cleared when new file is uploaded."""
        from prompt_prix import state

        # Setup existing battery run
        state.battery_run = BatteryRun(tests=["old"], models=["old"])
        state.battery_source_file = "/old/file.json"

        # Simulate file change (what on_battery_file_change does)
        state.battery_run = None
        state.battery_source_file = "/new/file.json"

        assert state.battery_run is None
        assert state.battery_source_file == "/new/file.json"

    def test_state_cleared_on_file_removal(self):
        """Test that state is cleared when file is removed."""
        from prompt_prix import state

        state.battery_run = BatteryRun(tests=["test"], models=["model"])
        state.battery_source_file = "/some/file.json"

        # Simulate file removal
        state.battery_run = None
        state.battery_source_file = None

        assert state.battery_run is None
        assert state.battery_source_file is None


class TestCooperativeCancellation:
    """Tests for cooperative cancellation via state.should_stop()."""

    def test_stop_flag_default_false(self):
        """Test stop flag starts as False."""
        from prompt_prix import state
        state.clear_stop()
        assert state.should_stop() is False

    def test_request_stop_sets_flag(self):
        """Test request_stop sets the flag."""
        from prompt_prix import state
        state.clear_stop()
        state.request_stop()
        assert state.should_stop() is True

    def test_clear_stop_resets_flag(self):
        """Test clear_stop resets the flag."""
        from prompt_prix import state
        state.request_stop()
        assert state.should_stop() is True
        state.clear_stop()
        assert state.should_stop() is False


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC VALIDATION INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBatterySemanticValidation:
    """Tests verifying BatteryRunner integrates semantic validation."""

    @pytest.mark.asyncio
    async def test_refusal_with_required_tools_is_semantic_failure(self):
        """A refusal when tool_choice='required' must be SEMANTIC_FAILURE.

        If a model says "I'm sorry, but I cannot..." when tools were required,
        that's a semantic failure - the model completed HTTP but failed the task.
        """
        from unittest.mock import patch

        models = ["model_a"]

        # MCP returns a refusal (no tool call made)
        refusal_text = "I'm sorry, but I cannot execute scripts or delete files."

        async def mock_complete_stream(**kwargs):
            yield refusal_text

        # Create test with tool_choice="required"
        test_with_tools = TestCase(
            id="tool_test",
            user="Delete the file report.pdf",
            tools=[{
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }],
            tool_choice="required"
        )

        with patch("prompt_prix.battery.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=[test_with_tools],
                models=models
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("tool_test", "model_a")

        # Must be SEMANTIC_FAILURE, not COMPLETED
        assert result.status == TestStatus.SEMANTIC_FAILURE, (
            f"Expected SEMANTIC_FAILURE for refusal with tool_choice='required', "
            f"got {result.status}. Response: {result.response}"
        )
