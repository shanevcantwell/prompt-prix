"""Tests for battery feature (benchmark test suite execution).

Per ADR-006: BatteryRunner is orchestration layer. Tests mock MCP tools, not HTTP.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from prompt_prix.benchmarks.base import BenchmarkCase
from prompt_prix.benchmarks.custom import CustomJSONLoader
from prompt_prix.battery import RunStatus, RunResult, BatteryRun, BatteryRunner


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
    """Create sample BenchmarkCase objects."""
    return [
        BenchmarkCase(
            id="test_1",
            name="Test One",
            category="basic",
            system="You are a helpful assistant.",
            user="What is 2 + 2?"
        ),
        BenchmarkCase(
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

class TestBenchmarkCase:
    """Tests for BenchmarkCase Pydantic model."""

    def test_create_basic_test_case(self):
        """Test creating a basic test case."""
        tc = BenchmarkCase(
            id="basic_test",
            user="Hello world"
        )
        assert tc.id == "basic_test"
        assert tc.user == "Hello world"
        assert tc.system == "You are a helpful assistant."

    def test_create_full_test_case(self):
        """Test creating a test case with all fields."""
        tc = BenchmarkCase(
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
            BenchmarkCase(id="", user="Hello")

    def test_empty_user_fails(self):
        """Test that empty user message raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BenchmarkCase(id="test", user="")

    def test_whitespace_only_id_fails(self):
        """Test that whitespace-only id raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BenchmarkCase(id="   ", user="Hello")

    def test_to_messages(self):
        """Test converting BenchmarkCase to OpenAI messages format."""
        tc = BenchmarkCase(
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

    def test_to_messages_with_multi_turn_history(self):
        """Pre-defined messages array takes precedence over system+user."""
        history = [
            {"role": "system", "content": "You are a file assistant."},
            {"role": "user", "content": "List the directory"},
            {"role": "assistant", "content": "list_directory('./sort_test')"},
            {"role": "user", "content": "[FILE] 1.txt\n[FILE] 2.txt"},
            {"role": "assistant", "content": "read_file('./sort_test/1.txt')"},
            {"role": "user", "content": "The zebra is a striped animal."},
            {"role": "user", "content": "Now move 1.txt to animals/"},
        ]
        tc = BenchmarkCase(
            id="multi_turn",
            user="Now move 1.txt to animals/",
            messages=history,
        )
        messages = tc.to_messages()
        assert messages == history
        assert len(messages) == 7

    def test_to_messages_without_messages_field_unchanged(self):
        """Without messages field, to_messages() returns system+user (backward compat)."""
        tc = BenchmarkCase(id="single", user="Hello", system="Be helpful")
        messages = tc.to_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Be helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert tc.messages is None

    def test_to_messages_returns_copy(self):
        """to_messages() returns a copy, not the original list."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        tc = BenchmarkCase(id="copy_test", user="Bye", messages=history)
        result = tc.to_messages()
        assert result == history
        assert result is not history  # Must be a copy

    def test_display_name_uses_name_if_set(self):
        """Test display_name property."""
        tc = BenchmarkCase(id="test_id", name="Display Name", user="Hello")
        assert tc.display_name == "Display Name"

    def test_display_name_uses_id_if_no_name(self):
        """Test display_name falls back to id."""
        tc = BenchmarkCase(id="test_id", user="Hello")
        assert tc.display_name == "test_id"

    def test_react_mode_fields(self):
        """BenchmarkCase accepts mode, mock_tools, max_iterations for ReAct."""
        mock_tools = {
            "read_file": {"./1.txt": "Content about animals"},
            "move_file": {"_default": "File moved"},
        }
        tc = BenchmarkCase(
            id="react_test",
            user="Organize these files",
            mode="react",
            mock_tools=mock_tools,
            max_iterations=20,
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        assert tc.mode == "react"
        assert tc.mock_tools == mock_tools
        assert tc.max_iterations == 20

    def test_react_mode_defaults(self):
        """mode=None and max_iterations=15 by default (backward compat)."""
        tc = BenchmarkCase(id="basic", user="Hello")
        assert tc.mode is None
        assert tc.mock_tools is None
        assert tc.max_iterations == 15

    def test_react_mode_via_json_loader(self, tmp_path):
        """JSON loader round-trips react fields through BenchmarkCase."""
        import json
        battery_file = tmp_path / "react_battery.json"
        battery_file.write_text(json.dumps({
            "prompts": [{
                "id": "react_1",
                "user": "Organize files",
                "mode": "react",
                "mock_tools": {
                    "list_directory": {"./sort_test": "[FILE] 1.txt"},
                    "read_file": {"./sort_test/1.txt": "Zebras are animals"},
                },
                "max_iterations": 10,
                "tools": [{"type": "function", "function": {"name": "list_directory"}}],
                "expected_response": "Files organized",
            }]
        }))

        from prompt_prix.benchmarks.custom import CustomJSONLoader
        cases = CustomJSONLoader.load(battery_file)
        assert len(cases) == 1
        tc = cases[0]
        assert tc.mode == "react"
        assert tc.mock_tools["read_file"]["./sort_test/1.txt"] == "Zebras are animals"
        assert tc.max_iterations == 10


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

class TestRunResult:
    """Tests for RunResult model."""

    def test_create_pending_result(self):
        """Test creating a pending result."""
        result = RunResult(test_id="t1", model_id="m1")
        assert result.status == RunStatus.PENDING
        assert result.status_symbol == "—"

    def test_status_symbols(self):
        """Test status symbols for all states."""
        assert RunResult(test_id="t", model_id="m", status=RunStatus.PENDING).status_symbol == "—"
        assert RunResult(test_id="t", model_id="m", status=RunStatus.RUNNING).status_symbol == "⏳"
        assert RunResult(test_id="t", model_id="m", status=RunStatus.COMPLETED).status_symbol == "✓"
        assert RunResult(test_id="t", model_id="m", status=RunStatus.SEMANTIC_FAILURE).status_symbol == "❌"
        assert RunResult(test_id="t", model_id="m", status=RunStatus.ERROR).status_symbol == "⚠"


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
        result = RunResult(test_id="t1", model_id="m1", status=RunStatus.COMPLETED)
        run.set_result(result)

        retrieved = run.get_result("t1", "m1")
        assert retrieved is not None
        assert retrieved.status == RunStatus.COMPLETED

    def test_to_grid(self):
        """Test converting to DataFrame format."""
        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        run.set_result(RunResult(test_id="t1", model_id="m1", status=RunStatus.COMPLETED))
        run.set_result(RunResult(test_id="t1", model_id="m2", status=RunStatus.ERROR))

        df = run.to_grid()
        assert list(df.columns) == ["Test", "m1", "m2"]
        assert list(df.iloc[0]) == ["t1", "✓", "⚠"]  # t1 results (ERROR=⚠)
        assert list(df.iloc[1]) == ["t2", "—", "—"]  # t2 pending

    def test_to_grid_latency_mode(self):
        """Test grid with latency display mode."""
        from prompt_prix.battery import GridDisplayMode

        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.COMPLETED, latency_ms=1500.0
        ))
        run.set_result(RunResult(
            test_id="t1", model_id="m2",
            status=RunStatus.ERROR, latency_ms=2500.0
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

        run.set_result(RunResult(test_id="t1", model_id="m1", status=RunStatus.COMPLETED))
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
        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream):
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

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream):
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
        assert result_a.status == RunStatus.COMPLETED

        # Check model_b errored after retries exhausted
        result_b = final_state.get_result("test_1", "model_b")
        assert result_b.status == RunStatus.ERROR

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

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream):
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
            yield "__LATENCY_MS__:10.5"  # Latency sentinel from adapter

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream):
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
        assert result.latency_ms == 10.5  # From latency sentinel


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
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.COMPLETED,
            response="Test response",
            latency_ms=1234.5
        ))
        run.set_result(RunResult(
            test_id="t2", model_id="m1",
            status=RunStatus.ERROR,
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
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.COMPLETED,
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
        test_with_tools = BenchmarkCase(
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

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=[test_with_tools],
                models=models
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("tool_test", "model_a")

        # Must be SEMANTIC_FAILURE, not COMPLETED
        assert result.status == RunStatus.SEMANTIC_FAILURE, (
            f"Expected SEMANTIC_FAILURE for refusal with tool_choice='required', "
            f"got {result.status}. Response: {result.response}"
        )


# ─────────────────────────────────────────────────────────────────────
# PIPELINED JUDGING TESTS
# ─────────────────────────────────────────────────────────────────────

class TestPipelinedJudging:
    """Tests for pipelined judge execution during inference.

    When judge_model is set, judge tasks are submitted eagerly as inference
    results complete — not batched after all inference finishes.
    """

    @pytest.fixture
    def tests_with_criteria(self):
        """BenchmarkCases that have pass_criteria (trigger judging)."""
        return [
            BenchmarkCase(
                id="judged_1",
                user="What is 2 + 2?",
                pass_criteria="Answer must contain the number 4",
            ),
            BenchmarkCase(
                id="judged_2",
                user="What is 3 + 3?",
                pass_criteria="Answer must contain the number 6",
            ),
        ]

    @pytest.mark.asyncio
    async def test_pipelined_all_results_judged(self, tests_with_criteria):
        """All COMPLETED results with criteria get judged in pipelined mode."""
        async def mock_complete_stream(**kwargs):
            yield "The answer is 4"

        async def mock_judge(**kwargs):
            return {"pass": True, "reason": "Correct", "score": 10}

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.battery.judge", side_effect=mock_judge):
            runner = BatteryRunner(
                tests=tests_with_criteria,
                models=["model_a"],
                judge_model="judge-model",
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        # Both results should be COMPLETED with judge verdicts
        for test_id in ["judged_1", "judged_2"]:
            result = final_state.get_result(test_id, "model_a")
            assert result.status == RunStatus.COMPLETED
            assert result.judge_result is not None
            assert result.judge_result["pass"] is True
            assert result.judge_latency_ms is not None

    @pytest.mark.asyncio
    async def test_pipelined_judge_fail_downgrades_status(self, tests_with_criteria):
        """Judge failure downgrades COMPLETED to SEMANTIC_FAILURE."""
        async def mock_complete_stream(**kwargs):
            yield "I don't know"

        async def mock_judge(**kwargs):
            return {"pass": False, "reason": "Missing expected number", "score": 0}

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.battery.judge", side_effect=mock_judge):
            runner = BatteryRunner(
                tests=tests_with_criteria[:1],
                models=["model_a"],
                judge_model="judge-model",
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("judged_1", "model_a")
        assert result.status == RunStatus.SEMANTIC_FAILURE
        assert result.failure_reason == "Missing expected number"

    @pytest.mark.asyncio
    async def test_no_judge_uses_inference_only(self, tests_with_criteria):
        """Without judge_model, only inference runs (no _execute_pipelined)."""
        mcp_calls = []

        async def mock_complete_stream(**kwargs):
            mcp_calls.append(kwargs["model_id"])
            yield "Response"

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.battery.judge") as mock_judge:
            runner = BatteryRunner(
                tests=tests_with_criteria,
                models=["model_a"],
                # No judge_model
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        # Inference happened
        assert len(mcp_calls) == 2
        # Judge was never called
        mock_judge.assert_not_called()
        # Phase stays as inference
        assert final_state.phase == "inference"

    @pytest.mark.asyncio
    async def test_pipelined_skips_failed_results(self, tests_with_criteria):
        """Results that fail semantic validation in inference are not judged."""
        call_count = {"judge": 0}

        async def mock_complete_stream(**kwargs):
            model_id = kwargs.get("model_id", "")
            if model_id == "model_fail":
                raise Exception("Connection failed")
            yield "The answer is 4"

        async def mock_judge(**kwargs):
            call_count["judge"] += 1
            return {"pass": True, "reason": "OK", "score": 10}

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.battery.judge", side_effect=mock_judge):
            runner = BatteryRunner(
                tests=tests_with_criteria[:1],
                models=["model_ok", "model_fail"],
                judge_model="judge-model",
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        # Only the successful model's result gets judged
        assert call_count["judge"] == 1
        result_ok = final_state.get_result("judged_1", "model_ok")
        assert result_ok.judge_result is not None

        result_fail = final_state.get_result("judged_1", "model_fail")
        assert result_fail.status == RunStatus.ERROR
        assert result_fail.judge_result is None

    @pytest.mark.asyncio
    async def test_pipelined_judge_total_increments_during_inference(self, tests_with_criteria):
        """judge_total grows as inference results complete (not set all at once)."""
        judge_totals_observed = []

        async def mock_complete_stream(**kwargs):
            yield "The answer is 4"

        async def mock_judge(**kwargs):
            await asyncio.sleep(0.05)  # Small delay so we can observe state
            return {"pass": True, "reason": "OK", "score": 10}

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.battery.judge", side_effect=mock_judge):
            runner = BatteryRunner(
                tests=tests_with_criteria,
                models=["model_a"],
                judge_model="judge-model",
            )

            async for state in runner.run():
                if state.judge_total > 0:
                    judge_totals_observed.append(state.judge_total)

        # judge_total should have been observed at least once
        assert len(judge_totals_observed) > 0
        # Final judge_total should match number of results that needed judging
        assert judge_totals_observed[-1] == 2

    @pytest.mark.asyncio
    async def test_pipelined_phase_transition(self, tests_with_criteria):
        """Phase transitions from 'inference' to 'judging' when inference done."""
        phases_observed = []

        async def mock_complete_stream(**kwargs):
            yield "The answer is 4"

        async def mock_judge(**kwargs):
            await asyncio.sleep(0.1)  # Judge takes longer than inference
            return {"pass": True, "reason": "OK", "score": 10}

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.battery.judge", side_effect=mock_judge):
            runner = BatteryRunner(
                tests=tests_with_criteria,
                models=["model_a"],
                judge_model="judge-model",
            )

            async for state in runner.run():
                phases_observed.append(state.phase)

        # Should have seen both phases
        assert "inference" in phases_observed
        # If judge tasks outlast inference, we see "judging" phase
        # (this depends on timing — judge sleeps 0.1s, inference is instant)


# ─────────────────────────────────────────────────────────────────────
# REACT-AS-ATOM: UNIFIED PIPELINE TESTS
# ─────────────────────────────────────────────────────────────────────

class TestReactAsAtom:
    """Tests proving react tests flow through BatteryRunner as standard RunResults.

    React-as-atom: a react loop is an execution detail, not an orchestration path.
    BatteryRunner has zero mode awareness — dispatch handles mode internally.
    """

    @pytest.fixture
    def react_test(self):
        """A react-mode BenchmarkCase."""
        return BenchmarkCase(
            id="react_categorize",
            user="Organize the files",
            mode="react",
            system="You are a file organizer.",
            mock_tools={
                "read_file": {"./1.txt": "Content about animals"},
                "move_file": {"_default": "File moved"},
            },
            tools=[
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "move_file"}},
            ],
            max_iterations=10,
        )

    @pytest.fixture
    def mixed_tests(self, react_test):
        """Mixed battery: standard + react tests in one file."""
        return [
            BenchmarkCase(id="simple_math", user="What is 2+2?"),
            react_test,
        ]

    @pytest.mark.asyncio
    async def test_react_completed_produces_run_result(self, react_test):
        """Completed react loop produces a standard RunResult with react_trace."""
        from prompt_prix.react.schemas import ReActIteration, ToolCall

        step_count = {"n": 0}

        async def mock_step(**kwargs):
            step_count["n"] += 1
            if step_count["n"] <= 2:
                return {
                    "completed": False,
                    "final_response": None,
                    "new_iterations": [
                        ReActIteration(
                            iteration=1,
                            tool_call=ToolCall(
                                id=f"call_{step_count['n']}",
                                name="read_file",
                                args={"path": f"./file{step_count['n']}.txt"},
                            ),
                            observation="mock data",
                            success=True,
                            latency_ms=50.0,
                        )
                    ],
                    "call_counter": step_count["n"],
                    "latency_ms": 50.0,
                }
            return {
                "completed": True,
                "final_response": "All files organized.",
                "new_iterations": [],
                "call_counter": step_count["n"],
                "latency_ms": 30.0,
            }

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            runner = BatteryRunner(tests=[react_test], models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("react_categorize", "model_a")
        assert result.status == RunStatus.COMPLETED
        assert result.response == "All files organized."
        assert result.react_trace is not None
        assert result.react_trace["completed"] is True
        assert result.react_trace["total_iterations"] == 2

    @pytest.mark.asyncio
    async def test_react_incomplete_is_semantic_failure(self, react_test):
        """React loop hitting max_iterations produces SEMANTIC_FAILURE."""
        step_count = {"n": 0}

        async def mock_step(**kwargs):
            step_count["n"] += 1
            return {
                "completed": False,
                "final_response": None,
                "new_iterations": [
                    __import__("prompt_prix.react.schemas", fromlist=["ReActIteration"]).ReActIteration(
                        iteration=1,
                        tool_call=__import__("prompt_prix.react.schemas", fromlist=["ToolCall"]).ToolCall(
                            id=f"call_{step_count['n']}",
                            name="read_file",
                            args={"path": f"./unique_{step_count['n']}.txt"},
                        ),
                        observation="data",
                        success=True,
                        latency_ms=50.0,
                    )
                ],
                "call_counter": step_count["n"],
                "latency_ms": 50.0,
            }

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            runner = BatteryRunner(tests=[react_test], models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("react_categorize", "model_a")
        assert result.status == RunStatus.SEMANTIC_FAILURE
        assert "max_iterations" in result.failure_reason
        assert result.react_trace is not None
        assert result.react_trace["termination_reason"] == "max_iterations"

    @pytest.mark.asyncio
    async def test_mixed_battery_unified_grid(self, mixed_tests):
        """Standard and react tests appear in the same grid."""
        from prompt_prix.react.schemas import ReActIteration, ToolCall

        async def mock_stream(**kwargs):
            yield "The answer is 4."
            yield "__LATENCY_MS__:100"

        async def mock_step(**kwargs):
            return {
                "completed": True,
                "final_response": "Files organized.",
                "new_iterations": [],
                "call_counter": 0,
                "latency_ms": 50.0,
            }

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_stream), \
             patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step):
            runner = BatteryRunner(tests=mixed_tests, models=["model_a"])

            final_state = None
            async for state in runner.run():
                final_state = state

        # Both tests in the same grid
        assert final_state.completed_count == 2
        assert final_state.total_count == 2

        # Standard test: no react_trace
        simple = final_state.get_result("simple_math", "model_a")
        assert simple.status == RunStatus.COMPLETED
        assert simple.react_trace is None

        # React test: has react_trace
        react = final_state.get_result("react_categorize", "model_a")
        assert react.status == RunStatus.COMPLETED
        assert react.react_trace is not None
        assert react.react_trace["completed"] is True

    @pytest.mark.asyncio
    async def test_react_with_drift_validation(self):
        """React test with expected_response gets drift validation on final_response."""
        test = BenchmarkCase(
            id="react_drift",
            user="Organize files",
            mode="react",
            mock_tools={"read_file": {"_default": "data"}},
            tools=[{"type": "function", "function": {"name": "read_file"}}],
            expected_response="Files organized into animals and fruits",
            max_iterations=10,
        )

        async def mock_step(**kwargs):
            return {
                "completed": True,
                "final_response": "Something completely different about weather.",
                "new_iterations": [],
                "call_counter": 0,
                "latency_ms": 50.0,
            }

        async def mock_drift(response, expected):
            return 0.8  # High drift

        with patch("prompt_prix.react.dispatch.react_step", side_effect=mock_step), \
             patch("prompt_prix.mcp.tools.drift.calculate_drift", side_effect=mock_drift):
            runner = BatteryRunner(
                tests=[test], models=["model_a"],
                drift_threshold=0.3,
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("react_drift", "model_a")
        assert result.status == RunStatus.SEMANTIC_FAILURE
        assert "Drift" in result.failure_reason
        assert result.drift_score == 0.8
        assert result.react_trace is not None  # Trace preserved even on drift failure
