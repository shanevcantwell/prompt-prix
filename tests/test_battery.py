"""Tests for battery feature (benchmark test suite execution)."""

import json
import pytest
from pathlib import Path
from typing import AsyncGenerator
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


class MockServerState:
    """Mock server state for testing (matches scheduler.ServerState)."""

    def __init__(self, url: str, manifest_models: list[str]):
        self.url = url
        self.manifest_models = manifest_models
        self.loaded_models: list[str] = []  # No models loaded by default
        self.is_busy = False


class MockServerPool:
    """Mock server pool for testing BatchRunner (matches scheduler.ServerPool API)."""

    def __init__(self, models: list[str]):
        # Create a single mock server with all models
        self.servers = {
            "http://mock-server:1234": MockServerState("http://mock-server:1234", models)
        }
        self._locks = {}

    def find_server(self, model_id: str, require_loaded: bool = False, preferred_url: str | None = None) -> str | None:
        """Find a server that can run the model."""
        # Check preferred URL first if specified
        if preferred_url and preferred_url in self.servers:
            server = self.servers[preferred_url]
            if not server.is_busy:
                if model_id in server.loaded_models or model_id in server.manifest_models:
                    return preferred_url

        for url, server in self.servers.items():
            if server.is_busy:
                continue
            if require_loaded:
                if model_id in server.loaded_models:
                    return url
            else:
                if model_id in server.manifest_models:
                    return url
        return None

    def get_available_models(self, only_loaded: bool = False) -> set[str]:
        """Get all models that can run."""
        if only_loaded:
            result = set()
            for server in self.servers.values():
                result.update(server.loaded_models)
            return result
        result = set()
        for server in self.servers.values():
            result.update(server.manifest_models)
        return result

    async def acquire(self, url: str):
        self.servers[url].is_busy = True

    def release(self, url: str):
        self.servers[url].is_busy = False

    async def refresh(self):
        pass  # No-op for tests


class MockAdapter:
    """Mock adapter for testing BatteryRunner."""

    def __init__(self, responses: dict[str, str] = None, errors: dict[str, str] = None):
        self.responses = responses or {}
        self.errors = errors or {}
        self.calls = []
        # Create mock pool with all response models available
        all_models = list(self.responses.keys()) + list(self.errors.keys())
        self._pool = MockServerPool(all_models)

    @property
    def pool(self):
        """Expose mock pool for work-stealing dispatcher."""
        return self._pool

    async def get_available_models(self):
        return list(self.responses.keys())

    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tools=None
    ) -> AsyncGenerator[str, None]:
        self.calls.append((model_id, messages))

        if model_id in self.errors:
            raise RuntimeError(self.errors[model_id])

        response = self.responses.get(model_id, "Default response")
        # Simulate streaming by yielding chunks
        for word in response.split():
            yield word + " "


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
        assert TestResult(test_id="t", model_id="m", status=TestStatus.SEMANTIC_FAILURE).status_symbol == "❌"
        assert TestResult(test_id="t", model_id="m", status=TestStatus.ERROR).status_symbol == "⚠"


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
        """Test converting to grid format."""
        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        run.set_result(TestResult(test_id="t1", model_id="m1", status=TestStatus.COMPLETED))
        run.set_result(TestResult(test_id="t1", model_id="m2", status=TestStatus.ERROR))

        grid = run.to_grid()
        assert grid[0] == ["Test", "m1", "m2"]  # Header
        assert grid[1] == ["t1", "✓", "⚠"]      # t1: m1=completed, m2=error (technical issue)
        assert grid[2] == ["t2", "—", "—"]      # t2 pending

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

        grid = run.to_grid(GridDisplayMode.LATENCY)
        assert grid[0] == ["Test", "m1", "m2"]  # Header
        assert grid[1] == ["t1", "✓ 1.5s", "⚠ 2.5s"]  # t1: symbol + latencies
        assert grid[2] == ["t2", "—", "—"]             # t2 pending

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
    """Tests for BatteryRunner orchestrator."""

    @pytest.mark.asyncio
    async def test_run_completes_all_tests(self, sample_test_cases):
        """Test that runner completes all (test, model) combinations."""
        adapter = MockAdapter(responses={
            "model_a": "Response A",
            "model_b": "Response B"
        })

        runner = BatteryRunner(
            adapter=adapter,
            tests=sample_test_cases,
            models=["model_a", "model_b"]
        )

        final_state = None
        async for state in runner.run():
            final_state = state

        assert final_state is not None
        assert final_state.completed_count == 4  # 2 tests × 2 models

    @pytest.mark.asyncio
    async def test_run_handles_errors(self, sample_test_cases):
        """Test that runner handles model errors gracefully."""
        from unittest.mock import patch, AsyncMock

        adapter = MockAdapter(
            responses={"model_a": "Success"},
            errors={"model_b": "Connection failed"}
        )

        runner = BatteryRunner(
            adapter=adapter,
            tests=sample_test_cases[:1],  # Just one test
            models=["model_a", "model_b"]
        )

        # Mock stream_completion to use adapter's behavior
        async def mock_stream(server_url, model_id, messages, **kwargs):
            async for chunk in adapter.stream_completion(
                model_id=model_id,
                messages=messages,
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 100),
                timeout_seconds=kwargs.get("timeout_seconds", 30),
                tools=kwargs.get("tools")
            ):
                yield chunk

        with patch("prompt_prix.battery.stream_completion", mock_stream):
            final_state = None
            async for state in runner.run():
                final_state = state

        # Check model_a succeeded
        result_a = final_state.get_result("test_1", "model_a")
        assert result_a.status == TestStatus.COMPLETED

        # Check model_b errored
        result_b = final_state.get_result("test_1", "model_b")
        assert result_b.status == TestStatus.ERROR
        assert "Connection failed" in result_b.error

    @pytest.mark.asyncio
    async def test_run_yields_state_updates(self, sample_test_cases):
        """Test that runner yields state updates for UI."""
        adapter = MockAdapter(responses={"m1": "Response"})

        runner = BatteryRunner(
            adapter=adapter,
            tests=sample_test_cases[:1],
            models=["m1"]
        )

        state_count = 0
        async for state in runner.run():
            state_count += 1

        # Should yield: RUNNING, then COMPLETED, then final
        assert state_count >= 2

    @pytest.mark.asyncio
    async def test_run_records_latency(self, sample_test_cases):
        """Test that runner records latency for completed tests."""
        adapter = MockAdapter(responses={"m1": "Response"})

        runner = BatteryRunner(
            adapter=adapter,
            tests=sample_test_cases[:1],
            models=["m1"]
        )

        final_state = None
        async for state in runner.run():
            final_state = state

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
        # Returns gr.update(visible=False, value=filepath) - File hidden, JS triggers download
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

        # Cleanup
        state.battery_run = None

    def test_export_csv_with_results(self):
        """Test CSV export creates file with results."""
        import os
        import csv
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
        # Returns gr.update(visible=False, value=filepath) - File hidden, JS triggers download
        assert file_update["visible"] is False
        filepath = file_update["value"]
        assert filepath is not None
        assert filepath.endswith(".csv")

        # Read CSV properly using csv module
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check header
        assert rows[0] == ["test_id", "model_id", "status", "latency_ms", "response"]
        # Check data row
        assert rows[1][0] == "t1"
        assert rows[1][1] == "m1"
        assert rows[1][2] == "completed"
        assert rows[1][3] == "500"
        assert "Hello\nWorld" in rows[1][4]  # Newlines preserved in response

        state.battery_run = None

    def test_export_image_with_results(self):
        """Test image export creates valid PNG file."""
        import os
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import export_grid_image

        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        run.set_result(TestResult(
            test_id="t1", model_id="m1",
            status=TestStatus.COMPLETED,
            response="Test response",
            latency_ms=500.0
        ))
        run.set_result(TestResult(
            test_id="t1", model_id="m2",
            status=TestStatus.ERROR,
            error="Test error"
        ))
        state.battery_run = run

        status, file_update = export_grid_image()

        assert "✅" in status
        # Returns gr.update(visible=False, value=filepath) - File hidden, JS triggers download
        assert file_update["visible"] is False
        filepath = file_update["value"]
        assert filepath is not None
        assert filepath.endswith(".png")
        assert os.path.exists(filepath)

        # Verify it's a valid PNG (check magic bytes)
        with open(filepath, "rb") as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG', "File should have PNG magic bytes"

        state.battery_run = None

    def test_export_image_no_results(self):
        """Test image export returns error when no battery run exists."""
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import export_grid_image

        state.battery_run = None
        status, file_update = export_grid_image()

        assert "❌" in status
        assert file_update["visible"] is False
        assert file_update["value"] is None

    def test_export_basename_from_source_file(self):
        """Test export filename derives from source file."""
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import _get_export_basename

        state.battery_source_file = "/path/to/my_test_suite.jsonl"
        basename = _get_export_basename()
        assert basename == "my_test_suite_results"

        state.battery_source_file = None
        basename = _get_export_basename()
        assert basename == "battery_results"


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


class TestGetCellDetailWithPrefix:
    """Tests for get_cell_detail with GPU-prefixed model names.

    Bug #23: Response Detail shows 'No result' when model dropdown
    contains prefixed names like '0: model-name' but results are keyed
    by stripped model ID 'model-name'.
    """

    def test_get_cell_detail_strips_gpu_prefix(self):
        """Test that get_cell_detail works with GPU-prefixed model names.

        The dropdown passes '0: model-name' but results are stored
        with key 'model-name'. The handler must strip the prefix.
        """
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import get_cell_detail

        # Setup battery run with stripped model IDs (as run_handler stores them)
        run = BatteryRun(tests=["test_1"], models=["lfm2-1.2b-tool"])
        run.set_result(TestResult(
            test_id="test_1",
            model_id="lfm2-1.2b-tool",  # Stored without prefix
            status=TestStatus.COMPLETED,
            response="Test response",
            latency_ms=100.0
        ))
        state.battery_run = run

        # Call with prefixed model name (as dropdown would provide)
        result = get_cell_detail("0: lfm2-1.2b-tool", "test_1")

        # Should find the result, not return "No result"
        assert "No result" not in result
        assert "Test response" in result

        # Cleanup
        state.battery_run = None

    def test_get_cell_detail_handles_model_with_slashes(self):
        """Test prefix stripping with complex model paths containing slashes."""
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import get_cell_detail

        model_id = "openai/gpt-oss-20b-gguf/gpt-oss-20b-router-mxfp4.gguf"
        run = BatteryRun(tests=["test_1"], models=[model_id])
        run.set_result(TestResult(
            test_id="test_1",
            model_id=model_id,
            status=TestStatus.COMPLETED,
            response="Complex model response",
            latency_ms=500.0
        ))
        state.battery_run = run

        # Call with prefixed complex model name
        result = get_cell_detail(f"1: {model_id}", "test_1")

        assert "No result" not in result
        assert "Complex model response" in result

        state.battery_run = None

    def test_get_cell_detail_without_prefix_still_works(self):
        """Test that non-prefixed model names still work (backwards compat)."""
        from prompt_prix import state
        from prompt_prix.tabs.battery.handlers import get_cell_detail

        run = BatteryRun(tests=["test_1"], models=["simple-model"])
        run.set_result(TestResult(
            test_id="test_1",
            model_id="simple-model",
            status=TestStatus.COMPLETED,
            response="Simple response",
            latency_ms=50.0
        ))
        state.battery_run = run

        # Call without prefix (should still work)
        result = get_cell_detail("simple-model", "test_1")

        assert "No result" not in result
        assert "Simple response" in result

        state.battery_run = None


class TestImportPromptfoo:
    """Tests for promptfoo import handler."""

    def test_import_valid_promptfoo_config(self, tmp_path):
        """Test importing a valid promptfoo YAML config."""
        import yaml
        from prompt_prix.tabs.battery.handlers import import_promptfoo

        config = {
            "prompts": ["What is 2+2?", "What is the capital of France?"]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        validation, temp_file, test_ids = import_promptfoo(str(config_file))

        assert "✅" in validation
        assert "2 tests" in validation
        assert temp_file is not None
        assert len(test_ids) == 2

    def test_import_no_file(self):
        """Test importing with no file returns error."""
        from prompt_prix.tabs.battery.handlers import import_promptfoo

        validation, temp_file, test_ids = import_promptfoo(None)

        assert "Select a promptfoo" in validation
        assert temp_file is None
        assert test_ids == []

    def test_import_empty_config(self, tmp_path):
        """Test importing config with no prompts returns error."""
        import yaml
        from prompt_prix.tabs.battery.handlers import import_promptfoo

        config = {"providers": ["openai:gpt-4"]}
        config_file = tmp_path / "empty.yaml"
        config_file.write_text(yaml.dump(config))

        validation, temp_file, test_ids = import_promptfoo(str(config_file))

        assert "❌" in validation
        assert "No tests found" in validation

    def test_imported_file_is_valid_json(self, tmp_path):
        """Test that imported temp file is valid Battery JSON."""
        import yaml
        from prompt_prix.tabs.battery.handlers import import_promptfoo
        from prompt_prix.benchmarks.custom import CustomJSONLoader

        config = {
            "prompts": ["Test prompt 1", "Test prompt 2"]
        }
        config_file = tmp_path / "promptfooconfig.yaml"
        config_file.write_text(yaml.dump(config))

        validation, temp_file, test_ids = import_promptfoo(str(config_file))

        # Verify the temp file is valid for Battery
        valid, message = CustomJSONLoader.validate(temp_file)
        assert valid is True
        assert "2 tests" in message


# ─────────────────────────────────────────────────────────────────────
# BATTERYRUNNER ADAPTER INTERFACE TESTS (#73)
# ─────────────────────────────────────────────────────────────────────

class HostAdapterMock:
    """
    Mock adapter implementing ONLY the HostAdapter protocol.

    This mock does NOT have a .pool attribute. Tests using this mock
    will FAIL until BatteryRunner is refactored to use HostAdapter
    interface instead of accessing adapter.pool directly.

    Part of #73 Phase 4 - tests for adapter refactor.
    """

    def __init__(self, responses: dict[str, str] = None, errors: dict[str, str] = None):
        self.responses = responses or {}
        self.errors = errors or {}
        self.calls = []
        self.acquired = []
        self.released = []

    # NOTE: No .pool property! This is intentional.

    async def get_available_models(self) -> list[str]:
        return list(self.responses.keys())

    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tools=None
    ):
        self.calls.append((model_id, messages))

        if model_id in self.errors:
            raise RuntimeError(self.errors[model_id])

        response = self.responses.get(model_id, "Default response")
        for word in response.split():
            yield word + " "

    def get_concurrency_limit(self) -> int:
        return 2

    async def acquire(self, model_id: str) -> None:
        self.acquired.append(model_id)

    async def release(self, model_id: str) -> None:
        self.released.append(model_id)


class TestBatteryRunnerWithHostAdapter:
    """
    Tests for BatteryRunner using the HostAdapter interface.

    These tests document the expected behavior after #73 refactor.
    They FAIL initially because BatteryRunner currently accesses adapter.pool.

    Phase 4 of #73 adapter refactor.
    """

    @pytest.fixture
    def host_adapter_mock(self):
        """Create a HostAdapter-compliant mock (no .pool)."""
        return HostAdapterMock(responses={
            "model_a": "Response A",
            "model_b": "Response B"
        })

    @pytest.fixture
    def simple_test_cases(self):
        """Simple test cases for adapter tests."""
        return [
            TestCase(id="t1", user="Test prompt 1"),
            TestCase(id="t2", user="Test prompt 2"),
        ]

    @pytest.mark.asyncio
    async def test_runner_works_without_pool_attribute(
        self, host_adapter_mock, simple_test_cases
    ):
        """BatteryRunner should work with adapter that has no .pool attribute.

        This is the key test for #73 - the abstraction leak is accessing adapter.pool.
        After refactor, BatteryRunner should use adapter methods directly.
        """
        runner = BatteryRunner(
            adapter=host_adapter_mock,
            tests=simple_test_cases,
            models=["model_a"]
        )

        # This should NOT raise AttributeError: 'HostAdapterMock' has no attribute 'pool'
        final_state = None
        async for state in runner.run():
            final_state = state

        assert final_state is not None
        assert final_state.completed_count == 2  # 2 tests × 1 model

    @pytest.mark.asyncio
    async def test_runner_calls_acquire_release(
        self, host_adapter_mock, simple_test_cases
    ):
        """BatteryRunner should call adapter.acquire/release for concurrency."""
        runner = BatteryRunner(
            adapter=host_adapter_mock,
            tests=simple_test_cases,
            models=["model_a"]
        )

        async for _ in runner.run():
            pass

        # Verify acquire/release were called for each model execution
        assert len(host_adapter_mock.acquired) > 0
        assert len(host_adapter_mock.released) > 0
        # Each acquire should have a matching release
        assert len(host_adapter_mock.acquired) == len(host_adapter_mock.released)

    @pytest.mark.asyncio
    async def test_runner_uses_adapter_stream_completion(
        self, host_adapter_mock, simple_test_cases
    ):
        """BatteryRunner should use adapter.stream_completion(), not core.stream_completion()."""
        runner = BatteryRunner(
            adapter=host_adapter_mock,
            tests=simple_test_cases,
            models=["model_a"]
        )

        async for _ in runner.run():
            pass

        # Verify adapter.stream_completion was called
        assert len(host_adapter_mock.calls) == 2  # 2 tests
        # Each call should be for model_a
        for model_id, messages in host_adapter_mock.calls:
            assert model_id == "model_a"

    @pytest.mark.asyncio
    async def test_runner_respects_concurrency_limit(
        self, simple_test_cases
    ):
        """BatteryRunner should respect adapter.get_concurrency_limit()."""
        import asyncio

        concurrent_count = 0
        max_concurrent = 0

        class ConcurrencyTrackingAdapter(HostAdapterMock):
            async def acquire(self, model_id: str) -> None:
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await super().acquire(model_id)

            async def release(self, model_id: str) -> None:
                nonlocal concurrent_count
                concurrent_count -= 1
                await super().release(model_id)

            async def stream_completion(self, model_id, messages, **kwargs):
                # Add delay to ensure concurrency is measurable
                await asyncio.sleep(0.02)
                yield "response"

            def get_concurrency_limit(self) -> int:
                return 1  # Strict limit of 1

        adapter = ConcurrencyTrackingAdapter(responses={"model_a": "ok", "model_b": "ok"})

        runner = BatteryRunner(
            adapter=adapter,
            tests=simple_test_cases,
            models=["model_a", "model_b"]
        )

        async for _ in runner.run():
            pass

        # With limit of 1, max concurrent should never exceed 1
        assert max_concurrent <= 1
