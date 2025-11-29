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


class MockServerConfig:
    """Mock server config for testing."""

    def __init__(self, available_models: list[str]):
        self.available_models = available_models
        self.is_busy = False


class MockServerPool:
    """Mock server pool for testing work-stealing dispatcher."""

    def __init__(self, models: list[str]):
        # Create a single mock server with all models
        self.servers = {
            "http://mock-server:1234": MockServerConfig(models)
        }

    async def acquire_server(self, url: str):
        self.servers[url].is_busy = True

    def release_server(self, url: str):
        self.servers[url].is_busy = False

    async def refresh_all_manifests(self):
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
        """Test converting to grid format."""
        run = BatteryRun(tests=["t1", "t2"], models=["m1", "m2"])
        run.set_result(TestResult(test_id="t1", model_id="m1", status=TestStatus.COMPLETED))
        run.set_result(TestResult(test_id="t1", model_id="m2", status=TestStatus.ERROR))

        grid = run.to_grid()
        assert grid[0] == ["Test", "m1", "m2"]  # Header
        assert grid[1] == ["t1", "✓", "❌"]      # t1 results
        assert grid[2] == ["t2", "—", "—"]      # t2 pending

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
