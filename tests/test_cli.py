"""Tests for prompt_prix.cli module.

CLI tests mock at two levels:
- `models` command: mock adapter via registry (same as test_mcp_list_models)
- `run-battery` command: mock the runner's run() to avoid full adapter stack
"""

import asyncio
import json
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prompt_prix.mcp.registry import register_adapter, clear_adapter


# ─────────────────────────────────────────────────────────────────────
# TEST DATA
# ─────────────────────────────────────────────────────────────────────

SAMPLE_BENCHMARK = {
    "test_suite": "cli_test",
    "version": "1.0",
    "prompts": [
        {
            "id": "test_1",
            "name": "Test One",
            "user": "What is 2 + 2?",
        },
        {
            "id": "test_2",
            "name": "Test Two",
            "user": "What is 3 + 3?",
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_adapter():
    """Register a mock adapter for CLI tests."""
    adapter = MagicMock()
    adapter.get_available_models = AsyncMock(
        return_value=["model-a", "model-b"]
    )
    adapter.get_models_by_server = MagicMock(
        return_value={"http://localhost:1234": ["model-a", "model-b"]}
    )
    adapter.get_unreachable_servers = MagicMock(return_value=[])

    async def default_stream(*args, **kwargs):
        yield "response"

    adapter.stream_completion = default_stream

    register_adapter(adapter)
    yield adapter
    clear_adapter()


@pytest.fixture
def benchmark_file(tmp_path):
    """Create a temporary benchmark JSON file."""
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(SAMPLE_BENCHMARK))
    return path


# ─────────────────────────────────────────────────────────────────────
# MODELS COMMAND
# ─────────────────────────────────────────────────────────────────────


class TestModelsCommand:
    """Tests for `prompt-prix-cli models`."""

    @pytest.mark.asyncio
    async def test_models_text(self, mock_adapter):
        """Text mode prints one model ID per line."""
        from prompt_prix.cli import _cmd_models

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            code = await _cmd_models(json_output=False)

        assert code == 0
        lines = mock_stdout.getvalue().strip().split("\n")
        assert "model-a" in lines
        assert "model-b" in lines

    @pytest.mark.asyncio
    async def test_models_json(self, mock_adapter):
        """JSON mode prints full discovery result."""
        from prompt_prix.cli import _cmd_models

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            code = await _cmd_models(json_output=True)

        assert code == 0
        result = json.loads(mock_stdout.getvalue())
        assert "models" in result
        assert "servers" in result
        assert "unreachable" in result
        assert "model-a" in result["models"]

    @pytest.mark.asyncio
    async def test_models_unreachable_to_stderr(self, mock_adapter):
        """Unreachable servers are reported on stderr."""
        mock_adapter.get_unreachable_servers.return_value = ["http://dead:1234"]

        from prompt_prix.cli import _cmd_models

        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            code = await _cmd_models(json_output=False)

        assert code == 0
        assert "dead:1234" in mock_stderr.getvalue()


# ─────────────────────────────────────────────────────────────────────
# RUN-BATTERY COMMAND
# ─────────────────────────────────────────────────────────────────────


def _make_battery_state(tests, models):
    """Create a completed BatteryRun state for mocking."""
    from prompt_prix.battery import BatteryRun, RunResult, RunStatus

    state = BatteryRun(tests=tests, models=models)
    for test_id in tests:
        for model_id in models:
            state.set_result(RunResult(
                test_id=test_id,
                model_id=model_id,
                status=RunStatus.COMPLETED,
                response=f"Response for {test_id} x {model_id}",
                latency_ms=1500.0,
            ))
    return state


def _make_consistency_state(tests, models, runs=3):
    """Create a completed ConsistencyRun state for mocking."""
    from prompt_prix.battery import RunResult, RunStatus
    from prompt_prix.consistency import ConsistencyRun

    state = ConsistencyRun(tests=tests, models=models, runs_total=runs)
    for test_id in tests:
        for model_id in models:
            for i in range(runs):
                result = RunResult(
                    test_id=test_id,
                    model_id=model_id,
                    status=RunStatus.COMPLETED,
                    response=f"Run {i} response",
                    latency_ms=1200.0 + i * 100,
                )
                state.add_result(result)
    return state


class TestRunBatteryCommand:
    """Tests for `prompt-prix-cli run-battery`."""

    @pytest.mark.asyncio
    async def test_run_battery_output(self, mock_adapter, benchmark_file, tmp_path):
        """Battery run writes valid JSON with correct structure."""
        from prompt_prix.cli import _cmd_run_battery

        output_file = tmp_path / "results.json"
        final_state = _make_battery_state(["test_1", "test_2"], ["model-a"])

        async def mock_run(self):
            yield final_state

        with patch("prompt_prix.battery.BatteryRunner", autospec=False) as MockRunner:
            instance = MagicMock()
            instance.run = lambda: mock_run(instance)
            instance.state = final_state
            MockRunner.return_value = instance

            code = await _cmd_run_battery(
                tests_path=str(benchmark_file),
                models_csv="model-a",
                output=str(output_file),
            )

        assert code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert data["tests"] == ["test_1", "test_2"]
        assert data["models"] == ["model-a"]
        assert len(data["results"]) == 2
        assert data["results"][0]["status"] == "completed"
        assert data["results"][0]["latency_ms"] == 1500.0

    @pytest.mark.asyncio
    async def test_run_battery_consistency(self, mock_adapter, benchmark_file, tmp_path):
        """Consistency run (runs>1) writes aggregates format."""
        from prompt_prix.cli import _cmd_run_battery

        output_file = tmp_path / "results.json"
        final_state = _make_consistency_state(["test_1", "test_2"], ["model-a"], runs=3)

        async def mock_run(self):
            yield final_state

        with patch("prompt_prix.consistency.ConsistencyRunner", autospec=False) as MockRunner:
            instance = MagicMock()
            instance.run = lambda: mock_run(instance)
            instance.state = final_state
            MockRunner.return_value = instance

            code = await _cmd_run_battery(
                tests_path=str(benchmark_file),
                models_csv="model-a",
                runs=3,
                output=str(output_file),
            )

        assert code == 0
        data = json.loads(output_file.read_text())
        assert data["runs_total"] == 3
        assert "aggregates" in data
        assert len(data["aggregates"]) == 2
        assert data["aggregates"][0]["passes"] == 3
        assert data["aggregates"][0]["total"] == 3

    @pytest.mark.asyncio
    async def test_run_battery_missing_models(self, mock_adapter, benchmark_file):
        """Unknown model ID exits with error."""
        from prompt_prix.cli import _cmd_run_battery

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            code = await _cmd_run_battery(
                tests_path=str(benchmark_file),
                models_csv="nonexistent-model",
            )

        assert code == 1
        assert "not available" in mock_stderr.getvalue()

    @pytest.mark.asyncio
    async def test_run_battery_bad_file(self, mock_adapter):
        """Nonexistent file exits with error."""
        from prompt_prix.cli import _cmd_run_battery

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            code = await _cmd_run_battery(
                tests_path="/nonexistent/file.json",
                models_csv="model-a",
            )

        assert code == 1
        assert "not found" in mock_stderr.getvalue()

    @pytest.mark.asyncio
    async def test_run_battery_stdout(self, mock_adapter, benchmark_file):
        """Without -o, results go to stdout."""
        from prompt_prix.cli import _cmd_run_battery

        final_state = _make_battery_state(["test_1", "test_2"], ["model-a"])

        async def mock_run(self):
            yield final_state

        with patch("prompt_prix.battery.BatteryRunner", autospec=False) as MockRunner, \
             patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            instance = MagicMock()
            instance.run = lambda: mock_run(instance)
            instance.state = final_state
            MockRunner.return_value = instance

            code = await _cmd_run_battery(
                tests_path=str(benchmark_file),
                models_csv="model-a",
            )

        assert code == 0
        data = json.loads(mock_stdout.getvalue())
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_progress_to_stderr(self, mock_adapter, benchmark_file, tmp_path):
        """Progress lines are emitted to stderr as results complete."""
        from prompt_prix.cli import _cmd_run_battery

        final_state = _make_battery_state(["test_1", "test_2"], ["model-a"])

        async def mock_run(self):
            yield final_state

        output_file = tmp_path / "results.json"

        with patch("prompt_prix.battery.BatteryRunner", autospec=False) as MockRunner, \
             patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            instance = MagicMock()
            instance.run = lambda: mock_run(instance)
            instance.state = final_state
            MockRunner.return_value = instance

            code = await _cmd_run_battery(
                tests_path=str(benchmark_file),
                models_csv="model-a",
                output=str(output_file),
            )

        assert code == 0
        stderr_output = mock_stderr.getvalue()
        # Should have progress lines for each completed result
        assert "model-a" in stderr_output
        assert "test_1" in stderr_output


# ─────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────


class TestArgParsing:
    """Tests for CLI argument parsing."""

    def test_build_parser_models(self):
        from prompt_prix.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["models"])
        assert args.command == "models"
        assert not args.json_output

    def test_build_parser_models_json(self):
        from prompt_prix.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["models", "--json"])
        assert args.json_output is True

    def test_build_parser_run_battery(self):
        from prompt_prix.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "run-battery",
            "--tests", "file.json",
            "--models", "model-a,model-b",
            "--runs", "3",
            "--drift-threshold", "0.15",
            "--judge-model", "judge-model",
            "--max-tokens", "4096",
            "--timeout", "600",
            "-o", "output.json",
        ])
        assert args.command == "run-battery"
        assert args.tests == "file.json"
        assert args.models == "model-a,model-b"
        assert args.runs == 3
        assert args.drift_threshold == 0.15
        assert args.judge_model == "judge-model"
        assert args.max_tokens == 4096
        assert args.timeout == 600
        assert args.output == "output.json"

    def test_build_parser_run_battery_defaults(self):
        from prompt_prix.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "run-battery", "--tests", "f.json", "--models", "m"
        ])
        assert args.runs == 1
        assert args.drift_threshold == 0.0
        assert args.judge_model is None
        assert args.max_tokens == 2048
        assert args.timeout == 300
        assert args.output is None
