"""Tests for drift-based validation in battery and consistency runners.

Tests mock the drift calculation (no embedding server needed for unit tests).
Per ADR-006: orchestration tests mock MCP tools.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from prompt_prix.benchmarks.base import BenchmarkCase
from prompt_prix.battery import (
    RunStatus,
    RunResult,
    BatteryRun,
    BatteryRunner,
    GridDisplayMode,
)
from prompt_prix.consistency import ConsistencyRunner, ConsistencyRun


# ─────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def tests_with_criteria():
    """BenchmarkCases that have expected_response (trigger drift validation)."""
    return [
        BenchmarkCase(
            id="drift_1",
            user="What is 2 + 2?",
            expected_response="The answer is 4",
        ),
        BenchmarkCase(
            id="drift_2",
            user="What is the capital of France?",
            expected_response="The capital of France is Paris",
        ),
    ]


@pytest.fixture
def tests_without_criteria():
    """BenchmarkCases without expected_response (drift skipped)."""
    return [
        BenchmarkCase(id="no_criteria", user="Hello world"),
    ]


# ─────────────────────────────────────────────────────────────────────
# BATTERY RUN RECALCULATION TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBatteryRunRecalculation:
    """Tests for BatteryRun.recalculate_drift_threshold."""

    def test_recalculate_flips_pass_to_fail(self):
        """Lower threshold flips a passing result to SEMANTIC_FAILURE."""
        run = BatteryRun(tests=["t1"], models=["m1"])
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.COMPLETED,
            response="some response",
            drift_score=0.35,
        ))

        run.recalculate_drift_threshold(0.3)
        result = run.get_result("t1", "m1")
        assert result.status == RunStatus.SEMANTIC_FAILURE
        assert "Drift" in result.failure_reason

    def test_recalculate_flips_fail_to_pass(self):
        """Higher threshold restores a drift-failed result to COMPLETED."""
        run = BatteryRun(tests=["t1"], models=["m1"])
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.SEMANTIC_FAILURE,
            response="some response",
            drift_score=0.35,
            failure_reason="Drift 0.350 exceeds threshold 0.3",
        ))

        run.recalculate_drift_threshold(0.5)
        result = run.get_result("t1", "m1")
        assert result.status == RunStatus.COMPLETED
        assert result.failure_reason is None

    def test_recalculate_ignores_no_drift_score(self):
        """Results without drift_score are unaffected."""
        run = BatteryRun(tests=["t1"], models=["m1"])
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.COMPLETED,
            response="some response",
        ))

        run.recalculate_drift_threshold(0.1)
        result = run.get_result("t1", "m1")
        assert result.status == RunStatus.COMPLETED

    def test_recalculate_ignores_non_drift_failures(self):
        """Semantic failures from refusal detection are not flipped."""
        run = BatteryRun(tests=["t1"], models=["m1"])
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.SEMANTIC_FAILURE,
            response="I'm sorry I cannot",
            drift_score=0.2,
            failure_reason="Refusal detected: I'm sorry",
        ))

        run.recalculate_drift_threshold(1.0)
        result = run.get_result("t1", "m1")
        # Still SEMANTIC_FAILURE because failure was refusal, not drift
        assert result.status == RunStatus.SEMANTIC_FAILURE
        assert "Refusal" in result.failure_reason

    def test_recalculate_threshold_zero_disables(self):
        """Threshold 0 disables drift — restores drift failures to pass."""
        run = BatteryRun(tests=["t1"], models=["m1"])
        run.set_result(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.SEMANTIC_FAILURE,
            response="some response",
            drift_score=0.35,
            failure_reason="Drift 0.350 exceeds threshold 0.3",
        ))

        run.recalculate_drift_threshold(0.0)
        result = run.get_result("t1", "m1")
        assert result.status == RunStatus.COMPLETED
        assert result.failure_reason is None


# ─────────────────────────────────────────────────────────────────────
# BATTERY RUNNER INLINE DRIFT TESTS
# ─────────────────────────────────────────────────────────────────────

class TestBatteryRunnerDrift:
    """Tests for inline drift validation in BatteryRunner._execute_test."""

    @pytest.mark.asyncio
    async def test_drift_below_threshold_passes(self, tests_with_criteria):
        """Response close to expected_response passes drift check."""
        async def mock_complete_stream(**kwargs):
            yield "The answer is 4"

        async def mock_drift(text_a, text_b):
            return 0.15  # Below threshold

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.mcp.tools.drift.calculate_drift", side_effect=mock_drift):
            runner = BatteryRunner(
                tests=tests_with_criteria[:1],
                models=["model_a"],
                drift_threshold=0.3,
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("drift_1", "model_a")
        assert result.status == RunStatus.COMPLETED
        assert result.drift_score == 0.15

    @pytest.mark.asyncio
    async def test_drift_above_threshold_fails(self, tests_with_criteria):
        """Response far from expected_response fails drift check."""
        async def mock_complete_stream(**kwargs):
            yield "The weather is lovely today"

        async def mock_drift(text_a, text_b):
            return 0.65  # Above threshold

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.mcp.tools.drift.calculate_drift", side_effect=mock_drift):
            runner = BatteryRunner(
                tests=tests_with_criteria[:1],
                models=["model_a"],
                drift_threshold=0.3,
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("drift_1", "model_a")
        assert result.status == RunStatus.SEMANTIC_FAILURE
        assert result.drift_score == 0.65
        assert "Drift" in result.failure_reason

    @pytest.mark.asyncio
    async def test_drift_disabled_when_threshold_zero(self, tests_with_criteria):
        """drift_threshold=0 disables drift validation entirely."""
        async def mock_complete_stream(**kwargs):
            yield "Some response"

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=tests_with_criteria[:1],
                models=["model_a"],
                drift_threshold=0.0,  # Disabled
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("drift_1", "model_a")
        assert result.status == RunStatus.COMPLETED
        assert result.drift_score is None  # Never calculated

    @pytest.mark.asyncio
    async def test_drift_skipped_without_criteria(self, tests_without_criteria):
        """Tests without expected_response skip drift even when threshold > 0."""
        async def mock_complete_stream(**kwargs):
            yield "Hello!"

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream):
            runner = BatteryRunner(
                tests=tests_without_criteria,
                models=["model_a"],
                drift_threshold=0.3,
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("no_criteria", "model_a")
        assert result.status == RunStatus.COMPLETED
        assert result.drift_score is None

    @pytest.mark.asyncio
    async def test_drift_error_fails_open(self, tests_with_criteria):
        """Drift calculation error fails open (result still COMPLETED)."""
        async def mock_complete_stream(**kwargs):
            yield "The answer is 4"

        async def mock_drift(text_a, text_b):
            raise RuntimeError("Embedding server unreachable")

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.mcp.tools.drift.calculate_drift", side_effect=mock_drift):
            runner = BatteryRunner(
                tests=tests_with_criteria[:1],
                models=["model_a"],
                drift_threshold=0.3,
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        result = final_state.get_result("drift_1", "model_a")
        assert result.status == RunStatus.COMPLETED
        assert result.drift_score is None  # Calculation failed


# ─────────────────────────────────────────────────────────────────────
# CONSISTENCY RUNNER DRIFT TESTS
# ─────────────────────────────────────────────────────────────────────

class TestConsistencyRunnerDrift:
    """Tests for drift validation in ConsistencyRunner."""

    @pytest.mark.asyncio
    async def test_consistency_drift_validation(self, tests_with_criteria):
        """ConsistencyRunner applies drift to each individual run."""
        call_count = {"drift": 0}

        async def mock_complete_stream(**kwargs):
            yield "The answer is 4"

        async def mock_drift(text_a, text_b):
            call_count["drift"] += 1
            return 0.15

        with patch("prompt_prix.react.dispatch.complete_stream", side_effect=mock_complete_stream), \
             patch("prompt_prix.mcp.tools.drift.calculate_drift", side_effect=mock_drift):
            runner = ConsistencyRunner(
                tests=tests_with_criteria[:1],
                models=["model_a"],
                runs=3,
                drift_threshold=0.3,
            )

            final_state = None
            async for state in runner.run():
                final_state = state

        # Drift calculated for each run (3 runs × 1 test × 1 model)
        assert call_count["drift"] == 3
        agg = final_state.get_aggregate("drift_1", "model_a")
        assert agg.passes == 3


class TestConsistencyRunRecalculation:
    """Tests for ConsistencyRun.recalculate_drift_threshold."""

    def test_recalculate_updates_pass_counts(self):
        """Recalculating drift threshold adjusts aggregate pass counts."""
        run = ConsistencyRun(tests=["t1"], models=["m1"], runs_total=3)
        agg = run.ensure_aggregate("t1", "m1")

        # 3 results, all passing with drift scores
        for i in range(3):
            result = RunResult(
                test_id="t1", model_id="m1",
                status=RunStatus.COMPLETED,
                response=f"response {i}",
                drift_score=0.2 + (i * 0.1),  # 0.2, 0.3, 0.4
            )
            agg.results.append(result)
            agg.passes += 1

        assert agg.passes == 3

        # Lower threshold to 0.25 — only first result passes
        run.recalculate_drift_threshold(0.25)
        assert agg.passes == 1
        assert agg.results[0].status == RunStatus.COMPLETED
        assert agg.results[1].status == RunStatus.SEMANTIC_FAILURE
        assert agg.results[2].status == RunStatus.SEMANTIC_FAILURE

        # Raise threshold back to 0.5 — all pass again
        run.recalculate_drift_threshold(0.5)
        assert agg.passes == 3
        assert all(r.status == RunStatus.COMPLETED for r in agg.results)


# ─────────────────────────────────────────────────────────────────────
# DRIFT TOOL UNIT TESTS
# ─────────────────────────────────────────────────────────────────────

class TestDriftTool:
    """Tests for the mcp/tools/drift.py wrapper.

    semantic-chunker is not installed in the test venv (Docker-only dep),
    so we mock the entire module hierarchy via tests.sc_mock helpers.
    """

    @pytest.mark.asyncio
    async def test_drift_returns_float(self):
        """calculate_drift returns a float from semantic-chunker."""
        import sys
        from tests.sc_mock import make_semantic_chunker_modules, reset_semantic_chunker

        mock_result = {"drift": 0.42, "interpretation": "Moderate"}

        modules_dict, embeddings_mod = make_semantic_chunker_modules("embeddings")
        embeddings_mod.calculate_drift = AsyncMock(return_value=mock_result)

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.drift import calculate_drift
            score = await calculate_drift("hello", "world")

        assert score == 0.42

    @pytest.mark.asyncio
    async def test_drift_raises_on_error(self):
        """calculate_drift raises RuntimeError when semantic-chunker returns error."""
        import sys
        from tests.sc_mock import make_semantic_chunker_modules, reset_semantic_chunker

        mock_result = {"error": "Both text_a and text_b are required"}

        modules_dict, embeddings_mod = make_semantic_chunker_modules("embeddings")
        embeddings_mod.calculate_drift = AsyncMock(return_value=mock_result)

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.drift import calculate_drift
            with pytest.raises(RuntimeError, match="required"):
                await calculate_drift("", "")

    @pytest.mark.asyncio
    async def test_drift_raises_import_error_when_unavailable(self):
        """calculate_drift raises ImportError when semantic-chunker absent."""
        from tests.sc_mock import reset_semantic_chunker
        reset_semantic_chunker()
        import prompt_prix.mcp.tools._semantic_chunker as sc_mod
        sc_mod._available = False

        from prompt_prix.mcp.tools.drift import calculate_drift
        with pytest.raises(ImportError, match="not available"):
            await calculate_drift("hello", "world")
