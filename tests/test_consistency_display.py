"""Tests for CellAggregate grid display with error distinction.

Verifies that inconsistent and consistent-fail cells surface
infrastructure error counts separately from semantic failures.
"""

import pytest

from prompt_prix.battery import GridDisplayMode, RunResult, RunStatus
from prompt_prix.consistency import CellAggregate, ConsistencyStatus


def _make_agg(passes: int, errors: int, semantic_failures: int) -> CellAggregate:
    """Build a CellAggregate with the given result mix."""
    total = passes + errors + semantic_failures
    results = []
    for _ in range(passes):
        results.append(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.COMPLETED, response="ok", latency_ms=100.0,
        ))
    for _ in range(errors):
        results.append(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.ERROR, error="Connection refused",
        ))
    for _ in range(semantic_failures):
        results.append(RunResult(
            test_id="t1", model_id="m1",
            status=RunStatus.SEMANTIC_FAILURE, response="refused",
            latency_ms=50.0, failure_reason="Refusal detected",
        ))
    return CellAggregate(
        test_id="t1", model_id="m1",
        passes=passes, errors=errors,
        total=total, results=results,
    )


class TestConsistentPass:
    def test_display_shows_checkmark(self):
        agg = _make_agg(passes=10, errors=0, semantic_failures=0)
        assert agg.get_display(GridDisplayMode.SYMBOLS) == "✓"

    def test_status_symbol(self):
        agg = _make_agg(passes=5, errors=0, semantic_failures=0)
        assert agg.status_symbol == "✓"


class TestConsistentFailAllSemantic:
    def test_display_shows_x(self):
        """All semantic failures, no errors → ❌."""
        agg = _make_agg(passes=0, errors=0, semantic_failures=10)
        assert agg.get_display(GridDisplayMode.SYMBOLS) == "❌"

    def test_status_symbol(self):
        agg = _make_agg(passes=0, errors=0, semantic_failures=5)
        assert agg.status_symbol == "❌"


class TestConsistentFailAllErrors:
    def test_display_shows_warning(self):
        """All infrastructure errors → ⚠ (not ❌)."""
        agg = _make_agg(passes=0, errors=10, semantic_failures=0)
        assert agg.get_display(GridDisplayMode.SYMBOLS) == "⚠"

    def test_status_symbol(self):
        agg = _make_agg(passes=0, errors=5, semantic_failures=0)
        assert agg.status_symbol == "⚠"


class TestConsistentFailMixed:
    def test_display_shows_x_with_error_count(self):
        """Mixed errors + semantic failures → ❌ ⚠{N}."""
        agg = _make_agg(passes=0, errors=3, semantic_failures=7)
        assert agg.get_display(GridDisplayMode.SYMBOLS) == "❌ ⚠3"


class TestInconsistentNoErrors:
    def test_display_shows_pass_rate(self):
        """Inconsistent with no errors → 🟣 N/M (no ⚠)."""
        agg = _make_agg(passes=6, errors=0, semantic_failures=4)
        assert agg.get_display(GridDisplayMode.SYMBOLS) == "🟣 6/10"


class TestInconsistentWithErrors:
    def test_display_shows_pass_rate_and_error_count(self):
        """Inconsistent with errors → 🟣 N/M ⚠E."""
        agg = _make_agg(passes=6, errors=3, semantic_failures=1)
        assert agg.get_display(GridDisplayMode.SYMBOLS) == "🟣 6/10 ⚠3"


class TestLatencyMode:
    def test_latency_mode_unaffected(self):
        """Latency display is unchanged by error distinction."""
        agg = _make_agg(passes=6, errors=3, semantic_failures=1)
        display = agg.get_display(GridDisplayMode.LATENCY)
        # Only passed runs have latency (100ms each)
        assert "0.1s" in display
