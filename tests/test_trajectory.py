"""Tests for prompt_prix.mcp.tools.trajectory module.

Both tools delegate to semantic-chunker. Tests mock the module hierarchy
(same pattern as test_drift.py TestDriftTool).
"""

import pytest
import sys
from unittest.mock import AsyncMock, MagicMock

from tests.sc_mock import make_semantic_chunker_modules, reset_semantic_chunker


# ─────────────────────────────────────────────────────────────────────
# ANALYZE TRAJECTORY TESTS
# ─────────────────────────────────────────────────────────────────────

class TestAnalyzeTrajectory:
    """Tests for analyze_trajectory MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_metrics(self):
        """analyze_trajectory returns trajectory metrics from semantic-chunker."""
        from unittest.mock import patch

        mock_result = {
            "n_sentences": 5,
            "mean_velocity": 0.42,
            "mean_acceleration": 0.15,
            "max_acceleration": 0.38,
            "acceleration_spikes": [{"magnitude": 0.38, "isolation_score": 0.7, "position_ratio": 0.6}],
            "deadpan_score": 0.65,
            "heller_score": 0.30,
            "circularity_score": 0.20,
            "tautology_density": 0.15,
            "deceleration_score": 0.10,
            "adams_interpretation": "Moderate deadpan structure",
            "heller_interpretation": "Low circularity",
        }

        modules_dict, trajectory_mod = make_semantic_chunker_modules("trajectory")
        trajectory_mod.analyze_trajectory = AsyncMock(return_value=mock_result)

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.trajectory import analyze_trajectory
            result = await analyze_trajectory(
                text="Ford suffered through the Vogon poetry. Arthur calmly praised the imagery. The Vogon ejected them into space.",
            )

        assert result["deadpan_score"] == 0.65
        assert result["heller_score"] == 0.30
        assert result["n_sentences"] == 5
        assert len(result["acceleration_spikes"]) == 1

    @pytest.mark.asyncio
    async def test_forwards_args_correctly(self):
        """analyze_trajectory passes correct args dict to semantic-chunker."""
        from unittest.mock import patch

        captured_args = {}

        async def capture_call(manager, args):
            captured_args.update(args)
            return {"n_sentences": 2, "mean_velocity": 0.5, "deadpan_score": 0,
                    "heller_score": 0, "mean_acceleration": 0, "max_acceleration": 0,
                    "acceleration_spikes": [], "circularity_score": 0,
                    "tautology_density": 0, "deceleration_score": 0,
                    "adams_interpretation": "", "heller_interpretation": ""}

        modules_dict, trajectory_mod = make_semantic_chunker_modules("trajectory")
        trajectory_mod.analyze_trajectory = capture_call

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.trajectory import analyze_trajectory
            await analyze_trajectory(
                text="Sentence one. Sentence two.",
                acceleration_threshold=0.5,
                include_sentences=True,
            )

        assert captured_args["text"] == "Sentence one. Sentence two."
        assert captured_args["acceleration_threshold"] == 0.5
        assert captured_args["include_sentences"] is True

    @pytest.mark.asyncio
    async def test_raises_on_error_result(self):
        """analyze_trajectory raises RuntimeError on semantic-chunker error."""
        from unittest.mock import patch

        modules_dict, trajectory_mod = make_semantic_chunker_modules("trajectory")
        trajectory_mod.analyze_trajectory = AsyncMock(
            return_value={"error": "Text must have at least 2 sentences"}
        )

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.trajectory import analyze_trajectory
            with pytest.raises(RuntimeError, match="2 sentences"):
                await analyze_trajectory(text="Just one sentence.")

    @pytest.mark.asyncio
    async def test_raises_import_error_when_unavailable(self):
        """analyze_trajectory raises ImportError when semantic-chunker absent."""
        reset_semantic_chunker()
        import prompt_prix.mcp.tools._semantic_chunker as sc_mod
        sc_mod._available = False

        from prompt_prix.mcp.tools.trajectory import analyze_trajectory
        with pytest.raises(ImportError, match="not available"):
            await analyze_trajectory(text="Some text here.")


# ─────────────────────────────────────────────────────────────────────
# COMPARE TRAJECTORIES TESTS
# ─────────────────────────────────────────────────────────────────────

class TestCompareTrajectories:
    """Tests for compare_trajectories MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_fitness_score(self):
        """compare_trajectories returns fitness score and components."""
        from unittest.mock import patch

        mock_result = {
            "fitness_score": 0.35,
            "synthetic_deadpan": 0.60,
            "synthetic_heller": 0.25,
            "synthetic_mean_isolation": 0.55,
            "acceleration_dtw": 0.20,
            "acceleration_correlation": 0.70,
            "spike_position_match": 0.80,
            "spike_count_match": 0.90,
            "interpretation": "Good structure, some rhythm deviation",
            "golden_summary": {"n_sentences": 5, "deadpan_score": 0.72},
            "synthetic_summary": {"n_sentences": 4, "deadpan_score": 0.60},
        }

        modules_dict, trajectory_mod = make_semantic_chunker_modules("trajectory")
        trajectory_mod.compare_trajectories_handler = AsyncMock(return_value=mock_result)

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.trajectory import compare_trajectories
            result = await compare_trajectories(
                golden_text="Ford suffered. Arthur praised. The Vogon ejected.",
                synthetic_text="The cat sat. The dog barked. Thunder struck.",
            )

        assert result["fitness_score"] == 0.35
        assert result["synthetic_deadpan"] == 0.60
        assert "golden_summary" in result

    @pytest.mark.asyncio
    async def test_forwards_both_texts(self):
        """compare_trajectories passes both texts to semantic-chunker."""
        from unittest.mock import patch

        captured_args = {}

        async def capture_call(manager, args):
            captured_args.update(args)
            return {"fitness_score": 0.5, "interpretation": "ok",
                    "synthetic_deadpan": 0, "synthetic_heller": 0,
                    "synthetic_mean_isolation": 0, "acceleration_dtw": 0,
                    "acceleration_correlation": 0, "spike_position_match": 0,
                    "spike_count_match": 0,
                    "golden_summary": {}, "synthetic_summary": {}}

        modules_dict, trajectory_mod = make_semantic_chunker_modules("trajectory")
        trajectory_mod.compare_trajectories_handler = capture_call

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.trajectory import compare_trajectories
            await compare_trajectories(
                golden_text="Golden text here.",
                synthetic_text="Synthetic text here.",
                acceleration_threshold=0.4,
            )

        assert captured_args["golden_text"] == "Golden text here."
        assert captured_args["synthetic_text"] == "Synthetic text here."
        assert captured_args["acceleration_threshold"] == 0.4

    @pytest.mark.asyncio
    async def test_raises_on_error_result(self):
        """compare_trajectories raises RuntimeError on semantic-chunker error."""
        from unittest.mock import patch

        modules_dict, trajectory_mod = make_semantic_chunker_modules("trajectory")
        trajectory_mod.compare_trajectories_handler = AsyncMock(
            return_value={"error": "Golden text too short"}
        )

        with patch.dict(sys.modules, modules_dict):
            reset_semantic_chunker()
            from prompt_prix.mcp.tools.trajectory import compare_trajectories
            with pytest.raises(RuntimeError, match="too short"):
                await compare_trajectories(
                    golden_text="Short.", synthetic_text="Also short.",
                )

    @pytest.mark.asyncio
    async def test_raises_import_error_when_unavailable(self):
        """compare_trajectories raises ImportError when semantic-chunker absent."""
        reset_semantic_chunker()
        import prompt_prix.mcp.tools._semantic_chunker as sc_mod
        sc_mod._available = False

        from prompt_prix.mcp.tools.trajectory import compare_trajectories
        with pytest.raises(ImportError, match="not available"):
            await compare_trajectories(
                golden_text="Text.", synthetic_text="Other text.",
            )
