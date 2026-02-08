"""MCP tools: semantic trajectory — text as particle in embedding space.

analyze_trajectory: Compute velocity/acceleration/curvature profiles,
    deadpan score (Adams-style absurdism), and heller score (circular reasoning).
compare_trajectories: Fitness score comparing synthetic text rhythm
    against a golden reference via DTW and acceleration correlation.

Both delegate to semantic-chunker (require embedding server + spaCy).
"""

from prompt_prix.mcp.tools._semantic_chunker import ensure_importable, get_manager


async def analyze_trajectory(
    text: str,
    acceleration_threshold: float = 0.3,
    include_sentences: bool = False,
) -> dict:
    """
    Analyze semantic trajectory of a text passage.

    Treats each sentence as a point in embedding space and computes
    kinematic quantities along the path: velocity (magnitude of semantic
    shift), acceleration (rate of pacing change), curvature (angular change).

    Args:
        text: Text passage to analyze (needs 2+ sentences)
        acceleration_threshold: Threshold for flagging acceleration spikes (default 0.3)
        include_sentences: Include sentence breakdown and profiles in output

    Returns:
        {
            "n_sentences": int,
            "mean_velocity": float,
            "mean_acceleration": float,
            "max_acceleration": float,
            "acceleration_spikes": [{"magnitude": float, "isolation_score": float, "position_ratio": float}, ...],
            "deadpan_score": float,  # Adams-style (isolated spikes in stable background)
            "heller_score": float,   # Heller-style (circular, decelerating)
            "circularity_score": float,
            "tautology_density": float,
            "deceleration_score": float,
            "adams_interpretation": str,
            "heller_interpretation": str,
            ...
        }

    Raises:
        ImportError: If semantic-chunker is not available.
        RuntimeError: If embedding server or spaCy returns an error.
    """
    if not ensure_importable():
        raise ImportError("semantic-chunker is not available")

    from semantic_chunker.mcp.commands.trajectory import analyze_trajectory as _analyze_trajectory
    result = await _analyze_trajectory(get_manager(), {
        "text": text,
        "acceleration_threshold": acceleration_threshold,
        "include_sentences": include_sentences,
    })
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


async def compare_trajectories(
    golden_text: str,
    synthetic_text: str,
    acceleration_threshold: float = 0.3,
) -> dict:
    """
    Compare semantic trajectory of synthetic text against a golden reference.

    Computes a fitness score (lower = better match) based on acceleration
    profile similarity via DTW, spike position matching, and structural
    metrics (deadpan, isolation, correlation).

    Args:
        golden_text: Golden reference passage (the target structure)
        synthetic_text: Synthetic passage to evaluate
        acceleration_threshold: Threshold for acceleration spikes

    Returns:
        {
            "fitness_score": float,  # 0.0-1.0, lower = better
            "synthetic_deadpan": float,
            "synthetic_heller": float,
            "acceleration_dtw": float,
            "acceleration_correlation": float,
            "spike_position_match": float,
            "spike_count_match": float,
            "interpretation": str,
            "golden_summary": {"n_sentences": int, "deadpan_score": float, ...},
            "synthetic_summary": {"n_sentences": int, "deadpan_score": float, ...},
        }

    Raises:
        ImportError: If semantic-chunker is not available.
        RuntimeError: If embedding server or spaCy returns an error.
    """
    if not ensure_importable():
        raise ImportError("semantic-chunker is not available")

    from semantic_chunker.mcp.commands.trajectory import compare_trajectories_handler
    result = await compare_trajectories_handler(get_manager(), {
        "golden_text": golden_text,
        "synthetic_text": synthetic_text,
        "acceleration_threshold": acceleration_threshold,
    })
    if "error" in result:
        raise RuntimeError(result["error"])
    return result
