"""MCP tool implementations.

Each tool is a self-contained async function that uses the adapter layer
for provider-agnostic model interactions. All tools are stateless.
"""

from prompt_prix.mcp.tools.complete import complete, complete_stream
from prompt_prix.mcp.tools.drift import calculate_drift
from prompt_prix.mcp.tools.geometry import analyze_variants, generate_variants
from prompt_prix.mcp.tools.judge import judge
from prompt_prix.mcp.tools.list_models import list_models
from prompt_prix.mcp.tools.react_step import react_step
from prompt_prix.mcp.tools.trajectory import analyze_trajectory, compare_trajectories

__all__ = [
    "analyze_trajectory",
    "analyze_variants",
    "calculate_drift",
    "compare_trajectories",
    "complete",
    "complete_stream",
    "generate_variants",
    "judge",
    "list_models",
    "react_step",
]
