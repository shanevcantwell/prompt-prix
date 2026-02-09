"""MCP protocol server for prompt-prix.

Exposes the prompt-prix tool layer over MCP stdio transport.
Agents (LAS, Claude Desktop, any MCP client) launch this as a
subprocess and call tools via JSON-RPC.

Entry points:
    prompt-prix-mcp          (console script)
    python -m prompt_prix.mcp.server
"""

from mcp.server import FastMCP

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

from prompt_prix.mcp.tools import (
    list_models,
    complete,
    calculate_drift,
    analyze_variants,
    generate_variants,
    analyze_trajectory,
    compare_trajectories,
    judge,
)
from prompt_prix.mcp.tools.react_step import react_step


# ─────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "prompt-prix",
    instructions=(
        "Visual fan-out for comparing LLM responses. "
        "Provides model completion, LLM-as-judge evaluation, "
        "embedding drift measurement, prompt geometry analysis, "
        "semantic trajectory analysis, and ReAct loop execution."
    ),
)

# Register all tools. FastMCP auto-generates JSON schemas from
# type annotations (including Pydantic models like ReActIteration).
mcp.add_tool(list_models)
mcp.add_tool(complete)
mcp.add_tool(calculate_drift)
mcp.add_tool(analyze_variants)
mcp.add_tool(generate_variants)
mcp.add_tool(analyze_trajectory)
mcp.add_tool(compare_trajectories)
mcp.add_tool(judge)
mcp.add_tool(react_step)


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def run():
    """Entry point for prompt-prix MCP server."""
    load_dotenv()

    from prompt_prix.mcp.registry import register_default_adapter
    register_default_adapter()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
