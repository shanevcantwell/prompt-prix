"""
prompt-prix MCP server.

Exposes prompt-prix capabilities as MCP tools for agentic systems.
Run with: python -m prompt_prix.mcp

Requires:
    pip install mcp  # or: pip install prompt-prix[mcp]
"""

from mcp.server.fastmcp import FastMCP

# Create MCP server instance
# Tools are registered via @mcp.tool() decorators in tools/*.py
mcp = FastMCP("prompt-prix")


def run_server():
    """
    Entry point for MCP server.

    Imports tools to trigger registration, then starts stdio server.
    """
    # Import tools to register them (decorators run on import)
    from prompt_prix.mcp.tools import judge  # noqa: F401

    # Run with stdio transport (default)
    mcp.run()
