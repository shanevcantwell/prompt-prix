"""
prompt-prix MCP server package.

Exposes prompt-prix capabilities as MCP tools for agentic systems.

Usage:
    python -m prompt_prix.mcp

Requires LM Studio server(s) configured via environment:
    LM_STUDIO_SERVER_1=http://localhost:1234
    LM_STUDIO_SERVER_2=http://localhost:1235  # optional
"""

from prompt_prix.mcp.server import mcp, run_server

__all__ = ["mcp", "run_server"]
