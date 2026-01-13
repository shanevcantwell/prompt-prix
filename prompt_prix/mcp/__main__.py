"""
Run prompt-prix as MCP server.

Usage:
    python -m prompt_prix.mcp

Requires:
    - MCP SDK: pip install mcp
    - LM Studio server(s) configured via environment:
        LM_STUDIO_SERVER_1=http://localhost:1234
        LM_STUDIO_SERVER_2=http://localhost:1235  # optional
"""

from prompt_prix.mcp.server import run_server

if __name__ == "__main__":
    run_server()
