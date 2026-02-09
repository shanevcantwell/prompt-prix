"""Shared helpers for mocking the semantic-chunker module hierarchy.

Used by test_drift.py, test_geometry.py, and test_trajectory.py.
"""

import types
from unittest.mock import MagicMock


def make_semantic_chunker_modules(subcommand: str):
    """Create fake semantic_chunker module hierarchy for sys.modules patching.

    Args:
        subcommand: The commands submodule name (e.g., "embeddings", "geometry", "trajectory").

    Returns:
        (modules_dict, subcommand_module) — patch sys.modules with modules_dict,
        then attach mock functions to subcommand_module.
    """
    sc = types.ModuleType("semantic_chunker")
    sc_mcp = types.ModuleType("semantic_chunker.mcp")
    sc_mcp_commands = types.ModuleType("semantic_chunker.mcp.commands")
    sc_mcp_commands_sub = types.ModuleType(f"semantic_chunker.mcp.commands.{subcommand}")
    sc_mcp_state = types.ModuleType("semantic_chunker.mcp.state_manager")

    sc_mcp_state.StateManager = MagicMock

    sc.mcp = sc_mcp
    sc_mcp.commands = sc_mcp_commands
    sc_mcp.state_manager = sc_mcp_state
    setattr(sc_mcp_commands, subcommand, sc_mcp_commands_sub)

    modules_dict = {
        "semantic_chunker": sc,
        "semantic_chunker.mcp": sc_mcp,
        "semantic_chunker.mcp.commands": sc_mcp_commands,
        f"semantic_chunker.mcp.commands.{subcommand}": sc_mcp_commands_sub,
        "semantic_chunker.mcp.state_manager": sc_mcp_state,
    }
    return modules_dict, sc_mcp_commands_sub


def reset_semantic_chunker():
    """Reset cached state on the shared _semantic_chunker module."""
    import prompt_prix.mcp.tools._semantic_chunker as sc_mod
    sc_mod._manager = None
    sc_mod._available = None
