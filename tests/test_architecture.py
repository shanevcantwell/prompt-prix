"""
Architecture enforcement tests.

These tests verify the MCP -> Adapter -> Provider layering is respected.
They MUST FAIL until the layer bypass refactor is complete.

See: .claude/plans/harmonic-hatching-shamir.md for architecture diagrams.
"""
import ast
from pathlib import Path

import pytest


class TestLayerBoundaries:
    """Verify consumers use MCP primitives, not core directly."""

    def test_battery_does_not_import_stream_completion_from_core(self):
        """battery.py must not import stream_completion from core."""
        source = Path("prompt_prix/battery.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "prompt_prix.core":
                for alias in node.names:
                    if alias.name == "stream_completion":
                        pytest.fail(
                            f"battery.py imports stream_completion from core (line {node.lineno}). "
                            "Should use MCP complete_stream instead."
                        )

    def test_compare_handlers_does_not_import_stream_completion_from_core(self):
        """compare/handlers.py must not import stream_completion from core."""
        source = Path("prompt_prix/tabs/compare/handlers.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "prompt_prix.core":
                for alias in node.names:
                    if alias.name == "stream_completion":
                        pytest.fail(
                            f"compare/handlers.py imports stream_completion from core (line {node.lineno}). "
                            "Should use MCP complete_stream instead."
                        )

    def test_core_does_not_export_stream_completion(self):
        """core.py must not define stream_completion (should be in adapter)."""
        source = Path("prompt_prix/core.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "stream_completion":
                pytest.fail(
                    f"core.py defines stream_completion (line {node.lineno}). "
                    "This function should be deleted - logic belongs in LMStudioAdapter."
                )

    def test_core_does_not_export_get_completion(self):
        """core.py must not define get_completion (deprecated)."""
        source = Path("prompt_prix/core.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_completion":
                pytest.fail(
                    f"core.py defines get_completion (line {node.lineno}). "
                    "This function should be deleted."
                )


class TestMCPSignatures:
    """Verify MCP primitives accept required parameters for shared state."""

    def test_complete_stream_accepts_pool_parameter(self):
        """MCP complete_stream must accept pool for shared ServerPool state."""
        import inspect

        from prompt_prix.mcp.tools.complete import complete_stream

        params = list(inspect.signature(complete_stream).parameters.keys())
        assert "pool" in params, (
            f"complete_stream missing 'pool' parameter for shared ServerPool state. "
            f"Current params: {params}"
        )

    def test_complete_accepts_pool_parameter(self):
        """MCP complete must accept pool for shared ServerPool state."""
        import inspect

        from prompt_prix.mcp.tools.complete import complete

        params = list(inspect.signature(complete).parameters.keys())
        assert "pool" in params, (
            f"complete missing 'pool' parameter for shared ServerPool state. "
            f"Current params: {params}"
        )
