"""
Architecture enforcement tests.

These tests verify the Orchestration -> Adapter -> Provider layering is respected.
Per ADR-006, adapters own their resource management (ServerPool is internal to LMStudioAdapter).

See: docs/adr/006-adapter-resource-ownership.md
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


class TestServerPoolEncapsulation:
    """Verify ServerPool is encapsulated inside adapters per ADR-006."""

    def test_mcp_complete_does_not_accept_pool_parameter(self):
        """MCP complete must NOT accept pool - adapter owns resource management."""
        import inspect

        from prompt_prix.mcp.tools.complete import complete

        params = list(inspect.signature(complete).parameters.keys())
        assert "pool" not in params, (
            f"complete accepts 'pool' parameter - violates ADR-006. "
            f"Adapter should own ServerPool internally. Current params: {params}"
        )

    def test_mcp_complete_stream_does_not_accept_pool_parameter(self):
        """MCP complete_stream must NOT accept pool - adapter owns resource management."""
        import inspect

        from prompt_prix.mcp.tools.complete import complete_stream

        params = list(inspect.signature(complete_stream).parameters.keys())
        assert "pool" not in params, (
            f"complete_stream accepts 'pool' parameter - violates ADR-006. "
            f"Adapter should own ServerPool internally. Current params: {params}"
        )

    def test_serverpool_not_imported_outside_adapters(self):
        """ServerPool must only be imported within adapters/ directory."""
        import subprocess

        result = subprocess.run(
            ["grep", "-r", "from prompt_prix.core import.*ServerPool", "prompt_prix/"],
            capture_output=True,
            text=True,
        )

        violations = []
        for line in result.stdout.strip().split("\n"):
            if line and "adapters/" not in line:
                violations.append(line)

        if violations:
            pytest.fail(
                f"ServerPool imported outside adapters/ - violates ADR-006:\n"
                + "\n".join(violations)
            )
