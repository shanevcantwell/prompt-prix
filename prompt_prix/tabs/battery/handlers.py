"""
Battery tab event handlers.

Handles benchmark test suite execution, exports, and result display.
"""

import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from prompt_prix import state


def _get_export_basename() -> str:
    """Get base name for export files from source filename with timestamp."""
    import time
    timestamp = int(time.time())
    if state.battery_source_file:
        stem = Path(state.battery_source_file).stem
        return f"{stem}_results_{timestamp}"
    return f"battery_results_{timestamp}"


def validate_file(file_obj) -> str:
    """
    Validate benchmark file before enabling Run button.

    Returns validation message string. Starts with ✅ if valid, ❌ if not.
    """
    if file_obj is None:
        return "Upload a benchmark JSON file"

    from prompt_prix.benchmarks import CustomJSONLoader

    valid, message = CustomJSONLoader.validate(file_obj)
    return message


def get_test_ids(file_obj) -> list[str]:
    """Extract test IDs from benchmark file for dropdown population."""
    if file_obj is None:
        return []

    from prompt_prix.benchmarks import CustomJSONLoader

    try:
        tests = CustomJSONLoader.load(file_obj)
        return [t.id for t in tests]
    except Exception:
        return []


async def run_handler(
    file_obj,
    models_selected: list[str],
    servers_text: str,
    timeout: int,
    max_tokens: int,
    system_prompt: str,
    judge_model: str = None
):
    """
    Run battery tests across selected models.

    Yields (status, grid_data) tuples for streaming UI updates.

    Note:
        Temperature is fixed at 0.0 for evaluation reproducibility.
        Judge model is reserved for future LLM-as-judge evaluation.
    """
    # Clear any previous stop request so we can run again
    state.clear_stop()

    # Fail-fast validation
    if file_obj is None:
        yield "❌ No benchmark file uploaded", []
        return

    if not models_selected:
        yield "❌ No models selected", []
        return

    from prompt_prix.benchmarks import CustomJSONLoader
    from prompt_prix.battery import BatteryRunner
    from prompt_prix.mcp.tools.list_models import list_models
    from prompt_prix.parsers import parse_servers_input
    from prompt_prix.handlers import _ensure_adapter_registered

    # Load test cases
    try:
        tests = CustomJSONLoader.load(file_obj)
    except Exception as e:
        yield f"❌ Failed to load tests: {e}", []
        return

    # Parse and validate servers
    servers = parse_servers_input(servers_text)
    if not servers:
        yield "❌ No servers configured", []
        return

    # Register adapter with current servers before using MCP tools
    _ensure_adapter_registered(servers)

    # Validate models using MCP primitive (uses registry internally)
    # Models may have server affinity prefix (e.g., "0:model_name")
    from prompt_prix.server_affinity import strip_server_prefix, extract_server_indices

    result = await list_models()
    available = set(result["models"])
    actual_names = [strip_server_prefix(m) for m in models_selected]
    missing = [m for m in actual_names if m not in available]
    if missing:
        yield f"❌ Models not available: {', '.join(missing)}", []
        return

    # Create and run battery (temperature=0.0 for reproducibility)
    # BatteryRunner calls MCP tools internally - doesn't need servers
    # max_concurrent = number of unique servers (from affinity prefixes)
    # This enables parallel execution across GPUs while keeping each GPU serialized
    server_indices = extract_server_indices(models_selected)
    max_concurrent = len(server_indices) if server_indices else 1

    runner = BatteryRunner(
        tests=tests,
        models=models_selected,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        max_concurrent=max_concurrent,
        judge_model=judge_model
    )

    # Store state for later detail retrieval
    state.battery_run = runner.state

    # Stream state updates to UI
    async for battery_state in runner.run():
        grid = battery_state.to_grid()
        progress = f"⏳ Running... ({battery_state.completed_count}/{battery_state.total_count})"
        yield progress, grid

    # Final status
    yield f"✅ Battery complete ({battery_state.completed_count} tests)", grid


def export_json():
    """Export battery results as JSON file."""
    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    export_data = {
        "tests": state.battery_run.tests,
        "models": state.battery_run.models,
        "results": []
    }

    for test_id in state.battery_run.tests:
        for model_id in state.battery_run.models:
            result = state.battery_run.get_result(test_id, model_id)
            if result:
                export_data["results"].append({
                    "test_id": result.test_id,
                    "model_id": result.model_id,
                    "status": result.status.value,
                    "response": result.response,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                    "failure_reason": result.failure_reason
                })

    # Write to temp file with meaningful name
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.json")

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    return f"✅ Exported {len(export_data['results'])} results", gr.update(visible=False, value=filepath)


def export_csv():
    """Export battery results as CSV file."""
    import csv

    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    # Write to temp file with meaningful name
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.csv")

    row_count = 0
    with open(filepath, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["test_id", "model_id", "status", "latency_ms",
                         "error", "failure_reason", "response"])

        for test_id in state.battery_run.tests:
            for model_id in state.battery_run.models:
                result = state.battery_run.get_result(test_id, model_id)
                if result:
                    latency = f"{result.latency_ms:.0f}" if result.latency_ms else ""
                    writer.writerow([
                        result.test_id,
                        result.model_id,
                        result.status.value,
                        latency,
                        result.error or "",
                        result.failure_reason or "",
                        result.response or ""
                    ])
                    row_count += 1

    return f"✅ Exported {row_count} results", gr.update(visible=False, value=filepath)


def get_cell_detail(model: str, test: str) -> str:
    """Get response detail for a (model, test) cell."""
    from prompt_prix.battery import RunStatus

    if not state.battery_run:
        return "*No battery run available*"

    if not model or not test:
        return "*Select a model and test to view the response*"

    result = state.battery_run.get_result(test, model)
    if not result:
        return f"*No result for {model} × {test}*"

    if result.status == RunStatus.ERROR:
        return f"**Status:** ⚠ Error\n\n**Error:** {result.error}"

    if result.status == RunStatus.PENDING:
        return f"**Status:** — Pending"

    if result.status == RunStatus.RUNNING:
        return f"**Status:** ⏳ Running..."

    if result.status == RunStatus.SEMANTIC_FAILURE:
        latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
        failure = result.failure_reason or "Unknown semantic failure"
        judge_info = ""
        if result.judge_result:
            score = result.judge_result.get("score")
            score_str = f" (score: {score})" if score is not None else ""
            judge_info = f"\n\n**Judged by:** LLM{score_str}"
        return (
            f"**Status:** ❌ Semantic Failure\n\n"
            f"**Reason:** {failure}{judge_info}\n\n"
            f"**Latency:** {latency}\n\n"
            f"---\n\n{result.response}"
        )

    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
    return f"**Status:** ✓ Completed\n\n**Latency:** {latency}\n\n---\n\n{result.response}"


def refresh_grid(display_mode_str: str) -> list:
    """Refresh the battery grid with the selected display mode."""
    from prompt_prix.battery import GridDisplayMode

    if not state.battery_run:
        return []

    if "Latency" in display_mode_str:
        mode = GridDisplayMode.LATENCY
    else:
        mode = GridDisplayMode.SYMBOLS

    return state.battery_run.to_grid(mode)


def export_grid_image():
    """Export battery results grid as PNG image.

    Note: Image export requires additional dependencies (matplotlib or similar).
    This is a placeholder for future implementation.
    """
    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    # TODO: Implement grid-to-image conversion
    # For now, return a message indicating the feature is not yet implemented
    return "⚠️ Image export not yet implemented", gr.update(visible=False, value=None)
