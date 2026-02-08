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


def _is_yaml_file(file_path) -> bool:
    """Check if file has YAML extension."""
    return Path(file_path).suffix.lower() in ['.yaml', '.yml']


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

    Supports JSON, JSONL, and promptfoo YAML formats.
    Returns validation message string. Starts with ✅ if valid, ❌ if not.
    """
    if file_obj is None:
        return "Upload a benchmark file (JSON/JSONL/YAML)"

    if _is_yaml_file(file_obj):
        from prompt_prix.benchmarks import PromptfooLoader
        valid, message = PromptfooLoader.validate(file_obj)
    else:
        from prompt_prix.benchmarks import CustomJSONLoader
        valid, message = CustomJSONLoader.validate(file_obj)

    return message


def get_test_ids(file_obj) -> list[str]:
    """Extract test IDs from benchmark file for dropdown population."""
    if file_obj is None:
        return []

    try:
        if _is_yaml_file(file_obj):
            from prompt_prix.benchmarks import PromptfooLoader
            tests = PromptfooLoader.load(file_obj)
        else:
            from prompt_prix.benchmarks import CustomJSONLoader
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
    judge_model: str = None,
    runs: int = 1,
    display_mode_str: str = "Symbols (✓/❌)",
    parallel_slots: int = 1,
    drift_threshold: float = 0.0
):
    """
    Run battery tests across selected models.

    Yields (status, grid_data) tuples for streaming UI updates.

    Args:
        runs: Number of runs per test (1 = standard, >1 = consistency testing)

    Note:
        Temperature is fixed at 0.0 for evaluation reproducibility.
        Judge model is reserved for future LLM-as-judge evaluation.
    """
    # Parse display mode for grid rendering
    from prompt_prix.battery import GridDisplayMode
    if "Latency" in display_mode_str:
        display_mode = GridDisplayMode.LATENCY
    else:
        display_mode = GridDisplayMode.SYMBOLS

    # Ensure runs is an int (Gradio slider returns float)
    runs = int(runs) if runs else 1

    # Clear any previous stop request so we can run again
    state.clear_stop()

    # Fail-fast validation
    if file_obj is None:
        yield "❌ No benchmark file uploaded", []
        return

    if not models_selected:
        yield "❌ No models selected", []
        return

    from prompt_prix.battery import BatteryRunner
    from prompt_prix.mcp.tools.list_models import list_models
    from prompt_prix.parsers import parse_servers_input
    from prompt_prix.handlers import _ensure_adapter_registered

    # Load test cases (auto-detect format by extension)
    try:
        if _is_yaml_file(file_obj):
            from prompt_prix.benchmarks import PromptfooLoader
            tests = PromptfooLoader.load(file_obj)
        else:
            from prompt_prix.benchmarks import CustomJSONLoader
            tests = CustomJSONLoader.load(file_obj)
    except Exception as e:
        yield f"❌ Failed to load tests: {e}", []
        return

    # Parse and validate servers
    servers = parse_servers_input(servers_text)
    if not servers:
        yield "❌ No servers configured", []
        return

    import logging
    logger = logging.getLogger(__name__)

    # Ensure parallel_slots is an int (Gradio slider returns float)
    parallel_slots = int(parallel_slots) if parallel_slots else 1

    # Register adapter with current servers before using MCP tools
    _ensure_adapter_registered(servers, parallel_slots=parallel_slots)

    # Validate models using MCP primitive (uses registry internally)
    result = await list_models()
    available = set(result["models"])
    missing = [m for m in models_selected if m not in available]
    if missing:
        yield f"❌ Models not available: {', '.join(missing)}", []
        return

    # All tests go through one runner — mode is handled by dispatch
    if runs > 1:
        # Multi-run consistency testing
        from prompt_prix.consistency import ConsistencyRunner
        logger.info(f"Consistency Run Config: Models={models_selected}, Runs={runs}")

        runner = ConsistencyRunner(
            tests=tests,
            models=models_selected,
            runs=runs,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            judge_model=judge_model,
            drift_threshold=drift_threshold
        )

        # Store state for later detail retrieval
        state.consistency_run = runner.state
        state.battery_run = None

        # Stream state updates to UI
        async for consistency_state in runner.run():
            grid = consistency_state.to_grid(display_mode)
            if consistency_state.phase == "judging":
                progress = f"⏳ Judging... ({consistency_state.judge_completed}/{consistency_state.judge_total})"
            elif consistency_state.judge_total > 0:
                progress = (
                    f"⏳ Running {runs}x... ({consistency_state.completed_runs}/{consistency_state.total_runs})"
                    f" | Judging ({consistency_state.judge_completed}/{consistency_state.judge_total})"
                )
            else:
                progress = f"⏳ Running {runs}x... ({consistency_state.completed_runs}/{consistency_state.total_runs})"
            yield progress, grid

        # Final status with consistency summary
        total_cells = consistency_state.total_count
        inconsistent = sum(
            1 for agg in consistency_state.aggregates.values()
            if agg.status.value == "inconsistent"
        )
        if inconsistent > 0:
            yield f"✅ Complete - {inconsistent}/{total_cells} cells inconsistent", grid
        else:
            yield f"✅ Complete - all {total_cells} cells consistent", grid

    else:
        # Standard single-run battery
        logger.info(f"Battery Run Config: Models={models_selected}")

        runner = BatteryRunner(
            tests=tests,
            models=models_selected,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            judge_model=judge_model,
            drift_threshold=drift_threshold
        )

        # Store state for later detail retrieval
        state.battery_run = runner.state
        state.consistency_run = None

        # Stream state updates to UI
        async for battery_state in runner.run():
            grid = battery_state.to_grid(display_mode)
            if battery_state.phase == "judging":
                progress = f"⏳ Judging... ({battery_state.judge_completed}/{battery_state.judge_total})"
            elif battery_state.judge_total > 0:
                progress = (
                    f"⏳ Running tests... ({battery_state.completed_count}/{battery_state.total_count})"
                    f" | Judging ({battery_state.judge_completed}/{battery_state.judge_total})"
                )
            else:
                progress = f"⏳ Running tests... ({battery_state.completed_count}/{battery_state.total_count})"
            yield progress, grid

        yield f"✅ Battery complete ({battery_state.completed_count} tests)", grid


def export_json():
    """Export battery results as JSON file."""
    # Support both single-run and multi-run modes
    if state.consistency_run:
        return _export_consistency_json()
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
                    "judge_latency_ms": result.judge_latency_ms,
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


def _export_consistency_json():
    """Export consistency run results as JSON."""
    run = state.consistency_run
    export_data = {
        "tests": run.tests,
        "models": run.models,
        "runs_total": run.runs_total,
        "aggregates": []
    }

    for test_id in run.tests:
        for model_id in run.models:
            agg = run.get_aggregate(test_id, model_id)
            if agg:
                export_data["aggregates"].append({
                    "test_id": agg.test_id,
                    "model_id": agg.model_id,
                    "status": agg.status.value,
                    "passes": agg.passes,
                    "total": agg.total,
                    "avg_latency_ms": agg.avg_latency_ms,
                    "results": [
                        {
                            "status": r.status.value,
                            "response": r.response,
                            "latency_ms": r.latency_ms,
                            "error": r.error,
                            "failure_reason": r.failure_reason
                        }
                        for r in agg.results
                    ]
                })

    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.json")

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    return f"✅ Exported {len(export_data['aggregates'])} cells ({run.runs_total} runs each)", gr.update(visible=False, value=filepath)


def export_csv():
    """Export battery results as CSV file."""
    import csv

    # Support both single-run and multi-run modes
    if state.consistency_run:
        return _export_consistency_csv()
    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    # Write to temp file with meaningful name
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.csv")

    row_count = 0
    with open(filepath, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["test_id", "model_id", "status", "latency_ms", "judge_latency_ms",
                         "error", "failure_reason", "response"])

        for test_id in state.battery_run.tests:
            for model_id in state.battery_run.models:
                result = state.battery_run.get_result(test_id, model_id)
                if result:
                    latency = f"{result.latency_ms:.0f}" if result.latency_ms else ""
                    judge_latency = f"{result.judge_latency_ms:.0f}" if result.judge_latency_ms else ""
                    writer.writerow([
                        result.test_id,
                        result.model_id,
                        result.status.value,
                        latency,
                        judge_latency,
                        result.error or "",
                        result.failure_reason or "",
                        result.response or ""
                    ])
                    row_count += 1

    return f"✅ Exported {row_count} results", gr.update(visible=False, value=filepath)


def _export_consistency_csv():
    """Export consistency run results as CSV."""
    import csv

    run = state.consistency_run
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.csv")

    row_count = 0
    with open(filepath, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["test_id", "model_id", "status", "passes", "total",
                         "avg_latency_ms", "run_statuses"])

        for test_id in run.tests:
            for model_id in run.models:
                agg = run.get_aggregate(test_id, model_id)
                if agg:
                    avg_lat = f"{agg.avg_latency_ms:.0f}" if agg.avg_latency_ms else ""
                    run_statuses = ",".join(r.status.value for r in agg.results)
                    writer.writerow([
                        agg.test_id,
                        agg.model_id,
                        agg.status.value,
                        agg.passes,
                        agg.total,
                        avg_lat,
                        run_statuses
                    ])
                    row_count += 1

    return f"✅ Exported {row_count} cells ({run.runs_total} runs each)", gr.update(visible=False, value=filepath)


def get_cell_detail(model: str, test: str) -> str:
    """Get response detail for a (model, test) cell."""
    from prompt_prix.battery import RunStatus

    if not model or not test:
        return "*Select a model and test to view the response*"

    # Check for consistency run first (multi-run mode)
    if state.consistency_run:
        return _get_consistency_cell_detail(model, test)

    if not state.battery_run:
        return "*No battery run available*"

    result = state.battery_run.get_result(test, model)
    if not result:
        return f"*No result for {model} × {test}*"

    return _format_single_result(result)


def _format_run_scores(result) -> str:
    """Format drift/judge scores for a single run result.

    Returns a string like '\\n**Drift:** 0.150\\n**Judge:** Passed (score: 8)'
    or '' if no scores are available.
    """
    parts = []
    if result.drift_score is not None:
        parts.append(f"**Drift:** {result.drift_score:.3f}")
    if result.judge_result:
        score = result.judge_result.get("score")
        score_str = f" (score: {score})" if score is not None else ""
        judge_latency = f" in {result.judge_latency_ms:.0f}ms" if result.judge_latency_ms else ""
        parts.append(f"**Judge:** Passed{score_str}{judge_latency}")
    if not parts:
        return ""
    return "\n" + "\n".join(parts)


def _get_consistency_cell_detail(model: str, test: str) -> str:
    """Get detail for a consistency run cell (multiple runs)."""
    from prompt_prix.battery import RunStatus
    from prompt_prix.consistency import ConsistencyStatus

    agg = state.consistency_run.get_aggregate(test, model)
    if not agg:
        return f"*No result for {model} × {test}*"

    # Header with aggregate stats
    status_emoji = agg.status_symbol
    status_name = agg.status.value.replace("_", " ").title()
    avg_latency = f"{agg.avg_latency_ms:.0f}ms" if agg.avg_latency_ms else "N/A"

    header = (
        f"## {status_emoji} {status_name}\n\n"
        f"**Pass Rate:** {agg.passes}/{agg.total} "
        f"({agg.passes * 100 // agg.total}%)\n\n"
        f"**Avg Latency:** {avg_latency}\n\n"
        f"---\n\n"
    )

    # Individual run details
    run_details = []
    for i, result in enumerate(agg.results, 1):
        status_sym = result.status_symbol
        latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
        scores = _format_run_scores(result)

        if result.status == RunStatus.ERROR:
            detail = f"### Run {i}: {status_sym} Error\n**Error:** {result.error}"
        elif result.status == RunStatus.SEMANTIC_FAILURE:
            reason = result.failure_reason or "Unknown"
            detail = (
                f"### Run {i}: {status_sym} Failed ({latency})\n"
                f"**Reason:** {reason}{scores}\n\n"
                f"```\n{result.response}\n```"
            )
        else:
            detail = (
                f"### Run {i}: {status_sym} Passed ({latency}){scores}\n\n"
                f"```\n{result.response}\n```"
            )
        if result.react_trace:
            detail += "\n\n" + _format_react_trace(result.react_trace)
        run_details.append(detail)

    return header + "\n\n".join(run_details)


def _format_react_trace(react_trace: dict) -> str:
    """Format a react trace for detail view display."""
    parts = []
    completed = react_trace.get("completed", False)
    total = react_trace.get("total_iterations", 0)
    valid = react_trace.get("valid_iterations", 0)
    invalid = react_trace.get("invalid_iterations", 0)
    cycle = react_trace.get("cycle_detected", False)
    reason = react_trace.get("termination_reason")

    status = "Completed" if completed else f"Incomplete ({reason})"
    parts.append(f"**ReAct Loop:** {status}")
    parts.append(f"**Iterations:** {valid} valid / {total} total" +
                 (f" ({invalid} invalid)" if invalid else ""))
    if cycle:
        parts.append("**Cycle Detected:** Yes")

    iterations = react_trace.get("iterations", [])
    if iterations:
        parts.append("\n---\n")
        for i, step in enumerate(iterations, 1):
            tc = step.get("tool_call", {})
            name = tc.get("name", "?")
            args = tc.get("args", {})
            obs = step.get("observation", "")
            success = step.get("success", True)
            latency = step.get("latency_ms")

            status_mark = "+" if success else "-"
            lat_str = f" ({latency:.0f}ms)" if latency else ""
            parts.append(f"**Step {i}:** `{name}({args})`{lat_str}")
            parts.append(f"```\n[{status_mark}] {obs[:200]}\n```")

    return "\n".join(parts)


def _format_single_result(result) -> str:
    """Format a single RunResult for display."""
    from prompt_prix.battery import RunStatus

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
            judge_latency = f" in {result.judge_latency_ms:.0f}ms" if result.judge_latency_ms else ""
            judge_info = f"\n\n**Judged by:** LLM{score_str}{judge_latency}"
        drift_info = f"\n\n**Drift:** {result.drift_score:.3f}" if result.drift_score is not None else ""
        base = (
            f"**Status:** ❌ Semantic Failure\n\n"
            f"**Reason:** {failure}{judge_info}{drift_info}\n\n"
            f"**Latency:** {latency}\n\n"
            f"---\n\n{result.response}"
        )
        if result.react_trace:
            base += "\n\n---\n\n" + _format_react_trace(result.react_trace)
        return base

    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
    judge_info = ""
    if result.judge_result:
        judge_latency = f" (judged in {result.judge_latency_ms:.0f}ms)" if result.judge_latency_ms else ""
        judge_info = f"\n\n**Judge:** ✓ Passed{judge_latency}"
    drift_info = f"\n\n**Drift:** {result.drift_score:.3f}" if result.drift_score is not None else ""
    base = f"**Status:** ✓ Completed\n\n**Latency:** {latency}{judge_info}{drift_info}\n\n---\n\n{result.response}"
    if result.react_trace:
        base += "\n\n---\n\n" + _format_react_trace(result.react_trace)
    return base


def refresh_grid(display_mode_str: str = None) -> list:
    """Refresh the battery grid with the selected display mode.

    Stores display mode in global state to prevent reset on grid cell clicks.
    """
    from prompt_prix.battery import GridDisplayMode

    # Store in global state if provided, otherwise use stored value
    if display_mode_str:
        state.battery_display_mode = display_mode_str
    else:
        display_mode_str = state.battery_display_mode

    if "Latency" in display_mode_str:
        mode = GridDisplayMode.LATENCY
    else:
        mode = GridDisplayMode.SYMBOLS

    # Check which run state is active
    if state.consistency_run:
        return state.consistency_run.to_grid(mode)
    elif state.battery_run:
        return state.battery_run.to_grid(mode)
    else:
        return []


def export_grid_image():
    """Export battery results grid as PNG image using PIL."""
    from PIL import Image, ImageDraw, ImageFont
    from prompt_prix.battery import RunStatus

    # Support both single-run and multi-run modes
    run_state = state.consistency_run or state.battery_run
    if not run_state:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    tests = run_state.tests
    models = run_state.models

    if not tests or not models:
        return "❌ No results to render", gr.update(visible=False, value=None)

    # Layout constants
    cell_width = 120
    cell_height = 30
    header_col_width = 200
    padding = 5

    # Colors
    colors = {
        'header_bg': '#e0e0e0',
        'pass': '#d1fae5',
        'fail': '#fee2e2',
        'error': '#fef3c7',
        'pending': '#f3f4f6',
        'border': '#9ca3af',
        'text': '#1f2937',
    }

    # Calculate image dimensions
    img_width = header_col_width + (len(models) * cell_width)
    img_height = cell_height + (len(tests) * cell_height)

    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    # Load font with fallback
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except (IOError, OSError):
            font = ImageFont.load_default()

    def draw_cell(x, y, width, text, bg_color):
        """Draw a single cell with centered text."""
        draw.rectangle([x, y, x + width - 1, y + cell_height - 1],
                       fill=bg_color, outline=colors['border'])
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (width - text_width) // 2
        draw.text((text_x, y + padding), text, fill=colors['text'], font=font)

    # Draw header row (model names)
    draw_cell(0, 0, header_col_width, "Test", colors['header_bg'])
    for col, model_id in enumerate(models):
        x = header_col_width + (col * cell_width)
        # Truncate model name to fit
        display_name = model_id[:12] if len(model_id) > 12 else model_id
        draw_cell(x, 0, cell_width, display_name, colors['header_bg'])

    # Add color for inconsistent results
    colors['inconsistent'] = '#e9d5ff'  # Light purple

    # Draw data rows
    for row, test_id in enumerate(tests):
        y = cell_height + (row * cell_height)

        # Test ID column (truncate to fit)
        display_test = test_id[:25] if len(test_id) > 25 else test_id
        draw_cell(0, y, header_col_width, display_test, colors['header_bg'])

        # Result cells
        for col, model_id in enumerate(models):
            x = header_col_width + (col * cell_width)

            # Handle both consistency runs and battery runs
            if state.consistency_run:
                from prompt_prix.consistency import ConsistencyStatus
                agg = state.consistency_run.get_aggregate(test_id, model_id)
                if agg:
                    status = agg.status
                    if status == ConsistencyStatus.CONSISTENT_PASS:
                        bg_color = colors['pass']
                        cell_text = f"OK {agg.pass_rate_display}"
                    elif status == ConsistencyStatus.CONSISTENT_FAIL:
                        bg_color = colors['fail']
                        cell_text = f"FAIL {agg.pass_rate_display}"
                    elif status == ConsistencyStatus.INCONSISTENT:
                        bg_color = colors['inconsistent']
                        cell_text = f"~{agg.pass_rate_display}"
                    else:
                        bg_color = colors['pending']
                        cell_text = "..."
                else:
                    bg_color = colors['pending']
                    cell_text = "..."
            else:
                result = state.battery_run.get_result(test_id, model_id)
                if result:
                    if result.status == RunStatus.COMPLETED:
                        bg_color = colors['pass']
                        latency_s = f"{result.latency_ms / 1000:.1f}" if result.latency_ms else "?"
                        cell_text = f"OK {latency_s}s"
                    elif result.status == RunStatus.SEMANTIC_FAILURE:
                        bg_color = colors['fail']
                        latency_s = f"{result.latency_ms / 1000:.1f}" if result.latency_ms else "?"
                        cell_text = f"FAIL {latency_s}s"
                    elif result.status == RunStatus.ERROR:
                        bg_color = colors['error']
                        cell_text = "ERR"
                    else:
                        bg_color = colors['pending']
                        cell_text = "..."
                else:
                    bg_color = colors['pending']
                    cell_text = "..."

            draw_cell(x, y, cell_width, cell_text, bg_color)

    # Save to temp file
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.png")

    try:
        img.save(filepath, 'PNG')
    except Exception as e:
        return f"❌ Failed to save image: {e}", gr.update(visible=False, value=None)

    return f"✅ Exported grid image", gr.update(visible=False, value=filepath)


def handle_cell_select(evt: gr.SelectData) -> tuple:
    """Handle grid cell selection, return (dialog_visible, detail_content).

    ADR-009: Click a cell to see response detail in dismissible dialog.
    """
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"handle_cell_select called with evt.index={evt.index}, evt.value={evt.value}")

    # Determine which run state is active
    run_state = state.consistency_run or state.battery_run
    if not run_state:
        logger.warning("No run state available")
        return gr.update(visible=False), "*No battery run available*"

    row, col = evt.index
    logger.info(f"Cell select: row={row}, col={col}, value={evt.value}")

    # Col 0 is test ID column - don't show detail for that
    if col == 0:
        logger.info("Clicked on test ID column, hiding dialog")
        return gr.update(visible=False), ""

    # Map indices to identifiers
    # Note: Gradio DataFrame indices are 0-based for data rows (header not included)
    try:
        test_id = run_state.tests[row]
        model_id = run_state.models[col - 1]  # col 0 is Test column
        logger.info(f"Mapped to: test_id={test_id}, model_id={model_id}")
    except IndexError:
        logger.warning(f"Invalid cell: row={row}, col={col}, tests={len(run_state.tests)}, models={len(run_state.models)}")
        return gr.update(visible=False), "*Invalid cell selection*"

    detail = get_cell_detail(model_id, test_id)
    logger.info(f"Got detail content, length={len(detail)}, showing dialog")
    return gr.update(visible=True), detail


def recalculate_drift(drift_threshold: float, display_mode_str: str = None) -> list:
    """
    Recalculate pass/fail for stored results based on new drift threshold.

    Called when the drift_threshold slider changes. No re-inference needed —
    drift_score is already stored on each RunResult.
    """
    from prompt_prix.battery import GridDisplayMode

    if display_mode_str and "Latency" in display_mode_str:
        mode = GridDisplayMode.LATENCY
    else:
        mode = GridDisplayMode.SYMBOLS

    if state.consistency_run:
        state.consistency_run.recalculate_drift_threshold(drift_threshold)
        return state.consistency_run.to_grid(mode)
    elif state.battery_run:
        state.battery_run.recalculate_drift_threshold(drift_threshold)
        return state.battery_run.to_grid(mode)
    else:
        return []
