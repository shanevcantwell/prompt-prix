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

    # Register adapter with current servers before using MCP tools
    _ensure_adapter_registered(servers)

    # Validate models using MCP primitive (uses registry internally)
    result = await list_models()
    available = set(result["models"])
    missing = [m for m in models_selected if m not in available]
    if missing:
        yield f"❌ Models not available: {', '.join(missing)}", []
        return

    # Pre-flight check: warn if tests have criteria but no judge model (#103)
    tests_with_criteria = [t for t in tests if t.pass_criteria or t.fail_criteria]
    if tests_with_criteria and not judge_model:
        yield (
            f"⚠️ {len(tests_with_criteria)} tests have pass/fail criteria but no judge model selected. "
            "Results will show ✓ but won't be evaluated against criteria.",
            []
        )

    # Create and run battery (temperature=0.0 for reproducibility)
    # BatteryRunner calls MCP tools internally - doesn't need servers
    # Adapter handles concurrency via per-server locks
    logger.info(f"Battery Run Config: Models={models_selected}")

    runner = BatteryRunner(
        tests=tests,
        models=models_selected,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        judge_model=judge_model
    )

    # Store state for later detail retrieval
    state.battery_run = runner.state

    # Stream state updates to UI
    async for battery_state in runner.run():
        grid = battery_state.to_grid()
        # Show different status based on phase (ADR-008 two-phase execution)
        if battery_state.phase == "judging":
            progress = f"⏳ Judging... ({battery_state.judge_completed}/{battery_state.judge_total})"
        else:
            progress = f"⏳ Running tests... ({battery_state.completed_count}/{battery_state.total_count})"
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
            judge_latency = f" in {result.judge_latency_ms:.0f}ms" if result.judge_latency_ms else ""
            judge_info = f"\n\n**Judged by:** LLM{score_str}{judge_latency}"
        return (
            f"**Status:** ❌ Semantic Failure\n\n"
            f"**Reason:** {failure}{judge_info}\n\n"
            f"**Latency:** {latency}\n\n"
            f"---\n\n{result.response}"
        )

    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
    judge_info = ""
    if result.judge_result:
        judge_latency = f" (judged in {result.judge_latency_ms:.0f}ms)" if result.judge_latency_ms else ""
        judge_info = f"\n\n**Judge:** ✓ Passed{judge_latency}"
    return f"**Status:** ✓ Completed\n\n**Latency:** {latency}{judge_info}\n\n---\n\n{result.response}"


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
    """Export battery results grid as PNG image using PIL."""
    from PIL import Image, ImageDraw, ImageFont
    from prompt_prix.battery import RunStatus

    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    tests = state.battery_run.tests
    models = state.battery_run.models

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

    # Draw data rows
    for row, test_id in enumerate(tests):
        y = cell_height + (row * cell_height)

        # Test ID column (truncate to fit)
        display_test = test_id[:25] if len(test_id) > 25 else test_id
        draw_cell(0, y, header_col_width, display_test, colors['header_bg'])

        # Result cells
        for col, model_id in enumerate(models):
            x = header_col_width + (col * cell_width)
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
    if not state.battery_run:
        return gr.update(visible=False), "*No battery run available*"

    row, col = evt.index

    # Row 0 is header, col 0 is test ID column
    if row == 0 or col == 0:
        return gr.update(visible=False), ""

    # Map indices to identifiers (adjust for header row and test ID column)
    try:
        test_id = state.battery_run.tests[row - 1]
        model_id = state.battery_run.models[col - 1]
    except IndexError:
        return gr.update(visible=False), "*Invalid cell selection*"

    detail = get_cell_detail(model_id, test_id)
    return gr.update(visible=True), detail
