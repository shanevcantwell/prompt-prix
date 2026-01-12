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
from prompt_prix.handlers import _init_pool_and_validate
from prompt_prix.parsers import parse_prefixed_model


def _strip_gpu_prefix(model_id: str) -> str:
    """Strip GPU prefix for API calls.

    Converts "0: llama-3.2" -> "llama-3.2" for server/API compatibility.
    Prefixed IDs are used internally for uniqueness (same model on multiple GPUs),
    but APIs expect the raw model name.
    """
    if ': ' in model_id:
        prefix, rest = model_id.split(': ', 1)
        if prefix.isdigit():
            return rest
    return model_id


def _get_export_basename() -> str:
    """Get base name for export files from source filename.

    Includes timestamp to avoid browser/Gradio caching issues when
    exporting multiple runs of the same benchmark file.
    """
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
    YAML files are auto-converted to JSON for Battery consumption.

    Returns validation message string. Starts with ✅ if valid, ❌ if not.
    """
    if file_obj is None:
        return "Upload a benchmark file"

    file_path = Path(file_obj)

    # Auto-detect and convert YAML (promptfoo) format
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        return _validate_and_convert_yaml(file_obj)

    # JSON/JSONL format
    from prompt_prix.benchmarks import CustomJSONLoader
    valid, message = CustomJSONLoader.validate(file_obj)
    return message


def _validate_and_convert_yaml(file_obj) -> str:
    """
    Validate promptfoo YAML and convert to JSON for Battery.

    Stores converted JSON path in state for run_handler to use.
    """
    try:
        from prompt_prix.promptfoo import parse_tests

        tests = parse_tests(Path(file_obj))

        if not tests:
            return "❌ No tests found in promptfoo config"

        # Convert to JSON format for Battery consumption
        tests_data = {
            "test_suite": f"promptfoo_{Path(file_obj).stem}",
            "prompts": [
                {
                    "id": t.id,
                    "name": t.name,
                    "user": t.user,
                    "system": t.system,
                    "pass_criteria": t.pass_criteria,
                }
                for t in tests
            ]
        }

        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            prefix='promptfoo_converted_',
            delete=False,
            encoding='utf-8'
        )
        json.dump(tests_data, temp_file, indent=2)
        temp_file.close()

        # Store converted path for run_handler
        state.battery_converted_file = temp_file.name
        state.battery_source_file = file_obj

        return f"✅ Valid promptfoo config: {len(tests)} tests"

    except Exception as e:
        return f"❌ Failed to parse: {str(e)}"


def import_promptfoo(file_obj) -> tuple[str, str | None, list[str]]:
    """
    Import tests from a promptfoo YAML config file.

    Converts promptfoo format to internal TestCase format and writes
    to a temp JSON file for Battery to use.

    Returns:
        (validation_message, temp_file_path, test_ids)
    """
    if file_obj is None:
        return "Select a promptfoo YAML config file", None, []

    try:
        from prompt_prix.promptfoo import parse_tests

        tests = parse_tests(Path(file_obj))

        if not tests:
            return "❌ No tests found in promptfoo config", None, []

        # Convert to JSON format for Battery consumption
        tests_data = {
            "test_suite": f"promptfoo_import_{Path(file_obj).stem}",
            "prompts": [
                {
                    "id": t.id,
                    "name": t.name,
                    "user": t.user,
                    "system": t.system,
                    "pass_criteria": t.pass_criteria,
                }
                for t in tests
            ]
        }

        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            prefix='promptfoo_import_',
            delete=False,
            encoding='utf-8'
        )
        json.dump(tests_data, temp_file, indent=2)
        temp_file.close()

        # Store for export naming
        state.battery_source_file = temp_file.name

        test_ids = [t.id for t in tests]
        return f"✅ Imported {len(tests)} tests from promptfoo config", temp_file.name, test_ids

    except Exception as e:
        return f"❌ Failed to import: {str(e)}", None, []


def get_test_ids(file_obj) -> list[str]:
    """Extract test IDs from benchmark file for dropdown population."""
    if file_obj is None:
        return []

    file_path = Path(file_obj)

    # For YAML files, use the converted JSON
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        if state.battery_converted_file:
            file_obj = state.battery_converted_file
        else:
            # Not yet converted - try to parse directly
            try:
                from prompt_prix.promptfoo import parse_tests
                tests = parse_tests(file_path)
                return [t.id for t in tests]
            except Exception:
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
    judge_model: str | None = None
):
    """
    Run battery tests across selected models.

    Yields (status, grid_data) tuples for streaming UI updates.

    Args:
        file_obj: Benchmark file (JSON/JSONL/YAML)
        models_selected: List of model IDs to test
        servers_text: Server URLs (one per line)
        timeout: Timeout in seconds per request
        max_tokens: Max tokens per response
        system_prompt: Optional system prompt override
        judge_model: Optional model ID for LLM-as-judge validation

    Note: Temperature is not passed; LM Studio uses per-model defaults.
    """
    import pandas as pd
    from prompt_prix.benchmarks import CustomJSONLoader
    from prompt_prix.adapters import LMStudioAdapter
    from prompt_prix.battery import BatteryRunner

    # Fail-fast validation
    if file_obj is None:
        yield "❌ No benchmark file uploaded", pd.DataFrame()
        return

    if not models_selected:
        yield "❌ No models selected", pd.DataFrame()
        return

    # Prevent concurrent battery runs (fixes state.battery_run race condition)
    if not state.start_battery_run():
        yield "❌ Battery run already in progress - wait for it to complete or click Stop", pd.DataFrame()
        return

    # Clear any previous stop request so we can run again
    state.clear_stop()

    # For YAML files, use the converted JSON
    actual_file = file_obj
    if Path(file_obj).suffix.lower() in ['.yaml', '.yml']:
        if state.battery_converted_file:
            actual_file = state.battery_converted_file
        else:
            state.end_battery_run()  # Release lock on early return
            yield "❌ YAML file not converted - re-upload file", pd.DataFrame()
            return

    # Load test cases
    try:
        tests = CustomJSONLoader.load(actual_file)
    except Exception as e:
        state.end_battery_run()  # Release lock on load failure
        yield f"❌ Failed to load tests: {e}", pd.DataFrame()
        return

    # Parse prefixed selections and build server hints
    # Key insight: prefixed ID (e.g., "0: llama-3.2") is the unique identifier
    # Same model on multiple GPUs must remain distinct throughout the system
    hints = {}
    prefixed_models = []
    stripped_for_validation = set()  # Unique stripped names for pool validation
    for selection in models_selected:
        idx, stripped_id = parse_prefixed_model(selection)
        url = state.get_server_url(idx)
        if url:
            hints[selection] = url  # Use full prefixed ID as key (no collision)
        prefixed_models.append(selection)  # Keep full prefixed ID
        stripped_for_validation.add(stripped_id)

    state.set_server_hints(hints)

    # Validate servers and models (use stripped model IDs for server lookup)
    pool, error = await _init_pool_and_validate(servers_text, list(stripped_for_validation))
    if error:
        state.end_battery_run()  # Release lock on validation failure
        yield error, pd.DataFrame()
        return

    adapter = LMStudioAdapter(pool)

    # Clear previous state to avoid stale columns (#81)
    state.battery_run = None

    # Create and run battery (temperature omitted - use per-model defaults)
    # Pass server hints for orchestrated fan-out dispatch (GPU prefix routing)
    # Use prefixed_models to keep "0: llama" and "1: llama" as distinct identifiers
    runner = BatteryRunner(
        adapter=adapter,
        tests=tests,
        models=prefixed_models,
        server_hints=hints,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        judge_model=judge_model if judge_model else None
    )

    # Store state for later detail retrieval
    state.battery_run = runner.state

    # Yield initial empty grid with correct columns before starting
    initial_headers = ["Test"] + prefixed_models
    initial_rows = [[t.id] + ["—"] * len(prefixed_models) for t in tests]
    initial_grid = pd.DataFrame(initial_rows, columns=initial_headers)
    yield "Starting...", initial_grid

    try:
        # Stream state updates to UI
        async for battery_state in runner.run():
            grid = battery_state.to_grid()
            progress = f"⏳ Running... ({battery_state.completed_count}/{battery_state.total_count})"
            yield progress, grid

        # Final status
        yield f"✅ Battery complete ({battery_state.completed_count} tests)", grid
    finally:
        # Always release lock when run completes (success, error, or cancellation)
        state.end_battery_run()


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
    """Export battery results as CSV file.

    Note: Uses csv module with proper quoting for reliable cross-platform behavior.
    """
    import csv

    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    # Write to temp file with meaningful name
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.csv")

    # Use csv module with proper quoting and newline handling
    # newline='' is required for csv module on Windows to avoid double line endings
    with open(filepath, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["test_id", "model_id", "status", "latency_ms", "response"])

        row_count = 0
        for test_id in state.battery_run.tests:
            for model_id in state.battery_run.models:
                result = state.battery_run.get_result(test_id, model_id)
                if result:
                    response = result.response or ""
                    latency = f"{result.latency_ms:.0f}" if result.latency_ms else ""
                    writer.writerow([
                        test_id,
                        model_id,
                        result.status.value,
                        latency,
                        response
                    ])
                    row_count += 1

    return f"✅ Exported {row_count} results", gr.update(visible=False, value=filepath)


def export_grid_image():
    """Export battery grid as PNG image.

    Renders the grid using Pillow with colored cells for pass/fail status.
    """
    from PIL import Image, ImageDraw, ImageFont

    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    run = state.battery_run
    models = run.models
    tests = run.tests

    if not models or not tests:
        return "❌ No results to render", gr.update(visible=False, value=None)

    # Layout constants
    cell_width = 120  # Wider to fit latency text
    cell_height = 30
    header_col_width = 200  # First column for test names
    padding = 5

    # Calculate dimensions
    img_width = header_col_width + (len(models) * cell_width)
    img_height = cell_height + (len(tests) * cell_height)  # Header row + test rows

    # Create image with white background
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Colors
    colors = {
        'header_bg': '#e0e0e0',
        'pass': '#d1fae5',      # Light green
        'fail': '#fee2e2',      # Light red
        'error': '#fef3c7',     # Light yellow
        'pending': '#f3f4f6',   # Light gray
        'border': '#9ca3af',
        'text': '#1f2937',
    }

    def truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 2] + ".."

    # Draw header row (model names)
    x = header_col_width
    for model_id in models:
        draw.rectangle([x, 0, x + cell_width - 1, cell_height - 1],
                       fill=colors['header_bg'], outline=colors['border'])
        # Strip GPU prefix for display (e.g., "0: llama-3.2" -> "llama-3.2")
        stripped_name = _strip_gpu_prefix(model_id)
        display_name = truncate_text(stripped_name.split('/')[-1], 12)
        draw.text((x + padding, padding), display_name, fill=colors['text'], font=font)
        x += cell_width

    # Draw first column header
    draw.rectangle([0, 0, header_col_width - 1, cell_height - 1],
                   fill=colors['header_bg'], outline=colors['border'])
    draw.text((padding, padding), "Test", fill=colors['text'], font=font)

    # Draw grid rows
    y = cell_height
    for test_id in tests:
        # Test name column
        draw.rectangle([0, y, header_col_width - 1, y + cell_height - 1],
                       fill=colors['header_bg'], outline=colors['border'])
        display_test = truncate_text(test_id, 25)
        draw.text((padding, y + padding), display_test, fill=colors['text'], font=font)

        # Result cells
        x = header_col_width
        for model_id in models:
            result = run.get_result(test_id, model_id)

            if result:
                status = result.status.value
                # Format latency if available
                latency_str = ""
                if result.latency_ms is not None:
                    latency_str = f" {result.latency_ms / 1000:.1f}s"

                # Use ASCII symbols that render in any font
                if status == "completed":
                    bg_color = colors['pass']
                    cell_text = f"OK{latency_str}"
                elif status == "semantic_failure":
                    bg_color = colors['fail']
                    cell_text = f"FAIL{latency_str}"
                elif status == "error":
                    bg_color = colors['error']
                    cell_text = "ERR"
                else:
                    bg_color = colors['pending']
                    cell_text = "..."
            else:
                bg_color = colors['pending']
                cell_text = "..."

            draw.rectangle([x, y, x + cell_width - 1, y + cell_height - 1],
                           fill=bg_color, outline=colors['border'])
            # Center the text
            text_bbox = draw.textbbox((0, 0), cell_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (cell_width - text_width) // 2
            draw.text((text_x, y + padding), cell_text, fill=colors['text'], font=font)
            x += cell_width

        y += cell_height

    # Save to temp file
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.png")
    img.save(filepath, 'PNG')

    # Verify file was created
    if not os.path.exists(filepath):
        return "❌ Failed to save image file", gr.update(visible=False, value=None)

    return f"✅ Exported: {filepath}", gr.update(visible=False, value=filepath)


def get_cell_detail(model: str, test: str) -> str:
    """Get response detail for a (model, test) cell.

    Args:
        model: Model ID, possibly with GPU prefix like '0: model-name'
        test: Test ID

    Note:
        The model dropdown may contain GPU-prefixed values (e.g., '0: lfm2-1.2b-tool')
        but battery results are keyed by stripped model ID. We strip the prefix before lookup.
    """
    from prompt_prix.battery import TestStatus

    if not state.battery_run:
        return "*No battery run available*"

    if not model or not test:
        return "*Select a model and test to view the response*"

    # Strip GPU prefix if present (dropdown shows '0: model-name', results keyed by 'model-name')
    if ": " in model:
        _, model = parse_prefixed_model(model)

    result = state.battery_run.get_result(test, model)
    if not result:
        return f"*No result for {model} × {test}*"

    if result.status == TestStatus.ERROR:
        return f"**Status:** ❌ Error\n\n**Error:** {result.error}"

    if result.status == TestStatus.PENDING:
        return f"**Status:** — Pending"

    if result.status == TestStatus.RUNNING:
        return f"**Status:** ⏳ Running..."

    if result.status == TestStatus.SEMANTIC_FAILURE:
        latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
        failure = result.failure_reason or "Unknown semantic failure"
        return (
            f"**Status:** ⚠ Semantic Failure\n\n"
            f"**Reason:** {failure}\n\n"
            f"**Latency:** {latency}\n\n"
            f"---\n\n{result.response}"
        )

    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
    return f"**Status:** ✓ Completed\n\n**Latency:** {latency}\n\n---\n\n{result.response}"


def refresh_grid(display_mode_str: str):
    """Refresh the battery grid with the selected display mode.

    Returns:
        pandas DataFrame with grid data, or empty DataFrame if no results.
    """
    import pandas as pd
    from prompt_prix.battery import GridDisplayMode

    if not state.battery_run:
        return pd.DataFrame()

    if "Latency" in display_mode_str:
        mode = GridDisplayMode.LATENCY
    else:
        mode = GridDisplayMode.SYMBOLS

    return state.battery_run.to_grid(mode)
