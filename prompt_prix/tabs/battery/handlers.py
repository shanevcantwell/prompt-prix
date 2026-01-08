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


def _get_export_basename() -> str:
    """Get base name for export files from source filename."""
    if state.battery_source_file:
        stem = Path(state.battery_source_file).stem
        return f"{stem}_results"
    return "battery_results"


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
    temperature: float,
    timeout: int,
    max_tokens: int,
    system_prompt: str
):
    """
    Run battery tests across selected models.

    Yields (status, grid_data) tuples for streaming UI updates.
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
    from prompt_prix.adapters import LMStudioAdapter
    from prompt_prix.battery import BatteryRunner

    # Load test cases
    try:
        tests = CustomJSONLoader.load(file_obj)
    except Exception as e:
        yield f"❌ Failed to load tests: {e}", []
        return

    # Parse prefixed selections and build server hints
    hints = {}
    stripped_models = []
    for selection in models_selected:
        idx, model_id = parse_prefixed_model(selection)
        url = state.get_server_url(idx)
        if url:
            hints[model_id] = url
        stripped_models.append(model_id)

    state.set_server_hints(hints)

    # Validate servers and models (use stripped model IDs)
    pool, error = await _init_pool_and_validate(servers_text, stripped_models)
    if error:
        yield error, []
        return

    adapter = LMStudioAdapter(pool)

    # Create and run battery
    runner = BatteryRunner(
        adapter=adapter,
        tests=tests,
        models=stripped_models,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout
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


async def quick_prompt_handler(
    prompt: str,
    models_selected: list[str],
    servers_text: str,
    temperature: float,
    timeout: int,
    max_tokens: int,
    system_prompt: str
):
    """Run a single prompt against selected models for quick ad-hoc testing."""
    state.clear_stop()

    if not prompt or not prompt.strip():
        yield "*Enter a prompt to test*"
        return

    if not models_selected:
        yield "❌ No models selected"
        return

    from prompt_prix.core import stream_completion

    # Parse prefixed selections and build server hints
    hints = {}
    stripped_models = []
    for selection in models_selected:
        idx, model_id = parse_prefixed_model(selection)
        url = state.get_server_url(idx)
        if url:
            hints[model_id] = url
        stripped_models.append(model_id)

    state.set_server_hints(hints)

    pool, error = await _init_pool_and_validate(servers_text, stripped_models)
    if error:
        yield error
        return

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    results = {}
    output_lines = [f"**Prompt:** {prompt.strip()}\n\n---\n"]

    for model_id in stripped_models:
        if state.should_stop():
            output_lines.append("---\n🛑 **Stopped by user**")
            yield "\n".join(output_lines)
            return

        hint = state.get_server_hint(model_id)
        server_url = pool.find_server(model_id, preferred_url=hint)
        if not server_url:
            results[model_id] = f"❌ No server available"
            output_lines.append(f"### {model_id}\n{results[model_id]}\n\n")
            yield "\n".join(output_lines)
            continue

        output_lines.append(f"### {model_id}\n⏳ *Generating...*\n\n")
        yield "\n".join(output_lines)

        try:
            await pool.acquire_server(server_url)
            response = ""
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout
            ):
                if state.should_stop():
                    output_lines[-1] = f"### {model_id}\n{response}\n\n*(stopped)*\n\n"
                    output_lines.append("---\n🛑 **Stopped by user**")
                    yield "\n".join(output_lines)
                    return
                response += chunk

            results[model_id] = response
            output_lines[-1] = f"### {model_id}\n{response}\n\n"
            yield "\n".join(output_lines)

        except Exception as e:
            results[model_id] = f"❌ Error: {e}"
            output_lines[-1] = f"### {model_id}\n{results[model_id]}\n\n"
            yield "\n".join(output_lines)
        finally:
            pool.release_server(server_url)

    output_lines.append("---\n✅ **Complete**")
    yield "\n".join(output_lines)


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
                    "error": result.error
                })

    # Write to temp file with meaningful name
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.json")

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    return f"✅ Exported {len(export_data['results'])} results", gr.update(visible=True, value=filepath)


def export_csv():
    """Export battery results as CSV file."""
    if not state.battery_run:
        return "❌ No battery results to export", gr.update(visible=False, value=None)

    lines = ["test_id,model_id,status,latency_ms,response"]

    for test_id in state.battery_run.tests:
        for model_id in state.battery_run.models:
            result = state.battery_run.get_result(test_id, model_id)
            if result:
                response = result.response or ""
                response = response.replace('"', '""')
                response = response.replace('\n', '\\n')
                latency = f"{result.latency_ms:.0f}" if result.latency_ms else ""
                lines.append(f'"{test_id}","{model_id}","{result.status.value}",{latency},"{response}"')

    # Write to temp file with meaningful name
    basename = _get_export_basename()
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{basename}.csv")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return f"✅ Exported {len(lines) - 1} results", gr.update(visible=True, value=filepath)


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
