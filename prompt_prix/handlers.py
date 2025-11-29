"""
Gradio event handlers for prompt-prix.
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import gradio as gr

from prompt_prix import state
from prompt_prix.config import get_beyond_compare_path
from prompt_prix.core import ServerPool, ComparisonSession
from prompt_prix.export import generate_markdown_report, generate_json_report, save_report
from prompt_prix.parsers import parse_models_input, parse_servers_input, parse_prompts_file, load_system_prompt


def handle_stop():
    """Handle Stop button click - signal cancellation."""
    state.request_stop()
    return "🛑 Stop requested..."


async def fetch_available_models(servers_text: str) -> tuple[str, dict]:
    """
    Query all configured servers and return available models.
    Returns (status_message, gr.update for CheckboxGroup choices).
    """
    servers = parse_servers_input(servers_text)

    if not servers:
        return "❌ No servers configured", gr.update(choices=[])

    # Create a temporary server pool to fetch manifests
    pool = ServerPool(servers)
    await pool.refresh_all_manifests()

    # Collect all models with their server info
    models_by_server: dict[str, list[str]] = {}
    for url, server in pool.servers.items():
        if server.available_models:
            models_by_server[url] = server.available_models

    if not models_by_server:
        return "⚠️ No models found on any server. Are models loaded in LM Studio?", gr.update(choices=[])

    # Build output: list all unique models
    all_models = set()
    for models in models_by_server.values():
        all_models.update(models)

    # Sort for consistent ordering
    sorted_models = sorted(all_models)

    # Build status showing which server has which models
    status_parts = [f"✅ Found {len(all_models)} model(s):"]
    for url, models in models_by_server.items():
        status_parts.append(f"  {url}: {len(models)} model(s)")

    return " | ".join(status_parts), gr.update(choices=sorted_models)


async def initialize_session(
    servers_text: str,
    models_selected: list[str],
    system_prompt_text: str,
    temperature: float,
    timeout: int,
    max_tokens: int
) -> tuple:
    """
    Initialize or reinitialize the comparison session.
    Returns tuple of (status_message, *model_tab_contents)

    Args:
        servers_text: Newline-separated server URLs
        models_selected: List of selected model IDs from CheckboxGroup
        system_prompt_text: System prompt text directly from textbox
        temperature: Model temperature setting
        timeout: Timeout in seconds
        max_tokens: Maximum tokens to generate
    """
    servers = parse_servers_input(servers_text)
    models = models_selected if models_selected else []
    system_prompt = system_prompt_text.strip() if system_prompt_text else ""

    if not servers:
        return ("❌ No servers configured",) + tuple("" for _ in range(10))
    if not models:
        return ("❌ No models configured",) + tuple("" for _ in range(10))

    # Initialize server pool and refresh manifests
    state.server_pool = ServerPool(servers)
    await state.server_pool.refresh_all_manifests()

    # Check which models are actually available
    available = state.server_pool.get_all_available_models()
    missing = [m for m in models if m not in available]

    if missing:
        return (
            f"⚠️ Models not found on any server: {', '.join(missing)}",
        ) + tuple("" for _ in range(10))

    # Create session
    state.session = ComparisonSession(
        models=models,
        server_pool=state.server_pool,
        system_prompt=system_prompt,
        temperature=temperature,
        timeout_seconds=timeout,
        max_tokens=max_tokens
    )

    return (f"✅ Session initialized with {len(models)} models",) + tuple(
        "" for _ in range(10)
    )


async def send_single_prompt(prompt: str, tools_json: str = ""):
    """
    Send a single prompt to all models with streaming output.
    Yields updates as responses stream in from each model.
    """
    session = state.session

    if session is None:
        yield ("❌ Session not initialized", []) + tuple("" for _ in range(10))
        return

    if session.state.halted:
        yield (
            f"❌ Session halted: {session.state.halt_reason}",
            ["completed"] * len(session.state.models) + [""] * (10 - len(session.state.models)),
        ) + tuple(
            session.get_context_display(m) if m in session.state.contexts else ""
            for m in session.state.models[:10]
        ) + tuple("" for _ in range(10 - len(session.state.models)))
        return

    if not prompt.strip():
        yield ("❌ Empty prompt", []) + tuple("" for _ in range(10))
        return

    # Parse tools JSON if provided
    tools = None
    if tools_json and tools_json.strip():
        try:
            tools = json.loads(tools_json)
            if not isinstance(tools, list):
                yield ("❌ Tools must be a JSON array", []) + tuple("" for _ in range(10))
                return
        except json.JSONDecodeError as e:
            yield (f"❌ Invalid tools JSON: {e}", []) + tuple("" for _ in range(10))
            return

    # Track streaming state for each model
    streaming_responses: dict[str, str] = {m: "" for m in session.state.models}
    streaming_started: set[str] = set()  # Models that have started receiving data
    completed_models: set[str] = set()

    def build_tab_states():
        """Build tab state list: pending, streaming, or completed."""
        states = []
        for i in range(10):
            if i < len(session.state.models):
                model_id = session.state.models[i]
                if model_id in completed_models:
                    states.append("completed")
                elif model_id in streaming_started:
                    states.append("streaming")
                else:
                    states.append("pending")
            else:
                states.append("")
        return states

    def build_output():
        """Build current output tuple with streaming state."""
        contexts = []
        for i in range(10):
            if i < len(session.state.models):
                model_id = session.state.models[i]
                if model_id in completed_models:
                    # Model finished - show final context (user + assistant already in context)
                    contexts.append(session.state.contexts[model_id].to_display_format())
                else:
                    # Model still streaming - existing has user message, append streaming response
                    existing = session.state.contexts[model_id].to_display_format()
                    current = streaming_responses.get(model_id, "")
                    if existing:
                        # User message already in context, just append assistant streaming
                        if current:
                            contexts.append(f"{existing}\n\n**Assistant:** {current}")
                        else:
                            contexts.append(f"{existing}\n\n**Assistant:** ...")
                    else:
                        # No messages yet (shouldn't happen, but handle gracefully)
                        contexts.append(f"### {model_id}\n\n**User:** {prompt.strip()}\n\n**Assistant:** ...")
            else:
                contexts.append("")
        return contexts

    # Add user messages to all contexts upfront (before streaming starts)
    for model_id in session.state.models:
        session.state.contexts[model_id].add_user_message(prompt.strip())

    # Refresh manifests once before starting
    await session.server_pool.refresh_all_manifests()

    # Initial state - show waiting status with all tabs pending
    pending = len(session.state.models)
    yield (f"⏳ Generating responses... (0/{pending} complete)", build_tab_states()) + tuple(build_output())

    # Work-stealing dispatcher: queue of models, servers pull work
    model_queue = list(session.state.models)  # Models waiting to be processed
    active_tasks: dict[str, asyncio.Task] = {}  # server_url -> task

    async def run_model_on_server(model_id: str, server_url: str):
        """Run a single model on a specific server."""
        nonlocal streaming_responses, streaming_started, completed_models
        context = session.state.contexts[model_id]

        streaming_started.add(model_id)

        # Show which server is handling this model
        server_label = server_url.split("//")[-1]
        streaming_responses[model_id] = f"*[Server: {server_label}]*\n\n"

        try:
            from prompt_prix.core import stream_completion
            messages = context.to_openai_messages(session.state.system_prompt)

            full_response = f"*[Server: {server_label}]*\n\n"
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=model_id,
                messages=messages,
                temperature=session.state.temperature,
                max_tokens=session.state.max_tokens,
                timeout_seconds=session.state.timeout_seconds,
                tools=tools
            ):
                full_response += chunk
                streaming_responses[model_id] = full_response

            context.add_assistant_message(full_response)
            completed_models.add(model_id)

        except Exception as e:
            context.error = str(e)
            session.state.halted = True
            session.state.halt_reason = f"Model {model_id} failed: {e}"
            streaming_responses[model_id] = f"[ERROR: {e}]"
            completed_models.add(model_id)
        finally:
            session.server_pool.release_server(server_url)

    def find_work_for_server(server_url: str) -> str | None:
        """Find first model in queue that this server can run."""
        server = session.server_pool.servers[server_url]
        for model_id in model_queue:
            if model_id in server.available_models:
                return model_id
        return None

    # Dispatcher loop - keeps servers busy
    while model_queue or active_tasks:
        # Try to assign work to any idle server
        for server_url, server in session.server_pool.servers.items():
            if server.is_busy:
                continue  # Server already working

            # Find a model this server can run
            model_id = find_work_for_server(server_url)
            if model_id:
                model_queue.remove(model_id)
                await session.server_pool.acquire_server(server_url)
                task = asyncio.create_task(run_model_on_server(model_id, server_url))
                active_tasks[server_url] = task

        # Wait a bit, yield UI update
        await asyncio.sleep(0.1)
        done = len(completed_models)
        total = len(session.state.models)
        yield (f"⏳ Generating responses... ({done}/{total} complete)", build_tab_states()) + tuple(build_output())

        # Clean up completed tasks
        for server_url in list(active_tasks.keys()):
            if active_tasks[server_url].done():
                del active_tasks[server_url]

        # Refresh manifests occasionally if we have queued work but no progress
        if model_queue and not active_tasks:
            await session.server_pool.refresh_all_manifests()

    # Wait for any remaining tasks
    if active_tasks:
        await asyncio.gather(*active_tasks.values(), return_exceptions=True)

    # Final output with completed conversations - all tabs completed
    status = "✅ All responses complete"
    if session.state.halted:
        status = f"⚠️ Session halted: {session.state.halt_reason}"

    # All models completed
    final_tab_states = ["completed"] * len(session.state.models) + [""] * (10 - len(session.state.models))

    contexts = []
    for i in range(10):
        if i < len(session.state.models):
            model_id = session.state.models[i]
            contexts.append(session.get_context_display(model_id))
        else:
            contexts.append("")

    yield (status, final_tab_states) + tuple(contexts)


async def run_batch_prompts(file_obj) -> tuple:
    """
    Run a batch of prompts from uploaded file.
    Returns tuple of (status_message, *model_tab_contents)
    """
    session = state.session

    if session is None:
        return ("❌ Session not initialized",) + tuple("" for _ in range(10))

    if file_obj is None:
        return ("❌ No file uploaded",) + tuple("" for _ in range(10))

    # Read file content
    content = Path(file_obj.name).read_text(encoding="utf-8")
    prompts = parse_prompts_file(content)

    if not prompts:
        return ("❌ No prompts found in file",) + tuple("" for _ in range(10))

    # Run each prompt in sequence
    completed = 0
    for i, prompt in enumerate(prompts):
        if session.state.halted:
            break
        await session.send_prompt_to_all(prompt)
        completed = i + 1

    # Build result
    status = f"✅ Completed {completed}/{len(prompts)} prompts"
    if session.state.halted:
        status = f"⚠️ Halted after {completed}/{len(prompts)} prompts: {session.state.halt_reason}"

    contexts = []
    for j in range(10):
        if j < len(session.state.models):
            model_id = session.state.models[j]
            contexts.append(session.get_context_display(model_id))
        else:
            contexts.append("")

    return (status,) + tuple(contexts)


def export_markdown() -> tuple[str, str]:
    """Export current session as Markdown."""
    session = state.session

    if session is None:
        return "❌ No session to export", ""

    report = generate_markdown_report(session.state)
    filename = f"prompt-prix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    save_report(report, filename)

    return f"✅ Exported to {filename}", report


def export_json() -> tuple[str, str]:
    """Export current session as JSON."""
    session = state.session

    if session is None:
        return "❌ No session to export", ""

    report = generate_json_report(session.state)
    filename = f"prompt-prix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_report(report, filename)

    return f"✅ Exported to {filename}", report


def launch_beyond_compare(model_a: str, model_b: str) -> str:
    """
    Launch Beyond Compare with two model outputs for side-by-side diff.
    """
    session = state.session

    if session is None:
        return "❌ No session - initialize first"

    if not model_a or not model_b:
        return "❌ Select two models to compare"

    if model_a == model_b:
        return "❌ Select two different models"

    if model_a not in session.state.contexts:
        return f"❌ Model '{model_a}' not in session"
    if model_b not in session.state.contexts:
        return f"❌ Model '{model_b}' not in session"

    # Get the conversation content
    content_a = session.get_context_display(model_a)
    content_b = session.get_context_display(model_b)

    if not content_a.strip() and not content_b.strip():
        return "❌ No conversation content to compare"

    # Write to temp files with model names in filename for clarity
    try:
        # Create temp files that persist (delete=False)
        safe_name_a = model_a.replace("/", "_").replace("\\", "_")[:50]
        safe_name_b = model_b.replace("/", "_").replace("\\", "_")[:50]

        file_a = tempfile.NamedTemporaryFile(
            mode='w', suffix=f'_{safe_name_a}.txt',
            delete=False, encoding='utf-8'
        )
        file_a.write(f"# Model: {model_a}\n\n{content_a}")
        file_a.close()

        file_b = tempfile.NamedTemporaryFile(
            mode='w', suffix=f'_{safe_name_b}.txt',
            delete=False, encoding='utf-8'
        )
        file_b.write(f"# Model: {model_b}\n\n{content_b}")
        file_b.close()

        # Launch Beyond Compare
        bc_path = get_beyond_compare_path()
        subprocess.Popen([bc_path, file_a.name, file_b.name])

        return f"✅ Launched Beyond Compare: {model_a} vs {model_b}"

    except FileNotFoundError:
        return f"❌ Beyond Compare not found. Set BEYOND_COMPARE_PATH in .env"
    except Exception as e:
        return f"❌ Failed to launch Beyond Compare: {e}"


# ─────────────────────────────────────────────────────────────────────
# BATTERY HANDLERS
# ─────────────────────────────────────────────────────────────────────

def battery_validate_file(file_obj) -> str:
    """
    Validate benchmark file before enabling Run button.

    Returns validation message string. Starts with ✅ if valid, ❌ if not.
    Fail-fast: validation happens before any execution.
    """
    if file_obj is None:
        return "Upload a benchmark JSON file"

    from prompt_prix.benchmarks import CustomJSONLoader

    valid, message = CustomJSONLoader.validate(file_obj)
    return message


def battery_get_test_ids(file_obj) -> list[str]:
    """
    Extract test IDs from benchmark file for dropdown population.

    Returns list of test IDs, or empty list on error.
    """
    if file_obj is None:
        return []

    from prompt_prix.benchmarks import CustomJSONLoader

    try:
        tests = CustomJSONLoader.load(file_obj)
        return [t.id for t in tests]
    except Exception:
        return []


async def battery_run_handler(
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

    Args:
        file_obj: Uploaded benchmark JSON file
        models_selected: List of model IDs to test
        servers_text: Newline-separated server URLs
        temperature: Sampling temperature
        timeout: Timeout per request in seconds
        max_tokens: Maximum tokens per response
        system_prompt: Optional override for test-defined system prompts
    """
    # Fail-fast validation
    if file_obj is None:
        yield "❌ No benchmark file uploaded", []
        return

    if not models_selected:
        yield "❌ No models selected", []
        return

    servers = parse_servers_input(servers_text)
    if not servers:
        yield "❌ No servers configured", []
        return

    # Import here to avoid circular imports
    from prompt_prix.benchmarks import CustomJSONLoader
    from prompt_prix.adapters import LMStudioAdapter
    from prompt_prix.battery import BatteryRunner
    from prompt_prix.core import ServerPool

    # Load test cases
    try:
        tests = CustomJSONLoader.load(file_obj)
    except Exception as e:
        yield f"❌ Failed to load tests: {e}", []
        return

    # Create server pool and adapter
    pool = ServerPool(servers)
    await pool.refresh_all_manifests()

    # Verify models are available
    available = pool.get_all_available_models()
    missing = [m for m in models_selected if m not in available]
    if missing:
        yield f"❌ Models not available: {', '.join(missing)}", []
        return

    adapter = LMStudioAdapter(pool)

    # Create and run battery
    runner = BatteryRunner(
        adapter=adapter,
        tests=tests,
        models=models_selected,
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


async def battery_quick_prompt_handler(
    prompt: str,
    models_selected: list[str],
    servers_text: str,
    temperature: float,
    timeout: int,
    max_tokens: int,
    system_prompt: str
):
    """
    Run a single prompt against selected models for quick ad-hoc testing.

    Yields markdown-formatted results as each model completes.
    """
    state.clear_stop()  # Reset stop flag at start

    if not prompt or not prompt.strip():
        yield "*Enter a prompt to test*"
        return

    if not models_selected:
        yield "❌ No models selected"
        return

    servers = parse_servers_input(servers_text)
    if not servers:
        yield "❌ No servers configured"
        return

    from prompt_prix.core import ServerPool, stream_completion

    pool = ServerPool(servers)
    await pool.refresh_all_manifests()

    # Verify models are available
    available = pool.get_all_available_models()
    missing = [m for m in models_selected if m not in available]
    if missing:
        yield f"❌ Models not available: {', '.join(missing)}"
        return

    # Build messages
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    results = {}
    output_lines = [f"**Prompt:** {prompt.strip()}\n\n---\n"]

    for model_id in models_selected:
        # Check for stop request
        if state.should_stop():
            output_lines.append("---\n🛑 **Stopped by user**")
            yield "\n".join(output_lines)
            return

        # Find available server for this model
        server_url = pool.find_available_server(model_id)
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
                # Check for stop during streaming
                if state.should_stop():
                    output_lines[-1] = f"### {model_id}\n{response}\n\n*(stopped)*\n\n"
                    output_lines.append("---\n🛑 **Stopped by user**")
                    yield "\n".join(output_lines)
                    return
                response += chunk

            results[model_id] = response
            # Update the last entry
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


def battery_export_json() -> tuple[str, str]:
    """Export battery results as JSON."""
    if not state.battery_run:
        return "❌ No battery results to export", ""

    from prompt_prix.battery import TestStatus

    # Build export structure
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

    json_str = json.dumps(export_data, indent=2)
    return f"✅ Exported {len(export_data['results'])} results", json_str


def battery_export_csv() -> tuple[str, str]:
    """Export battery results as CSV."""
    if not state.battery_run:
        return "❌ No battery results to export", ""

    from prompt_prix.battery import TestStatus

    lines = ["test_id,model_id,status,latency_ms,response"]

    for test_id in state.battery_run.tests:
        for model_id in state.battery_run.models:
            result = state.battery_run.get_result(test_id, model_id)
            if result:
                # Escape CSV fields
                response = result.response or ""
                response = response.replace('"', '""')  # Escape quotes
                response = response.replace('\n', '\\n')  # Escape newlines
                latency = f"{result.latency_ms:.0f}" if result.latency_ms else ""
                lines.append(f'"{test_id}","{model_id}","{result.status.value}",{latency},"{response}"')

    csv_str = "\n".join(lines)
    return f"✅ Exported {len(lines) - 1} results", csv_str


def battery_get_cell_detail(model: str, test: str) -> str:
    """
    Get response detail for a (model, test) cell.

    Returns markdown-formatted detail including latency and response content.
    """
    from prompt_prix.battery import TestStatus

    if not state.battery_run:
        return "*No battery run available*"

    if not model or not test:
        return "*Select a model and test to view the response*"

    result = state.battery_run.get_result(test, model)
    if not result:
        return f"*No result for {model} × {test}*"

    if result.status == TestStatus.ERROR:
        return f"**Status:** ❌ Error\n\n**Error:** {result.error}"

    if result.status == TestStatus.PENDING:
        return f"**Status:** — Pending"

    if result.status == TestStatus.RUNNING:
        return f"**Status:** ⏳ Running..."

    # Completed
    latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "N/A"
    return f"**Status:** ✓ Completed\n\n**Latency:** {latency}\n\n---\n\n{result.response}"
