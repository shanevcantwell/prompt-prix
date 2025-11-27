"""
Gradio event handlers for prompt-prix.
"""

import asyncio
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

from prompt_prix import state
from prompt_prix.config import get_beyond_compare_path
from prompt_prix.core import ServerPool, ComparisonSession
from prompt_prix.export import generate_markdown_report, generate_json_report, save_report
from prompt_prix.parsers import parse_models_input, parse_servers_input, parse_prompts_file, load_system_prompt


async def fetch_available_models(servers_text: str) -> tuple[str, str]:
    """
    Query all configured servers and return available models.
    Returns (status_message, models_text).
    """
    servers = parse_servers_input(servers_text)

    if not servers:
        return "❌ No servers configured", ""

    # Create a temporary server pool to fetch manifests
    pool = ServerPool(servers)
    await pool.refresh_all_manifests()

    # Collect all models with their server info
    models_by_server: dict[str, list[str]] = {}
    for url, server in pool.servers.items():
        if server.available_models:
            models_by_server[url] = server.available_models

    if not models_by_server:
        return "⚠️ No models found on any server. Are models loaded in LM Studio?", ""

    # Build output: list all unique models
    all_models = set()
    for models in models_by_server.values():
        all_models.update(models)

    # Sort for consistent ordering
    sorted_models = sorted(all_models)
    models_text = "\n".join(sorted_models)

    # Build status showing which server has which models
    status_parts = [f"✅ Found {len(all_models)} model(s):"]
    for url, models in models_by_server.items():
        status_parts.append(f"  {url}: {len(models)} model(s)")

    return " | ".join(status_parts), models_text


async def initialize_session(
    servers_text: str,
    models_text: str,
    system_prompt_file: Optional[str],
    temperature: float,
    timeout: int,
    max_tokens: int
) -> tuple:
    """
    Initialize or reinitialize the comparison session.
    Returns tuple of (status_message, *model_tab_contents)
    """
    servers = parse_servers_input(servers_text)
    models = parse_models_input(models_text)
    system_prompt = load_system_prompt(system_prompt_file)

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


async def send_single_prompt(prompt: str):
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

    # Initial state - show waiting status with all tabs pending
    pending = len(session.state.models)
    yield (f"⏳ Generating responses... (0/{pending} complete)", build_tab_states()) + tuple(build_output())

    # Process each model with streaming
    async def stream_model(model_id: str):
        nonlocal streaming_responses, streaming_started, completed_models
        context = session.state.contexts[model_id]

        # Find server
        server_url = None
        while server_url is None:
            await session.server_pool.refresh_all_manifests()
            server_url = session.server_pool.find_available_server(model_id)
            if server_url is None:
                await asyncio.sleep(0.5)

        await session.server_pool.acquire_server(server_url)
        streaming_started.add(model_id)  # Mark as streaming once server acquired

        try:
            from prompt_prix.core import stream_completion
            messages = context.to_openai_messages(session.state.system_prompt)

            full_response = ""
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=model_id,
                messages=messages,
                temperature=session.state.temperature,
                max_tokens=session.state.max_tokens,
                timeout_seconds=session.state.timeout_seconds
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

    # Start all models as tasks
    tasks = [asyncio.create_task(stream_model(m)) for m in session.state.models]

    # Poll and yield updates while tasks are running
    while len(completed_models) < len(session.state.models):
        await asyncio.sleep(0.1)  # Update rate
        done = len(completed_models)
        total = len(session.state.models)
        yield (f"⏳ Generating responses... ({done}/{total} complete)", build_tab_states()) + tuple(build_output())

    # Wait for all tasks to fully complete
    await asyncio.gather(*tasks, return_exceptions=True)

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
