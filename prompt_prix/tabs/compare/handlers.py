"""
Compare tab event handlers.

Handles interactive multi-turn model comparison.
"""

import asyncio
import json
import subprocess
import tempfile
from datetime import datetime

from prompt_prix import state
from prompt_prix.scheduler import ServerPool
from prompt_prix.core import ComparisonSession
from prompt_prix.export import generate_markdown_report, generate_json_report, save_report
from prompt_prix.parsers import parse_servers_input


def _empty_tabs(n: int = 10) -> tuple:
    """Return tuple of n empty strings for model tabs."""
    return tuple("" for _ in range(n))


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
    """
    servers = parse_servers_input(servers_text)
    models = models_selected if models_selected else []
    system_prompt = system_prompt_text.strip() if system_prompt_text else ""

    if not servers:
        return ("❌ No servers configured",) + _empty_tabs()
    if not models:
        return ("❌ No models configured",) + _empty_tabs()

    state.server_pool = ServerPool(servers)
    await state.server_pool.refresh()

    available = state.server_pool.get_available_models()
    missing = [m for m in models if m not in available]

    if missing:
        return (
            f"⚠️ Models not found on any server: {', '.join(missing)}",
        ) + _empty_tabs()

    state.session = ComparisonSession(
        models=models,
        server_pool=state.server_pool,
        system_prompt=system_prompt,
        temperature=temperature,
        timeout_seconds=timeout,
        max_tokens=max_tokens
    )

    return (f"✅ Session initialized with {len(models)} models",) + _empty_tabs()


def clear_session() -> tuple:
    """Clear the current session and all conversation contexts.

    Preserves UI settings (models, temperature, etc.) but clears conversation history.
    Returns tuple of (status_message, tab_states, *model_tab_contents).
    """
    state.session = None

    return (
        "🗑️ Conversation cleared. Ready for new prompts.",
        [],  # tab_states
    ) + _empty_tabs()


async def send_single_prompt(prompt: str, tools_json: str = "", image_path: str = None, seed: int = None, repeat_penalty: float = None):
    """Send a single prompt to all models with streaming output.

    Args:
        prompt: The user's text prompt
        tools_json: Optional JSON string defining function tools
        image_path: Optional path to an image file for vision models
        seed: Optional seed for reproducible outputs
        repeat_penalty: Optional repeat penalty (1.0 = off)
    """
    session = state.session

    if session is None:
        yield ("❌ Session not initialized", []) + _empty_tabs()
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
        yield ("❌ Empty prompt", []) + _empty_tabs()
        return

    # Parse tools JSON if provided
    tools = None
    if tools_json and tools_json.strip():
        try:
            tools = json.loads(tools_json)
            if not isinstance(tools, list):
                yield ("❌ Tools must be a JSON array", []) + _empty_tabs()
                return
        except json.JSONDecodeError as e:
            yield (f"❌ Invalid tools JSON: {e}", []) + _empty_tabs()
            return

    streaming_responses: dict[str, str] = {m: "" for m in session.state.models}
    streaming_started: set[str] = set()
    completed_models: set[str] = set()

    def build_tab_states():
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
        contexts = []
        for i in range(10):
            if i < len(session.state.models):
                model_id = session.state.models[i]
                if model_id in completed_models:
                    contexts.append(session.state.contexts[model_id].to_display_format())
                else:
                    existing = session.state.contexts[model_id].to_display_format()
                    current = streaming_responses.get(model_id, "")
                    if existing:
                        if current:
                            contexts.append(f"{existing}\n\n**Assistant:** {current}")
                        else:
                            contexts.append(f"{existing}\n\n**Assistant:** ...")
                    else:
                        user_prefix = "**User:** 🖼️" if image_path else "**User:**"
                        contexts.append(f"### {model_id}\n\n{user_prefix} {prompt.strip()}\n\n**Assistant:** ...")
            else:
                contexts.append("")
        return contexts

    # Add user messages to all contexts upfront
    for model_id in session.state.models:
        session.state.contexts[model_id].add_user_message(prompt.strip(), image_path=image_path)

    await session.server_pool.refresh()

    pending = len(session.state.models)
    yield (f"⏳ Generating responses... (0/{pending} complete)", build_tab_states()) + tuple(build_output())

    model_queue = list(session.state.models)
    active_tasks: dict[str, asyncio.Task] = {}

    async def run_model_on_server(model_id: str, server_url: str):
        nonlocal streaming_responses, streaming_started, completed_models
        context = session.state.contexts[model_id]

        streaming_started.add(model_id)

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
                tools=tools,
                seed=seed,
                repeat_penalty=repeat_penalty
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
            session.server_pool.release(server_url)

    def find_work_for_server(server_url: str) -> str | None:
        server = session.server_pool.servers[server_url]
        for model_id in model_queue:
            if model_id in server.manifest_models:
                return model_id
        return None

    while model_queue or active_tasks:
        for server_url, server in session.server_pool.servers.items():
            if server.is_busy:
                continue

            model_id = find_work_for_server(server_url)
            if model_id:
                model_queue.remove(model_id)
                await session.server_pool.acquire(server_url)
                task = asyncio.create_task(run_model_on_server(model_id, server_url))
                active_tasks[server_url] = task

        await asyncio.sleep(0.1)
        done = len(completed_models)
        total = len(session.state.models)
        yield (f"⏳ Generating responses... ({done}/{total} complete)", build_tab_states()) + tuple(build_output())

        for server_url in list(active_tasks.keys()):
            if active_tasks[server_url].done():
                del active_tasks[server_url]

        if model_queue and not active_tasks:
            await session.server_pool.refresh()

    if active_tasks:
        await asyncio.gather(*active_tasks.values(), return_exceptions=True)

    status = "✅ All responses complete"
    if session.state.halted:
        status = f"⚠️ Session halted: {session.state.halt_reason}"

    final_tab_states = ["completed"] * len(session.state.models) + [""] * (10 - len(session.state.models))

    contexts = []
    for i in range(10):
        if i < len(session.state.models):
            model_id = session.state.models[i]
            contexts.append(session.get_context_display(model_id))
        else:
            contexts.append("")

    yield (status, final_tab_states) + tuple(contexts)


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
    """Launch Beyond Compare with two model outputs for side-by-side diff."""
    from prompt_prix.config import get_beyond_compare_path

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

    content_a = session.get_context_display(model_a)
    content_b = session.get_context_display(model_b)

    if not content_a.strip() and not content_b.strip():
        return "❌ No conversation content to compare"

    try:
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

        bc_path = get_beyond_compare_path()
        subprocess.Popen([bc_path, file_a.name, file_b.name])

        return f"✅ Launched Beyond Compare: {model_a} vs {model_b}"

    except FileNotFoundError:
        return f"❌ Beyond Compare not found. Set BEYOND_COMPARE_PATH in .env"
    except Exception as e:
        return f"❌ Failed to launch Beyond Compare: {e}"
