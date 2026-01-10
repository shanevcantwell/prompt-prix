"""
Compare tab event handlers.

Handles interactive multi-turn model comparison using HuggingFace Inference API.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from datetime import datetime

import gradio as gr

from prompt_prix import state
from prompt_prix.core import ComparisonSession
from prompt_prix.adapters import HuggingFaceAdapter
from prompt_prix.export import generate_markdown_report, generate_json_report, save_report


def _empty_tabs(n: int = 10) -> tuple:
    """Return tuple of n empty strings for model tabs."""
    return tuple("" for _ in range(n))


async def initialize_session(
    models_selected: list[str],
    system_prompt_text: str,
    timeout: int,
    max_tokens: int
) -> tuple:
    """
    Initialize or reinitialize the comparison session with HuggingFace.
    Returns tuple of (status_message, *model_tab_contents)

    Uses HF_TOKEN from environment for authentication.
    """
    models = models_selected if models_selected else []
    system_prompt = system_prompt_text.strip() if system_prompt_text else ""

    if not models:
        return ("❌ No models selected",) + _empty_tabs()

    # Validate HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        return ("❌ HF_TOKEN environment variable not set",) + _empty_tabs()

    try:
        adapter = HuggingFaceAdapter(models=models)
    except ValueError as e:
        return (f"❌ {e}",) + _empty_tabs()

    state.session = ComparisonSession(
        models=models,
        adapter=adapter,
        system_prompt=system_prompt,
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
    """Send a single prompt to all models with streaming output via HuggingFace.

    Args:
        prompt: The user's text prompt
        tools_json: Optional JSON string defining function tools
        image_path: Optional path to an image file for vision models
        seed: Optional seed for reproducible outputs (not supported by all HF models)
        repeat_penalty: Optional repeat penalty (not supported by HF API)
    """
    session = state.session

    if session is None:
        yield ("❌ Session not initialized", []) + _empty_tabs()
        return

    if session.state.halted:
        halted_states = [
            {"status": "completed", "name": m} for m in session.state.models
        ] + [{"status": "", "name": ""} for _ in range(10 - len(session.state.models))]
        yield (
            f"❌ Session halted: {session.state.halt_reason}",
            halted_states,
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
        """Build tab states with both status and model name for dynamic tabs."""
        states = []
        for i in range(10):
            if i < len(session.state.models):
                model_id = session.state.models[i]
                if model_id in completed_models:
                    status = "completed"
                elif model_id in streaming_started:
                    status = "streaming"
                else:
                    status = "pending"
                states.append({"status": status, "name": model_id})
            else:
                states.append({"status": "", "name": ""})
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

    pending = len(session.state.models)
    yield (f"⏳ Generating responses... (0/{pending} complete)", build_tab_states()) + tuple(build_output())

    async def run_model(model_id: str):
        nonlocal streaming_responses, streaming_started, completed_models
        context = session.state.contexts[model_id]

        streaming_started.add(model_id)

        try:
            messages = context.to_openai_messages(session.state.system_prompt)

            raw_response = ""
            async for chunk in session.adapter.stream_completion(
                model_id=model_id,
                messages=messages,
                temperature=session.state.temperature,
                max_tokens=session.state.max_tokens,
                timeout_seconds=session.state.timeout_seconds,
                tools=tools
            ):
                raw_response += chunk
                streaming_responses[model_id] = raw_response

            context.add_assistant_message(raw_response)
            completed_models.add(model_id)

        except Exception as e:
            context.error = str(e)
            session.state.halted = True
            session.state.halt_reason = f"Model {model_id} failed: {e}"
            streaming_responses[model_id] = f"[ERROR: {e}]"
            completed_models.add(model_id)

    # Run all models concurrently
    tasks = [asyncio.create_task(run_model(m)) for m in session.state.models]

    while not all(t.done() for t in tasks):
        await asyncio.sleep(0.1)
        done = len(completed_models)
        total = len(session.state.models)
        yield (f"⏳ Generating responses... ({done}/{total} complete)", build_tab_states()) + tuple(build_output())

    await asyncio.gather(*tasks, return_exceptions=True)

    status = "✅ All responses complete"
    if session.state.halted:
        status = f"⚠️ Session halted: {session.state.halt_reason}"

    final_tab_states = [
        {"status": "completed", "name": m} for m in session.state.models
    ] + [{"status": "", "name": ""} for _ in range(10 - len(session.state.models))]

    contexts = []
    for i in range(10):
        if i < len(session.state.models):
            model_id = session.state.models[i]
            contexts.append(session.get_context_display(model_id))
        else:
            contexts.append("")

    yield (status, final_tab_states) + tuple(contexts)


def export_markdown() -> tuple[str, dict]:
    """Export current session as Markdown file for download."""
    session = state.session
    if session is None:
        return "❌ No session to export", gr.update(visible=False)

    report = generate_markdown_report(session.state)
    filename = f"prompt-prix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    # Write to temp file for download
    filepath = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.md',
        prefix=filename.replace('.md', '_'),
        delete=False,
        encoding='utf-8'
    ).name
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    return f"✅ Exported: {filename}", gr.update(visible=False, value=filepath)


def export_json() -> tuple[str, dict]:
    """Export current session as JSON file for download."""
    session = state.session
    if session is None:
        return "❌ No session to export", gr.update(visible=False)

    report = generate_json_report(session.state)
    filename = f"prompt-prix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Write to temp file for download
    filepath = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        prefix=filename.replace('.json', '_'),
        delete=False,
        encoding='utf-8'
    ).name
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    return f"✅ Exported: {filename}", gr.update(visible=False, value=filepath)


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
