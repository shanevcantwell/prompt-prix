"""
Report generation in Markdown and JSON formats.
"""

import json
from datetime import datetime
from prompt_prix.config import SessionState, REPORT_DIVIDER, CONVERSATION_SEPARATOR


def generate_markdown_report(state: SessionState) -> str:
    """
    Generate a Markdown report with all model contexts.
    """
    lines = []

    # Header
    lines.append("# LLM Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Models:** {', '.join(state.models)}")
    lines.append(f"**Temperature:** {state.temperature}")
    lines.append(f"**Max Tokens:** {state.max_tokens}")
    lines.append("")

    if state.halted:
        lines.append(f"**Session Halted:** {state.halt_reason}")
        lines.append("")

    # System prompt
    lines.append("## System Prompt")
    lines.append("")
    lines.append("```")
    lines.append(state.system_prompt)
    lines.append("```")
    lines.append("")

    # Model sections
    for model_id in state.models:
        context = state.contexts.get(model_id)
        if not context:
            continue

        lines.append(REPORT_DIVIDER.strip())
        lines.append("")
        lines.append(f"## Model: {model_id}")
        lines.append("")

        if context.error:
            lines.append(f"**Error:** {context.error}")
            lines.append("")

        for msg in context.messages:
            if msg.role == "user":
                lines.append("### User")
                lines.append("")
                lines.append(msg.content)
                lines.append("")
            elif msg.role == "assistant":
                lines.append("### Assistant")
                lines.append("")
                lines.append(msg.content)
                lines.append("")

    return "\n".join(lines)


def generate_json_report(state: SessionState) -> str:
    """
    Generate a JSON report with all model contexts.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "configuration": {
            "models": state.models,
            "system_prompt": state.system_prompt,
            "temperature": state.temperature,
            "max_tokens": state.max_tokens,
            "timeout_seconds": state.timeout_seconds
        },
        "halted": state.halted,
        "halt_reason": state.halt_reason,
        "conversations": {}
    }

    for model_id in state.models:
        context = state.contexts.get(model_id)
        if not context:
            continue

        report["conversations"][model_id] = {
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in context.messages
            ],
            "error": context.error
        }

    return json.dumps(report, indent=2)


def save_report(content: str, filepath: str) -> None:
    """Save report content to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
