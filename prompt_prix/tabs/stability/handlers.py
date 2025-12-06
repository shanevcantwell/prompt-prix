"""
Stability tab event handlers.

Handles regeneration runs - running the same prompt multiple times
against a model to capture output variance.

No metrics computed internally - just capture and export for external analysis.
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Optional

import gradio as gr

from prompt_prix import state
from prompt_prix.handlers import _init_pool_and_validate


@dataclass
class Regeneration:
    """Single regeneration result."""
    index: int
    response: str
    thinking_blocks: Optional[list[dict]] = None
    latency_ms: float = 0.0
    timestamp: str = ""


@dataclass
class StabilityRun:
    """State for a stability analysis run."""
    prompt: str
    model_id: str
    temperature: float
    max_tokens: int
    system_prompt: str
    regenerations: list[Regeneration] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_export_dict(self) -> dict:
        """Convert to exportable dictionary."""
        return {
            "prompt": self.prompt,
            "model_id": self.model_id,
            "config": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system_prompt": self.system_prompt,
            },
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "regenerations": [
                {
                    "index": r.index,
                    "response": r.response,
                    "thinking_blocks": r.thinking_blocks,
                    "latency_ms": r.latency_ms,
                    "timestamp": r.timestamp,
                }
                for r in self.regenerations
            ],
        }


# Global state for current stability run
stability_run: Optional[StabilityRun] = None


def is_gemini_model(model_id: str) -> bool:
    """Check if the model is a Gemini Web UI model."""
    return "gemini" in model_id.lower() and "web ui" in model_id.lower()


async def run_regenerations(
    use_gemini: bool,
    model_id: str,
    prompt: str,
    regen_count: int,
    servers_text: str,
    temperature: float,
    timeout: int,
    max_tokens: int,
    system_prompt: str,
    capture_thinking: bool
) -> AsyncGenerator[tuple, None]:
    """
    Run multiple regenerations of the same prompt.

    Yields (status, *regen_outputs) tuples for streaming UI updates.
    """
    global stability_run

    # Clear stop flag
    state.clear_stop()

    # Validation - prompt is always required
    if not prompt or not prompt.strip():
        yield ("Enter a prompt",) + tuple(["*Waiting...*"] * 20)
        return

    # Gemini checkbox bypasses model dropdown
    if use_gemini:
        model_id = "gemini-2.0-flash-thinking (Web UI)"

    # Model required if not using Gemini
    if not model_id:
        yield ("Select a model",) + tuple(["*Waiting...*"] * 20)
        return

    # Initialize run state
    stability_run = StabilityRun(
        prompt=prompt.strip(),
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        started_at=datetime.now().isoformat(),
    )

    # Check if this is a Gemini model (requires special adapter)
    if use_gemini or is_gemini_model(model_id):
        async for result in _run_gemini_regenerations(
            prompt, regen_count, temperature, max_tokens, system_prompt, capture_thinking
        ):
            yield result
        return

    # LM Studio model - use standard completion
    from prompt_prix.core import stream_completion

    # Show initial status
    outputs = ["*Waiting...*"] * 20
    yield (f"Initializing for {model_id}...",) + tuple(outputs)

    pool, error = await _init_pool_and_validate(servers_text, [model_id])
    if error:
        yield (error,) + tuple(["*Error*"] * 20)
        return

    yield (f"Starting {regen_count} regenerations...",) + tuple(outputs)

    for i in range(int(regen_count)):
        if state.should_stop():
            stability_run.completed_at = datetime.now().isoformat()
            yield (f"Stopped at regeneration {i}/{regen_count}",) + tuple(outputs)
            return

        # Find server for this model
        server_url = pool.find_available_server(model_id)
        if not server_url:
            outputs[i] = f"*Error: No server available for {model_id}*"
            yield (f"Running regeneration {i + 1}/{regen_count}...",) + tuple(outputs)
            continue

        outputs[i] = "*Generating...*"
        yield (f"Running regeneration {i + 1}/{regen_count}...",) + tuple(outputs)

        # Build messages
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        start_time = time.time()
        response = ""

        try:
            await pool.acquire_server(server_url)
            async for chunk in stream_completion(
                server_url=server_url,
                model_id=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout
            ):
                if state.should_stop():
                    break
                response += chunk

            latency_ms = (time.time() - start_time) * 1000

            # Record regeneration
            regen = Regeneration(
                index=i + 1,
                response=response,
                latency_ms=latency_ms,
                timestamp=datetime.now().isoformat(),
            )
            stability_run.regenerations.append(regen)

            # Format output for display
            outputs[i] = f"**Latency:** {latency_ms:.0f}ms\n\n---\n\n{response}"
            yield (f"Completed regeneration {i + 1}/{regen_count}",) + tuple(outputs)

        except Exception as e:
            outputs[i] = f"*Error: {e}*"
            yield (f"Error in regeneration {i + 1}: {e}",) + tuple(outputs)
        finally:
            pool.release_server(server_url)

    stability_run.completed_at = datetime.now().isoformat()
    yield (f"Completed {regen_count} regenerations",) + tuple(outputs)


async def _run_gemini_regenerations(
    prompt: str,
    regen_count: int,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    capture_thinking: bool
) -> AsyncGenerator[tuple, None]:
    """
    Run regenerations using Gemini adapter.

    Tries visual adapter (Fara-7B) first, falls back to DOM adapter.
    Visual adapter is more resilient to UI changes.
    """
    global stability_run

    outputs = ["*Waiting...*"] * 20

    # Try visual adapter first (uses Fara-7B for element location)
    adapter = None
    adapter_type = None

    try:
        from prompt_prix.adapters.gemini_visual import GeminiVisualAdapter
        adapter = GeminiVisualAdapter()
        adapter_type = "visual (Fara-7B)"
        yield (f"Using {adapter_type} adapter...",) + tuple(outputs)
    except ImportError:
        # Fara not available, fall back to DOM adapter
        pass
    except Exception as e:
        # Visual adapter failed to init, try DOM
        yield (f"Visual adapter unavailable ({e}), trying DOM...",) + tuple(outputs)

    # Fall back to DOM adapter
    if adapter is None:
        try:
            from prompt_prix.adapters.gemini_webui import GeminiWebUIAdapter
            adapter = GeminiWebUIAdapter()
            adapter_type = "DOM"
            yield (f"Using {adapter_type} adapter...",) + tuple(outputs)
        except ImportError as e:
            yield (f"No Gemini adapter available: {e}",) + tuple(outputs)
            return
        except Exception as e:
            yield (f"Failed to initialize Gemini adapter: {e}",) + tuple(outputs)
            return

    try:
        for i in range(regen_count):
            if state.should_stop():
                stability_run.completed_at = datetime.now().isoformat()
                yield (f"Stopped at regeneration {i}/{regen_count}",) + tuple(outputs)
                return

            outputs[i] = f"*Generating via {adapter_type}...*"
            yield (f"Running regeneration {i + 1}/{regen_count}...",) + tuple(outputs)

            start_time = time.time()

            try:
                # For regeneration, we need to either:
                # 1. Start a new conversation each time (independent samples)
                # 2. Use the regenerate button in the same conversation (dependent samples)
                # The ADR suggests regenerate button is what we want to study
                if i == 0:
                    result = await adapter.send_prompt(prompt, system_prompt)
                else:
                    result = await adapter.regenerate()

                latency_ms = (time.time() - start_time) * 1000

                response = result.get("response", "")
                thinking = result.get("thinking_blocks") if capture_thinking else None

                regen = Regeneration(
                    index=i + 1,
                    response=response,
                    thinking_blocks=thinking,
                    latency_ms=latency_ms,
                    timestamp=datetime.now().isoformat(),
                )
                stability_run.regenerations.append(regen)

                # Format output with thinking if present
                output_parts = [f"**Latency:** {latency_ms:.0f}ms"]
                if thinking:
                    output_parts.append(f"**Thinking Blocks:** {len(thinking)}")
                output_parts.append("---")
                output_parts.append(response)
                outputs[i] = "\n\n".join(output_parts)

                yield (f"Completed regeneration {i + 1}/{regen_count}",) + tuple(outputs)

            except Exception as e:
                outputs[i] = f"*Error: {e}*"
                yield (f"Error in regeneration {i + 1}: {e}",) + tuple(outputs)

        stability_run.completed_at = datetime.now().isoformat()
        yield (f"Completed {regen_count} regenerations",) + tuple(outputs)

    finally:
        # Always close the adapter to release browser resources
        if adapter and hasattr(adapter, 'close'):
            try:
                await adapter.close()
            except Exception:
                pass


def export_json():
    """Export stability run as JSON file."""
    if not stability_run:
        return "No stability run to export", gr.update(visible=False, value=None)

    export_data = stability_run.to_export_dict()

    temp_dir = tempfile.gettempdir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(temp_dir, f"stability_run_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    return f"Exported {len(stability_run.regenerations)} regenerations", gr.update(visible=True, value=filepath)


def export_markdown():
    """Export stability run as Markdown file (LLM-readable format)."""
    if not stability_run:
        return "No stability run to export", gr.update(visible=False, value=None)

    lines = [
        "# Stability Analysis Run",
        "",
        "## Configuration",
        f"- **Model:** {stability_run.model_id}",
        f"- **Temperature:** {stability_run.temperature}",
        f"- **Max Tokens:** {stability_run.max_tokens}",
        f"- **Started:** {stability_run.started_at}",
        f"- **Completed:** {stability_run.completed_at}",
        "",
        "## Prompt",
        "```",
        stability_run.prompt,
        "```",
        "",
    ]

    if stability_run.system_prompt:
        lines.extend([
            "## System Prompt",
            "```",
            stability_run.system_prompt,
            "```",
            "",
        ])

    lines.append("## Regenerations")
    lines.append("")

    for regen in stability_run.regenerations:
        lines.append(f"### Regeneration {regen.index}")
        lines.append(f"- **Latency:** {regen.latency_ms:.0f}ms")
        lines.append(f"- **Timestamp:** {regen.timestamp}")

        if regen.thinking_blocks:
            lines.append(f"- **Thinking Blocks:** {len(regen.thinking_blocks)}")
            for j, block in enumerate(regen.thinking_blocks, 1):
                stage_name = block.get("stage_name", f"Stage {j}")
                lines.append(f"  - {stage_name}")

        lines.append("")
        lines.append("**Response:**")
        lines.append("")
        lines.append(regen.response)
        lines.append("")
        lines.append("---")
        lines.append("")

    temp_dir = tempfile.gettempdir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(temp_dir, f"stability_run_{timestamp}.md")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return f"Exported {len(stability_run.regenerations)} regenerations", gr.update(visible=True, value=filepath)
