import gradio as gr
from types import SimpleNamespace
from prompt_prix.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS
)

def render_tab():
    """Render the Stability tab and return its components."""
    with gr.Tab("📊 Stability", id="stability-tab"):

        gr.Markdown("""
        Analyze regeneration stability. Run the same prompt multiple times
        against a model to observe output variance across regenerations.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                stability_gemini_checkbox = gr.Checkbox(
                    label="Use Gemini Web UI",
                    value=True,
                    info="Use Gemini instead of LM Studio models"
                )

                with gr.Row(visible=False) as stability_lmstudio_row:
                    stability_fetch_btn = gr.Button(
                        "🔄 Fetch",
                        variant="secondary",
                        size="sm"
                    )

                stability_model = gr.Dropdown(
                    label="LM Studio Model",
                    choices=[],
                    value=None,
                    info="Select model (or use Gemini above)",
                    visible=False
                )
                stability_regen_count = gr.Slider(
                    label="Regeneration Count",
                    minimum=2,
                    maximum=20,
                    step=1,
                    value=5,
                    info="Number of times to regenerate"
                )
                stability_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter the prompt to regenerate...",
                    lines=4
                )

                with gr.Row():
                    stability_run_btn = gr.Button(
                        "▶️ Run Regenerations",
                        variant="primary"
                    )
                    stability_stop_btn = gr.Button(
                        "⏹️ Stop",
                        variant="stop"
                    )

            with gr.Column(scale=1):
                stability_temp = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=DEFAULT_TEMPERATURE
                )
                stability_timeout = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=DEFAULT_TIMEOUT_SECONDS
                )
                stability_max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=DEFAULT_MAX_TOKENS
                )
                stability_system_prompt = gr.Textbox(
                    label="System Prompt (optional)",
                    placeholder="System instructions",
                    lines=2
                )
                stability_capture_thinking = gr.Checkbox(
                    label="Capture Thinking Blocks",
                    value=True,
                    info="Extract reasoning traces (Gemini only)"
                )

        stability_status = gr.Textbox(
            label="Status",
            value="Select a model and enter a prompt",
            interactive=False
        )

        gr.Markdown("### Regenerations")

        # Tabbed interface for each regeneration
        regen_outputs = []
        with gr.Tabs(elem_id="regen-tabs"):
            for i in range(20):  # Max regenerations
                with gr.Tab(f"Regen {i + 1}", visible=(i < 5)):
                    output = gr.Markdown(
                        value="*Waiting...*",
                        label=f"Regeneration {i + 1}"
                    )
                    regen_outputs.append(output)

        with gr.Row():
            stability_export_json_btn = gr.Button("Export JSON")
            stability_export_md_btn = gr.Button("Export Markdown")

        stability_export_file = gr.File(
            label="Download",
            visible=False
        )

    return SimpleNamespace(
        gemini_checkbox=stability_gemini_checkbox,
        lmstudio_row=stability_lmstudio_row,
        fetch_btn=stability_fetch_btn,
        model=stability_model,
        regen_count=stability_regen_count,
        prompt=stability_prompt,
        run_btn=stability_run_btn,
        stop_btn=stability_stop_btn,
        temp=stability_temp,
        timeout=stability_timeout,
        max_tokens=stability_max_tokens,
        system_prompt=stability_system_prompt,
        capture_thinking=stability_capture_thinking,
        status=stability_status,
        regen_outputs=regen_outputs,
        export_json_btn=stability_export_json_btn,
        export_md_btn=stability_export_md_btn,
        export_file=stability_export_file
    )
