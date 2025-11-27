"""
Gradio UI definition for prompt-prix.
"""

import gradio as gr

from prompt_prix import state
from prompt_prix.config import (
    get_default_servers,
    DEFAULT_MODELS, DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS, DEFAULT_MAX_TOKENS
)
from prompt_prix.handlers import (
    fetch_available_models,
    initialize_session,
    send_single_prompt,
    run_batch_prompts,
    export_markdown,
    export_json,
    launch_beyond_compare
)


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="prompt-prix", theme=gr.themes.Soft()) as app:
        gr.Markdown("# prompt-prix")
        gr.Markdown("Compare responses from multiple LLMs served via LM Studio.")

        # ─────────────────────────────────────────────────────────────
        # CONFIGURATION PANEL
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("Configuration", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    servers_input = gr.Textbox(
                        label="LM Studio Servers (one per line)",
                        value="\n".join(get_default_servers()),
                        lines=3,
                        placeholder="http://192.168.1.10:1234\nhttp://192.168.1.11:1234"
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        models_input = gr.Textbox(
                            label="Models to Compare (one per line)",
                            value="\n".join(DEFAULT_MODELS),
                            lines=5,
                            placeholder="llama-3.2-3b-instruct\nqwen2.5-7b-instruct"
                        )
                    fetch_models_button = gr.Button("🔄 Fetch Available Models", size="sm")

            with gr.Row():
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=DEFAULT_TEMPERATURE
                )
                timeout_slider = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=DEFAULT_TIMEOUT_SECONDS
                )
                max_tokens_slider = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=DEFAULT_MAX_TOKENS
                )

            with gr.Row():
                system_prompt_file = gr.File(
                    label="System Prompt File (optional)",
                    file_types=[".txt"],
                    type="filepath"
                )
                init_button = gr.Button("Initialize Session", variant="primary")

        # ─────────────────────────────────────────────────────────────
        # STATUS DISPLAY
        # ─────────────────────────────────────────────────────────────

        status_display = gr.Textbox(
            label="Status",
            value="Session not initialized",
            interactive=False
        )

        # ─────────────────────────────────────────────────────────────
        # INPUT PANEL
        # ─────────────────────────────────────────────────────────────

        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                send_button = gr.Button("Send Prompt", variant="primary")

            with gr.Column(scale=1):
                batch_file = gr.File(
                    label="Batch Prompts File",
                    file_types=[".txt"],
                    type="filepath"
                )
                batch_button = gr.Button("Run Batch")

        # ─────────────────────────────────────────────────────────────
        # MODEL OUTPUT TABS
        # ─────────────────────────────────────────────────────────────

        # Create tabs for up to 10 models (can be extended)
        model_outputs = []
        with gr.Tabs():
            for i in range(10):
                with gr.Tab(f"Model {i + 1}"):
                    output = gr.Markdown(
                        value="",
                        label=f"Conversation"
                    )
                    model_outputs.append(output)

        # ─────────────────────────────────────────────────────────────
        # EXPORT PANEL
        # ─────────────────────────────────────────────────────────────

        with gr.Row():
            export_md_button = gr.Button("Export Markdown")
            export_json_button = gr.Button("Export JSON")

        export_preview = gr.Textbox(
            label="Export Preview",
            lines=10,
            interactive=False,
            visible=False
        )

        # ─────────────────────────────────────────────────────────────
        # BEYOND COMPARE PANEL
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("Compare Models (Beyond Compare)", open=False):
            gr.Markdown("Select two models to open their outputs in Beyond Compare for side-by-side diff.")
            with gr.Row():
                compare_model_a = gr.Dropdown(
                    label="Model A",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True
                )
                compare_model_b = gr.Dropdown(
                    label="Model B",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True
                )
            compare_button = gr.Button("Open in Beyond Compare", variant="primary")

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS
        # ─────────────────────────────────────────────────────────────

        fetch_models_button.click(
            fn=fetch_available_models,
            inputs=[servers_input],
            outputs=[status_display, models_input]
        )

        # Beyond Compare - update dropdowns after init
        def get_model_choices():
            if state.session is None:
                return gr.update(choices=[]), gr.update(choices=[])
            models = state.session.state.models
            return gr.update(choices=models, value=models[0] if models else None), \
                   gr.update(choices=models, value=models[1] if len(models) > 1 else None)

        init_button.click(
            fn=initialize_session,
            inputs=[
                servers_input,
                models_input,
                system_prompt_file,
                temperature_slider,
                timeout_slider,
                max_tokens_slider
            ],
            outputs=[status_display] + model_outputs
        ).then(
            fn=get_model_choices,
            inputs=[],
            outputs=[compare_model_a, compare_model_b]
        )

        send_button.click(
            fn=send_single_prompt,
            inputs=[prompt_input],
            outputs=[status_display] + model_outputs
        )

        # Also send on Enter in prompt box
        prompt_input.submit(
            fn=send_single_prompt,
            inputs=[prompt_input],
            outputs=[status_display] + model_outputs
        )

        batch_button.click(
            fn=run_batch_prompts,
            inputs=[batch_file],
            outputs=[status_display] + model_outputs
        )

        export_md_button.click(
            fn=export_markdown,
            inputs=[],
            outputs=[status_display, export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[export_preview]
        )

        export_json_button.click(
            fn=export_json,
            inputs=[],
            outputs=[status_display, export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[export_preview]
        )

        compare_button.click(
            fn=launch_beyond_compare,
            inputs=[compare_model_a, compare_model_b],
            outputs=[status_display]
        )

    return app
