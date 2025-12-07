import gradio as gr
from types import SimpleNamespace
from prompt_prix.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS
)
from prompt_prix.parsers import get_default_system_prompt

def render_tab():
    """Render the Compare tab and return its components."""
    with gr.Tab("💬 Compare", id="compare-tab"):

        gr.Markdown("""
        Interactive multi-turn comparison. Send prompts to multiple models
        and see responses side-by-side with conversation history.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    compare_models = gr.CheckboxGroup(
                        label="Models to Compare",
                        choices=[],
                        value=[],
                        elem_id="compare-models",
                        scale=3
                    )
                    compare_fetch_btn = gr.Button(
                        "🔄 Fetch",
                        variant="secondary",
                        size="sm",
                        scale=1
                    )

            with gr.Column(scale=1):
                compare_temp = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=DEFAULT_TEMPERATURE
                )
                compare_timeout = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=DEFAULT_TIMEOUT_SECONDS
                )
                compare_max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=DEFAULT_MAX_TOKENS
                )
                compare_seed = gr.Number(
                    label="Seed (optional)",
                    value=None,
                    precision=0,
                    minimum=0,
                    maximum=2147483647,
                    info="Set for reproducible outputs"
                )
                compare_system_prompt = gr.Textbox(
                    label="System Prompt (optional)",
                    value=get_default_system_prompt(),
                    placeholder="System instructions for all models",
                    lines=3
                )

                with gr.Accordion("Tools (Function Calling)", open=False):
                    compare_tools = gr.Code(
                        label="Tools JSON (OpenAI format)",
                        language="json",
                        value="",
                        lines=8,
                        elem_id="tools"
                    )

                gr.Markdown("---")
                gr.Markdown("**Prompt**")
                compare_prompt = gr.Textbox(
                    label="User Message",
                    placeholder="Enter your prompt here...",
                    lines=2
                )
                compare_image = gr.Image(
                    label="Attach Image (optional)",
                    type="filepath",
                    height=150
                )
                compare_send_btn = gr.Button(
                    "⚡ Send to All",
                    variant="primary"
                )

        compare_status = gr.Textbox(
            label="Status",
            value="Select models and send a prompt",
            interactive=False
        )

        tab_states = gr.JSON(value=[], visible=False, elem_id="tab-states")

        model_outputs = []
        with gr.Tabs(elem_id="model-tabs"):
            for i in range(10):
                with gr.Tab(f"Model {i + 1}"):
                    output = gr.Markdown(value="", label="Conversation")
                    model_outputs.append(output)

        with gr.Row():
            compare_export_md_btn = gr.Button("Export Markdown")
            compare_export_json_btn = gr.Button("Export JSON")

        compare_export_preview = gr.Textbox(
            label="Export Preview",
            lines=10,
            interactive=False,
            visible=False
        )

    return SimpleNamespace(
        models=compare_models,
        fetch_btn=compare_fetch_btn,
        temp=compare_temp,
        timeout=compare_timeout,
        max_tokens=compare_max_tokens,
        seed=compare_seed,
        system_prompt=compare_system_prompt,
        tools=compare_tools,
        prompt=compare_prompt,
        image=compare_image,
        send_btn=compare_send_btn,
        status=compare_status,
        tab_states=tab_states,
        model_outputs=model_outputs,
        export_md_btn=compare_export_md_btn,
        export_json_btn=compare_export_json_btn,
        export_preview=compare_export_preview
    )
