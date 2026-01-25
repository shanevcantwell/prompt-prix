import gradio as gr
from types import SimpleNamespace
from prompt_prix.parsers import get_default_system_prompt


def render_tab():
    """Render the Compare tab and return its components.

    Server config, model selection, timeout, and max_tokens are now in
    the shared header above tabs.

    Compare is a workshop for constructing multi-turn test scenarios.
    Build conversation context, test tool calling, then export as Battery test cases.
    """
    with gr.Tab("💬 Compare", id="compare-tab"):

        gr.Markdown("""
        Multi-turn context engineering. Build conversation scenarios,
        test tool calling across models, then export as test cases for Battery.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                compare_system_prompt = gr.Textbox(
                    label="System Prompt",
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

                with gr.Accordion("Advanced", open=False):
                    compare_seed = gr.Number(
                        label="Seed (optional)",
                        value=None,
                        precision=0,
                        minimum=0,
                        maximum=2147483647,
                        info="Set for reproducible outputs"
                    )
                    compare_repeat_penalty = gr.Slider(
                        label="Repeat Penalty",
                        minimum=1.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.1,
                        info="Penalize repeated tokens (1.0 = off)"
                    )

            with gr.Column(scale=1):
                compare_prompt = gr.Textbox(
                    label="User Message",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                compare_image = gr.Image(
                    label="Attach Image (optional)",
                    type="filepath",
                    height=100
                )

        # Full-width button row (like Battery tab)
        with gr.Row():
            compare_send_btn = gr.Button(
                "⚡ Send to All",
                variant="primary",
                scale=2
            )
            compare_stop_btn = gr.Button(
                "🛑 Stop",
                variant="stop",
                scale=1
            )
            compare_clear_btn = gr.Button(
                "🗑️ Clear",
                variant="secondary",
                scale=1
            )

        compare_status = gr.Textbox(
            label="Status",
            value="Select models above and send a prompt",
            interactive=False
        )

        tab_states = gr.JSON(value=[], visible=False, elem_id="tab-states")

        model_outputs = []
        with gr.Tabs(elem_id="model-tabs"):
            for i in range(10):
                with gr.Tab(f"Model {i + 1}"):
                    output = gr.Markdown(
                        value="",
                        label="Conversation",
                        elem_classes=["model-output-content"]
                    )
                    model_outputs.append(output)

        with gr.Row():
            compare_export_md_btn = gr.Button("Export Markdown")
            compare_export_json_btn = gr.Button("Export JSON")

        compare_export_file = gr.File(
            label="Download",
            visible=False
        )

    return SimpleNamespace(
        system_prompt=compare_system_prompt,
        tools=compare_tools,
        seed=compare_seed,
        repeat_penalty=compare_repeat_penalty,
        prompt=compare_prompt,
        image=compare_image,
        send_btn=compare_send_btn,
        stop_btn=compare_stop_btn,
        clear_btn=compare_clear_btn,
        status=compare_status,
        tab_states=tab_states,
        model_outputs=model_outputs,
        export_md_btn=compare_export_md_btn,
        export_json_btn=compare_export_json_btn,
        export_file=compare_export_file
    )
