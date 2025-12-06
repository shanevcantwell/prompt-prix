import gradio as gr
from types import SimpleNamespace
from prompt_prix.config import (
    get_default_servers,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS
)

def render_tab():
    """Render the Battery tab and return its components."""
    with gr.Tab("🔋 Battery", id="battery-tab"):

        gr.Markdown("""
        Run benchmark test suites across multiple models.
        Upload a test file, select models, and see results in the grid below.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                battery_file = gr.File(
                    label="Test Suite (JSON/JSONL)",
                    file_types=[".json", ".jsonl"],
                    type="filepath"
                )
                battery_validation = gr.Textbox(
                    label="Validation",
                    value="Upload a benchmark file",
                    interactive=False,
                    lines=1
                )

                with gr.Row():
                    battery_fetch_btn = gr.Button(
                        "🔄 Fetch",
                        variant="secondary",
                        size="sm"
                    )
                    only_loaded_checkbox = gr.Checkbox(
                        label="Only Loaded",
                        value=False,
                        info="Filter to models currently in memory"
                    )
                    gemini_checkbox = gr.Checkbox(
                        label="Gemini",
                        value=False,
                        info="Include Gemini Web UI (requires session)"
                    )

                battery_models = gr.CheckboxGroup(
                    label="Models to Test",
                    choices=[],
                    value=[],
                    elem_id="battery-models"
                )

            with gr.Column(scale=1):
                servers_input = gr.Textbox(
                    label="LM Studio Servers (one per line)",
                    value="\n".join(get_default_servers()),
                    lines=2,
                    placeholder="http://localhost:1234",
                    elem_id="servers"
                )
                battery_temp = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=DEFAULT_TEMPERATURE
                )
                battery_timeout = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=DEFAULT_TIMEOUT_SECONDS
                )
                battery_max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=DEFAULT_MAX_TOKENS
                )
                battery_system_prompt = gr.Textbox(
                    label="System Prompt Override (optional)",
                    placeholder="Leave empty to use test-defined prompts",
                    lines=2
                )
                judge_model = gr.Dropdown(
                    label="Judge Model",
                    choices=[],
                    value=None,
                    info="Model for LLM-as-judge evaluation"
                )

                gr.Markdown("---")
                gr.Markdown("**Quick Prompt**")
                quick_prompt = gr.Textbox(
                    label="Manual Prompt",
                    placeholder="Enter a prompt to test against selected models",
                    lines=2
                )
                quick_prompt_btn = gr.Button(
                    "⚡ Run Prompt",
                    variant="secondary",
                    size="sm"
                )
                quick_prompt_output = gr.Markdown(
                    value="*Results will appear here*",
                    label="Quick Prompt Results"
                )

        with gr.Row():
            battery_run_btn = gr.Button(
                "▶️ Run Battery",
                variant="primary",
                interactive=False,
                scale=2
            )
            battery_stop_btn = gr.Button(
                "⏹️ Stop",
                variant="stop",
                scale=1
            )

        battery_status = gr.Textbox(
            label="Status",
            value="Ready",
            interactive=False
        )

        gr.Markdown("### Results")
        battery_display_mode = gr.Radio(
            label="Display",
            choices=["Symbols (✓/❌)", "Latency (seconds)"],
            value="Symbols (✓/❌)",
            interactive=True
        )

        battery_grid = gr.Dataframe(
            label="Model × Test Results",
            headers=["Model"],
            interactive=False,
            wrap=True,
            elem_id="battery-grid"
        )

        with gr.Accordion("📋 Response Detail", open=False):
            with gr.Row():
                detail_model = gr.Dropdown(
                    label="Model",
                    choices=[],
                    interactive=True
                )
                detail_test = gr.Dropdown(
                    label="Test",
                    choices=[],
                    interactive=True
                )
                detail_refresh_btn = gr.Button("Show", size="sm")

            detail_content = gr.Markdown(
                value="*Select a model and test to view the response*"
            )

        with gr.Row():
            battery_export_json_btn = gr.Button("Export JSON")
            battery_export_csv_btn = gr.Button("Export CSV")

        battery_export_file = gr.File(
            label="Download",
            visible=False
        )

    return SimpleNamespace(
        file=battery_file,
        validation=battery_validation,
        fetch_btn=battery_fetch_btn,
        only_loaded_checkbox=only_loaded_checkbox,
        gemini_checkbox=gemini_checkbox,
        models=battery_models,
        servers_input=servers_input,
        temp=battery_temp,
        timeout=battery_timeout,
        max_tokens=battery_max_tokens,
        system_prompt=battery_system_prompt,
        judge_model=judge_model,
        quick_prompt=quick_prompt,
        quick_prompt_btn=quick_prompt_btn,
        quick_prompt_output=quick_prompt_output,
        run_btn=battery_run_btn,
        stop_btn=battery_stop_btn,
        status=battery_status,
        display_mode=battery_display_mode,
        grid=battery_grid,
        detail_model=detail_model,
        detail_test=detail_test,
        detail_refresh_btn=detail_refresh_btn,
        detail_content=detail_content,
        export_json_btn=battery_export_json_btn,
        export_csv_btn=battery_export_csv_btn,
        export_file=battery_export_file
    )
