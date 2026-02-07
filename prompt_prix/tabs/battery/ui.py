import gradio as gr
from types import SimpleNamespace


def render_tab():
    """Render the Battery tab and return its components.

    Server config, model selection, timeout, and max_tokens are now in
    the shared header above tabs.
    """
    with gr.Tab("🔋 Battery", id="battery-tab"):

        with gr.Accordion("ℹ️ Battery Help", open=False):
            gr.Markdown("""
**Run benchmarks across multiple models simultaneously.**

### Quick Start
1. Upload a test file (JSON/JSONL/YAML) or click **Load Sample**
2. Click **Run Battery** to test all selected models
3. Click any cell to see the full response

### Understanding Results
| Symbol | Meaning |
|--------|---------|
| ✓ | Passed - correct tool call or expected response |
| ❌ | Failed - wrong tool, missing call, or semantic failure |
| ⚠ | Error - API timeout, parse error, or model refusal |
| 🟣 | Inconsistent - passed some runs but not all (multi-run mode)

### Judge Model
Select a judge model to evaluate responses against `pass_criteria`/`fail_criteria` fields in your test file. Without a judge, results show ✓ for any successful response.

### Test File Format
Each test needs: `id`, `user` (the prompt). Optional: `tools`, `system`, `pass_criteria`, `fail_criteria`

[View example test files →](https://github.com/shanevcantwell/prompt-prix/tree/main/examples)
            """)

        with gr.Row():
            with gr.Column(scale=1):
                battery_file = gr.File(
                    label="Test Suite",
                    file_types=[".json", ".jsonl", ".yaml", ".yml"],
                    type="filepath"
                )
                with gr.Row():
                    battery_validation = gr.Textbox(
                        label="Validation",
                        value="Upload a benchmark file or load sample",
                        interactive=False,
                        lines=1,
                        scale=3
                    )
                    load_sample_btn = gr.Button(
                        "📋 Load Sample",
                        size="sm",
                        scale=1
                    )

            with gr.Column(scale=1):
                battery_system_prompt = gr.Textbox(
                    label="System Prompt Override",
                    placeholder="Leave empty to use test-defined prompts",
                    lines=2,
                    info="Applies to all tests (overrides per-test prompts)"
                )
                judge_model = gr.Dropdown(
                    label="Judge Model (Optional)",
                    choices=[],
                    value=None,
                    visible=False,
                    info="If set, uses this LLM to evaluate responses against test's pass_criteria/fail_criteria fields"
                )
                drift_threshold = gr.Slider(
                    label="Drift Threshold",
                    minimum=0.0,
                    maximum=0.5,
                    step=0.025,
                    value=0.0,
                    info="Cosine distance to expected_response (0 = disabled). Lower = stricter."
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
        with gr.Row():
            battery_runs_slider = gr.Slider(
                label="Runs",
                minimum=1,
                maximum=10,
                step=1,
                value=1,
                scale=1,
                info="Run each test N times with different seeds to detect inconsistent models"
            )
            battery_display_mode = gr.Radio(
                label="Display Mode",
                choices=["Symbols (✓/❌)", "Latency (seconds)"],
                value="Symbols (✓/❌)",
                interactive=True,
                scale=2,
                info="Toggle between pass/fail symbols and response times"
            )

        battery_grid = gr.Dataframe(
            label="Model × Test Results",
            interactive=True,  # Required for cell selection in Gradio 6.x
            wrap=True,
            elem_id="battery-grid"
        )

        # Dismissible detail dialog (hidden by default, ADR-009)
        with gr.Column(visible=False) as detail_dialog:
            gr.Markdown("### Response Detail")
            detail_markdown = gr.Markdown()
            detail_close_btn = gr.Button("Close", size="sm")

        with gr.Accordion("📋 Response Detail", open=False, visible=False):
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
            battery_export_image_btn = gr.Button("Export Image")

        battery_export_file = gr.File(
            label="Download",
            visible=False
        )

    return SimpleNamespace(
        file=battery_file,
        validation=battery_validation,
        load_sample_btn=load_sample_btn,
        system_prompt=battery_system_prompt,
        judge_model=judge_model,
        drift_threshold=drift_threshold,
        runs_slider=battery_runs_slider,
        run_btn=battery_run_btn,
        stop_btn=battery_stop_btn,
        status=battery_status,
        display_mode=battery_display_mode,
        grid=battery_grid,
        # ADR-009: Dismissible detail dialog
        detail_dialog=detail_dialog,
        detail_markdown=detail_markdown,
        detail_close_btn=detail_close_btn,
        # Legacy accordion detail (hidden)
        detail_model=detail_model,
        detail_test=detail_test,
        detail_refresh_btn=detail_refresh_btn,
        detail_content=detail_content,
        export_json_btn=battery_export_json_btn,
        export_csv_btn=battery_export_csv_btn,
        export_image_btn=battery_export_image_btn,
        export_file=battery_export_file
    )
