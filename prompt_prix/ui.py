"""
Gradio UI definition for prompt-prix.

Battery-first design: Model × Test grid is the primary interface.
Interactive comparison is secondary (Compare tab).
"""

import gradio as gr

from prompt_prix import state
from prompt_prix.config import (
    get_default_servers,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS
)
from prompt_prix.handlers import (
    fetch_available_models,
    initialize_session,
    send_single_prompt,
    export_markdown,
    export_json,
    battery_validate_file,
    battery_run_handler,
    battery_get_cell_detail,
    battery_get_test_ids,
    battery_quick_prompt_handler,
    battery_export_json,
    battery_export_csv,
    handle_stop
)
from prompt_prix.parsers import get_default_system_prompt
from prompt_prix.ui_helpers import (
    CUSTOM_CSS,
    TAB_STATUS_JS,
    PERSISTENCE_LOAD_JS,
    SAVE_SERVERS_JS,
    SAVE_TEMPERATURE_JS,
)


# ─────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    # Gradio 5.x: theme/css on Blocks(); Gradio 6.x: on launch()
    import inspect
    blocks_params = inspect.signature(gr.Blocks).parameters
    blocks_kwargs = {"title": "prompt-prix"}

    if "theme" in blocks_params:
        # Gradio 5.x style
        blocks_kwargs["theme"] = gr.themes.Soft()
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as app:
        
        gr.Markdown("# prompt-prix")
        gr.Markdown("Find your optimal open-weights model.")

        # ─────────────────────────────────────────────────────────────
        # SHARED: SERVER CONFIGURATION
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("🖥️ Servers", open=False):
            with gr.Row():
                servers_input = gr.Textbox(
                    label="LM Studio Servers (one per line)",
                    value="\n".join(get_default_servers()),
                    lines=3,
                    placeholder="http://192.168.1.10:1234\nhttp://192.168.1.11:1234",
                    elem_id="servers",
                    scale=2
                )
                with gr.Column(scale=1):
                    fetch_models_btn = gr.Button("🔄 Fetch Models", variant="secondary")
                    server_status = gr.Textbox(
                        label="Status",
                        value="Click Fetch to discover models",
                        interactive=False,
                        lines=2
                    )
        
        # Shared model list (populated by fetch, used by both tabs)
        available_models = gr.State([])

        # ─────────────────────────────────────────────────────────────
        # MAIN TABS
        # ─────────────────────────────────────────────────────────────

        with gr.Tabs() as main_tabs:
            
            # ─────────────────────────────────────────────────────────
            # TAB 1: BATTERY (Primary)
            # ─────────────────────────────────────────────────────────
            
            with gr.Tab("🔋 Battery", id="battery-tab"):
                
                gr.Markdown("""
                Run benchmark test suites across multiple models. 
                Upload a test file, select models, and see results in the grid below.
                """)
                
                with gr.Row():
                    # Left column: Test suite + Models
                    with gr.Column(scale=1):
                        battery_file = gr.File(
                            label="Test Suite (JSON)",
                            file_types=[".json"],
                            type="filepath"
                        )
                        battery_validation = gr.Textbox(
                            label="Validation",
                            value="Upload a benchmark file",
                            interactive=False,
                            lines=1
                        )
                        
                        battery_models = gr.CheckboxGroup(
                            label="Models to Test",
                            choices=[],
                            value=[],
                            elem_id="battery-models"
                        )
                    
                    # Right column: Config
                    with gr.Column(scale=1):
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
                            lines=3
                        )

                        # Quick prompt for ad-hoc testing
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
                
                # ─────────────────────────────────────────────────────
                # HERO: Results Grid
                # ─────────────────────────────────────────────────────
                
                gr.Markdown("### Results")
                gr.Markdown("*Rows: Models | Columns: Tests | Cells: ✓ completed, ❌ error, ⏳ running, — pending*")
                
                battery_grid = gr.Dataframe(
                    label="Model × Test Results",
                    headers=["Model"],
                    interactive=False,
                    wrap=True,
                    elem_id="battery-grid"
                )
                
                # Detail view for clicked cell
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
                
                # Export
                with gr.Row():
                    battery_export_json_btn = gr.Button("Export JSON")
                    battery_export_csv_btn = gr.Button("Export CSV")

                battery_export_preview = gr.Textbox(
                    label="Export Preview",
                    lines=10,
                    interactive=False,
                    visible=False
                )

            # ─────────────────────────────────────────────────────────
            # TAB 2: COMPARE (Secondary)
            # ─────────────────────────────────────────────────────────
            
            with gr.Tab("💬 Compare", id="compare-tab"):

                gr.Markdown("""
                Interactive multi-turn comparison. Send prompts to multiple models
                and see responses side-by-side with conversation history.
                """)

                with gr.Row():
                    # Left column: Models
                    with gr.Column(scale=1):
                        compare_models = gr.CheckboxGroup(
                            label="Models to Compare",
                            choices=[],
                            value=[],
                            elem_id="compare-models"
                        )

                    # Right column: Config + Quick Prompt
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

                        # Quick prompt for comparison
                        gr.Markdown("---")
                        gr.Markdown("**Prompt**")
                        compare_prompt = gr.Textbox(
                            label="User Message",
                            placeholder="Enter your prompt here...",
                            lines=2
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
                
                # Hidden state for tab colors
                tab_states = gr.JSON(value=[], visible=False, elem_id="tab-states")
                
                # Model output tabs (up to 10)
                model_outputs = []
                with gr.Tabs(elem_id="model-tabs"):
                    for i in range(10):
                        with gr.Tab(f"Model {i + 1}"):
                            output = gr.Markdown(value="", label="Conversation")
                            model_outputs.append(output)
                
                # Export
                with gr.Row():
                    compare_export_md_btn = gr.Button("Export Markdown")
                    compare_export_json_btn = gr.Button("Export JSON")
                
                compare_export_preview = gr.Textbox(
                    label="Export Preview",
                    lines=10,
                    interactive=False,
                    visible=False
                )

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Shared
        # ─────────────────────────────────────────────────────────────

        async def on_fetch_models(servers_text):
            """Fetch models and update both tabs' checkboxes."""
            status, models_update = await fetch_available_models(servers_text)
            choices = models_update.get("choices", []) if isinstance(models_update, dict) else []
            return (
                status,
                choices,  # State
                gr.update(choices=choices),  # Battery models
                gr.update(choices=choices),  # Compare models
                gr.update(choices=choices),  # Detail model dropdown
            )

        fetch_models_btn.click(
            fn=on_fetch_models,
            inputs=[servers_input],
            outputs=[
                server_status,
                available_models,
                battery_models,
                compare_models,
                detail_model
            ]
        )

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Battery Tab
        # ─────────────────────────────────────────────────────────────

        def on_battery_file_change(file_obj):
            """Validate file and enable/disable run button."""
            if file_obj is None:
                return "Upload a benchmark file", gr.update(interactive=False), gr.update(choices=[])

            validation_result = battery_validate_file(file_obj)
            is_valid = validation_result.startswith("✅")

            # Extract test IDs for detail dropdown (if valid)
            test_choices = battery_get_test_ids(file_obj) if is_valid else []

            return (
                validation_result,
                gr.update(interactive=is_valid),
                gr.update(choices=test_choices)
            )

        battery_file.change(
            fn=on_battery_file_change,
            inputs=[battery_file],
            outputs=[battery_validation, battery_run_btn, detail_test]
        )

        battery_run_btn.click(
            fn=battery_run_handler,
            inputs=[
                battery_file,
                battery_models,
                servers_input,
                battery_temp,
                battery_timeout,
                battery_max_tokens,
                battery_system_prompt
            ],
            outputs=[battery_status, battery_grid]
        )

        # Quick prompt
        quick_prompt_btn.click(
            fn=battery_quick_prompt_handler,
            inputs=[
                quick_prompt,
                battery_models,
                servers_input,
                battery_temp,
                battery_timeout,
                battery_max_tokens,
                battery_system_prompt
            ],
            outputs=[quick_prompt_output]
        )

        # Stop button - signals cancellation to running handlers
        battery_stop_btn.click(
            fn=handle_stop,
            inputs=[],
            outputs=[battery_status]
        )

        # Detail view
        detail_refresh_btn.click(
            fn=battery_get_cell_detail,
            inputs=[detail_model, detail_test],
            outputs=[detail_content]
        )

        # Battery export
        battery_export_json_btn.click(
            fn=battery_export_json,
            inputs=[],
            outputs=[battery_status, battery_export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[battery_export_preview]
        )

        battery_export_csv_btn.click(
            fn=battery_export_csv,
            inputs=[],
            outputs=[battery_status, battery_export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[battery_export_preview]
        )

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Compare Tab
        # ─────────────────────────────────────────────────────────────

        async def compare_send_with_auto_init(
            prompt, tools, servers_text, models_selected,
            system_prompt, temperature, timeout, max_tokens
        ):
            """Auto-initialize session if needed, then send prompt."""
            # Auto-initialize if session doesn't exist or models changed
            if (state.session is None or
                set(state.session.state.models) != set(models_selected)):
                init_status, *init_outputs = await initialize_session(
                    servers_text, models_selected, system_prompt,
                    temperature, timeout, max_tokens
                )
                if "❌" in init_status or "⚠️" in init_status:
                    # Initialization failed
                    yield (init_status, []) + tuple(init_outputs)
                    return

            # Now send the prompt
            async for result in send_single_prompt(prompt, tools):
                yield result

        compare_send_btn.click(
            fn=compare_send_with_auto_init,
            inputs=[
                compare_prompt, compare_tools, servers_input, compare_models,
                compare_system_prompt, compare_temp, compare_timeout, compare_max_tokens
            ],
            outputs=[compare_status, tab_states] + model_outputs
        )

        compare_prompt.submit(
            fn=compare_send_with_auto_init,
            inputs=[
                compare_prompt, compare_tools, servers_input, compare_models,
                compare_system_prompt, compare_temp, compare_timeout, compare_max_tokens
            ],
            outputs=[compare_status, tab_states] + model_outputs
        )

        tab_states.change(
            fn=None,
            inputs=[tab_states],
            outputs=[tab_states],
            js=TAB_STATUS_JS
        )

        compare_export_md_btn.click(
            fn=export_markdown,
            inputs=[],
            outputs=[compare_status, compare_export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[compare_export_preview]
        )

        compare_export_json_btn.click(
            fn=export_json,
            inputs=[],
            outputs=[compare_status, compare_export_preview]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[compare_export_preview]
        )

        # ─────────────────────────────────────────────────────────────
        # PERSISTENCE: Load from localStorage, save on change
        # ─────────────────────────────────────────────────────────────

        # Load saved settings (returns undefined to preserve .env defaults if empty)
        app.load(
            fn=None,
            inputs=[],
            outputs=[servers_input, battery_temp, compare_temp],
            js=PERSISTENCE_LOAD_JS
        )

        # Save servers to localStorage on change (debounced by Gradio for textboxes)
        servers_input.change(
            fn=None,
            inputs=[servers_input],
            outputs=[servers_input],
            js=SAVE_SERVERS_JS
        )

        # Save temperature to localStorage on change (syncs both sliders)
        battery_temp.change(
            fn=None,
            inputs=[battery_temp],
            outputs=[battery_temp],
            js=SAVE_TEMPERATURE_JS
        )
        compare_temp.change(
            fn=None,
            inputs=[compare_temp],
            outputs=[compare_temp],
            js=SAVE_TEMPERATURE_JS
        )

    return app