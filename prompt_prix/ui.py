"""
Gradio UI definition for prompt-prix.

Battery-first design: Model × Test grid is the primary interface.
Interactive comparison is secondary (Compare tab).

Tab-specific handlers are in prompt_prix.tabs.{battery,compare}.handlers
Tab-specific UI layouts are in prompt_prix.tabs.{battery,compare}.ui
"""

import gradio as gr

from prompt_prix import state
from prompt_prix.handlers import fetch_available_models, handle_stop
from prompt_prix.config import (
    get_default_servers,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    TOGETHER_DEFAULT_MODELS,
)
from prompt_prix.ui_helpers import (
    CUSTOM_CSS,
    TAB_STATUS_JS,
    PERSISTENCE_LOAD_JS,
    SAVE_SERVERS_JS,
    AUTO_DOWNLOAD_JS,
)

# Import tab-specific handlers
from prompt_prix.tabs.battery import handlers as battery_handlers
from prompt_prix.tabs.compare import handlers as compare_handlers

# Import tab-specific UI layouts
from prompt_prix.tabs.battery import ui as battery_ui
from prompt_prix.tabs.compare import ui as compare_ui


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    # Gradio 5.x: theme/css on Blocks(); Gradio 6.x: on launch()
    import inspect
    blocks_params = inspect.signature(gr.Blocks).parameters
    blocks_kwargs = {"title": "prompt-prix"}

    if "theme" in blocks_params:
        blocks_kwargs["theme"] = gr.themes.Soft()
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as app:

        gr.Markdown("# prompt-prix")
        gr.Markdown("Audit LLM function calling reliability across multiple models.")

        available_models = gr.State([])

        # ─────────────────────────────────────────────────────────────
        # SHARED HEADER: Model selection (HF Spaces mode - simplified)
        # ─────────────────────────────────────────────────────────────

        # Hidden servers input for handler compatibility
        servers_input = gr.Textbox(value="huggingface-inference", visible=False)

        with gr.Accordion("⚙️ Model Configuration", open=True):
            with gr.Row():
                models_checkbox = gr.CheckboxGroup(
                    label="Select Models to Compare",
                    choices=TOGETHER_DEFAULT_MODELS,
                    value=TOGETHER_DEFAULT_MODELS,  # All selected by default
                    elem_id="models",
                    info="Models run via Together AI"
                )

            with gr.Row():
                timeout_slider = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=120,
                    scale=1,
                    interactive=False,
                    info="Fixed at 120s for demo"
                )
                max_tokens_slider = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=256,
                    scale=1,
                    interactive=False,
                    info="Fixed at 256 for demo"
                )

        # Hidden components for handler compatibility (run_handler expects 11 inputs)
        parallel_slots_hidden = gr.Slider(value=1, visible=False)
        drift_threshold_hidden = gr.Slider(value=0.0, visible=False)

        gr.Markdown("---")

        # ─────────────────────────────────────────────────────────────
        # MAIN TABS
        # ─────────────────────────────────────────────────────────────

        with gr.Tabs() as main_tabs:

            # Render tabs and get their components
            battery = battery_ui.render_tab()
            compare = compare_ui.render_tab()

        # ─────────────────────────────────────────────────────────────
        # HF SPACES MODE: Pre-populate dropdowns with Together models
        # ─────────────────────────────────────────────────────────────

        # Initialize battery dropdowns with Together models on load
        app.load(
            fn=lambda: (
                gr.update(choices=TOGETHER_DEFAULT_MODELS),
                gr.update(choices=TOGETHER_DEFAULT_MODELS),
            ),
            outputs=[battery.detail_model, battery.judge_model]
        )

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Battery Tab
        # ─────────────────────────────────────────────────────────────

        def on_battery_file_change(file_obj):
            # Clear previous battery run state when file changes
            state.battery_run = None

            if file_obj is None:
                state.battery_source_file = None
                return (
                    "Upload a benchmark file",
                    gr.update(interactive=False),
                    gr.update(choices=[]),
                    []  # Clear grid
                )

            # Store source filename for export naming
            state.battery_source_file = file_obj

            validation_result = battery_handlers.validate_file(file_obj)
            is_valid = validation_result.startswith("✅")
            test_choices = battery_handlers.get_test_ids(file_obj) if is_valid else []

            return (
                validation_result,
                gr.update(interactive=is_valid),
                gr.update(choices=test_choices),
                []  # Clear grid for new file
            )

        battery.file.change(
            fn=on_battery_file_change,
            inputs=[battery.file],
            outputs=[battery.validation, battery.run_btn, battery.detail_test, battery.grid]
        )

        # Load sample tests button
        import os
        SAMPLE_FILE = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples", "tool_competence_tests.json"
        )

        def load_sample_tests():
            """Load the bundled sample test file."""
            if os.path.exists(SAMPLE_FILE):
                return SAMPLE_FILE
            return None

        battery.load_sample_btn.click(
            fn=load_sample_tests,
            inputs=[],
            outputs=[battery.file]
        )

        battery.run_btn.click(
            fn=battery_handlers.run_handler,
            inputs=[
                battery.file, models_checkbox, servers_input,
                timeout_slider, max_tokens_slider, battery.system_prompt,
                battery.judge_model, battery.runs_slider, battery.display_mode,
                parallel_slots_hidden, drift_threshold_hidden
            ],
            outputs=[battery.status, battery.grid]
        )

        battery.stop_btn.click(fn=handle_stop, inputs=[], outputs=[battery.status])

        battery.display_mode.change(
            fn=battery_handlers.refresh_grid,
            inputs=[battery.display_mode],
            outputs=[battery.grid]
        )

        battery.detail_refresh_btn.click(
            fn=battery_handlers.get_cell_detail,
            inputs=[battery.detail_model, battery.detail_test],
            outputs=[battery.detail_content]
        )

        battery.export_json_btn.click(
            fn=battery_handlers.export_json,
            inputs=[],
            outputs=[battery.status, battery.export_file]
        )

        battery.export_csv_btn.click(
            fn=battery_handlers.export_csv,
            inputs=[],
            outputs=[battery.status, battery.export_file]
        )

        battery.export_image_btn.click(
            fn=battery_handlers.export_grid_image,
            inputs=[],
            outputs=[battery.status, battery.export_file]
        )

        # Auto-download when export file is ready
        battery.export_file.change(
            fn=None,
            inputs=[battery.export_file],
            outputs=[battery.export_file],
            js=AUTO_DOWNLOAD_JS
        )

        # ADR-009: Show dialog on cell select
        battery.grid.select(
            fn=battery_handlers.handle_cell_select,
            inputs=[],
            outputs=[battery.detail_dialog, battery.detail_markdown]
        )

        # ADR-009: Hide dialog on close button
        battery.detail_close_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[battery.detail_dialog]
        )

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Compare Tab
        # ─────────────────────────────────────────────────────────────

        async def compare_send_with_auto_init(
            prompt, tools, image, seed, repeat_penalty, servers_text,
            models_selected,
            system_prompt, timeout, max_tokens
        ):
            # Temperature fixed at 0.7 for interactive comparison (model default)
            temperature = 0.7

            if (state.session is None or
                set(state.session.state.models) != set(models_selected or [])):
                init_status, *init_outputs = await compare_handlers.initialize_session(
                    servers_text, models_selected or [], system_prompt,
                    temperature, timeout, max_tokens
                )
                if "❌" in init_status or "⚠️" in init_status:
                    yield (init_status, []) + tuple(init_outputs)
                    return

            async for result in compare_handlers.send_single_prompt(prompt, tools, image, seed, repeat_penalty):
                yield result

        compare.send_btn.click(
            fn=compare_send_with_auto_init,
            inputs=[
                compare.prompt, compare.tools, compare.image, compare.seed, compare.repeat_penalty,
                servers_input, models_checkbox,
                compare.system_prompt, timeout_slider, max_tokens_slider
            ],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.prompt.submit(
            fn=compare_send_with_auto_init,
            inputs=[
                compare.prompt, compare.tools, compare.image, compare.seed, compare.repeat_penalty,
                servers_input, models_checkbox,
                compare.system_prompt, timeout_slider, max_tokens_slider
            ],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.stop_btn.click(fn=handle_stop, inputs=[], outputs=[compare.status])

        compare.clear_btn.click(
            fn=compare_handlers.clear_session,
            inputs=[],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.tab_states.change(fn=None, inputs=[compare.tab_states], outputs=[compare.tab_states], js=TAB_STATUS_JS)

        # Re-apply tab styling when returning to Compare tab (fixes #115)
        main_tabs.select(
            fn=lambda tab_states: tab_states,
            inputs=[compare.tab_states],
            outputs=[compare.tab_states]
        )

        compare.export_md_btn.click(
            fn=compare_handlers.export_markdown,
            inputs=[],
            outputs=[compare.status, compare.export_file]
        )

        compare.export_json_btn.click(
            fn=compare_handlers.export_json,
            inputs=[],
            outputs=[compare.status, compare.export_file]
        )

        # Auto-download when export file is ready
        compare.export_file.change(
            fn=None,
            inputs=[compare.export_file],
            outputs=[compare.export_file],
            js=AUTO_DOWNLOAD_JS
        )

        # ─────────────────────────────────────────────────────────────
        # PERSISTENCE
        # ─────────────────────────────────────────────────────────────

        app.load(
            fn=None,
            inputs=[],
            outputs=[servers_input],
            js=PERSISTENCE_LOAD_JS
        )

        servers_input.change(fn=None, inputs=[servers_input], outputs=[servers_input], js=SAVE_SERVERS_JS)

    return app
