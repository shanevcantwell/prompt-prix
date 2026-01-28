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
        gr.Markdown("Find your optimal open-weights model.")

        available_models = gr.State([])

        # ─────────────────────────────────────────────────────────────
        # SHARED HEADER: Server config + Model selection (collapsible)
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("⚙️ Server & Model Configuration", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    servers_input = gr.Textbox(
                        label="LM Studio Servers (one per line)",
                        value="\n".join(get_default_servers()),
                        lines=2,
                        placeholder="http://localhost:1234",
                        elem_id="servers"
                    )
                    with gr.Row():
                        fetch_btn = gr.Button(
                            "🔄 Fetch Models",
                            variant="secondary",
                            size="sm"
                        )
                        only_loaded_checkbox = gr.Checkbox(
                            label="Only Loaded",
                            value=False,
                            info="Filter to models in memory"
                        )

                with gr.Column(scale=2):
                    models_checkbox = gr.CheckboxGroup(
                        label="Models",
                        choices=[],
                        value=[],
                        elem_id="models"
                    )

            with gr.Row():
                timeout_slider = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=DEFAULT_TIMEOUT_SECONDS,
                    scale=1
                )
                max_tokens_slider = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=DEFAULT_MAX_TOKENS,
                    scale=1
                )

        gr.Markdown("---")

        # ─────────────────────────────────────────────────────────────
        # MAIN TABS
        # ─────────────────────────────────────────────────────────────

        with gr.Tabs() as main_tabs:

            # Render tabs and get their components
            battery = battery_ui.render_tab()
            compare = compare_ui.render_tab()

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Shared Header
        # ─────────────────────────────────────────────────────────────

        async def on_fetch_models(servers_text, only_loaded):
            """Fetch models and update model selectors.

            Applies 'only loaded' filter locally for models_checkbox.
            Judge dropdown always shows all available models (unfiltered).
            """
            result = await fetch_available_models(servers_text)
            all_models = result["all_models"]
            loaded_models = result.get("loaded_models") or set()

            # Filter for checkbox if requested
            if only_loaded and loaded_models:
                checkbox_models = [m for m in all_models if m in loaded_models]
            else:
                checkbox_models = all_models

            return (
                all_models,
                gr.update(choices=sorted(checkbox_models), value=[]),  # models_checkbox - filtered
                gr.update(choices=sorted(all_models)),  # battery.detail_model - unfiltered
                gr.update(choices=sorted(all_models)),  # battery.judge_model - unfiltered
            )

        fetch_btn.click(
            fn=on_fetch_models,
            inputs=[servers_input, only_loaded_checkbox],
            outputs=[
                available_models,
                models_checkbox,
                battery.detail_model,
                battery.judge_model,
            ]
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

        battery.run_btn.click(
            fn=battery_handlers.run_handler,
            inputs=[
                battery.file, models_checkbox, servers_input,
                timeout_slider, max_tokens_slider, battery.system_prompt,
                battery.judge_model
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
