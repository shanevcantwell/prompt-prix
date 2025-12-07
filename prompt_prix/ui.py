"""
Gradio UI definition for prompt-prix.

Battery-first design: Model × Test grid is the primary interface.
Interactive comparison is secondary (Compare tab).

Tab-specific handlers are in prompt_prix.tabs.{battery,compare}.handlers
Tab-specific UI layouts are in prompt_prix.tabs.{battery,compare,stability}.ui
"""

import gradio as gr

from prompt_prix import state
from prompt_prix.handlers import fetch_available_models, handle_stop
from prompt_prix.ui_helpers import (
    CUSTOM_CSS,
    TAB_STATUS_JS,
    PERSISTENCE_LOAD_JS,
    SAVE_SERVERS_JS,
    SAVE_TEMPERATURE_JS,
)

# Import tab-specific handlers
from prompt_prix.tabs.battery import handlers as battery_handlers
from prompt_prix.tabs.compare import handlers as compare_handlers
from prompt_prix.tabs.stability import handlers as stability_handlers

# Import tab-specific UI layouts
from prompt_prix.tabs.battery import ui as battery_ui
from prompt_prix.tabs.compare import ui as compare_ui
from prompt_prix.tabs.stability import ui as stability_ui


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
        # MAIN TABS
        # ─────────────────────────────────────────────────────────────

        with gr.Tabs() as main_tabs:
            
            # Render tabs and get their components
            battery = battery_ui.render_tab()
            compare = compare_ui.render_tab()
            stability = stability_ui.render_tab()

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Shared
        # ─────────────────────────────────────────────────────────────

        async def on_fetch_models(servers_text, only_loaded, include_gemini):
            """Fetch models and update all tabs' model selectors."""
            status, models_update = await fetch_available_models(servers_text, only_loaded)
            choices = models_update.get("choices", []) if isinstance(models_update, dict) else []

            # Add Gemini if checkbox is checked
            if include_gemini:
                gemini_model = "gemini-2.0-flash-thinking (Web UI)"
                if gemini_model not in choices:
                    choices = [gemini_model] + list(choices)

            return (
                choices,
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
            )

        battery.fetch_btn.click(
            fn=on_fetch_models,
            inputs=[battery.servers_input, battery.only_loaded_checkbox, battery.gemini_checkbox],
            outputs=[
                available_models, 
                battery.models, 
                compare.models, 
                battery.detail_model, 
                battery.judge_model, 
                stability.model
            ]
        )

        compare.fetch_btn.click(
            fn=on_fetch_models,
            inputs=[battery.servers_input, battery.only_loaded_checkbox, battery.gemini_checkbox],
            outputs=[
                available_models, 
                battery.models, 
                compare.models, 
                battery.detail_model, 
                battery.judge_model, 
                stability.model
            ]
        )

        async def on_stability_fetch(servers_text, include_gemini):
            """Fetch models for Stability tab."""
            status, models_update = await fetch_available_models(servers_text, only_loaded=False)
            choices = models_update.get("choices", []) if isinstance(models_update, dict) else []

            if include_gemini:
                gemini_model = "gemini-2.0-flash-thinking (Web UI)"
                if gemini_model not in choices:
                    choices = [gemini_model] + list(choices)

            return gr.update(choices=choices)

        stability.fetch_btn.click(
            fn=on_stability_fetch,
            inputs=[battery.servers_input, stability.gemini_checkbox],
            outputs=[stability.model]
        )

        def on_gemini_checkbox_change(use_gemini):
            """Toggle visibility of LM Studio controls based on Gemini checkbox."""
            return (
                gr.update(visible=not use_gemini),  # stability_lmstudio_row
                gr.update(visible=not use_gemini),  # stability_model
            )

        stability.gemini_checkbox.change(
            fn=on_gemini_checkbox_change,
            inputs=[stability.gemini_checkbox],
            outputs=[stability.lmstudio_row, stability.model]
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
                battery.file, battery.models, battery.servers_input,
                battery.temp, battery.timeout, battery.max_tokens, battery.system_prompt
            ],
            outputs=[battery.status, battery.grid]
        )

        battery.quick_prompt_btn.click(
            fn=battery_handlers.quick_prompt_handler,
            inputs=[
                battery.quick_prompt, battery.models, battery.servers_input,
                battery.temp, battery.timeout, battery.max_tokens, battery.system_prompt
            ],
            outputs=[battery.quick_prompt_output]
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

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Compare Tab
        # ─────────────────────────────────────────────────────────────

        async def compare_send_with_auto_init(
            prompt, tools, image, seed, repeat_penalty, servers_text, models_selected,
            system_prompt, temperature, timeout, max_tokens
        ):
            if (state.session is None or
                set(state.session.state.models) != set(models_selected)):
                init_status, *init_outputs = await compare_handlers.initialize_session(
                    servers_text, models_selected, system_prompt,
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
                battery.servers_input, compare.models,
                compare.system_prompt, compare.temp, compare.timeout, compare.max_tokens
            ],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.prompt.submit(
            fn=compare_send_with_auto_init,
            inputs=[
                compare.prompt, compare.tools, compare.image, compare.seed, compare.repeat_penalty,
                battery.servers_input, compare.models,
                compare.system_prompt, compare.temp, compare.timeout, compare.max_tokens
            ],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.clear_btn.click(
            fn=compare_handlers.clear_session,
            inputs=[],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.tab_states.change(fn=None, inputs=[compare.tab_states], outputs=[compare.tab_states], js=TAB_STATUS_JS)

        compare.export_md_btn.click(
            fn=compare_handlers.export_markdown,
            inputs=[],
            outputs=[compare.status, compare.export_preview]
        ).then(fn=lambda: gr.update(visible=True), outputs=[compare.export_preview])

        compare.export_json_btn.click(
            fn=compare_handlers.export_json,
            inputs=[],
            outputs=[compare.status, compare.export_preview]
        ).then(fn=lambda: gr.update(visible=True), outputs=[compare.export_preview])

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Stability Tab
        # ─────────────────────────────────────────────────────────────

        stability.run_btn.click(
            fn=stability_handlers.run_regenerations,
            inputs=[
                stability.gemini_checkbox, stability.model, stability.prompt, stability.regen_count,
                battery.servers_input, stability.temp, stability.timeout,
                stability.max_tokens, stability.system_prompt, stability.capture_thinking
            ],
            outputs=[stability.status] + stability.regen_outputs
        )

        stability.stop_btn.click(fn=handle_stop, inputs=[], outputs=[stability.status])

        stability.export_json_btn.click(
            fn=stability_handlers.export_json,
            inputs=[],
            outputs=[stability.status, stability.export_file]
        )

        stability.export_md_btn.click(
            fn=stability_handlers.export_markdown,
            inputs=[],
            outputs=[stability.status, stability.export_file]
        )

        # ─────────────────────────────────────────────────────────────
        # PERSISTENCE
        # ─────────────────────────────────────────────────────────────

        app.load(
            fn=None,
            inputs=[],
            outputs=[battery.servers_input, battery.temp, compare.temp],
            js=PERSISTENCE_LOAD_JS
        )

        battery.servers_input.change(fn=None, inputs=[battery.servers_input], outputs=[battery.servers_input], js=SAVE_SERVERS_JS)
        battery.temp.change(fn=None, inputs=[battery.temp], outputs=[battery.temp], js=SAVE_TEMPERATURE_JS)
        compare.temp.change(fn=None, inputs=[compare.temp], outputs=[compare.temp], js=SAVE_TEMPERATURE_JS)

    return app
