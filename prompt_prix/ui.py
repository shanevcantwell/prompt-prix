"""
Gradio UI definition for prompt-prix.

v2 Simplified: Shared header with model selection, two tabs (Battery/Compare).
Battery runs test suites. Compare builds multi-turn test scenarios for export.

Tab-specific handlers are in prompt_prix.tabs.{battery,compare}.handlers
Tab-specific UI layouts are in prompt_prix.tabs.{battery,compare}.ui
"""

import gradio as gr

from prompt_prix import state
from prompt_prix.handlers import fetch_available_models, handle_stop
from prompt_prix.config import (
    get_default_servers,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS
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
        gr.Markdown("Audit local LLM function calling and agentic reliability.")

        available_models = gr.State([])

        # ─────────────────────────────────────────────────────────────
        # SHARED HEADER: Server config + Model selection (collapsible)
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("⚙️ Model Configuration", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    models_input = gr.Textbox(
                        label="Available Models (one per line)",
                        value="meta-llama/Llama-3.2-3B-Instruct\nmistralai/Mistral-7B-Instruct-v0.3",
                        lines=3,
                        placeholder="meta-llama/Llama-3.2-3B-Instruct",
                        elem_id="models-input",
                        info="HuggingFace model IDs"
                    )
                with gr.Column(scale=2):
                    models_selector = gr.CheckboxGroup(
                        label="Models to Test",
                        choices=[],
                        value=[],
                        elem_id="models-selector"
                    )
                    sync_models_btn = gr.Button(
                        "🔄 Load Models",
                        variant="secondary",
                        size="sm"
                    )

            # Hidden: LM Studio config (teased for future)
            with gr.Accordion("🔧 LM Studio (Local - Coming Soon)", open=False, visible=False):
                servers_input = gr.Textbox(
                    label="LM Studio Servers (one per line)",
                    value="\n".join(get_default_servers()),
                    lines=2,
                    placeholder="http://localhost:1234",
                    elem_id="servers",
                    interactive=False
                )
                with gr.Row():
                    fetch_btn = gr.Button(
                        "🔄 Fetch Models",
                        variant="secondary",
                        size="sm",
                        interactive=False
                    )
                    only_loaded_checkbox = gr.Checkbox(
                        label="Only Loaded",
                        value=False,
                        info="Filter to models in memory",
                        interactive=False
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

        def on_sync_models(models_text):
            """Sync model selector from text input."""
            lines = [line.strip() for line in models_text.strip().split("\n") if line.strip()]
            return (
                lines,
                gr.update(choices=lines, value=lines),  # Select all by default
                gr.update(choices=lines),  # battery.detail_model
                gr.update(choices=lines),  # battery.judge_model
            )

        sync_models_btn.click(
            fn=on_sync_models,
            inputs=[models_input],
            outputs=[
                available_models,
                models_selector,
                battery.detail_model,
                battery.judge_model,
            ]
        )

        # Auto-sync on load
        app.load(
            fn=on_sync_models,
            inputs=[models_input],
            outputs=[
                available_models,
                models_selector,
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
            state.battery_converted_file = None  # Clear converted file

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
                battery.file, models_selector,
                timeout_slider, max_tokens_slider, battery.system_prompt
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

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Compare Tab
        # ─────────────────────────────────────────────────────────────

        async def compare_send_with_auto_init(
            prompt, tools, image, seed, repeat_penalty, models_selected,
            system_prompt, timeout, max_tokens
        ):
            if (state.session is None or
                set(state.session.state.models) != set(models_selected)):
                init_status, *init_outputs = await compare_handlers.initialize_session(
                    models_selected, system_prompt,
                    timeout, max_tokens
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
                models_selector,
                compare.system_prompt, timeout_slider, max_tokens_slider
            ],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.prompt.submit(
            fn=compare_send_with_auto_init,
            inputs=[
                compare.prompt, compare.tools, compare.image, compare.seed, compare.repeat_penalty,
                models_selector,
                compare.system_prompt, timeout_slider, max_tokens_slider
            ],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.clear_btn.click(
            fn=compare_handlers.clear_session,
            inputs=[],
            outputs=[compare.status, compare.tab_states] + compare.model_outputs
        )

        compare.tab_states.change(fn=None, inputs=[compare.tab_states], outputs=[compare.tab_states], js=TAB_STATUS_JS)

        # Re-apply tab styling when returning to Compare tab (fixes #52)
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
        # PERSISTENCE (disabled for HF Spaces demo)
        # ─────────────────────────────────────────────────────────────
        # Note: LM Studio server persistence removed for HF Spaces version

    return app
