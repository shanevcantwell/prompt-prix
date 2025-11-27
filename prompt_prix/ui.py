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


# Custom CSS for tab status colors
TAB_STATUS_CSS = """
/* Tab status indicator colors */
#model-tabs button.tab-pending {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
    border-left: 4px solid #ef4444 !important;
}
#model-tabs button.tab-streaming {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
    border-left: 4px solid #f59e0b !important;
    animation: pulse 1.5s ease-in-out infinite;
}
#model-tabs button.tab-completed {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
    border-left: 4px solid #10b981 !important;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
"""

# JavaScript to update tab colors based on state
TAB_STATUS_JS = """
function updateTabColors(tabStates) {
    if (!tabStates) return tabStates;

    const tabContainer = document.getElementById('model-tabs');
    if (!tabContainer) return tabStates;

    const buttons = tabContainer.querySelectorAll('button');

    tabStates.forEach((status, index) => {
        if (index < buttons.length) {
            const btn = buttons[index];
            btn.classList.remove('tab-pending', 'tab-streaming', 'tab-completed');
            if (status === 'pending') {
                btn.classList.add('tab-pending');
            } else if (status === 'streaming') {
                btn.classList.add('tab-streaming');
            } else if (status === 'completed') {
                btn.classList.add('tab-completed');
            }
        }
    });

    return tabStates;
}
"""



def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(title="prompt-prix", theme=gr.themes.Soft(), css=TAB_STATUS_CSS) as app:
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
                        placeholder="http://192.168.1.10:1234\nhttp://192.168.1.11:1234",
                        elem_id="servers"
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        models_input = gr.Textbox(
                            label="Models to Compare (one per line)",
                            value="\n".join(DEFAULT_MODELS),
                            lines=5,
                            placeholder="llama-3.2-3b-instruct\nqwen2.5-7b-instruct",
                            elem_id="models"
                        )
                    fetch_models_button = gr.Button("🔄 Fetch Available Models", size="sm")

            with gr.Row():
                temperature_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=DEFAULT_TEMPERATURE,
                    elem_id="temperature"
                )
                timeout_slider = gr.Slider(
                    label="Timeout (seconds)",
                    minimum=30,
                    maximum=600,
                    step=30,
                    value=DEFAULT_TIMEOUT_SECONDS,
                    elem_id="timeout"
                )
                max_tokens_slider = gr.Slider(
                    label="Max Tokens",
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=DEFAULT_MAX_TOKENS,
                    elem_id="max_tokens"
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

        # Hidden component to track tab states and trigger JS updates
        tab_states = gr.JSON(value=[], visible=False, elem_id="tab-states")

        # Create tabs for up to 10 models (can be extended)
        model_outputs = []
        with gr.Tabs(elem_id="model-tabs"):
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
            outputs=[status_display, tab_states] + model_outputs
        )

        # Also send on Enter in prompt box
        prompt_input.submit(
            fn=send_single_prompt,
            inputs=[prompt_input],
            outputs=[status_display, tab_states] + model_outputs
        )

        # Update tab colors whenever tab_states changes (including during streaming)
        tab_states.change(
            fn=None,
            inputs=[tab_states],
            outputs=[tab_states],
            js=TAB_STATUS_JS
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

        # ─────────────────────────────────────────────────────────────
        # PERSISTENCE - Save/Load form values to localStorage
        # ─────────────────────────────────────────────────────────────

        # Save to localStorage on change
        servers_input.change(
            fn=None, inputs=[servers_input], outputs=[],
            js="(v) => { localStorage.setItem('promptprix_servers', v); }"
        )
        models_input.change(
            fn=None, inputs=[models_input], outputs=[],
            js="(v) => { localStorage.setItem('promptprix_models', v); }"
        )
        temperature_slider.change(
            fn=None, inputs=[temperature_slider], outputs=[],
            js="(v) => { localStorage.setItem('promptprix_temperature', v); }"
        )
        timeout_slider.change(
            fn=None, inputs=[timeout_slider], outputs=[],
            js="(v) => { localStorage.setItem('promptprix_timeout', v); }"
        )
        max_tokens_slider.change(
            fn=None, inputs=[max_tokens_slider], outputs=[],
            js="(v) => { localStorage.setItem('promptprix_max_tokens', v); }"
        )

        # Load from localStorage on page load
        def load_persisted_values():
            # Return None for each value - JavaScript will handle the actual loading
            return [None, None, None, None, None]

        app.load(
            fn=load_persisted_values,
            inputs=[],
            outputs=[servers_input, models_input, temperature_slider, timeout_slider, max_tokens_slider],
            js="""
            () => {
                const servers = localStorage.getItem('promptprix_servers');
                const models = localStorage.getItem('promptprix_models');
                const temp = localStorage.getItem('promptprix_temperature');
                const timeout = localStorage.getItem('promptprix_timeout');
                const max_tokens = localStorage.getItem('promptprix_max_tokens');
                return [
                    servers || null,
                    models || null,
                    temp ? parseFloat(temp) : null,
                    timeout ? parseInt(timeout) : null,
                    max_tokens ? parseInt(max_tokens) : null
                ];
            }
            """
        )

    return app
