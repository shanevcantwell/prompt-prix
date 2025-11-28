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
    launch_beyond_compare,
    battery_validate_file,
    battery_run_handler
)
from prompt_prix.parsers import get_default_system_prompt


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

# JavaScript to update tab colors based on state (uses inline styles for highest specificity)
TAB_STATUS_JS = """
function updateTabColors(tabStates) {
    if (!tabStates) return tabStates;

    const tabContainer = document.getElementById('model-tabs');
    if (!tabContainer) return tabStates;

    const buttons = tabContainer.querySelectorAll('button[role="tab"]');

    tabStates.forEach((status, index) => {
        if (index < buttons.length) {
            const btn = buttons[index];
            // Use inline styles for highest specificity (overcomes Gradio theme)
            if (status === 'pending') {
                btn.style.background = 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)';
                btn.style.borderLeft = '4px solid #ef4444';
                btn.style.animation = '';
            } else if (status === 'streaming') {
                btn.style.background = 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)';
                btn.style.borderLeft = '4px solid #f59e0b';
                btn.style.animation = 'pulse 1.5s ease-in-out infinite';
            } else if (status === 'completed') {
                btn.style.background = 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)';
                btn.style.borderLeft = '4px solid #10b981';
                btn.style.animation = '';
            } else {
                // Reset styles for empty/inactive tabs
                btn.style.background = '';
                btn.style.borderLeft = '';
                btn.style.animation = '';
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
                        placeholder="http://127.0.0.1:1234\nhttp://192.168.137.2:1234",
                        elem_id="servers"
                    )
                with gr.Column(scale=1):
                    models_checkboxes = gr.CheckboxGroup(
                        label="Models to Compare",
                        choices=[],  # Populated by Fetch button or localStorage
                        value=[],    # Selected models
                        elem_id="models"
                    )
                    with gr.Row():
                        fetch_models_button = gr.Button("🔄 Fetch Available Models", size="sm")
                        clear_models_btn = gr.Button("Clear Selection", size="sm")

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
                system_prompt_input = gr.Textbox(
                    label="System Prompt",
                    value=get_default_system_prompt(),
                    lines=50,
                    max_lines=50,
                    elem_id="system_prompt"
                )

            with gr.Row():
                save_state_btn = gr.Button("💾 Save State", variant="secondary")
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

        # Tools input for function calling tests
        with gr.Accordion("Tools (Function Calling)", open=False):
            tools_input = gr.Code(
                label="Tools JSON (OpenAI format)",
                language="json",
                value="",
                lines=10,
                elem_id="tools"
            )
            gr.Markdown("*Example: `[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", ...}}]`*")

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
        # BATTERY PANEL
        # ─────────────────────────────────────────────────────────────

        with gr.Accordion("Battery (Benchmark Suite)", open=False):
            gr.Markdown("""
            **Fan-out benchmark tests across all selected models.**

            Upload a JSON benchmark file (e.g., `tool_competence_tests.json`) and run
            all tests against your selected models. Results display in a grid showing
            pass/fail status for each (test, model) combination.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    battery_file = gr.File(
                        label="Benchmark JSON File",
                        file_types=[".json"],
                        type="filepath"
                    )
                with gr.Column(scale=1):
                    battery_validation_status = gr.Textbox(
                        label="Validation",
                        value="Upload a benchmark JSON file",
                        interactive=False
                    )
                    battery_run_btn = gr.Button(
                        "🔋 Run Battery",
                        variant="primary",
                        interactive=False
                    )

            battery_status = gr.Textbox(
                label="Battery Status",
                value="",
                interactive=False
            )

            battery_grid = gr.Dataframe(
                label="Results Grid",
                headers=["Test"],
                interactive=False,
                wrap=True
            )

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS
        # ─────────────────────────────────────────────────────────────

        fetch_models_button.click(
            fn=fetch_available_models,
            inputs=[servers_input],
            outputs=[status_display, models_checkboxes]
        )

        # Clear model selection
        clear_models_btn.click(
            fn=lambda: gr.update(value=[]),
            inputs=[],
            outputs=[models_checkboxes]
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
                models_checkboxes,
                system_prompt_input,
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
            inputs=[prompt_input, tools_input],
            outputs=[status_display, tab_states] + model_outputs
        )

        # Also send on Enter in prompt box
        prompt_input.submit(
            fn=send_single_prompt,
            inputs=[prompt_input, tools_input],
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

        # Battery event bindings
        battery_file.change(
            fn=battery_validate_file,
            inputs=[battery_file],
            outputs=[battery_validation_status, battery_run_btn]
        )

        battery_run_btn.click(
            fn=battery_run_handler,
            inputs=[battery_file, models_checkboxes, servers_input],
            outputs=[battery_status, battery_grid]
        )

        # ─────────────────────────────────────────────────────────────
        # PERSISTENCE - Save/Load form values to localStorage
        # ─────────────────────────────────────────────────────────────

        # Save State button - ONLY way to save (explicit save, no auto-save)
        save_state_btn.click(
            fn=lambda: "✅ State saved to browser storage",
            inputs=[],
            outputs=[status_display],
            js="""
            () => {
                // Get all current values from Gradio components
                const serversEl = document.querySelector('#servers textarea');
                const modelsEl = document.querySelector('#models');
                const tempEl = document.querySelector('#temperature input[type="range"]');
                const timeoutEl = document.querySelector('#timeout input[type="range"]');
                const maxTokensEl = document.querySelector('#max_tokens input[type="range"]');
                const toolsEl = document.querySelector('#tools textarea');
                const systemPromptEl = document.querySelector('#system_prompt textarea');

                // Save servers
                if (serversEl) {
                    localStorage.setItem('promptprix_servers', serversEl.value);
                }

                // Save model choices and selected models (CheckboxGroup)
                if (modelsEl) {
                    const checkboxes = modelsEl.querySelectorAll('input[type="checkbox"]');
                    const choices = [];
                    const selected = [];
                    checkboxes.forEach(cb => {
                        const label = cb.parentElement.textContent.trim();
                        choices.push(label);
                        if (cb.checked) {
                            selected.push(label);
                        }
                    });
                    localStorage.setItem('promptprix_model_choices', JSON.stringify(choices));
                    localStorage.setItem('promptprix_models', JSON.stringify(selected));
                }

                // Save sliders
                if (tempEl) {
                    localStorage.setItem('promptprix_temperature', tempEl.value);
                }
                if (timeoutEl) {
                    localStorage.setItem('promptprix_timeout', timeoutEl.value);
                }
                if (maxTokensEl) {
                    localStorage.setItem('promptprix_max_tokens', maxTokensEl.value);
                }

                // Save tools JSON
                if (toolsEl) {
                    localStorage.setItem('promptprix_tools', toolsEl.value);
                }

                // Save system prompt
                if (systemPromptEl) {
                    localStorage.setItem('promptprix_system_prompt', systemPromptEl.value);
                }

                return [];
            }
            """
        )

        # Load from localStorage on page load
        def load_persisted_values():
            # Return defaults - JS will override with localStorage values
            return [
                gr.update(),  # servers
                gr.update(),  # models_checkboxes (choices + value)
                gr.update(),  # temperature
                gr.update(),  # timeout
                gr.update(),  # max_tokens
                gr.update(),  # tools
                gr.update(),  # system_prompt
            ]

        app.load(
            fn=load_persisted_values,
            inputs=[],
            outputs=[
                servers_input, models_checkboxes, temperature_slider,
                timeout_slider, max_tokens_slider, tools_input, system_prompt_input
            ],
            js="""
            () => {
                const servers = localStorage.getItem('promptprix_servers');
                const modelChoices = localStorage.getItem('promptprix_model_choices');
                const modelSelected = localStorage.getItem('promptprix_models');
                const temp = localStorage.getItem('promptprix_temperature');
                const timeout = localStorage.getItem('promptprix_timeout');
                const maxTokens = localStorage.getItem('promptprix_max_tokens');
                const tools = localStorage.getItem('promptprix_tools');
                const systemPrompt = localStorage.getItem('promptprix_system_prompt');

                // Parse model data - CheckboxGroup needs {choices: [...], value: [...]}
                let modelUpdate = undefined;
                if (modelChoices || modelSelected) {
                    const choices = modelChoices ? JSON.parse(modelChoices) : [];
                    const selected = modelSelected ? JSON.parse(modelSelected) : [];
                    // Filter selected to only include models that exist in choices
                    const validSelected = selected.filter(s => choices.includes(s));
                    modelUpdate = {choices: choices, value: validSelected};
                }

                return [
                    servers ? servers : undefined,
                    modelUpdate,
                    temp ? parseFloat(temp) : undefined,
                    timeout ? parseInt(timeout) : undefined,
                    maxTokens ? parseInt(maxTokens) : undefined,
                    tools ? tools : undefined,
                    systemPrompt ? systemPrompt : undefined
                ];
            }
            """
        )

    return app
