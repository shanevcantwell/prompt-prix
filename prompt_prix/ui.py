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
    battery_get_test_ids
)
from prompt_prix.parsers import get_default_system_prompt


# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Battery grid styling */
#battery-grid table {
    font-family: monospace;
    font-size: 14px;
}
#battery-grid td {
    text-align: center;
    min-width: 80px;
}

/* Status colors for Compare tab */
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

/* Hero grid prominence */
#battery-grid {
    min-height: 300px;
}

/* Compact config panels */
.config-row {
    gap: 1rem;
}
"""

# JavaScript for Compare tab colors
TAB_STATUS_JS = """
function updateTabColors(tabStates) {
    if (!tabStates) return tabStates;
    const tabContainer = document.getElementById('model-tabs');
    if (!tabContainer) return tabStates;
    const buttons = tabContainer.querySelectorAll('button[role="tab"]');
    tabStates.forEach((status, index) => {
        if (index < buttons.length) {
            const btn = buttons[index];
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
                btn.style.background = '';
                btn.style.borderLeft = '';
                btn.style.animation = '';
            }
        }
    });
    return tabStates;
}
"""


# ─────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(
        title="prompt-prix",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    ) as app:
        
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

            # ─────────────────────────────────────────────────────────
            # TAB 2: COMPARE (Secondary)
            # ─────────────────────────────────────────────────────────
            
            with gr.Tab("💬 Compare", id="compare-tab"):
                
                gr.Markdown("""
                Interactive prompt comparison. Send the same prompt to multiple models
                and see responses side-by-side in real-time.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_models = gr.CheckboxGroup(
                            label="Models to Compare",
                            choices=[],
                            value=[],
                            elem_id="compare-models"
                        )
                    
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
                
                with gr.Accordion("System Prompt", open=False):
                    compare_system_prompt = gr.Textbox(
                        label="System Prompt",
                        value=get_default_system_prompt(),
                        lines=10
                    )
                
                with gr.Accordion("Tools (Function Calling)", open=False):
                    compare_tools = gr.Code(
                        label="Tools JSON (OpenAI format)",
                        language="json",
                        value="",
                        lines=10,
                        elem_id="tools"
                    )
                
                compare_init_btn = gr.Button("Initialize Session", variant="secondary")
                compare_status = gr.Textbox(
                    label="Status",
                    value="Session not initialized",
                    interactive=False
                )
                
                with gr.Row():
                    compare_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3,
                        scale=3
                    )
                    compare_send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1
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

        # Detail view
        detail_refresh_btn.click(
            fn=battery_get_cell_detail,
            inputs=[detail_model, detail_test],
            outputs=[detail_content]
        )

        # ─────────────────────────────────────────────────────────────
        # EVENT BINDINGS: Compare Tab
        # ─────────────────────────────────────────────────────────────

        compare_init_btn.click(
            fn=initialize_session,
            inputs=[
                servers_input,
                compare_models,
                compare_system_prompt,
                compare_temp,
                compare_timeout,
                compare_max_tokens
            ],
            outputs=[compare_status] + model_outputs
        )

        compare_send_btn.click(
            fn=send_single_prompt,
            inputs=[compare_prompt, compare_tools],
            outputs=[compare_status, tab_states] + model_outputs
        )

        compare_prompt.submit(
            fn=send_single_prompt,
            inputs=[compare_prompt, compare_tools],
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
        # PERSISTENCE
        # ─────────────────────────────────────────────────────────────

        app.load(
            fn=None,
            inputs=[],
            outputs=[servers_input, battery_temp, compare_temp],
            js="""
            () => {
                const servers = localStorage.getItem('promptprix_servers');
                const temp = localStorage.getItem('promptprix_temperature');
                return [
                    servers ? servers : null,
                    temp ? parseFloat(temp) : null,
                    temp ? parseFloat(temp) : null
                ];
            }
            """
        )

    return app