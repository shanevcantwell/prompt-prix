# ADR-009: Interactive Battery Grid (Cell Selection)

**Status:** Accepted
**Date:** 2026-01-24
**Related:** Demo polish, UX enhancement

## Context

The Battery tab displays a grid of test results (rows=tests, columns=models) with ✓/❌/⚠️ symbols. Users naturally want to click a cell to see the response details - what did that model actually output for that test? Currently, clicking a cell selects it visually but triggers no action.

**Existing capability:**
- `get_cell_detail(model, test)` in `tabs/battery/handlers.py:252` already returns formatted markdown with status, latency, judge info, and full response text
- This function is fully implemented but not wired to any UI trigger

**User expectation:**
- Click a failure cell → see why it failed (the actual response)
- Click a success cell → see the response that passed
- This is demo-level functionality, not a reporting system

## Decision Drivers

1. **Demo polish**: Users expect interactive grids to respond to clicks
2. **Minimal scope**: Show detail on click, nothing more
3. **Gradio-native**: Use standard patterns, avoid custom JS overlays
4. **Existing code**: Handler already exists, just needs wiring

## Options Considered

### Option A: Dataframe.select() → Detail Panel Below

Use Gradio's native `.select()` event on the Dataframe component. When a cell is clicked, update a Markdown component below the grid with the response details.

```python
# In ui.py Battery tab wiring
battery.grid.select(
    fn=handle_cell_select,
    inputs=[],
    outputs=[battery.detail_markdown]
)

# In handlers.py
def handle_cell_select(evt: gr.SelectData) -> str:
    # evt.index = (row, col), evt.value = cell content
    # Map row/col to test_id/model_id
    return get_cell_detail(model_id, test_id)
```

**Pros:**
- Gradio-native approach
- Handler already exists
- Simple vertical layout (grid above, detail below)
- No custom JS

**Cons:**
- Need to map row/col indices to test_id/model_id (header row handling)
- Multi-cell selection shows only last clicked (acceptable for demo)

### Option B: Cell Overlay Buttons

Create a dynamic overlay of clickable elements positioned over each cell.

**Pros:**
- Could support hover preview

**Cons:**
- Fragile positioning with dynamic grid sizes
- Heavy implementation for minimal benefit
- Not a Gradio pattern

### Option C: Separate Model/Test Dropdowns

Add dropdowns for model and test, update detail when selection changes.

**Pros:**
- Explicit selection
- Works without cell click event

**Cons:**
- Disconnected from grid (click vs dropdown mismatch)
- Extra UI clutter
- Not the natural interaction users expect

### Option D: Dismissible Dialog

Use Gradio's visibility toggle pattern to create a dialog-like UX. A hidden column containing detail markdown + close button appears on cell click, dismisses on button click.

```python
# Hidden dialog container
with gr.Column(visible=False) as detail_dialog:
    detail_markdown = gr.Markdown()
    close_btn = gr.Button("Close")

# Show on cell select, hide on close
grid.select(..., outputs=[detail_dialog, detail_markdown])
close_btn.click(fn=lambda: gr.update(visible=False), outputs=[detail_dialog])
```

**Pros:**
- No permanent UI clutter
- Click → view → dismiss → continue exploring
- Standard Gradio visibility pattern
- Grid stays prominent

**Cons:**
- Slightly more wiring (show/hide logic)

## Decision

**Option D (Dismissible Dialog)** selected:
- Cleaner UX: click cell → dialog appears → dismiss when done
- Grid remains the hero element (no permanent detail panel)
- Uses Gradio's visibility toggle pattern (no custom JS)
- Leverages existing `get_cell_detail()` handler

### Multi-Cell Selection Behavior

Gradio Dataframe allows selecting multiple cells. For this demo use case:
- Show detail for the most recently clicked cell only
- Dialog replaces previous content on new cell click
- No concatenation or batch detail view (scope creep)

## Implementation

### 1. Add dialog components to Battery layout

In `tabs/battery/ui.py`:

```python
# Dismissible detail dialog (hidden by default)
with gr.Column(visible=False, elem_id="battery-detail-dialog") as detail_dialog:
    detail_header = gr.Markdown("### Response Detail")
    detail_markdown = gr.Markdown()
    close_btn = gr.Button("Close", size="sm")
```

### 2. Add selection handler to handlers.py

```python
def handle_cell_select(evt: gr.SelectData) -> tuple:
    """Handle grid cell selection, return (dialog_visible, detail_content)."""
    if not state.battery_run:
        return gr.update(visible=False), "*No battery run available*"

    row, col = evt.index

    # Row 0 is header, col 0 is test ID column
    if row == 0 or col == 0:
        return gr.update(visible=False), ""

    # Map indices to identifiers
    test_id = state.battery_run.tests[row - 1]  # -1 for header
    model_id = state.battery_run.models[col - 1]  # -1 for test ID col

    detail = get_cell_detail(model_id, test_id)
    return gr.update(visible=True), detail
```

### 3. Wire in ui.py

```python
# Show dialog on cell select
battery.grid.select(
    fn=handlers.handle_cell_select,
    inputs=[],
    outputs=[battery.detail_dialog, battery.detail_markdown]
)

# Hide dialog on close button
battery.close_btn.click(
    fn=lambda: gr.update(visible=False),
    inputs=[],
    outputs=[battery.detail_dialog]
)
```

## Files to Modify

| File | Change |
|------|--------|
| `prompt_prix/tabs/battery/ui.py` | Add `detail_dialog`, `detail_markdown`, `detail_close_btn` |
| `prompt_prix/tabs/battery/handlers.py` | Add `handle_cell_select()` function |
| `prompt_prix/ui.py` | Wire `.select()` and close button events |

## Verification

1. Run `prompt-prix`
2. Battery tab → upload test file → run battery
3. Click any result cell (✓/❌/⚠️)
4. Dialog appears with: status, latency, judge info (if any), full response
5. Click "Close" → dialog dismisses
6. Click different cell → dialog shows new detail

## Out of Scope

- Multi-cell detail concatenation
- Hover preview
- Reporting/export from detail view
- Historical comparison
- Modal styling (CSS polish can come later)
