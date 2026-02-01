# ADR-012: Compare to Battery Export Pipeline

**Status**: Proposed
**Date**: 2026-02-01
**Related**:
- ADR-009 (Interactive Battery Grid)
- A1111 WebUI inter-tab workflow patterns

---

## Context

### Two Modes, One Workflow

prompt-prix has two primary modes:

| Tab | Purpose | Interaction |
|-----|---------|-------------|
| **Compare** | Ad-hoc prompt engineering | Interactive, exploratory |
| **Battery** | Systematic evaluation | Batch, regression testing |

Currently these operate independently. A user discovering a working prompt format in Compare must manually recreate it as a Battery test case.

### The Workflow Gap

```
Current workflow (manual):
┌─────────────────────────────────────────────────────────────────┐
│ Compare Tab                                                      │
│   1. Set system prompt                                          │
│   2. Add tools JSON                                             │
│   3. Send prompt, see responses                                 │
│   4. Iterate until working                                      │
│   5. ??? manually copy to JSON/YAML ???                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (manual effort)
┌─────────────────────────────────────────────────────────────────┐
│ Battery Tab                                                      │
│   1. Create test file with same system prompt                   │
│   2. Add same tools                                             │
│   3. Add expected output / pass criteria                        │
│   4. Run battery                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Inspiration: A1111 WebUI

Automatic1111's Stable Diffusion WebUI pioneered inter-tab data flow:
- "Send to img2img" button transfers image + parameters
- "Send to extras" for upscaling
- Each tab is a specialized tool, connected by one-click transfers

This pattern enables **pipeline workflows** where each tab is a stage.

---

## Decision

### Phase 1: Export Compare Session to Battery Format

Add "Export to Battery" button in Compare tab that generates a test file from the current session.

### Phase 2 (Future): Inter-Tab Data Shuffling

Enable one-click transfer of data between tabs, treating each tab as a pipeline stage.

---

## Phase 1: Export to Battery

### UI Addition

```
┌─────────────────────────────────────────────────────────────────┐
│ Compare Tab                                                      │
│                                                                  │
│  [System Prompt]     [User Message]                             │
│  [Tools JSON]        [Image]                                    │
│                                                                  │
│  [⚡ Send to All]  [⏹ Stop]  [🗑 Clear]  [📤 Export to Battery] │
│                                                                  │
│  Model 1 | Model 2 | Model 3 | ...                              │
└─────────────────────────────────────────────────────────────────┘
```

### Export Format

Generate JSON with placeholders for expected outputs:

```json
{
  "prompts": [
    {
      "id": "compare-export-001",
      "name": "Exported from Compare - 2026-02-01 01:30",
      "system": "You are a helpful assistant with access to tools.",
      "user": "What's the weather in Tokyo?",
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
              "type": "object",
              "properties": {
                "city": {"type": "string"}
              },
              "required": ["city"]
            }
          }
        }
      ],
      "tool_choice": "auto",
      "expected": "TODO: Add expected output or pass_criteria",
      "pass_criteria": null,
      "_compare_responses": {
        "gpt-oss-20b": "**Tool Call:** get_weather({\"city\": \"Tokyo\"})",
        "lfm2-350m": "I don't have access to weather tools..."
      }
    }
  ],
  "_metadata": {
    "exported_from": "compare",
    "timestamp": "2026-02-01T01:30:00Z",
    "models_tested": ["gpt-oss-20b", "lfm2-350m", "..."]
  }
}
```

### Key Features

1. **Include actual responses**: `_compare_responses` shows what each model returned, helping user write accurate `expected` or `pass_criteria`

2. **Multi-turn support**: If Compare session has multiple exchanges, export as multiple test cases:
   ```json
   {
     "prompts": [
       {"id": "turn-1", "user": "What's the weather?", ...},
       {"id": "turn-2", "user": "And tomorrow?", ...}
     ]
   }
   ```

3. **Placeholder prompts**: `expected: "TODO"` signals user must fill in validation criteria

4. **Metadata preservation**: Track origin for debugging

### Implementation

```python
# prompt_prix/tabs/compare/handlers.py

def export_to_battery(
    system_prompt: str,
    user_messages: list[str],  # Multi-turn history
    tools_json: str,
    model_responses: dict[str, list[str]],  # model_id -> responses
) -> tuple[str, str]:
    """
    Export Compare session to Battery-compatible JSON.

    Returns:
        (status_message, file_path)
    """
    prompts = []

    for idx, user_msg in enumerate(user_messages):
        test_case = {
            "id": f"compare-export-{idx+1:03d}",
            "name": f"Compare Export Turn {idx+1}",
            "system": system_prompt,
            "user": user_msg,
            "expected": "TODO: Add expected output",
            "pass_criteria": None,
            "_compare_responses": {
                model: responses[idx] if idx < len(responses) else None
                for model, responses in model_responses.items()
            }
        }

        # Include tools if present
        if tools_json.strip():
            try:
                test_case["tools"] = json.loads(tools_json)
                test_case["tool_choice"] = "auto"
            except json.JSONDecodeError:
                pass

        prompts.append(test_case)

    export_data = {
        "prompts": prompts,
        "_metadata": {
            "exported_from": "compare",
            "timestamp": datetime.now().isoformat(),
            "models_tested": list(model_responses.keys())
        }
    }

    # Save to examples/ directory
    filename = f"compare_export_{int(time.time())}.json"
    filepath = Path("examples") / filename

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    return f"✅ Exported {len(prompts)} test cases to {filepath}", str(filepath)
```

### UI Binding

```python
# In compare tab UI
export_battery_btn = gr.Button("📤 Export to Battery", size="sm")

export_battery_btn.click(
    fn=export_to_battery,
    inputs=[system_prompt, conversation_history, tools_json, model_responses_state],
    outputs=[status_display, export_file]
)
```

---

## Phase 2: Inter-Tab Data Shuffling (Future)

### Vision

Each tab becomes a **pipeline stage** with standardized inputs/outputs:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Compare    │ ──▶ │   Battery    │ ──▶ │   Results    │
│  (explore)   │     │  (validate)  │     │  (analyze)   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  "Send to Battery"   "Send to Compare"   "Export Report"
                      (debug failures)
```

### Potential Flows

| From | To | Data Transferred |
|------|----|------------------|
| Compare | Battery | System prompt, tools, prompts → test cases |
| Battery | Compare | Failed test → pre-populated for debugging |
| Battery | Results | Raw results → visualization/analysis |
| Results | Compare | Outlier response → inspect in detail |

### A1111-Style Button Patterns

```
Compare Tab:
  [📤 Export to Battery]     - Save as file
  [➡️ Send to Battery]       - Direct transfer, switch tabs

Battery Tab (on cell click):
  [🔍 Debug in Compare]      - Load this test case into Compare

Results Tab:
  [📊 Visualize in Charts]   - Future analysis tab
  [📥 Re-run in Battery]     - Retry failed subset
```

### State Management

Requires shared state accessible across tabs:

```python
# prompt_prix/state.py

@dataclass
class InterTabTransfer:
    """Data packet for cross-tab transfers."""
    source_tab: str
    target_tab: str
    payload: dict
    timestamp: datetime

class AppState:
    # ... existing state ...

    pending_transfer: Optional[InterTabTransfer] = None

    def send_to_tab(self, target: str, payload: dict):
        self.pending_transfer = InterTabTransfer(
            source_tab=self.current_tab,
            target_tab=target,
            payload=payload,
            timestamp=datetime.now()
        )

    def receive_transfer(self, tab: str) -> Optional[dict]:
        if self.pending_transfer and self.pending_transfer.target_tab == tab:
            payload = self.pending_transfer.payload
            self.pending_transfer = None
            return payload
        return None
```

---

## Implementation Plan

### Phase 1 (Short Term)

| Task | Effort | Files |
|------|--------|-------|
| Add "Export to Battery" button | 1 hour | `tabs/compare/ui.py` |
| Implement export handler | 2 hours | `tabs/compare/handlers.py` |
| Include model responses in export | 1 hour | Handler updates |
| Test with multi-turn conversations | 1 hour | Manual testing |

### Phase 2 (Future)

| Task | Effort | Files |
|------|--------|-------|
| Design inter-tab state protocol | Design | `state.py` |
| Add "Debug in Compare" to Battery | 2 hours | `tabs/battery/handlers.py` |
| Tab switch with data transfer | 2 hours | `ui.py`, Gradio JS |
| Bidirectional flow testing | 2 hours | Integration tests |

---

## Trade-offs

### Phase 1 Advantages
- Simple file-based transfer (no shared state complexity)
- User can edit file before loading in Battery
- Works with existing Battery file upload

### Phase 1 Limitations
- Two-step process (export, then upload)
- No direct "click and switch" flow

### Phase 2 Advantages
- Seamless A1111-style workflow
- Faster iteration cycles
- Pipeline-oriented thinking

### Phase 2 Complexity
- Shared state management
- Tab synchronization
- UI state preservation on switch

---

## Success Criteria

### Phase 1
- [ ] "Export to Battery" button visible in Compare tab
- [ ] Export includes system prompt, tools, all user messages
- [ ] Export includes `_compare_responses` for reference
- [ ] Exported file loads successfully in Battery tab
- [ ] Multi-turn conversations export as multiple test cases

### Phase 2
- [ ] "Debug in Compare" loads failed test with one click
- [ ] Tab switches preserve context
- [ ] Bidirectional flow works (Compare ↔ Battery)

---

## References

- [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Inter-tab "Send to" pattern
- [Gradio Tab Events](https://www.gradio.app/docs/tab) - Tab selection and state
