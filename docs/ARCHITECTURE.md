# prompt-prix Architecture

**Purpose:** Audit local LLM function calling and agentic reliability.

---

## Layers

```
┌─────────────────────────────────┐
│  UI                             │
│  Battery tab / Compare tab      │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Orchestration                  │
│  Battery: model-sequential      │
│  Compare: model-parallel        │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Primitives (MCP tools)         │
│  complete · judge · list_models │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Adapters                       │
│  LMStudio / HuggingFace / Surf  │
└─────────────────────────────────┘
```

---

## Primitives

Three MCP tools. Everything else is orchestration or UI.

| Primitive | Signature | Purpose |
|-----------|-----------|---------|
| `complete` | `(model_id, messages, params) → response` | Single model completion |
| `judge` | `(response, criteria, judge_model) → {pass, reason}` | Semantic evaluation |
| `list_models` | `(servers, only_loaded?) → model_id[]` | Discovery |

---

## Adapters

Adapters implement these primitives for different backends.

| Adapter | Backend | Status |
|---------|---------|--------|
| `LMStudioAdapter` | LM Studio (local GPU) | Working |
| `HuggingFaceAdapter` | HF Inference API | Exists |
| `SurfMCPAdapter` | surf-mcp → browser | Planned |

---

## Orchestration

### Battery

Executes a test plan across selected models. Optimized for model loading.

```
for model in selected_models:
    for test in test_plan:
        response = complete(model, test.messages)
        if judge_model:
            judge(response, test.criteria, judge_model)
```

**Input:** Test suite (promptfoo YAML, JSON, JSONL) + selected models
**Output:** Results grid (test × model)

### Compare

Fan-out to all models in parallel.

```
for model in selected_models:  # parallel
    complete(model, [system_prompt, user_message])
```

**Input:** System prompt + user message + selected models
**Output:** Streaming responses per model

---

## UI

### Battery Tab
- Server/model configuration
- Test suite upload + validation
- Optional judge model selection
- Results grid (✓/✗/⚠)
- Export: JSON, CSV, Image

### Compare Tab
- Server/model configuration
- System prompt + user message
- Optional image attachment
- Streaming response tabs
- Export: Markdown, JSON

---

## References

- [surf-mcp](https://github.com/shanevcantwell/surf-mcp) - Browser automation adapter
- [promptfoo](https://promptfoo.dev) - Test suite format
