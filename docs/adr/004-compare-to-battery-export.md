# ADR-004: Compare to Battery Export Workflow

## Status

Proposed

## Context

The Compare tab serves as a "workshop" for constructing conversation scenarios to test model behavior. Once a user identifies an interesting pattern or failure mode, they should be able to export that scenario as a test case for the Battery tab.

Currently:
- Compare exports full session reports (Markdown or JSON)
- Battery expects individual TestCase objects in BFCL-compatible format
- There is no direct path from "I found a bug in model X's behavior" to "This is now a regression test"

The workflow we want to enable:

```
Encounter weird behavior in production
    ↓
Reconstruct scenario in Compare (multi-turn context)
    ↓
Test against N models to isolate the pattern
    ↓
Export as TestCase
    ↓
Add to Battery suite for regression testing
```

## Decision Drivers

1. **Minimal friction** - Export should be one click after you've identified the scenario
2. **Tool support** - Many interesting scenarios involve tool calls, so Compare needs tool definition capability
3. **Annotation flexibility** - User may want to add pass/fail criteria after seeing model responses
4. **Format compatibility** - Output must be valid BFCL/Battery JSON

## Options Considered

### Option A: Export Last Turn Only

Export only the final user message as a single-shot TestCase.

```json
{
  "id": "explore-2024-01-08-001",
  "user": "Delete the file report.pdf",
  "system": "You have three tools...",
  "tools": [...],
  "tool_choice": "required"
}
```

**Pros:**
- Simple implementation
- Matches common use case (testing one specific prompt)
- Clean TestCase format

**Cons:**
- Loses multi-turn context that led to the scenario
- Can't test "after 3 tool calls, the model fails on the 4th"

### Option B: Export Full Conversation as Context

Export with conversation history embedded in system prompt or as message array.

```json
{
  "id": "explore-2024-01-08-001",
  "system": "...",
  "context": [
    {"role": "user", "content": "First message"},
    {"role": "assistant", "content": "First response"},
    {"role": "user", "content": "Second message"}
  ],
  "user": "Third message (the one being tested)",
  "tools": [...]
}
```

**Pros:**
- Preserves the context that triggers the behavior
- Can test multi-turn edge cases

**Cons:**
- Requires extending TestCase schema
- More complex Battery execution (needs to inject history)
- BFCL compatibility unclear

### Option C: Export as Prompt Template with Injected History

Flatten conversation history into system prompt.

```json
{
  "id": "explore-2024-01-08-001",
  "system": "You are a helpful assistant.\n\n[Prior conversation]\nUser: First message\nAssistant: First response\nUser: Second message\nAssistant: Second response",
  "user": "Third message",
  "tools": [...]
}
```

**Pros:**
- Works with existing TestCase schema
- No Battery changes needed
- Context is preserved in a portable way

**Cons:**
- Prompt formatting may affect model behavior differently than native multi-turn
- System prompt gets long/messy

## Proposed Decision

**Option A for v1, with Option C as enhancement.**

Start with single-turn export:
1. Add tool definition UI to Compare (collapsible JSON editor)
2. "Export as Test Case" button exports last user message + tools + system prompt
3. Auto-generate ID from timestamp
4. User manually adds `expected`, `pass_criteria`, `fail_criteria` if desired

Later, add "Export with Context" that uses Option C flattening for multi-turn scenarios.

## Implementation Plan

### Phase 1: Tool Support in Compare

Add to Compare tab:
- Collapsible "Tools" accordion with JSON editor
- `tool_choice` dropdown (auto/required/none)
- Tools are sent with each prompt to all models

### Phase 2: Export as Test Case

Add "Export as Test Case" button that:
1. Takes current system prompt, last user message, tools, tool_choice
2. Generates TestCase JSON with auto-ID
3. Shows preview in modal/textbox
4. Copy to clipboard or download as file

### Phase 3: Export with Context (future)

Add "Export with Context" option that:
1. Flattens conversation history into system prompt (Option C)
2. Exports the final turn as the `user` field
3. Preserves full context for regression testing

## Consequences

**Positive:**
- Clear path from exploration to regression suite
- Compare becomes the "front end" of test creation
- Tools become first-class in interactive testing

**Negative:**
- Tool JSON editor adds UI complexity
- Users need to understand TestCase schema for annotations

## Related

- ADR-001: Use existing benchmarks (BFCL format)
- ADR-002: Fan-out pattern as core abstraction
