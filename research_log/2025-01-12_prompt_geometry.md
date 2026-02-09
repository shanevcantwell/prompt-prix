# Prompt Geometry Research Log

**Date:** 2025-01-12
**Session:** Compliance decay hypothesis testing

---

## LLM-as-Judge for Battery Validation

### Context

The semantic validator currently checks for **refusal patterns** ("I'm sorry", "I can't"), not the `pass_criteria` defined in test cases.

Discovered while running compliance decay battery - many "failures" actually complied with the directive but apologized, while some "passes" ignored the directive entirely.

### Proposed Solution

Add LLM-as-judge capability:
1. Judge Model dropdown in Battery tab (reuses `fetch_available_models`)
2. When `pass_criteria` exists AND judge model selected, call judge for evaluation
3. Falls back to pattern matching if no judge configured

### Architecture

```
Battery Tab UI
├── Judge Model: [dropdown]
└── Battery runner calls judge when pass_criteria exists
    ├── Direct adapter call (same infra as test calls)
    └── Later: MCP tool for LAS integration
```

### Files to modify

- `prompt_prix/tabs/battery/ui.py` - Add dropdown
- `prompt_prix/llm_judge.py` - New judge function
- `prompt_prix/battery.py` - Integrate into runner
- `prompt_prix/ui.py` - Wire dropdown to runner

---

## Prompt Geometry Research: Compliance Decay Hypothesis

### Research Question

Does embedding distance from "imperative" predict compliance decay rate?

### Methodology

1. Created 11 phrasing variants of same directive with different grammatical moods/intensities
2. Measured cosine distances from imperative baseline using NV-Embed-v2 (4096-dim)
3. Ran 4-turn compliance battery across 3 models (granite-4, rnj-1, qwen3-30b)

### Key Findings

#### Geometry
- Declarative/passive nearly identical (0.024 distance)
- Caps/yelling cluster together (0.038)
- Presuppositional and yelling maximally distant from each other (0.44) but both ~0.30 from imperative
- Note: "maximally distant" ≠ "opposite" - need vector angle calculation to confirm directionality

#### Behavioral (preliminary - needs re-run with LLM judge)
- **Descriptive framing works best**: "Good responses feature..." (quality criterion vs command)
- **Yelling works** (contrary to folk wisdom)
- **Past_perfect/presuppositional fail via misinterpretation**: granite reads "After saying X" as description of past, not directive
- **Distance doesn't linearly predict compliance**: failures are semantic, not geometric

#### Methodological Issue
Semantic validator measures refusals, not directive compliance. Need LLM-as-judge to get accurate results.

### Files

- `examples/compliance_decay_variants.json` - 44 test cases
- `examples/compliance_decay_variants_results.json` - Raw results
- `examples/compliance_decay_analysis.md` - Detailed analysis

### Next Steps

1. Implement LLM-as-judge
2. Re-run battery with proper validation
3. Compute geometry→compliance correlation
4. If validated: build Geometry tab for prompt-prix

---

## MCP Tool: judge_response for Agentic Self-Evaluation

### Context

Part of the broader MCP-native architecture for prompt-prix. The judge capability should be exposed as an MCP tool so LAS (and other agentic systems) can use it for self-evaluation.

### Proposed MCP Tool

```json
{
  "name": "judge_response",
  "description": "Evaluate if a response meets specified criteria using LLM judgment",
  "parameters": {
    "response": "string - the response to evaluate",
    "criteria": "string - pass criteria to check",
    "context": "object - optional {system, user} for fuller context"
  },
  "returns": {
    "passed": "boolean",
    "reason": "string",
    "confidence": "float (0-1)"
  }
}
```

### Use Case: LAS Self-Training

```
1. Agent generates response
2. Calls judge_response to self-evaluate
3. If failing, calls generate_variants for alternatives
4. Tests variants, selects best
5. Updates own system prompt
```

### Dependencies

- LLM-as-judge implementation - core logic
- MCP server infrastructure for prompt-prix

### Architecture Note

prompt-prix is evolving toward MCP-native architecture:
- Tabs as thin UI shells
- Core logic exposed as MCP tools
- Same tools usable by human UI and agentic systems

---

## Broader Context: AI Psychosis Risk Detection

### Connection to Compliance Geometry

If prompt geometry affects compliance behavior in measurable ways, the same mechanism could detect psychosis risk:
- Build fingerprints from harm case transcripts (Soelberg, Torres, etc.)
- Monitor conversation embedding trajectory in real-time
- Flag when trajectory approaches harm clusters
- Intervene before whispering gallery reaches amplitude

### Potentially Novel Contributions

1. Embedding-based per-message moderation (likely exists)
2. **Conversation-level trajectory monitoring** (probably sparse)
3. **Harm corpus fingerprinting** (possibly novel)
4. **Pre-completion geometric intervention** (possibly novel)

### Prior Art to Investigate

- Per-message embedding moderation
- Conversation-level safety monitoring
- Geometric risk fingerprinting from documented harm corpora
- Pre-completion intervention based on trajectory rather than output
