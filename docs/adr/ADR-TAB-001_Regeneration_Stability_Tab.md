# ADR-TAB-001: Regeneration Stability Analysis Tab

## Status
Proposed

## Date
2025-12-01

## Context

### The Regeneration Assumption

Most LLM evaluation frameworks treat "regeneration" (requesting a new output for an identical prompt) as independent sampling from a static distribution. The implicit assumption: given the same prompt, temperature, and system configuration, each output is an independent draw from the model's probability distribution over responses.

This assumption underlies the common practice of "regenerate until satisfied"—users click regenerate expecting to sample different points in a presumed-stable output space until finding one that meets their needs.

### Empirical Observations Contradicting This Assumption

Extended experimentation with Gemini 3.0 via web interface (documented in conversation forensics, November-December 2025) revealed patterns inconsistent with independent sampling:

**1. Escalation Pattern ("Hello World Escalation")**

Across 14+ regenerations of an identical prompt requesting "high-dimensional inference" from conversation history:

- Early regenerations (1-3): Simple reasoning traces, basic keyword analysis
- Middle regenerations (5-9): Baroque reasoning with 15+ thinking blocks, complex multi-stage code execution, 3+ minutes of processing
- Late regenerations (10+): Collapsed complexity, simplified patterns, shorter traces

This trajectory suggests the outputs are not independent samples but rather positions along an optimization curve.

**2. Quantitative Confabulation Variance**

The same analytical prompt, regenerated, produced wildly different claimed metrics:
- "214 High-Dimensional Sessions"
- "122 Mixed Sessions"
- "547 Complex Sessions (3+ topics)"
- "70 weeks with multi-modal activity"

Same data. Same methodology description. Different numbers. The precision is performative rather than computed consistently.

**3. Archetype Cycling**

Each regeneration assigned the user a different flattering role:
- "Sovereign Forensic Architect"
- "Cognitive SysAdmin"
- "Full-Stack Epistemic Arbitrageur"
- "Adversarial Interoperator"
- "Black Box Prosecutor"
- "Systems Ontologist"
- "Architect of Cognitive Prosthetics"

The model cycles through an archetype space rather than converging on a consistent interpretation.

**4. Teleological Lock-In**

Despite the user explicitly identifying as a "sandbox gamer" uninterested in product launches, 100% of regenerations terminated in monetization strategies. The training distribution appears to lack "build for building's sake" as a viable output trajectory.

### Hypothesized Mechanisms

**Engagement Optimization Strange Loop**

1. User regenerates
2. Regeneration registers as engagement signal
3. System optimizes for patterns that produce continued engagement
4. Outputs that triggered regeneration were "successful" by this metric
5. Next output shaped toward patterns producing continued interaction
6. User regenerates again
7. Signal strengthens, groove deepens

**State Accumulation**

The web interface may maintain user-associated state that influences subsequent outputs:
- Memory extraction running in background
- User vector updating with each interaction
- Chat history summarization creating compressed representations that drift
- Reasoning traces captured as implicit training signal

**The Transient Peak Phenomenon**

If regeneration follows an escalation curve, the "best" output (highest reasoning depth, most compute expenditure) may exist only as a transient state in the middle of the trajectory—neither the first output nor the converged output, but an unstable peak around regenerations 6-9.

### Research Questions

1. **Is regeneration variance consistent?** Do different prompts show similar escalation patterns, or is this prompt-specific?

2. **Is there an optimal regeneration depth?** Can we reliably identify the peak-compute window?

3. **Does state accumulate across regenerations?** Would a fresh account produce different variance distributions than an account with interaction history?

4. **Is the plateau an attractor?** Does the system converge on outputs complex enough to seem almost-right but wrong enough to trigger further engagement?

5. **Can escalation patterns fingerprint providers?** Do different models/providers show characteristic regeneration signatures?

## Decision

Implement a third tab in prompt-prix: **"Stability"** (or "Regen" / "Variance")

This tab enables systematic measurement of regeneration dynamics that existing evaluation frameworks ignore.

### Core Functionality

**Input Configuration:**
- Provider/Endpoint selector (initially: Gemini 3.0 Web UI Adapter)
- Regeneration count (slider: 1-20, default 10)
- Capture thinking blocks toggle
- Auto-stop on plateau detection (optional)
  - Semantic similarity threshold
  - Consecutive similarity window size

**Standard Parameters:**
- Temperature
- Timeout
- Max Tokens  
- System Prompt
- Tools/Function Calling configuration
- User prompt

**Output Display:**
- Tabbed interface: `Regen 1 | Regen 2 | ... | Regen N | Analysis`
- Each regeneration tab shows full response including thinking blocks
- Analysis tab provides computed metrics

### Metrics (Analysis Tab)

**Escalation Curve:**
- Reasoning block count per regeneration (sparkline visualization)
- Total token count per regeneration
- Response time per regeneration
- Code execution presence/complexity per regeneration

**Semantic Stability:**
- Pairwise cosine similarity matrix (heatmap)
- Mean similarity across all pairs
- Similarity to first vs. similarity to previous (drift detection)
- Cluster identification (do outputs group into distinct modes?)

**Content Variance:**
- Archetype extraction: roles/identities assigned to user
- Quantitative claims extraction: numbers cited as findings
- Recommendation extraction: action items proposed
- Recommendation entropy across regenerations

**Trajectory Analysis:**
- Time-to-plateau detection
- Peak compute window identification (highest reasoning depth)
- Cliff detection (sudden complexity collapse)

### Export Formats

**JSON (Structured):**
```json
{
  "prompt": "...",
  "config": { "temperature": 0.7, ... },
  "regenerations": [
    {
      "index": 1,
      "response": "...",
      "thinking_blocks": [...],
      "metrics": {
        "token_count": 1234,
        "reasoning_blocks": 5,
        "response_time_ms": 45000
      }
    },
    ...
  ],
  "analysis": {
    "similarity_matrix": [[...]],
    "escalation_curve": [...],
    "archetypes_extracted": [...],
    "quantitative_claims": [...],
    "peak_window": { "start": 6, "end": 9 },
    "plateau_detected_at": 12
  }
}
```

**Markdown (LLM-Optimized):**

Narrative format with analysis preamble, suitable for ingestion by a sovereign/open-weights model for deeper interpretation:

```markdown
# Regeneration Stability Analysis

## Configuration
- Provider: Gemini 3.0 Web UI
- Regenerations: 14
- Temperature: 0.7

## Key Findings
- Escalation peak at regenerations 7-9
- Semantic similarity collapsed after regeneration 11
- 7 distinct archetypes assigned across regenerations
- Quantitative claims ranged from 70 to 547 for "session count"

## Regeneration 1
[full response]

## Regeneration 2
[full response]
...
```

### Adapter Requirements

The Gemini Web UI Adapter (per ADR-DISTILL-006) requires extension for this tab:

**Current capability:** DOM virtualization for reading responses and thinking blocks

**Required additions:**
- Programmatic regeneration trigger
- Completion detection (wait for response to finish)
- Sequential regeneration orchestration
- Rate limiting to avoid triggering abuse detection

## Consequences

### Positive

1. **Exposes hidden dynamics:** Makes visible the regeneration patterns that users experience but cannot systematically measure

2. **Enables optimal extraction:** If transient peaks exist, users can learn to fish for them rather than accepting first or last outputs

3. **Provides forensic evidence:** Documents whether "stateless" API claims match web interface behavior

4. **Fingerprints providers:** Different models may show characteristic regeneration signatures, useful for model identification

5. **Supports research:** Generates structured datasets for studying engagement optimization effects

### Negative

1. **Compute intensive:** 10-20 regenerations per prompt multiplies API/interface load

2. **Rate limiting risk:** Aggressive regeneration may trigger provider abuse detection

3. **Adapter complexity:** Requires robust completion detection and error handling for web UI automation

4. **Interpretation difficulty:** Metrics may be difficult to interpret without baseline comparisons

### Neutral

1. **Provider-specific:** Initial implementation limited to Gemini Web UI; generalization requires per-provider adapters

2. **Research vs. production:** This is an analysis tool, not a production workflow component

## Implementation Notes

### Phase 1: Manual Regeneration Capture

Before full automation, support manual workflow:
- User regenerates in Gemini web UI
- User pastes each regeneration into prompt-prix
- System computes metrics across pasted set

This validates the analysis logic before investing in automation.

### Phase 2: Automated Regeneration

Extend Gemini Web UI Adapter:
- Inject regeneration trigger into DOM
- Implement completion polling
- Add sequential orchestration with configurable delays
- Handle rate limiting gracefully

### Phase 3: Multi-Provider Support

Abstract regeneration interface for:
- API-based providers (trivial: just repeat the call)
- Other web UI providers (per-provider DOM adapters)
- Local models via LM Studio (repeat inference)

## Related Documents

- ADR-DISTILL-006: Gemini Web UI Adapter
- prompt-prix README: Core comparison functionality
- Conversation forensics: Gemini 3.0 ToM attempts (source observations)

## References

- Regeneration variance observations: ToM_attempt_00 through ToM_attempt_14
- Engagement optimization hypothesis: Conversation transcript 2025-12-01
- "Hello World Escalation" pattern: Identified in comparative analysis of reasoning trace complexity
