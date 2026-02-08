# ADR-013: Semantic-Chunker MCP Primitives

**Status**: Accepted
**Date**: 2026-02-08
**Related**:
- ADR-006 (Adapter Resource Ownership)
- ADR-011 (Embedding-Based Semantic Validation)
- LAS ADR-CORE-066 (Sleeptime Autonomous Orchestration)
- LAS ADR-CORE-067 (Adapter Convergence — Shared Pool Architecture)
- #148 (implementation)

---

## Context

prompt-prix wraps one semantic-chunker tool today: `calculate_drift` in `mcp/tools/drift.py`. This enables battery tests to validate responses against expected exemplars via cosine distance.

The semantic-chunker repo provides four additional tools that operate on the same embedding substrate:

| Tool | Purpose | Dependencies |
|------|---------|--------------|
| `analyze_variants` | Embed prompt variants, compute pairwise distances from baseline | StateManager (embeddings) |
| `generate_variants` | LLM-rephrase a constraint across grammatical dimensions | LLM call (hardcoded in upstream) |
| `analyze_trajectory` | Velocity/acceleration/curvature of text as particle in embedding space | StateManager, spaCy, numpy |
| `compare_trajectories` | Fitness score comparing synthetic text rhythm against golden reference | Same as above + DTW |

### Cross-Repo Role

prompt-prix's MCP tool surface is the evaluation contract for the broader ecosystem:

- **LAS ADR-CORE-066** has LAS autonomously driving prompt-prix via MCP for background model evaluation, consistency batteries, and drift measurement. These tools are the programmatic surface LAS orchestrates against.
- **LAS ADR-CORE-067** ports prompt-prix's `_ServerPool` + `_ConcurrentDispatcher` into LAS. The pool infrastructure is validated and adopted wholesale.
- **ADR-011** envisions a `TrajectoryValidator` using `heller_score` to detect circular reasoning in model responses. Wrapping `analyze_trajectory` as an MCP primitive is the prerequisite.

These wrappers serve both internal pipelines and external agents.

---

## Decision

### Wrap four semantic-chunker capabilities as MCP primitives

**File grouping** mirrors semantic-chunker module boundaries:

| New file | Tools | Pattern |
|----------|-------|---------|
| `mcp/tools/geometry.py` | `analyze_variants`, `generate_variants` | drift.py + judge.py |
| `mcp/tools/trajectory.py` | `analyze_trajectory`, `compare_trajectories` | drift.py |

### Shared StateManager singleton

Extract lazy-init helpers from `drift.py` into `mcp/tools/_semantic_chunker.py`. Three files (`drift.py`, `geometry.py`, `trajectory.py`) share one StateManager instance.

### `generate_variants` re-implementation

The semantic-chunker version hardcodes `openai.OpenAI(base_url="http://localhost:1234/v1")` with `model="local-model"`. This violates ADR-006 (adapter agnosticism) and LAS's model-at-call-time pattern (ADR-CORE-067 Phase 2).

Re-implement using prompt-prix's `complete()` — same pattern as `judge.py`. Takes explicit `model_id` parameter. The `DIMENSION_PROMPTS` dict and prompt template are lifted from semantic-chunker.

### Optional dependency handling

`analyze_variants`, `analyze_trajectory`, and `compare_trajectories` require semantic-chunker (and transitively spaCy, numpy). These are optional — tools raise `ImportError` with a clear message when semantic-chunker is absent, matching the existing `drift.py` behavior.

`generate_variants` has no semantic-chunker dependency — it only needs the registered adapter via `complete()`.

---

## Consequences

### Positive

- prompt-prix gains variant analysis and trajectory analysis as first-class MCP primitives
- LAS can orchestrate all five embedding-based tools through a uniform MCP interface
- `generate_variants` is adapter-agnostic — works with any backend prompt-prix supports
- Single StateManager singleton eliminates redundant embedding model initialization
- Foundation for ADR-011 validators (TrajectoryValidator, RefusalClusterValidator)

### Negative

- Three files now depend on `_semantic_chunker.py` shared module (tight internal coupling)
- spaCy `en_core_web_sm` auto-downloads on first trajectory analysis call (potential delay)
- `generate_variants` prompt template must be kept in sync with semantic-chunker's version if upstream changes

---

## References

- `prompt_prix/mcp/tools/drift.py` — existing wrapper pattern
- `prompt_prix/mcp/tools/judge.py` — pattern for tools that call `complete()`
- `semantic_chunker/mcp/commands/geometry.py` — upstream analyze_variants, generate_variants
- `semantic_chunker/mcp/commands/trajectory.py` — upstream analyze_trajectory, compare_trajectories
- PROMPT_GEOMETRY.md — detailed formulas and interpretation
