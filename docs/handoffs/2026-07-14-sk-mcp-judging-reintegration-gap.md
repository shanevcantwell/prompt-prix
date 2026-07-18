# Handoff — 2026-07-14 — sk-mcp judging re-integration gap

**Repo:** prompt-prix · **Branch:** main (clean) · **Session type:** cold orient → scope & record

## Nominal target (what opened the session)

Revisit embedding-based **judging with sk-mcp** (`semantic-kinematics-mcp`), on the premise
that "nv-embed-v2 anisotropy nullification is finally proven out."

## Headline finding

prompt-prix's embedding-judging seam is **inert, obsolete, and uncorrected** — it predates
sk-mcp's rename *and* its control/data-plane redesign. This is a **re-integration project**,
not a threshold tweak. Recorded as three cross-linked bugs; remediation deferred to an ADR
(superseding ADR-PPX-011). No code changed this session by design (session just opened).

## The three gaps (durable: GitHub issues)

- **#159** `bug` — **inert seam.** `prompt_prix/mcp/tools/drift.py:26` imports the retired
  `semantic_chunker` package (sk-mcp is now `semantic_kinematics`, v0.3.0a0). `ensure_importable()`
  → False → `calculate_drift` raises `ImportError` → runners silently skip drift on every run
  (per #138). All ADR-PPX-011 validators are dead code since the rename.
- **#160** `bug,refactor` — **obsolete mechanism.** Even with the name fixed, `drift.py` /
  `_semantic_chunker.py` directly import `mcp.commands.*` + drive a `StateManager` singleton —
  the pattern sk-mcp's current contract **forbids** (`ONE-DOOR`). sk-mcp now splits control-plane
  (9 MCP JSON-RPC tools) from data-plane (`BulkEmbedder`, `embed_corpus(items)->{id: np.ndarray}`).
  **ADR-PPX-011 documents a vanished mechanism** (`SemanticChunkerMCP` / `sys.path.insert` +
  `StateManager`) → must be superseded, not patched.
- **#161** `bug,enhancement` — **uncorrected measurement.** ADR-PPX-011 fixed thresholds
  (0.3/0.25/0.35) and the #140 slider assume a stable cosine distribution. Raw
  `calculate_drift`/`embed_text` return **in-cone** vectors; nv-embed-v2 *preserves* the
  anisotropy cone (‖μ‖²≈0.31) that embeddinggemma's isotropy regularizer *suppresses*, so a
  threshold means different things per backend — and prompt-prix inherits whichever species
  sk-mcp loaded. Root-cause sibling of **#140**. Correction lives in `analyze_axis_alignment`
  (empirical-null z-score), which needs a null cache: corpus → `BulkEmbedder` →
  `build_axis_null.py` → `analyze_axis_alignment`.

Cross-links posted on each issue (#159↔#160↔#161, out to #138/#140).

## Load-bearing caveat for the remediation ADR

"Proven out" holds for sk-mcp's **cone measurement** (reproducible, N≈86,748), **not** for an
end-to-end validated judging pipeline: sk-mcp's ADRs are still *Proposed*, the nv-embed cone
re-test is "possible-but-deprioritized," and **contrastive drift is the surviving track**
(absolute-magnitude measures were falsified on embeddinggemma). Scope any adoption to the
contrastive / z-scored track the record supports.

## Design decisions the ADR should record (not yet decided)

1. Correct-measurement-over-model-swap — a naive embeddinggemma→nv-embed swap under raw
   `calculate_drift` makes discrimination **worse** (deeper cone), not better.
2. Cross the sk-mcp **MCP control-plane** door for single-text judging/drift; use **BulkEmbedder**
   (data-plane) for corpus embedding. No direct `mcp.commands.*` import.
3. Fix identity: `semantic_chunker` → `semantic_kinematics`; retire `SEMANTIC_CHUNKER_PATH`.
4. Whether/how to stand up the null-cache pipeline for `analyze_axis_alignment` judging.

## Pointers

- Consumer side: `prompt_prix/mcp/tools/drift.py`, `_semantic_chunker.py`;
  `docs/adr/ADR-PPX-011-embedding-based-validation.md` (Phase 1 Accepted — now obsolete mechanism).
- Provider side: `/srv/dev/shanevcantwell/semantic-kinematics-mcp` — `docs/ARCHITECTURE.md`
  (control/data-plane split), `semantic_kinematics/embeddings/bulk.py` (`BulkEmbedder`), MCP
  tool server (`analyze_axis_alignment`, `calculate_drift`, `embed_text`).

## Working ground at close

`git status`: clean on main. No code/config changed. Durable output = issues #159/#160/#161
(remote) + this handoff. Next phase (operator's call): author the remediation ADR.
