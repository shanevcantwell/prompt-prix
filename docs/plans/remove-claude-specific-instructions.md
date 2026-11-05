# Remove Claude-Specific Instruction Files

**Status:** PROPOSED / NOT_IMPLEMENTED

**Issue:** [#169](https://github.com/shanevcantwell/prompt-prix/issues/169)

**Inspection baseline:** `07e083f51b28d521cb9de1496e99f86770becbb8`

**Scope:** this repository only

> **Authorization boundary:** This document is a ratification draft. It does not authorize deleting `.claude/CLAUDE.md`, creating the final canonical guidance, changing current references, or editing any other repository. No implementation work should begin until the questions at the end are answered.

## Decision summary

Remove `.claude/CLAUDE.md` without a redirect or tombstone, but do so in one atomic migration that first gives its still-needed, provider-neutral working guidance a provider-neutral home.

The recommended end state is:

- root `AGENTS.md` is the concise canonical home for repository-local agent/contributor operating guidance;
- `README.md`, `docs/README.md`, and `docs/ARCHITECTURE.md` remain canonical for product, usage, and architecture facts;
- ADRs remain canonical for durable architectural decisions;
- code docstrings state local invariants directly or cite the relevant ADR, rather than citing an agent instruction file;
- `.claude/` and its ignore rules are absent from the active tree; and
- Claude Desktop and Anthropic references that describe supported clients, APIs, research, or historical trade-offs remain, because they are product documentation rather than provider-specific repository instructions.

This recommendation is necessary because inspection found **no `AGENTS.md` or equivalent provider-neutral operating guide**. The issue context says provider-neutral doctrine is already live elsewhere, and much product/architecture material is, but the communication, collaboration, failure-handling, and destructive-git safeguards are not duplicated in current tracked guidance. Ratification must decide whether to preserve or deliberately retire those gaps.

## Disposition of `.claude/CLAUDE.md`

Classifications used below:

- **provider-specific delete** — meaningful only because the file is Claude-specific, including self-references and preservation of `.claude/`;
- **provider-neutral duplicate** — already has a more appropriate active source;
- **provider-neutral unique** — no equivalent tracked source was found; it must be ratified for migration or explicit retirement.

| Current section | Classification and evidence | Proposed disposition / destination |
|---|---|---|
| Title and purpose (lines 1-3) | Provider-neutral duplicate. `README.md` and `docs/README.md` state the purpose and tool count. | Delete with the file; do not duplicate in `AGENTS.md`. |
| Communication Style (7-11) | Provider-neutral unique. No equivalent tracked guidance was found. | Preserve a shortened version in `AGENTS.md`, if ratified. |
| Working Model: Co-Architects (15-27) | Provider-neutral unique. No equivalent tracked guidance was found. | Preserve the inspect-first, explain-scope/risk, and maintainability expectations in `AGENTS.md`, if ratified. |
| Core Concept: Fan-Out Pattern (31-41) | Provider-neutral duplicate. Covered by `README.md` and `docs/ARCHITECTURE.md`. | Delete duplicate; retain existing product/architecture docs. |
| Architecture (45-72) | Provider-neutral duplicate, with stale detail: it says local-inference-pool `v0.1.0`, while `pyproject.toml` pins `v0.6.0`. `docs/ARCHITECTURE.md`, ADR-006, and architecture tests are the stronger sources. | Do not migrate this copy. Keep architecture in `docs/ARCHITECTURE.md`/ADRs and correct any independently confirmed stale architecture text as a separate concern. |
| Battery File Formats (76-84) | Provider-neutral duplicate and incomplete: `docs/README.md` and `docs/ARCHITECTURE.md` include current ReAct fields omitted here. | Delete duplicate. Point examples to `docs/README.md#battery-file-formats`. |
| Semantic Validation (88-100) | Provider-neutral duplicate and less complete than `docs/ARCHITECTURE.md` and `docs/README.md`. | Delete duplicate; retain existing documentation and implementation as sources. |
| Testing (104-120) | Mixed. Commands/default marker are evidenced by `pyproject.toml`, integration tests, and `docs/README.md`; mocking boundaries are evidenced by ADR-006 and test docstrings. The “failure is a failure” rule and mandatory local `.venv` wording are provider-neutral unique. The self-reference “lives here in CLAUDE.md” is provider-specific delete. | Put only ratified working rules in `AGENTS.md`: install `.[dev]`, run the repository unit gate, report failures honestly, and mock at layer boundaries. Do not copy the unverifiable claim that every checkout's `.venv` already contains all dependencies. |
| Environment Configuration (123-133) | Provider-neutral duplicate and partly stale relative to `.env.example`, which prefers `LOCAL_INFERENCE_SERVER_N` and documents legacy LM Studio variables. | Treat `.env.example` as canonical; keep concise setup in `docs/README.md`. Do not migrate this block. |
| Git Operations Safety (137-147) | Mixed. Destructive-command safeguards and preservation of configuration/secrets are provider-neutral unique. Preserving `.claude/` is provider-specific delete. | Preserve only ratified neutral safeguards in `AGENTS.md`; remove the `.claude/` preservation rule. |
| Bug Spotters' Guide (151-158) | Provider-neutral unique, but its unconditional issue creation/closure commands may exceed an agent's authorization in a given task. | Do not migrate verbatim. Either retire it or replace it in `AGENTS.md` with “write a regression test and mutate GitHub only when explicitly authorized.” |
| Development Workflow (162-168) | Mixed. Setup/run commands are duplicated in `docs/README.md`; “commit with descriptive messages” and the generic edit/test flow are provider-neutral unique. | Keep user setup in `docs/README.md`; preserve only ratified repository-working gates in `AGENTS.md`. |
| Adding a New Adapter (170-177) | Provider-neutral duplicate. The protocol is in `prompt_prix/adapters/base.py`; ownership rules are in ADR-006/`docs/ARCHITECTURE.md`; the integration marker is in `pyproject.toml`. | Delete duplicate; optionally add a contributor-oriented link from `AGENTS.md`, not a second architecture specification. |

### Gap requiring ratification

The following neutral material does **not** currently have another tracked canonical home:

1. level, non-urgent communication;
2. inspect and understand before implementation;
3. explain options with file/function scope and risk;
4. prioritize maintainability over plausible-looking output;
5. report dependency/import/test failures rather than dismissing them;
6. destructive git safeguards; and
7. GitHub workflow advice, which needs an explicit authorization qualifier.

The migration must not claim these rules already exist elsewhere. The operator must either approve their curated migration to `AGENTS.md` or explicitly retire them.

## Active consumers and references

Repository-wide tracked-file searches at the inspection baseline found the following direct references or path handling:

| Location | Evidence | Consumer type | Proposed treatment |
|---|---|---|---|
| `README.md:185` | Links `.claude/CLAUDE.md` as “AI assistant context.” | Human/agent documentation pointer | Replace with `AGENTS.md` only after that file is ratified and created; otherwise remove the bullet. |
| `examples/README.md:7` | Links the Claude file's battery-format anchor. | User documentation pointer | Retarget to `../docs/README.md#battery-file-formats`. |
| `prompt_prix/adapters/base.py:19,52` | Cites `CLAUDE.md` for adapter rationale and “fail loudly.” | Source docstring reference; not runtime loading | Make the rationale self-contained and cite ADR-006 where useful; remove the external instruction-file dependency. |
| `prompt_prix/battery.py:10` | Attributes state-management rules to `CLAUDE.md`; those exact rules are not in the current file. | Stale source docstring reference | State the implemented behavior directly; retain the ADR-006 citation for layer ownership. |
| `prompt_prix/benchmarks/base.py:4` | Attributes Pydantic state to `CLAUDE.md`; that rule is absent from the current file. | Stale source docstring reference | Make the docstring self-contained. |
| `prompt_prix/benchmarks/custom.py:8` | Attributes fail-fast parsing to `CLAUDE.md`; that rule is absent from the current file. | Stale source docstring reference | Make the docstring self-contained. |
| `.gitignore:62` | Ignores `.claude/` even though the current file is already tracked. | Path handling, not a content consumer | Remove after deleting the directory so it cannot silently return as local active guidance. |
| `.dockerignore:38-39` | Labels and excludes `.claude/`. | Build-context path handling | Remove the Claude-specific lines; Docker already copies only `pyproject.toml` and `prompt_prix/`. |

Checks of `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `scripts/`, tracked files, and the Python/UI/CLI/MCP entry points found **no runtime, packaging, script, or CI dependency** on `.claude/CLAUDE.md`:

- setuptools packages only `prompt_prix*`;
- Docker copies `pyproject.toml` and `prompt_prix/`, not the instruction file;
- `prompt-prix`, `prompt-prix-cli`, `prompt-prix-mcp`, `python -m prompt_prix`, and `app.py` do not read it;
- no tracked `.github/workflows` or other CI configuration exists at this baseline; and
- scripts contain no `.claude`/`CLAUDE.md` reference.

A broader `Claude|Anthropic` search also finds legitimate product/research references in the README, MCP client examples, ADRs, the MCP server docstring, and a notebook. They are not instruction-file consumers and should not be removed under #169.

## Alternatives and trade-offs

### A. Root `AGENTS.md` plus existing specialist docs — recommended

**Pros:** provider-neutral discovery path; preserves unique operating doctrine; gives the README a clear AI-assistant-context target; keeps architecture and usage details in their existing sources.

**Cons:** creates a new canonical file that must stay concise and non-duplicative; requires ratification of which unique rules survive.

### B. No replacement agent guide; distribute or retire all unique material

Move testing/setup facts into existing docs, make docstrings self-contained, and remove the README assistant-context entry.

**Pros:** least doctrine and least duplication.

**Cons:** silently loses currently active communication, collaboration, and safety expectations unless each is deliberately retired; no obvious provider-neutral entry point for repository agents.

### C. Create `CONTRIBUTING.md` instead

**Pros:** conventional home for human development/testing workflow.

**Cons:** weaker agent discovery, awkward home for communication/authorization rules, and no existing repository convention supports it. It may still be useful later, but it does not replace an agent entry point cleanly.

### D. Keep `.claude/CLAUDE.md` as a redirect

Rejected. It leaves provider-specific active-tree doctrine in place, preserves a Claude-only discovery mechanism, and conflicts with the issue direction to delete rather than retain a tombstone.

## Exact implementation operations (not authorized by this draft)

If Alternative A is ratified, implement as one reviewable commit:

1. **Create** `AGENTS.md` at repository root with only the ratified unique operating rules and links to `docs/README.md`, `docs/ARCHITECTURE.md`, ADR-006, `.env.example`, and `pyproject.toml`; do not copy product/architecture tables.
2. **Delete** `.claude/CLAUDE.md` and remove the now-empty `.claude/` directory; add no redirect.
3. **Edit** `README.md` to replace the `.claude/CLAUDE.md` assistant-context link with `AGENTS.md`.
4. **Edit** `examples/README.md` to point battery formats to `docs/README.md#battery-file-formats`.
5. **Edit** `prompt_prix/adapters/base.py`, `prompt_prix/battery.py`, `prompt_prix/benchmarks/base.py`, and `prompt_prix/benchmarks/custom.py` so their docstrings are self-contained and, where applicable, cite ADR-006 rather than `CLAUDE.md`. Make no runtime changes.
6. **Edit** `.gitignore` to remove `.claude/`.
7. **Edit** `.dockerignore` to remove the `# Claude`/`.claude/` block.
8. **Do not edit** Claude Desktop, Anthropic API/research, or historical ADR references unless a separate accuracy issue is found.
9. **Do not edit** any other repository or any path under `design-docs/`.

If Alternative B or C is ratified, the same deletion/reference/docstring/ignore cleanup applies, but operation 1 and the README destination change accordingly.

## Validation and gates

### Validation run for this proposal

- inspected `.claude/CLAUDE.md`, `README.md`, `docs/README.md`, `docs/ARCHITECTURE.md`, `.env.example`, `pyproject.toml`, Docker files, scripts, tracked files, entry points, ADR-006, architecture tests, and all direct tracked references;
- no implementation files were changed by this proposal; and
- proposal-only validation requires reference/path searches plus `git diff --check`, not pytest.

### Required implementation gate after ratification

1. `git ls-files '.claude/*'` returns no files.
2. `git grep -n -E 'CLAUDE\.md|\.claude/' -- . ':!docs/plans/remove-claude-specific-instructions.md'` returns no active references.
3. A case-sensitive `git grep -n -E 'Claude|Anthropic'` is reviewed; every remaining match is classified as product/client/research/history, not repository instruction loading.
4. README and examples Markdown links resolve to tracked files and the battery-format anchor exists.
5. `git diff --check` passes.
6. `.venv/bin/pytest` (the configured non-integration unit suite) passes because Python docstrings are touched, even though runtime behavior should not change. Integration tests are not required for this documentation/docstring migration.
7. Final diff confirms no runtime code change, no other-repository change, and no `design-docs/` path.

## Rollback

Make the implementation atomic so `git revert <migration-commit>` restores the prior file, references, docstrings, and ignore rules together. Do not roll back only the new canonical guide while leaving broken pointers.

If the chosen provider-neutral guide proves unsuitable but removal remains correct, prefer a follow-up migration to ratified existing docs over restoring `.claude/CLAUDE.md`. Restoring the Claude file should be reserved for a full migration rollback, not used as a placeholder.

## Ratification questions

1. **Canonical home:** Approve root `AGENTS.md` (Alternative A), choose `CONTRIBUTING.md`, or intentionally have no replacement agent guide?
2. **Unique doctrine:** Which of the seven gap items should be preserved, revised, or explicitly retired? The recommendation is to preserve items 1-6 in concise form and revise item 7 to require explicit GitHub-mutation authorization.
3. **Docstring policy:** Approve making code docstrings self-contained/ADR-linked rather than pointing them to `AGENTS.md`?
4. **Path cleanup:** Approve removal of both `.gitignore` and `.dockerignore` Claude-specific entries with the directory?
5. **Scope classification:** Confirm that Claude Desktop/Anthropic product, client, research, and ADR references remain out of scope?
6. **Gate:** Approve the atomic file-operation list and the unit-test/reference/link/diff validation gate above?

Until these questions are answered, the status remains **PROPOSED / NOT_IMPLEMENTED** and no removal or canonical-guidance creation is authorized.
