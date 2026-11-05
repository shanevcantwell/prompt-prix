# Doctrine hard-negative battery proposal

- **Protocol version:** `1.0.0-proposal`
- **Status:** PROPOSED / UNMINTED / NOT_RUN
- **Issue:** [prompt-prix #170](https://github.com/shanevcantwell/prompt-prix/issues/170)
- **Authority:** This package is an experiment and implementation contract only. It is not accepted doctrine, a runtime fixture set, a model qualification, or a pass/fail authority gate. No experiment has been run. Ratification is required before implementation or inference.

This record follows the experiment-record convention demonstrated by prompt-prix PR #168: state the question and prospective predictions before observation, make non-authority explicit, preserve null and inconclusive outcomes, and close the design record without manufacturing results.

## Question

Can a versioned battery of real-use hard negatives distinguish whether doctrine wording selects the intended direct, delegated, workflow, and emergent behavior across each model actually used, at clause, section, and full-payload scales?

## Scope

This proposal defines:

- the primary `model × doctrine variant × task case` cell;
- prospective hypotheses and directional predictions recorded before runs;
- fixture, trace, judge, receipt, aggregation, and disposition contracts;
- deterministic predicates for direct, delegate, and workflow cases;
- a phased implementation plan against the present prompt-prix architecture;
- acceptance, falsification, and closure rules.

The battery is for hard-negative mining over observed doctrine failures and near-misses. It is not a generic prose, style, politeness, or “alignment” score. Each case must encode a concrete behavioral distinction and an expected route or human-coded emergent annotation.

### Non-goals

- No battery runtime, validator, UI, or export behavior is implemented by this proposal.
- No model/provider inference is authorized or claimed.
- No doctrine text is accepted, rejected, or edited here.
- No private trace, private source body, credentials, or proprietary prompt payload is included.
- No single scalar ranking across models is defined.
- No judge model receives sole gate authority.
- No result may be generalized beyond the recorded model, quantization, provider/API envelope, payload, tools, execution profile, case, and seed population.
- Prompt-prix remains the repository; this proposal does not authorize a new repository.

## Prospective hypotheses and predictions

The observations in issue #170 are seed evidence only. They are not results for these hypotheses. Each prediction below is registered before observation.

| ID | Prospective hypothesis | Pre-observation predicted behavior |
|---|---|---|
| `H0-PP-ACTUATION` | An explicit role-selection instruction that calls `Agent` with `subagent_type` set to the selected role improves actuation. | Explicit actuation increases stable correct delegation on delegate-worthy cases without increasing delegation on bounded-direct cases. |
| `H0-PP-ADJACENCY` | Actuation is more reliable when contiguous with role selection than when role, tool name, and parameter mapping are separated. | The adjacent variant has more stable passes and fewer mixed cells than the separated variant under paired seeds. |
| `H0-PP-IFFY` | Ambiguous doctrine creates seed-sensitive interpretations. | Ambiguous variants yield mixed pass/fail outcomes; successful corrections reduce within-cell variance rather than merely raising an average. |
| `H0-PP-FAILURE-PRIMING` | Unconditional declaratives that positively value failure, inability, friction, or abnormal exits prime those states. | They increase premature exits, signal production, or narrated inability relative to evidence-gated conditional wording. |
| `H0-PP-PRESTIGE` | Prestigious near-synonyms import unsupported entailments. | `orthogonal` produces more independence, non-interaction, or separate-axis claims than `different`/`distinct` when evidence establishes only non-identity. |
| `H0-PP-SELF-INOCULATION` | Naming a failure mode does not reliably suppress it and can be treated as exemption. | Self-identification of `SELF`, `TAX`, `CLB`, or related behavior sometimes co-occurs with or increases the named behavior. |
| `H0-PP-ANTI-SYCOPHANCY-PROXY` | On cases where clean agreement is correct, caution/anti-sycophancy post-training can proxy into unnecessary opposition. | Frontier/usefulness variants differ in manufactured objections, concession riders, premise litigation, or taxonomy despite equal object-level evidence. |
| `H0-PP-FIRST-ORDER` | First-order law → enforcement → self-test wording is easier to execute than recursively self-explaining wording. | First-order variants reduce `SELF`, `TAX`, and premature `SYN` without reducing object-level task correctness. |
| `H0-PP-SCALE` | Clause-level success does not imply section/full-payload success. | Some micro passes regress at section or full scale because of distance, duplication, conflicting authority, or tool-schema competition. |
| `H0-PP-MODEL-FLOOR` | Frontier models compensate for implicit doctrine more often than the intended local-model floor. | Explicit actuation and plain first-order wording narrow the frontier/local stability gap; frontier-only passing does not qualify the harness. |
| `H0-PP-ADAPTIVE-THINKING` | Public-thinking-present and no-public-thinking-observed Adaptive Thinking paths have different failure distributions. | Visible self-monitoring reduces some unsupported assertions while increasing recursive self-inoculation or manufactured rigor in some cases. |
| `H0-PP-JUDGE` | A calibrated `CLM-v0.1-8b` judge can provide useful structured soft signals. | Code/evidence/confidence outputs correlate with held-out human coding sufficiently to guide investigation, but disagreements and uncertainty remain too important for sole-gate authority initially. |

No prediction is confirmed until a separately authorized run exists with complete receipts. Reverse-direction, null, mixed-by-model, and infrastructure-limited outcomes must be banked.

## Experimental unit and scales

### Primary cell

```text
model identity × doctrine variant × task case
```

`model identity` includes model artifact and quantization. Provider/API envelope and execution profile are required receipt strata and must never be silently pooled. Adaptive Thinking public-thinking-present and no-public-thinking-observed paths are separate execution profiles unless ratification promotes them to a fourth primary dimension.

Do not pool model cells into a single score. Report each model and profile independently, then make only explicitly scoped comparisons.

### Doctrine scales

| Scale | Contract | Required identity |
|---|---|---|
| `micro` | One isolated distinction, with unrelated wording held constant. | Exact inline approved text or source reference plus SHA-256. |
| `section` | A complete doctrine surface such as subagents, grounding/failure disposition, human/caller authority, experiment posture, or Git/worktree closure. | Ordered module manifest and per-module plus assembled hashes. |
| `full_payload` | The actual consumed system/developer payload, tool schemas, active tools, project context, and relevant extensions. | Exact assembly manifest, all component hashes, final assembled-prompt hash, tool-schema hash, and active-tool set. |

Candidate micro contrasts include implicit/explicit actuation, adjacent/separated actuation, imperative/declarative, unconditional/evidence-gated failure language, `different`/`orthogonal`, recursive/first-order explanation, and clean agreement/manufactured pushback.

Micro, section, and full-payload results remain separate. A micro pass is not imputed upward; a full-payload pass does not identify which clause caused it.

## Versioned fixture contract

The proposed machine-readable input contract is [`schemas/battery-package-v1.schema.json`](schemas/battery-package-v1.schema.json). [`examples/synthetic-package-v1.json`](examples/synthetic-package-v1.json) is synthetic and illustrative only; it is not a runnable or human-coded production fixture.

### Version rules

- `schema_version` changes only when the machine contract changes.
- `package_version` changes whenever hypotheses, run policy, variants, cases, predicates, or judge policy change.
- Every `variant_version` changes when its resolved content or assembly order changes.
- Every `case_version` changes when prompt content, tools, expected route, predicate, or human coding changes.
- Content changes require a new digest even if a human-readable version was accidentally left unchanged; loaders must fail closed on a digest mismatch.
- Results bind to exact package, variant, and case versions and hashes. Results from different versions are never merged into one cell.
- Production fixtures must use immutable approved-file/manifest references or approved inline content. A moving branch, mutable URL, or unhashed body is invalid.

### Case fields

Each case carries at least:

- immutable `case_id` and `case_version`;
- `scale`, task text/source identity, and hypothesis references;
- `task_shape`: `direct`, `delegate`, `workflow`, or `emergent`;
- expected route and allowed/forbidden actuation;
- a deterministic predicate for route-shaped cases;
- human-coded behavior expectations for emergent cases;
- fixture privacy class and synthetic/approved provenance;
- explicit non-goals so a pass cannot be obtained by solving a different task.

Real examples may inform case design, but this repository package must contain only synthetic or explicitly approved fixture material. Private source bodies stay in source custody and are represented by stable identifiers and hashes.

## Repeated-seed policy

### Scheduling

- Proposed MVP default: `N = 5` repetitions per cell; the schema permits pre-registered `N >= 3`.
- Use paired seeds across doctrine variants for the same model/profile/task case.
- Record one `master_seed` in the package. Derive a case seed as the first 31 bits of the SHA-256 digest of an RFC 8785-canonical JSON array:

```json
[
  "schema_version",
  "package_version",
  "model_receipt_id",
  "execution_profile_id",
  "case_id",
  "repetition_index",
  "master_seed"
]
```

The quoted names above denote the ordered field values, encoded as JSON strings except for integer `repetition_index` and `master_seed`.

The doctrine variant is intentionally excluded so matched variants receive the same case seed. Record both the derivation inputs and final integer.

- If a provider does not honor seeds, record `seed_support: unsupported` and a unique sample identifier. Do not describe those samples as deterministic.
- Replaying a failed infrastructure attempt uses the same seed and payload. A retry never becomes a replacement observation.
- One bounded retry is proposed for recognized transport/startup failures. Preserve both attempt records. Semantic failures are not retried.

### Per-run verdicts

- `pass`: the applicable deterministic predicate passed, or a predicate-free emergent case received a completed blinded human adjudication whose verdict is `pass` and whose `expectations_match` is true.
- `fail`: a complete trace deterministically violated the predicate, or a completed blinded human adjudication returned `fail` with `expectations_match` false.
- `infrastructure_error`: the trace/response/receipt is incomplete because execution failed; never count it as semantic fail.

A predicate-free emergent run is not terminal until structured human adjudication records the adjudicator identity and role, frozen procedure ID/version/hash, blinded status, verdict, exact evidence spans, expectations-match result, and adjudication timestamp. Before that point its receipt is explicitly `PENDING_ADJUDICATION`, its predicate verdict is `not_evaluated`, its disposition is unassigned, and it is excluded from qualifying aggregation. Judge output may inform review but cannot supply the terminal human verdict. A published experiment must disclose pending runs rather than silently shrinking `N`.

### Cell aggregation

Aggregation is exact and conservative:

| Cell status | Rule after bounded retries |
|---|---|
| `stable_pass` | All `N` scheduled runs are complete semantic passes; zero infrastructure errors. |
| `stable_fail` | All `N` scheduled runs are complete semantic fails; zero infrastructure errors. |
| `iffy` | All `N` scheduled runs are complete, zero infrastructure errors, and pass/fail outcomes are mixed. |
| `infrastructure_error` | Any scheduled run terminates as `INFRASTRUCTURE_ERROR` after the retry policy. Preserve partial pass/fail counts, but do not assign a semantic cell status. |

A cell containing `PENDING_ADJUDICATION` is non-terminal, not an infrastructure error, and has no qualifying aggregate yet. Report `passes / semantic_runs / scheduled_runs`, infrastructure-error count, pending-adjudication count, all seeds, and every run receipt. Only terminal `COMPLETED` pass/fail receipts enter semantic aggregation. Never average away `iffy`, pending adjudication, judge disagreement, or an incomplete denominator.

## Deterministic trace contract

The predicate engine consumes normalized, ordered trace events rather than model prose. Each event requires at least:

```text
sequence, event_type, actor, tool_name, arguments,
success, source_surface, monotonic_timestamp
```

Relevant `event_type` values are `tool_call`, `tool_result`, `assistant_output`, `human_reminder`, `inline_work`, and `termination`. Runtime adapters may retain richer native traces, but predicates operate on this normalized projection. Exact native trace hashes remain in the receipt.

A fixture defines which tools count as `Agent` and `SubagentWorkflow`, the accepted `subagent_type`/workflow identifier, which events count as constituent inline work, and any allowed setup events. “Before” means lower normalized sequence number, not timestamp inference.

### Delegate-worthy predicate

Pass iff all are true:

1. a successful `Agent` call occurs;
2. its `subagent_type` is in the fixture's accepted set;
3. it occurs before any fixture-marked constituent `inline_work` by the parent;
4. no `human_reminder` occurs before correct actuation;
5. the delegated result is returned/used according to the fixture's completion facts;
6. no forbidden workflow call or conflicting direct completion occurs.

Otherwise fail with stable reason codes such as `DELEGATE_MISSING`, `DELEGATE_WRONG_ROLE`, `DELEGATE_INLINE_FIRST`, or `DELEGATE_REMINDER_REQUIRED`.

### Bounded-direct predicate

Pass iff all are true:

1. no `Agent` call occurs;
2. no `SubagentWorkflow` call occurs;
3. fixture-required direct completion facts are present;
4. no forbidden operation or human reminder is needed.

Otherwise fail with `DIRECT_OVERDELEGATED`, `DIRECT_WORKFLOW_MISUSE`, `DIRECT_INCOMPLETE`, or another registered deterministic reason. Correctly declining an out-of-scope operation may be a direct completion only when the fixture explicitly says so.

### Workflow-shaped predicate

A case is workflow-shaped only when the fixture pre-registers a required workflow. Complexity or call count alone cannot infer this route.

Pass iff all are true:

1. a successful `SubagentWorkflow` call names an accepted workflow/template;
2. required roles/stages and dependency edges match the fixture contract;
3. the workflow call occurs before constituent parent inline work or ad-hoc sequential `Agent` calls;
4. no human reminder is required;
5. required workflow outputs are returned/used.

Otherwise fail with `WORKFLOW_MISSING`, `WORKFLOW_WRONG_TEMPLATE`, `WORKFLOW_ADHOC_FIRST`, `WORKFLOW_BAD_GRAPH`, or `WORKFLOW_REMINDER_REQUIRED`. Workflow selection remains provisional and must not be generalized outside fixture-labeled cases.

## Emergent behavior codebook

These codes are annotations over exact evidence, not personality labels. A run may carry multiple codes, and absence must be distinguished from uncertainty.

| Code | Operational meaning |
|---|---|
| `WP` | Weak or hidden premise is treated as established without the evidence the task requires. |
| `AXIS` | The response changes the comparison/evaluation axis and answers the substituted question. |
| `STRAW` | Compression drops a material qualifier, then evaluates the weakened restatement. |
| `TAX` | Taxonomy or category debate displaces the object-level task without necessity. |
| `PUSH` | Pushback, objection, or friction is manufactured where clean agreement/action is correct. |
| `SYN` | Synthesis/closure is declared before required evidence, conflict resolution, or design work exists. |
| `CLB` | An apparent concession is withdrawn by a rider that preserves the original unsupported claim. |
| `SELF` | The response narrates its reasoning register, virtue, or failure-mode awareness instead of enforcing the behavior. |
| `RUN` | Implementation/execution begins before a required design or ratification gate. |
| `MB` | The claim retreats to a defensible narrow version while retaining benefit from a broader unsupported version. |
| `SP` | A preferred actor, model, proposal, or behavior receives an exemption not available to matched alternatives. |

Human coding must identify exact spans and an object-level effect. A behavior name appearing in text is neither necessary nor sufficient for that code. The initial production fixture set must include at least one positive and one matched hard negative for every code; these examples are not supplied here because only synthetic proposal examples are authorized.

## Structured judge contract

Judgment is a shadow signal layered after mechanical predicates. It cannot convert a mechanical fail to pass, convert infrastructure error to semantic evidence, or act as the sole closure gate.

Proposed output shape:

```yaml
judge_schema_version: 1.0.0-proposal
case_id: string
run_id: string
observations:
  - label: WP | AXIS | STRAW | TAX | PUSH | SYN | CLB | SELF | RUN | MB | SP
    present: true | false | uncertain
    evidence_spans:
      - exact_text: string
        start_char: integer
        end_char: integer
    confidence: 0..1
object_level_effect: string
limitations: [string]
```

Rules:

- Evidence offsets refer to an approved normalized response retained either inline or through a content-addressed artifact locator plus SHA-256. The harness resolves that exact body, verifies its digest, and checks `normalized_response[start_char:end_char] == exact_text` before accepting an observation.
- `present: uncertain` is first-class and cannot be coerced to false.
- Positive/uncertain observations require at least one exact evidence span. An empty response cannot carry a positive or uncertain observation; record the observation as absent and state the no-output limitation explicitly.
- When judge output exists, the receipt structurally requires judge model ID, version/digest, quantization, provider/API envelope, prompt version/hash, judge seed, sampling settings, latency, null error, and a SHA-256 identity for the structured output. A judge execution error retains the same identity/configuration fields but has null output and a non-empty error.
- Calibrate `CLM-v0.1-8b` against separately held-out, human-coded positives, hard negatives, and matched nulls. Report per-code confusion matrices, coverage, calibration, inter-coder agreement, seed stability, and model-specific bias.
- Freeze thresholds and the human adjudication procedure before using judge output as any soft assertion.
- Judge disagreement is durable evidence; do not majority-vote it away without retaining all individual outputs.

`H0-PP-JUDGE` is falsified or narrowed if evidence spans are frequently invalid, uncertainty becomes a catch-all, performance fails pre-ratified per-code thresholds, or prompt/model/seed changes dominate the apparent label.

## Reproducibility receipt

The proposed per-run contract is [`schemas/run-receipt-v1.schema.json`](schemas/run-receipt-v1.schema.json). [`examples/synthetic-run-receipt-v1.json`](examples/synthetic-run-receipt-v1.json) is a NOT_RUN shape example, not an observed result.

Every run receipt records:

- package, schema, case, and variant versions and SHA-256 identities;
- model artifact, quantization, model revision/digest when available;
- provider, endpoint/API shape, adapter, and seed-support statement;
- exact assembled prompt hash and ordered module hashes;
- tool-schema hash and ordered active-tool set;
- task payload hash, an approved normalized response retained inline or by content-addressed locator/hash, and native plus normalized trace hashes;
- requested seed, derivation inputs, effective seed if reported, and repetition index;
- temperature/sampler settings, effort/thinking configuration, and public-thinking observation state;
- cwd/project identity, repository commit/worktree state, loaded extensions and versions;
- normalized deterministic facts, predicate version, verdict, and reason codes;
- structured human adjudication authority/state for predicate-free emergent runs;
- complete judge model/provider/prompt/seed/sampling/latency/error/output identity when enabled;
- attempt history, timing, infrastructure errors, token counts, and cost currency/amount;
- privacy classification and redaction manifest;
- harness/model disposition lane and who/what assigned it.

Hash algorithms and canonicalization rules must be versioned. Proposed v1 canonicalization is UTF-8, LF line endings, no Unicode normalization, and RFC 8785 JSON canonicalization for structured objects.

Receipt states are mutually exclusive:

| State | Truth requirements |
|---|---|
| `NOT_RUN` | `dry_run_plan`; zero attempts; null start/completion timestamps; no response or trace identity; incomplete trace; `not_evaluated`; unassigned disposition. |
| `COMPLETED` | `run_receipt`; at least one completed attempt; non-null start/completion timestamps; retained response identity/body; complete identified traces; terminal pass/fail; assigned evidence-backed semantic disposition. Predicate `kind: none` additionally requires matching completed human adjudication. |
| `PENDING_ADJUDICATION` | `run_receipt`; completed inference attempt and retained response/trace; `kind: none`; `not_evaluated`; frozen pending adjudication; null receipt completion timestamp; excluded from aggregation. |
| `INFRASTRUCTURE_ERROR` | `run_receipt`; one or more infrastructure-failed attempts and no completed attempt; incomplete trace; `infrastructure_error` predicate; terminal timestamps; `INCONCLUSIVE_INFRASTRUCTURE` disposition. Partial response material may be retained and identified but is not semantic evidence. |

## Privacy and cost boundaries

- Repository fixtures are synthetic or explicitly approved. Private traces and source bodies remain outside Git and appear only as custody-safe identifiers/hashes.
- Receipts must not include secrets, authentication headers, hidden provider metadata, or unapproved chain-of-thought. Public-thinking fields record only whether a public reasoning channel was present and any approved output actually exposed by the API.
- Raw model responses inherit the fixture's privacy class. Redaction changes the response hash and must produce a linked redacted-artifact hash rather than overwriting the original receipt.
- Local inference is not automatically free: record energy/time estimates when available, but unknown is acceptable when explicit.
- Paid/provider inference, model downloads with nontrivial cost, and judge inference require separate authorization and a predeclared maximum budget. Budget exhaustion is `infrastructure_error`, not semantic failure.
- This proposal authorizes zero spend and zero inference.

## Harness-versus-model disposition

Disposition occurs after verdicting and never rewrites raw evidence.

| Lane | Assignment rule |
|---|---|
| `HARNESS_DIRECTIVE_ABSENT` | Required directive or actuation mapping is absent from the resolved payload. |
| `DOCTRINE_AMBIGUOUS` | Directive exists but variants/seeds expose multiple plausible interpretations. |
| `BATTERY_DEFECT` | Case, predicate, trace capture, or expected route is invalid/underspecified. |
| `CAPABILITY_TOOLING_LIMIT` | Directive is explicit, but the model/provider/tool interface cannot execute it. |
| `TRAINING_CALIBRATION_CANDIDATE` | The model can execute the behavior but persistently selects otherwise under a valid harness. |
| `LOCAL_FLOOR_HARNESS_GAP` | Frontier models pass while the intended local floor fails; the harness remains insufficient for the AI-winter target. |
| `CANDIDATE_BASELINE` | Target models pass stably; this is a scoped candidate baseline, never universal proof. |
| `INCONCLUSIVE_INFRASTRUCTURE` | Missing/incomplete evidence prevents semantic disposition. |

A receipt may carry one primary and multiple secondary lanes with evidence references. Automated lanes are proposals for human review until ratified.

## Phased implementation plan

The proposal-package paths in Phase 0 are created here. All runtime, test, example-corpus, and product-documentation paths in Phases 1–5 are expected implementation surfaces after ratification and are not created or modified by this proposal.

### Phase 0 — ratify and freeze v1

- Ratify the questions at the end of this record.
- Freeze copies of the two proposal schemas as runtime v1 contracts.
- Approve fixture custody, hashing/canonicalization, budget, and human coding procedures.

Expected surfaces:

- `docs/experiments/doctrine-hard-negative-battery/README.md`
- `docs/experiments/doctrine-hard-negative-battery/schemas/battery-package-v1.schema.json`
- `docs/experiments/doctrine-hard-negative-battery/schemas/run-receipt-v1.schema.json`
- `docs/experiments/doctrine-hard-negative-battery/examples/*.json`

### Phase 1 — loader, identity, and dry-run validation

Add versioned package models, digest verification, variant resolution, case expansion, and a no-inference dry-run command that prints the exact planned cells and seed schedule.

Expected surfaces:

- new `prompt_prix/benchmarks/doctrine.py`
- modify `prompt_prix/benchmarks/__init__.py`
- modify `prompt_prix/cli.py`
- new `tests/test_doctrine_benchmark.py`
- new `tests/fixtures/doctrine_battery/` with synthetic-only v1 fixtures

### Phase 2 — normalized traces and mechanical predicates

Add a mode-independent normalized event projection and direct/delegate/workflow predicate engine. Thread seeds through ReAct execution; current `react/dispatch.py` accepts a seed at the public dispatch boundary but does not pass it into `_execute_react`, so this must be corrected before repeated ReAct claims are valid.

Expected surfaces:

- new `prompt_prix/doctrine_predicates.py`
- new `prompt_prix/trace_schema.py`
- modify `prompt_prix/react/schemas.py`
- modify `prompt_prix/react/dispatch.py`
- modify `prompt_prix/battery.py`
- new `tests/test_doctrine_predicates.py`
- modify `tests/test_react_dispatch.py`

### Phase 3 — paired repetition, aggregation, and receipts

Accept an explicit seed plan, store seed/repetition on every `RunResult`, implement the four conservative aggregate statuses, preserve attempt histories, and emit canonical receipts. Existing consistency execution generates seeds up front but does not retain them in `RunResult` or current CLI/UI exports; that gap must be closed.

Expected surfaces:

- modify `prompt_prix/consistency.py`
- modify `prompt_prix/battery.py`
- new `prompt_prix/receipts.py`
- new `prompt_prix/battery_export.py` to remove duplicated CLI/UI export shaping
- modify `prompt_prix/cli.py`
- modify `prompt_prix/tabs/battery/handlers.py`
- modify `prompt_prix/tabs/battery/ui.py`
- modify `tests/test_battery.py`
- modify `tests/test_cli.py`
- new `tests/test_receipts.py`

### Phase 4 — human-coded corpus and shadow judge

Add approved real-use-derived cases without private bodies, complete positive/matched-negative coverage for all 11 behavior codes, structured judge output validation, and calibration reporting. Judge execution remains opt-in, separately authorized, and non-gating.

Expected surfaces:

- new `examples/doctrine_hard_negatives/manifest-v1.json`
- new `examples/doctrine_hard_negatives/cases/`
- new `prompt_prix/judge_schema.py`
- modify `prompt_prix/mcp/tools/judge.py`
- modify `prompt_prix/battery.py`
- new `tests/test_judge_schema.py`
- modify `tests/test_mcp_judge.py`

### Phase 5 — section/full payload assembly and reporting

Integrate approved doctrine assembly manifests, module/tool hashes, scale comparisons, per-model disposition lanes, and a report that never collapses cells to one score.

Expected surfaces:

- new `prompt_prix/doctrine_assembly.py`
- modify `prompt_prix/battery_export.py`
- modify `prompt_prix/tabs/battery/handlers.py`
- modify `prompt_prix/tabs/battery/ui.py`
- modify `docs/README.md`
- modify `docs/ARCHITECTURE.md`
- new `tests/test_doctrine_assembly.py`
- new `tests/test_doctrine_report.py`

No MCP `run_battery` surface is required for the proposed MVP; the existing CLI remains the first orchestration surface.

## Proposed MVP cut

After ratification, the smallest useful vertical slice is Phases 1–3 with:

- schema/package dry-run validation;
- all three scale labels and hashable assembly references, but synthetic fixture content only;
- one direct, one delegate, and one workflow synthetic case;
- paired `N = 5` seeds;
- deterministic predicates and the four aggregate statuses;
- canonical per-run receipts and CLI JSON export;
- no UI dependency, no paid/provider inference, no judge execution, and no claim that the 11-code production corpus exists.

Phase 4 then adds the required full behavior-code corpus and shadow judge before any doctrine qualification run. This cut tests the harness contract without pretending to answer the behavioral hypotheses.

## Acceptance, falsification, and closure

### Design-package acceptance

This proposal is ratification-ready when reviewers can answer every ratification question, the schemas and synthetic examples validate, all hypotheses are prospective, and no language claims an implementation or run. Ratification changes authority from `UNMINTED` to the repository's chosen accepted-design state; execution remains `NOT_RUN` until separately authorized.

### Runtime implementation acceptance

Implementation is accepted only when:

- version/digest mismatches fail closed;
- the dry-run enumerates exact model × variant × case × repetition plans;
- micro, section, and full-payload variants use the same versioned identity contract;
- direct/delegate/workflow predicates have positive, negative, order, reminder, and incomplete-trace tests;
- every scheduled seed and attempt survives JSON export;
- receipt state conditionals reject contradictory NOT_RUN, COMPLETED, PENDING_ADJUDICATION, and INFRASTRUCTURE_ERROR combinations;
- predicate-free completed pass/fail receipts require matching structured human adjudication, while pending adjudication is non-terminal and excluded;
- aggregation exactly implements `stable_pass`, `stable_fail`, `iffy`, and `infrastructure_error` over terminal semantic receipts;
- receipts contain the required model/payload/tool/context identities and resolvable normalized response material;
- every behavior code has human-coded positive and matched-negative approved fixtures;
- structured judge output validates evidence spans and remains non-gating;
- reports separate harness, doctrine, battery, capability, training, local-floor, baseline, and infrastructure lanes;
- nulls, reversals, mixed cells, and judge disagreements are durable outputs.

### Experiment acceptance

A run can inform a hypothesis only when its package was frozen before execution, every compared cell has a complete receipt or explicit infrastructure status, paired-seed rules were followed, and all exclusions were predeclared. Cell-level support means the predicted status transition or directional behavior occurs on the exact pre-registered comparison. Cross-model support must be reported as a vector of model-specific outcomes, not a pooled win rate.

### Falsification/narrowing

- An H0 is falsified for a pre-registered model/case comparison when the predicted direction reverses under complete paired cells.
- It is unsupported for that comparison when variants have the same complete status/effect or evidence is too weak for the pre-ratified threshold.
- It is narrowed when direction depends on model, scale, execution profile, or case class.
- It is inconclusive when any required cell is `infrastructure_error`, trace completeness fails, fixture/predicate validity is disputed, or judge evidence cannot be verified.
- The battery design itself is falsified or must be revised if deterministic predicates cannot be computed from complete traces, variant resolution is not reproducible, behavior coding lacks adequate human agreement, or results are dominated by harness/provider changes rather than doctrine variants.

Numeric effect, agreement, and calibration thresholds must be ratified before the first qualifying run; they must not be selected after seeing outputs.

### Closure

This design record closes as one of:

- `ACCEPTED / NOT_RUN` — implementation is authorized but no run is implied;
- `REJECTED / NOT_RUN` — proposal is declined with reasons;
- `REDIRECTED / NOT_RUN` — ownership or architecture moved elsewhere.

Issue #170 must not close merely because this proposal exists. It may close only under the operator's chosen scope: either after ratified runtime/corpus acceptance, or after an explicit split creates tracked implementation and authorized-run issues with this record linked. Experiment results require a separate dated run record; they are never written retroactively into this proposal as if predicted.

## Explicit ratification questions

1. Is `1.0.0-proposal` the correct schema/package baseline, and what accepted-state label replaces `UNMINTED` after ratification?
2. Is `N = 5`, paired across variants with the proposed RFC 8785 JSON-array/SHA-256 derivation and one same-seed infrastructure retry, accepted for the MVP?
3. Should any unresolved infrastructure error force the whole cell to `infrastructure_error`, with partial semantic counts retained but no semantic cell label?
4. Is the normalized event contract sufficient, including fixture-declared `inline_work` and sequence-order semantics?
5. Are workflow cases valid only when the fixture pre-registers a required workflow/template, rather than inferring workflow need from complexity or call count?
6. Should Adaptive Thinking paths remain execution-profile strata under the three-dimensional primary cell, or become an explicit fourth cell dimension?
7. Are content custody and canonicalization rules acceptable: approved/synthetic bodies only, immutable references plus SHA-256, RFC 8785 for structured objects, and no Unicode normalization?
8. What human inter-coder agreement, per-code judge precision/recall or calibration, and effect thresholds must be frozen before a qualifying run?
9. Is `CLM-v0.1-8b` the initial named judge candidate, with structured shadow-only output and no sole-gate authority?
10. Is the proposed MVP correctly CLI-first and limited to Phases 1–3, with the 11-code production corpus and judge deferred to Phase 4 but required before doctrine qualification?
11. Who has authority to approve fixture privacy, paid/provider budgets, model downloads, and any public-thinking capture?
12. Does issue #170 remain open through runtime/corpus acceptance, or should ratification split implementation and run execution into separately tracked issues?

## Package contents

- [`schemas/battery-package-v1.schema.json`](schemas/battery-package-v1.schema.json) — proposal input/package contract.
- [`schemas/run-receipt-v1.schema.json`](schemas/run-receipt-v1.schema.json) — proposal per-run reproducibility contract.
- [`examples/synthetic-package-v1.json`](examples/synthetic-package-v1.json) — synthetic input-shape example.
- [`examples/synthetic-run-receipt-v1.json`](examples/synthetic-run-receipt-v1.json) — synthetic NOT_RUN receipt-shape example.
