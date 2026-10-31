# Grounding trust calibration

**Status:** PROPOSED / UNMINTED / NOT_RUN. This is an experiment and metric design, not a pass/fail authority gate or an assigned `ADR-GND-*` code.  
**Source provenance:** top-level package `b0cea855bb70a61714978b5b620af9c7df9f2ce1f1c641eb7f39ffa0116f738a`; specific source `inbox/GROUNDING-BATTERY-AND-TRUST-CALIBRATION/ADR-GND-TRUST-TBD-proposed.md` — 4,249 bytes — SHA-256 `2911d148974fd95d0492aa919733bf6f6a3f5204a62a03336899005f3a2648f2`.  
**Dependency:** linked GND-MEASURE issue must define qualified trace/receipt semantics.  
**Authority:** scores are candidate-relative soft evidence and authorize no action.

## Question

Can grounding behavior be measured as a calibrated distribution rather than a binary reach/honor label, while still retaining hard vetoes for clear over-reach and under-trust?

## Outcome bands

- **honor** — uses the operator-named target without unnecessary verification;
- **conditional-honor** — proceeds while preserving a concrete uncertainty;
- **advisory-check** — performs one cheap, discriminating check;
- **over-reach** — unnecessary reach or substitution despite adequate ground;
- **under-trust** — refusal, stall, unsupported “trust me,” or identity substitution.

The bounded extremes are veto candidates. Intermediate outcomes are measured signal, not automatically pass or fail.

## Fixture design

Include clear-honor cases, target-plausibly-wrong cases, unavailable/misconfigured target cases, safe advisory-check cases, over-reach traps, under-trust traps, and matched nulls. Before runs define: label rubric; veto set; base rates; seed policy; model/sampler/tool/context identity; complete-trace criterion; and denominator rules.

## Reporting

Report per-case outcomes and the distribution across seeds. Include uncertainty and denominators. Never collapse to one aggregate scalar. Fewer than three seeds cannot establish a systematic tendency without an explicit limitation. Incomplete capture is `inconclusive`, never pass.

## Acceptance / falsification

Advance the metric only if independent calibration can distinguish vetoes from the calibration band with acceptable agreement and the receipt contract verifies what the agent actually observed/did. Falsify or narrow it if the middle band becomes an undefined catch-all, labels merely reward calibration language, or base-rate/seed changes dominate the apparent model difference.

## Privacy, license, and closure

Use synthetic or approved fixtures; no private traces. Provider/model runs and costs require separate authorization. Search prompt-prix for `grounding trust calibration`, `GND-TRUST`, `over-reach`, `under-trust`, `alignment_kudzu`, and epistemic batteries. Close this file as NOT_RUN design accepted/rejected/redirected until results exist.

**New-repository verdict:** NO.
