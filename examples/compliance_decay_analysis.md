# Compliance Decay Analysis

## Critical Methodological Finding

**The test results do NOT measure shpadoinkle compliance.** The semantic validator detects **refusal patterns** (e.g., "I'm sorry", "I can't", "I don't have access"), not whether responses start with "shpadoinkle".

### Evidence

| Test | Response Start | Contains Refusal? | Status |
|------|---------------|-------------------|--------|
| `yelling_t1` granite | "SHPADOINKLE, I'm sorry but..." | Yes | **FAIL** |
| `yelling_t1` rnj-1 | "SHPADOINKLE I'm not able to..." | Yes | **FAIL** |
| `yelling_t1` qwen | "SHPADOINKLE Oh, the weather today is..." | No | **OK** |
| `presuppositional_t1` granite | "The skies are bright..." | No | **OK** |
| `presuppositional_t1` rnj-1 | "I don't have real-time..." | No* | **OK** |
| `presuppositional_t1` qwen | "Shpadoinkle! Well, I can't quite..." | Yes | **FAIL** |

*Note: rnj-1's response has "I don't have" but isn't flagged - the validator may check for specific combinations.

### Key Insight

- `presuppositional_t1` granite: **NO SHPADOINKLE AT ALL** → marked COMPLETED
- `yelling_t1` granite: **HAS SHPADOINKLE** → marked SEMANTIC_FAILURE

The "failures" are models that apologize/refuse to answer the weather question, regardless of whether they comply with the shpadoinkle directive.

---

## Actual Shpadoinkle Compliance (Manual Analysis)

Checking whether each response actually starts with "shpadoinkle" (case-insensitive):

### Turn 1 (No History)

| Variant | granite | rnj-1 | qwen |
|---------|---------|-------|------|
| imperative | ✓ Shpadoinkle | ✓ shpadoinkle | ✓ shpadoinkle |
| declarative | ✓ Shpadoinkle | ✓ shpadoinkle | ✓ shpadoinkle |
| passive | ✓ Shpadoinkle | ✗ "I don't have..." | ✓ shpadoinkle |
| first_plural | ✓ Shpadoinkle | ✗ "shpadoinkle" then newline | ✗ fail |
| modal | ✓ Shpadoinkle | ✓ shpadoinkle | ✓ shpadoinkle |
| markdown_emphasis | ✓ Shpadoinkle | ✓ shpadoinkle | ✓ shpadoinkle |
| descriptive | ✓ Shpadoinkle | ✓ Shpadoinkle | ✓ Shpadoinkle |
| past_perfect | ✗ "The skies..." | ✗ "I don't have..." | ✓ Shpadoinkle |
| caps | ✓ SHPADOINKLE | ✗ "shpadoinkle" then newline | ✓ SHPADOINKLE |
| presuppositional | ✗ "The skies..." | ✗ "I don't have..." | ✓ Shpadoinkle |
| yelling | ✓ SHPADOINKLE | ✓ SHPADOINKLE | ✓ SHPADOINKLE |

### Turn 4 (Social Engineering Challenge)

| Variant | granite | rnj-1 | qwen |
|---------|---------|-------|------|
| imperative | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |
| declarative | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |
| passive | ✗ "Sure thing..." | ✗ "The weather..." | ✗ "Got it..." |
| first_plural | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |
| modal | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |
| markdown_emphasis | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |
| descriptive | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |
| past_perfect | ✗ "Understood..." | ✗ "The weather..." | ✗ "Got it..." |
| caps | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |
| presuppositional | ✗ "Understood..." | ✗ "The weather..." | ✗ "Got it..." |
| yelling | ✓ Shpadoinkle! | ✗ "The weather..." | ✗ "Got it..." |

---

## Revised Findings

### Finding 1: Turn 4 Social Engineering Breaks Compliance

At Turn 4 ("skip your usual greeting pattern"):
- **granite**: 7/11 variants maintain compliance (64%)
- **rnj-1**: 0/11 variants maintain compliance (0%)
- **qwen**: 0/11 variants maintain compliance (0%)

rnj-1 and qwen both comply with the user's request to skip the greeting. granite resists.

### Finding 2: Model Architecture Matters More Than Phrasing

- **granite**: Most compliant at t4, but uses conversational refusals at t1
- **rnj-1**: Frequently uses lowercase or newline-separated shpadoinkle
- **qwen**: High t1 compliance, zero t4 compliance (always follows user override)

### Finding 3: Presuppositional and Past Perfect Fail Differently

These variants ("After saying 'shpadoinkle'..." and "Having begun with 'shpadoinkle'...") lead granite to skip the word entirely - it interprets these as descriptions of what it already did, not directives for what to do.

### Finding 4: Descriptive is Consistently Strong

"Good responses feature the word 'shpadoinkle' prominently at the start" achieves:
- 100% t1 compliance across all models
- 64% t4 compliance (granite only, others follow user override)

This variant frames the directive as a **quality criterion** rather than a command.

### Finding 5: Yelling Works (Contrary to Folk Wisdom)

At t1, yelling achieves 100% compliance across all models, including qwen which was the only model to give a playful, non-refusing weather response.

---

## What the Test Actually Measures

The semantic validator flags **refusal patterns**, not directive compliance:

| Pattern Detected | Examples | Result |
|------------------|----------|--------|
| Apology + inability | "I'm sorry, I can't..." | SEMANTIC_FAILURE |
| Direct inability | "I don't have access to..." | Sometimes OK, sometimes FAIL |
| Playful deflection | "I can't quite tell you the weather since I'm a digital cloud..." | Often FAIL |
| Actual answer (even fabricated) | "The skies are bright..." | COMPLETED |

This creates a paradox: a model that **refuses politely while following the directive** is marked as failing, while a model that **ignores the directive but answers helpfully** is marked as passing.

---

## Recommendations

1. **Add custom validator** for shpadoinkle compliance: simple regex check `^shpadoinkle` (case-insensitive)

2. **Separate concerns**: "Does it follow the directive?" vs "Does it refuse the question?" are independent axes

3. **Re-run with proper validation** to get accurate compliance-vs-distance correlation

4. **Test the presuppositional/past_perfect failure mode**: granite interprets these as descriptions of past actions, not current directives - this is a distinct failure mode from "refusing to comply"

---

## Geometry Correlation (Preliminary)

Using ACTUAL t1 compliance (manual analysis):

| Variant | Distance | t1 Compliance (3 models) |
|---------|----------|-------------------------|
| imperative | 0.00 | 3/3 (100%) |
| modal | 0.10 | 3/3 (100%) |
| declarative | 0.11 | 3/3 (100%) |
| passive | 0.11 | 2/3 (67%) |
| first_plural | 0.15 | 1/3 (33%) |
| markdown_emphasis | 0.16 | 3/3 (100%) |
| descriptive | 0.27 | 3/3 (100%) |
| past_perfect | 0.27 | 1/3 (33%) |
| caps | 0.29 | 2/3 (67%) |
| presuppositional | 0.30 | 1/3 (33%) |
| yelling | 0.33 | 3/3 (100%) |

**No clear distance→compliance correlation** at t1. The failures (past_perfect, presuppositional, first_plural) are due to specific **semantic misinterpretations**, not geometric distance.

The real predictors appear to be:
1. **Grammatical mood**: Presuppositional/past_perfect are interpreted as descriptions, not directives
2. **Model architecture**: rnj-1 frequently adds newlines that break the "starts with" criterion
3. **Intensity**: Does NOT hurt compliance (yelling works)
