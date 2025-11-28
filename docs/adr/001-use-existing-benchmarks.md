# ADR-001: Use Existing Benchmarks Instead of Custom Eval Schema

**Status**: Accepted
**Date**: 2025-11-28

## Context

While building prompt-prix, we created `tool_competence_tests.json` as a custom evaluation schema for testing LLM tool-calling competence. The schema included:

- 15 test cases across 8 categories
- Custom `match_mode` values (exact, contains, exclusion, structure, etc.)
- Bespoke `expected` object structures

This raised the question: should prompt-prix define its own eval format, or adopt existing benchmark standards?

## Decision

**Adopt existing benchmark formats (BFCL, Inspect AI) rather than creating a bespoke schema.**

prompt-prix will:
1. Accept prompts in standard formats used by BFCL and Inspect AI
2. Not define or maintain its own evaluation schema specification
3. Treat any custom JSON (like `tool_competence_tests.json`) as sample input, not a specification

## Rationale

### Interoperability over Custom
- BFCL (Berkeley Function Calling Leaderboard) already has community adoption
- Inspect AI (UK AISI) has MIT license and `pip install inspect-ai`
- Using standard formats means users can reuse their existing test suites

### Inferred Schemas Fragment the Ecosystem
- AI assistants can trivially infer schemas from examples
- This leads to proliferation of "close enough" formats that don't quite interoperate
- A principled stance against custom schemas reduces fragmentation

### Focus on Value-Add
- prompt-prix's value is **visual comparison via fan-out**, not eval authoring
- Defining a schema puts us in competition with established frameworks
- Better to be a visual layer on top of those frameworks

## Consequences

### Positive
- Users with existing BFCL/Inspect AI test suites can use them directly
- No schema maintenance burden
- Clear positioning: "visual comparison tool, not eval framework"

### Negative
- No "prompt-prix format" for users to rally around
- May need adapters for different input formats
- `tool_competence_tests.json` becomes illustrative sample, not specification

### Migration
- Move `tool_competence_tests.json` to `examples/` directory
- Update documentation to reference BFCL/Inspect AI as upstream sources
- Consider building format detection/conversion utilities if demand arises

## References

- [BFCL GitHub](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [Inspect AI](https://inspect.ai-safety-institute.org.uk/)
- [NESTful Paper](https://arxiv.org/abs/2409.03797)
