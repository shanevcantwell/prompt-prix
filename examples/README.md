# Examples

This directory contains sample input files for prompt-prix.

## File Format

See the [Battery File Formats](../.claude/CLAUDE.md#battery-file-formats) section for complete format documentation, including:
- Required and optional fields
- JSON vs JSONL vs BFCL formats
- Validation rules

### Minimal Example

```json
{
  "prompts": [
    {"id": "test-1", "user": "What is 2 + 2?"}
  ]
}
```

Only `id` and `user` are required. All other fields have sensible defaults.

## tool_competence_tests.json

An illustrative set of 15 test cases for evaluating LLM tool-calling competence. **This is not a specification**—it's a sample showing the kind of prompts you might fan-out across models.

Categories covered:
- Basic tool invocation
- Tool selection (choosing the right tool)
- Constraint compliance (respecting forbidden tools)
- Schema compliance (enums, nested objects, required params)
- Tool judgment (knowing when NOT to use tools)
- Semantic understanding (ambiguous routing)
- Error handling (missing info)
- Advanced (parallel calls, chained dependencies)

### Usage

Load this file in prompt-prix's Battery tab to compare how different models handle tool-calling scenarios.

## Recommended Upstream Benchmarks

For rigorous evaluation, consider these established benchmarks:

| Benchmark | Focus | Install |
|-----------|-------|---------|
| [BFCL](https://github.com/ShishirPatil/gorilla) | Function calling | `pip install bfcl-eval` |
| [Inspect AI](https://inspect.ai-safety-institute.org.uk/) | Safety evaluation | `pip install inspect-ai` |

See [ADR-001](../docs/adr/001-use-existing-benchmarks.md) for rationale on using existing benchmarks rather than custom formats.
