# ADR-003: OpenAI-Compatible API as Integration Layer

**Status**: Accepted
**Date**: 2025-11-28

## Context

prompt-prix needs to communicate with LLM inference servers. The question: which API protocol(s) should it support?

Options considered:
1. Native SDKs for each provider (OpenAI, Anthropic, Google, etc.)
2. OpenAI-compatible endpoints only
3. Abstraction layer that adapts multiple protocols

## Decision

**Require OpenAI-compatible endpoints only.**

All inference servers must expose:
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion with streaming

## Rationale

### De Facto Standard
OpenAI's API has become the de facto standard for LLM inference:
- **LM Studio**: Native OpenAI-compatible server
- **Ollama**: Supports OpenAI-compatible mode
- **vLLM**: OpenAI-compatible by default
- **llama.cpp server**: OpenAI-compatible endpoints
- **Text Generation Inference**: OpenAI-compatible mode

### Simplicity
- Single HTTP client implementation (`httpx`)
- No SDK dependencies for each provider
- Consistent error handling

### Local-First Focus
prompt-prix is designed for local model evaluation:
- Users run models on their own hardware
- LM Studio is the primary target
- Cloud API access is secondary

### Proxy Pattern for Non-Compatible APIs
Users who need Anthropic/Google/etc. can:
1. Use a local proxy that translates to OpenAI format
2. Use LiteLLM or similar adapters
3. This keeps prompt-prix simple while enabling flexibility

## Consequences

### Positive
- Single, well-documented API to implement
- Works with most local inference servers out of the box
- No provider SDK maintenance burden

### Negative
- No native Anthropic Claude API support
- No native Google Gemini API support
- Users must set up proxies for non-OpenAI-compatible services

### Implementation
- `core.py`: `stream_completion()` uses OpenAI chat completions format
- `ServerPool`: Expects `/v1/models` endpoint for manifest refresh
- Tool/function calling uses OpenAI tools format

## Alternatives Considered

### Native Multi-Provider SDKs
Rejected because:
- SDK maintenance burden
- Different response formats to normalize
- prompt-prix is local-first, not cloud-first

### Abstraction Layer
Rejected because:
- Over-engineering for current scope
- OpenAI-compatible is sufficient for 90%+ of use cases
- Can add abstraction later if needed

## References

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [LM Studio Server Docs](https://lmstudio.ai/docs/local-server)
- [vLLM OpenAI Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
