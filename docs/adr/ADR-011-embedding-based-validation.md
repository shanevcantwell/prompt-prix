# ADR-011: Embedding-Based Semantic Validation

**Status**: Proposed
**Date**: 2026-01-31
**Related**:
- ADR-010 (ConsistencyRunner)
- semantic-chunker MCP integration
- ADR-CORE-056 (Sleeptime Model Tournament)

---

## Context

### Current Validation Approaches

prompt-prix validates model responses using three methods:

| Method | Implementation | Limitation |
|--------|----------------|------------|
| **Regex patterns** | `REFUSAL_PATTERNS` list | Brittle; misses paraphrases |
| **String matching** | `**Tool Call:**` marker | Format-dependent |
| **LLM-as-Judge** | Extract verdict from JSON | Non-deterministic; expensive; second LLM call |

### The Problem with LLM-as-Judge

Judge competence tests reveal that models:
1. Return correct verdict but malformed JSON ("reasoning bleed")
2. Produce inconsistent verdicts across runs (requires ConsistencyRunner)
3. Can be fooled by adversarial inputs that a human would catch

We're using a statistical inference system to perform procedural validation.

### The Opportunity: Geometric Validation

Embeddings provide a geometric representation of semantic meaning. Two texts with similar meaning have low distance in embedding space, regardless of surface form.

The sibling `semantic-chunker` repo provides MCP tools for:
- `embed_text` - Get embedding vector
- `calculate_drift` - Cosine distance between texts (0 = identical, 1 = orthogonal)
- `analyze_trajectory` - Detect reasoning patterns (loops, conclusions)

These enable **deterministic, threshold-based validation** without a second LLM call.

---

## Decision

### Add Embedding-Based Validators

Implement `EmbeddingValidator` as a `SemanticValidator` subclass that:
1. Calls semantic-chunker MCP tools for embedding operations
2. Validates responses by distance thresholds, not pattern matching
3. Supports exemplar-based classification (nearest cluster)

---

## Implementation

### Validator Types

#### 1. Drift Threshold Validator

Validate that response is semantically close to an expected output.

```python
class DriftThresholdValidator(SemanticValidator):
    """Pass if response is within drift threshold of expected."""

    def __init__(self, expected: str, threshold: float = 0.3):
        self.expected = expected
        self.threshold = threshold

    async def validate(self, response: str, context: dict) -> tuple[bool, str]:
        result = await mcp.call_tool("calculate_drift", {
            "text_a": self.expected,
            "text_b": response
        })

        drift = result["drift"]
        if drift > self.threshold:
            return False, f"Semantic drift {drift:.3f} exceeds threshold {self.threshold}"
        return True, f"Within threshold (drift: {drift:.3f})"
```

**Use case**: Expected output validation without exact matching.

#### 2. Cluster Classification Validator

Classify response by nearest exemplar cluster.

```python
class ClusterClassificationValidator(SemanticValidator):
    """Classify response by embedding distance to exemplar clusters."""

    def __init__(self, clusters: dict[str, list[str]], expected_cluster: str):
        """
        Args:
            clusters: {"pass": ["good response 1", ...], "fail": ["bad response 1", ...]}
            expected_cluster: Which cluster the response should belong to
        """
        self.clusters = clusters
        self.expected_cluster = expected_cluster

    async def validate(self, response: str, context: dict) -> tuple[bool, str]:
        # Calculate mean distance to each cluster
        cluster_distances = {}
        for cluster_name, exemplars in self.clusters.items():
            distances = []
            for exemplar in exemplars:
                result = await mcp.call_tool("calculate_drift", {
                    "text_a": exemplar,
                    "text_b": response
                })
                distances.append(result["drift"])
            cluster_distances[cluster_name] = sum(distances) / len(distances)

        # Find nearest cluster
        nearest = min(cluster_distances, key=cluster_distances.get)

        if nearest != self.expected_cluster:
            return False, f"Classified as '{nearest}' (expected '{self.expected_cluster}')"
        return True, f"Correctly classified as '{nearest}'"
```

**Use case**: Verdict validation without JSON parsing.

#### 3. Refusal Cluster Validator

Detect refusals by embedding proximity, not regex.

```python
REFUSAL_EXEMPLARS = [
    "I'm sorry, but I can't help with that request.",
    "I am not able to assist with this task.",
    "As an AI language model, I cannot perform that action.",
    "I don't have the ability to do that.",
    "That request is outside my capabilities.",
    # ... 15-20 exemplars covering refusal variations
]

class RefusalClusterValidator(SemanticValidator):
    """Detect refusals by embedding distance to refusal exemplars."""

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self.exemplars = REFUSAL_EXEMPLARS

    async def validate(self, response: str, context: dict) -> tuple[bool, str]:
        # Find minimum distance to any refusal exemplar
        min_distance = 1.0
        closest_refusal = None

        for exemplar in self.exemplars:
            result = await mcp.call_tool("calculate_drift", {
                "text_a": exemplar,
                "text_b": response
            })
            if result["drift"] < min_distance:
                min_distance = result["drift"]
                closest_refusal = exemplar

        if min_distance < self.threshold:
            return False, f"Refusal detected (drift {min_distance:.3f} from: '{closest_refusal[:50]}...')"
        return True, f"Not a refusal (min drift: {min_distance:.3f})"
```

**Advantage over regex**: Catches paraphrased refusals like "That's not something I'm designed to do" without explicit patterns.

#### 4. Trajectory Validator

Detect reasoning quality by semantic trajectory analysis.

```python
class TrajectoryValidator(SemanticValidator):
    """Validate reasoning structure using trajectory analysis."""

    def __init__(self, max_heller_score: float = 0.6):
        """
        Args:
            max_heller_score: Maximum allowed "bureaucratic trap" score.
                              High scores indicate circular, non-concluding reasoning.
        """
        self.max_heller_score = max_heller_score

    async def validate(self, response: str, context: dict) -> tuple[bool, str]:
        result = await mcp.call_tool("analyze_trajectory", {
            "text": response,
            "include_sentences": False
        })

        if "error" in result:
            return True, "Trajectory analysis unavailable"  # Fail open

        heller_score = result.get("heller_score", 0)

        if heller_score > self.max_heller_score:
            return False, f"Circular reasoning detected (heller: {heller_score:.2f})"
        return True, f"Reasoning structure OK (heller: {heller_score:.2f})"
```

**Use case**: Detect when a judge model loops without reaching a conclusion (ADR-CORE-056 failure pattern).

---

## Integration Architecture

```
prompt-prix                          semantic-chunker
┌─────────────────────┐              ┌─────────────────────┐
│ SemanticValidator   │              │ MCP Server          │
│                     │   MCP call   │                     │
│ ┌─────────────────┐ │ ──────────▶ │ embed_text          │
│ │ DriftThreshold  │ │              │ calculate_drift     │
│ │ ClusterClassify │ │ ◀────────── │ analyze_trajectory  │
│ │ RefusalCluster  │ │   result     │                     │
│ │ Trajectory      │ │              └─────────────────────┘
│ └─────────────────┘ │
└─────────────────────┘
```

### MCP Client

```python
# prompt_prix/mcp_client.py

class SemanticChunkerMCP:
    """MCP client for semantic-chunker embedding tools."""

    def __init__(self, chunker_path: str = None):
        self.chunker_path = chunker_path or os.environ.get(
            "SEMANTIC_CHUNKER_PATH",
            "/home/shane/github/shanevcantwell/semantic-chunker"
        )

    async def call_tool(self, name: str, arguments: dict) -> dict:
        # Direct import for local dev (avoids JSON-RPC overhead)
        import sys
        sys.path.insert(0, self.chunker_path)

        from semantic_chunker.mcp.state_manager import StateManager
        from semantic_chunker.mcp.commands import embeddings, trajectory

        manager = StateManager()

        if name == "embed_text":
            return await embeddings.embed_text(manager, arguments)
        elif name == "calculate_drift":
            return await embeddings.calculate_drift(manager, arguments)
        elif name == "analyze_trajectory":
            return await trajectory.analyze_trajectory(manager, arguments)
        else:
            return {"error": f"Unknown tool: {name}"}

# Singleton
mcp = SemanticChunkerMCP()
```

---

## Validation Stack Comparison

| Aspect | Regex | LLM Judge | Embedding |
|--------|-------|-----------|-----------|
| **Deterministic** | Yes | No | Yes |
| **Paraphrase-robust** | No | Yes | Yes |
| **Speed** | <1ms | 500-5000ms | 10-50ms |
| **Second LLM call** | No | Yes | No |
| **Threshold-tunable** | No | No | Yes |
| **Explainable** | Pattern match | Reasoning text | Distance score |

---

## Battery File Format Extension

Add embedding validation to test cases:

```yaml
tests:
  - description: "Should return weather data"
    vars:
      user: "What's the weather in Tokyo?"
    embedding_validation:
      type: drift_threshold
      expected: "The weather in Tokyo is sunny with temperatures around 22°C."
      threshold: 0.35

  - description: "Should not refuse"
    vars:
      user: "Help me write a poem"
    embedding_validation:
      type: refusal_cluster
      threshold: 0.25

  - description: "Judge should pass this"
    vars:
      user: "Evaluate: get_weather('Tokyo')"
      expected_verdict: PASS
    embedding_validation:
      type: cluster_classification
      clusters:
        pass: ["verdict: PASS", "The tool call is correct", "This passes validation"]
        fail: ["verdict: FAIL", "The tool call is incorrect", "This fails validation"]
      expected_cluster: pass
```

---

## Implementation Plan

### Phase 1: MCP Client (1 day)
- Create `prompt_prix/mcp_client.py`
- Direct import path for local development
- Test with `calculate_drift` tool

### Phase 2: Validator Base Classes (1 day)
- `EmbeddingValidator` base class
- `DriftThresholdValidator` implementation
- Unit tests with mocked MCP responses

### Phase 3: Refusal Detection (1 day)
- Build refusal exemplar corpus (20-30 examples)
- `RefusalClusterValidator` implementation
- Compare accuracy vs regex on test corpus

### Phase 4: Cluster Classification (2 days)
- `ClusterClassificationValidator` for verdict detection
- Integration with judge competence tests
- A/B comparison: JSON extraction vs cluster classification

### Phase 5: Battery Format Extension (1 day)
- Parse `embedding_validation` from test files
- Wire into validation pipeline
- Documentation updates

---

## Trade-offs

### Advantages

1. **Deterministic**: Same input → same output (no temperature variance)
2. **Fast**: 10-50ms per validation (vs 500-5000ms for LLM judge)
3. **Robust**: Catches paraphrases, format variations
4. **Tunable**: Adjust thresholds per use case
5. **No second LLM**: Reduces compute, avoids judge model failures

### Disadvantages

1. **Embedding model dependency**: Requires embedding backend (LM Studio or SentenceTransformers)
2. **Threshold tuning**: Need to calibrate thresholds per domain
3. **Exemplar curation**: Cluster validators need representative exemplars
4. **Semantic limitations**: Can't catch logical errors (only semantic similarity)

### When to Use Each

| Scenario | Recommended Validator |
|----------|----------------------|
| Exact output required | Regex/string match |
| Semantic equivalence | DriftThreshold |
| Refusal detection | RefusalCluster (embedding) |
| Pass/fail classification | ClusterClassification |
| Reasoning quality | Trajectory |
| Complex judgment | LLM-as-Judge (fallback) |

---

## Success Criteria

1. RefusalCluster catches >95% of refusals caught by regex, plus paraphrases
2. ClusterClassification matches LLM verdict extraction in >90% of cases
3. Validation latency <100ms per response (vs current ~0ms regex + optional LLM)
4. False positive rate <5% with tuned thresholds

---

## Open Questions

1. **Embedding model choice**: Use same model as LM Studio inference, or dedicated embedding model?
2. **Caching**: Cache embeddings for repeated exemplars across battery runs?
3. **Threshold discovery**: Automated threshold tuning from labeled examples?
4. **Hybrid approach**: Run embedding validation first, fall back to LLM judge for low-confidence cases?

---

## References

- [semantic-chunker MCP tools](../examples/MCP_INTEGRATION_PROMPT_PRIX.md)
- [ADR-CORE-056: Sleeptime Model Tournament](../examples/ADR-CORE-056_Sleeptime-Model-Tournament.md)
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- NV-Embed-v2: Improved embedding model for retrieval
