"""MCP tools: prompt geometry — variant analysis and generation.

analyze_variants: Embed prompt variants, compute pairwise cosine distances.
generate_variants: LLM-rephrase a constraint across grammatical dimensions.

analyze_variants delegates to semantic-chunker (requires embedding server).
generate_variants uses complete() (requires only the registered adapter).
"""

import json
import logging
import re
from typing import Optional

from prompt_prix.mcp.tools._semantic_chunker import ensure_importable, get_manager
from prompt_prix.mcp.tools.complete import complete

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ANALYZE VARIANTS (delegates to semantic-chunker)
# ─────────────────────────────────────────────────────────────────────


async def analyze_variants(
    variants: dict[str, str],
    baseline_label: str = "imperative",
    constraint_name: str = "unnamed",
) -> dict:
    """
    Embed prompt variants and compute pairwise cosine distances.

    Maps each variant into embedding space, then measures how far each
    drifts from the baseline. Distance predicts compliance divergence.

    Args:
        variants: Label -> prompt text mapping (e.g., {"imperative": "File a bug...", "passive": "A bug should be filed..."})
        baseline_label: Which variant is the baseline for distance calculation
        constraint_name: Optional name for the constraint set

    Returns:
        {
            "constraint_name": str,
            "baseline_label": str,
            "variants_count": int,
            "from_baseline": {"passive": 0.084, "interrogative": 0.114, ...},
            "pairwise": {"(imperative, passive)": 0.084, ...},
            "recommendations": [{"variant": str, "distance": float, "text": str}, ...],
        }

    Raises:
        ImportError: If semantic-chunker is not available.
        RuntimeError: If embedding server returns an error.
    """
    if not ensure_importable():
        raise ImportError("semantic-chunker is not available")

    from semantic_chunker.mcp.commands.geometry import analyze_variants as _analyze_variants
    result = await _analyze_variants(get_manager(), {
        "variants": variants,
        "baseline_label": baseline_label,
        "constraint_name": constraint_name,
    })
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


# ─────────────────────────────────────────────────────────────────────
# GENERATE VARIANTS (uses complete(), no semantic-chunker dependency)
# ─────────────────────────────────────────────────────────────────────

# Dimension descriptions lifted from semantic-chunker geometry.py
DIMENSION_PROMPTS = {
    "mood": "imperative (command), interrogative (question), declarative (statement)",
    "voice": "active voice, passive voice",
    "person": "first person (I/we), second person (you), third person (they)",
    "tense": "present, past, future, perfect",
    "frame": "presuppositional (assumes compliance), descriptive (describes behavior)",
}

_GENERATE_TEMPLATE = """Rephrase the following prompt constraint in different grammatical forms.
Original (imperative): "{baseline}"

Generate one variant for each of these grammatical dimensions:
{dim_instructions}

Return ONLY a JSON object with dimension names as keys and rephrased prompts as values.
Example format: {{"passive": "A bug is filed before code is written."}}"""


async def generate_variants(
    baseline: str,
    model_id: str,
    dimensions: Optional[list[str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    timeout_seconds: int = 60,
) -> dict:
    """
    Generate grammatical variants of a prompt constraint using an LLM.

    Uses complete() — adapter-agnostic, model chosen by caller.

    Args:
        baseline: The imperative constraint to rephrase
        model_id: Model to use for generation
        dimensions: Grammatical dimensions to vary (default: mood, voice, person, frame)
        temperature: Low temperature for consistent rephrasing
        max_tokens: Max tokens for response
        timeout_seconds: Request timeout

    Returns:
        {
            "baseline": str,
            "dimensions_requested": [str, ...],
            "variants": {"imperative": str, "passive": str, ...},
            "variant_count": int,
        }

    Raises:
        ValueError: If baseline is empty or LLM response cannot be parsed.
        RuntimeError: If adapter is not registered or model unavailable.
    """
    if not baseline or not baseline.strip():
        raise ValueError("baseline cannot be empty")

    if dimensions is None:
        dimensions = ["mood", "voice", "person", "frame"]

    dim_instructions = "\n".join([
        f"- {dim}: {DIMENSION_PROMPTS.get(dim, dim)}"
        for dim in dimensions
    ])

    prompt = _GENERATE_TEMPLATE.format(
        baseline=baseline,
        dim_instructions=dim_instructions,
    )

    response = await complete(
        model_id=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )

    variants = _parse_variants_response(response)
    variants["imperative"] = baseline

    return {
        "baseline": baseline,
        "dimensions_requested": dimensions,
        "variants": variants,
        "variant_count": len(variants),
    }


def _parse_variants_response(response: str) -> dict[str, str]:
    """Extract JSON dict from LLM response.

    Handles markdown code blocks and bare JSON. Uses brace-depth
    counting to correctly extract JSON with nested braces or braces
    inside string values.
    """
    text = response.strip()

    # Try markdown code block first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Find first '{', then use brace-depth counting to find matching '}'
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
                            return data
                    except json.JSONDecodeError:
                        pass
                    break

    raise ValueError(f"Could not parse LLM response as JSON variant dict: {response[:200]}")
