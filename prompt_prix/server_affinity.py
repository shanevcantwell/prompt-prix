"""
Server affinity prefix parsing utilities.

Server affinity uses a prefix format: "idx:model_name" where idx is a
0-based server index. This routes requests to specific servers instead
of load balancing.

Examples:
    "0:llama-3" → server 0, model "llama-3"
    "1:mistral" → server 1, model "mistral"
    "model-only" → no affinity, any server
"""

from typing import Optional, Tuple


def parse_server_prefix(model_id: str) -> Tuple[Optional[int], str]:
    """
    Parse server affinity prefix from model ID.

    Args:
        model_id: Model ID, possibly prefixed with "idx:"

    Returns:
        (server_index, actual_model_id) tuple.
        server_index is None if no valid prefix found.

    Examples:
        >>> parse_server_prefix("0:model")
        (0, "model")
        >>> parse_server_prefix("model")
        (None, "model")
        >>> parse_server_prefix("10:model-v2")
        (10, "model-v2")
        >>> parse_server_prefix(":model")
        (None, ":model")
        >>> parse_server_prefix("0:")
        (0, "")
    """
    if ":" not in model_id:
        return None, model_id

    parts = model_id.split(":", 1)
    prefix = parts[0]

    # Prefix must be a digit string
    if not prefix.isdigit():
        return None, model_id

    return int(prefix), parts[1]


def strip_server_prefix(model_id: str) -> str:
    """
    Strip server affinity prefix, returning just the model name.

    Args:
        model_id: Model ID, possibly prefixed with "idx:"

    Returns:
        Model ID without the server prefix.

    Examples:
        >>> strip_server_prefix("0:model")
        "model"
        >>> strip_server_prefix("model")
        "model"
        >>> strip_server_prefix("10:model-v2")
        "model-v2"
    """
    _, actual_model_id = parse_server_prefix(model_id)
    return actual_model_id


def extract_server_indices(model_ids: list[str]) -> set[int]:
    """
    Extract unique server indices from a list of model IDs.

    Args:
        model_ids: List of model IDs, possibly with prefixes

    Returns:
        Set of server indices found. Empty if no prefixes.

    Examples:
        >>> extract_server_indices(["0:a", "0:b", "1:c"])
        {0, 1}
        >>> extract_server_indices(["model1", "model2"])
        set()
    """
    indices = set()
    for model_id in model_ids:
        server_idx, _ = parse_server_prefix(model_id)
        if server_idx is not None:
            indices.add(server_idx)
    return indices
