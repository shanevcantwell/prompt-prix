"""
Text parsing utilities for prompt-prix.
"""

from pathlib import Path
from typing import Optional

from prompt_prix.config import DEFAULT_SYSTEM_PROMPT


def parse_models_input(models_text: str) -> list[str]:
    """Parse newline or comma-separated model list."""
    models = []
    for line in models_text.strip().split("\n"):
        for item in line.split(","):
            item = item.strip()
            if item:
                models.append(item)
    return models


def parse_servers_input(servers_text: str) -> list[str]:
    """Parse newline or comma-separated server list."""
    servers = []
    for line in servers_text.strip().split("\n"):
        for item in line.split(","):
            item = item.strip()
            if item:
                servers.append(item)
    return servers


def parse_prompts_file(file_content: str) -> list[str]:
    """Parse uploaded file into list of prompts (newline-separated)."""
    prompts = []
    for line in file_content.strip().split("\n"):
        line = line.strip()
        if line:
            prompts.append(line)
    return prompts


def load_system_prompt(file_path: Optional[str]) -> str:
    """Load system prompt from file or return default."""
    if file_path:
        path = Path(file_path)
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:  # Only use file if it has content
                return content
    # Try default file in package directory
    default_path = Path(__file__).parent / "system_prompt.txt"
    if default_path.exists():
        content = default_path.read_text(encoding="utf-8").strip()
        if content:  # Only use file if it has content
            return content
    return DEFAULT_SYSTEM_PROMPT


def get_default_system_prompt() -> str:
    """
    Get default system prompt for initial UI display.
    Tries system_prompt.txt in package directory first, then falls back to constant.
    """
    default_path = Path(__file__).parent / "system_prompt.txt"
    if default_path.exists():
        content = default_path.read_text(encoding="utf-8").strip()
        if content:  # Only use file if it has content
            return content
    return DEFAULT_SYSTEM_PROMPT
