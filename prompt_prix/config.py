"""
Configuration constants and Pydantic models for prompt-prix.
"""

import os
from pydantic import BaseModel
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────────────
# DEFAULTS - User-configurable via Gradio UI
# ─────────────────────────────────────────────────────────────────────

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TIMEOUT_SECONDS: int = 300  # 5 minutes
DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."

DEFAULT_MODELS: list[str] = [
    # Populate with model identifiers as they appear in LM Studio
    # Examples:
    # "llama-3.2-3b-instruct",
    # "qwen2.5-7b-instruct-q4_k_m",
    # "mistral-7b-instruct-v0.3",
]


# ─────────────────────────────────────────────────────────────────────
# INTERNAL CONSTANTS - Not exposed to UI
# ─────────────────────────────────────────────────────────────────────

MANIFEST_REFRESH_INTERVAL_SECONDS: int = 30
REPORT_DIVIDER: str = "\n\n" + "=" * 80 + "\n\n"
CONVERSATION_SEPARATOR: str = "\n\n---\n\n"


# ─────────────────────────────────────────────────────────────────────
# RETRY CONFIGURATION - For model loading failures
# ─────────────────────────────────────────────────────────────────────

def get_retry_attempts() -> int:
    """
    Get max retry attempts from environment or default.

    Set BATTERY_RETRY_ATTEMPTS in .env (default: 3).
    """
    try:
        return int(os.environ.get("BATTERY_RETRY_ATTEMPTS", "3"))
    except ValueError:
        return 3


def get_retry_min_wait() -> int:
    """
    Get minimum wait between retries in seconds.

    Set BATTERY_RETRY_MIN_WAIT in .env (default: 4).
    """
    try:
        return int(os.environ.get("BATTERY_RETRY_MIN_WAIT", "4"))
    except ValueError:
        return 4


def get_retry_max_wait() -> int:
    """
    Get maximum wait between retries in seconds.

    Set BATTERY_RETRY_MAX_WAIT in .env (default: 30).
    """
    try:
        return int(os.environ.get("BATTERY_RETRY_MAX_WAIT", "30"))
    except ValueError:
        return 30


# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT LOADING
# ─────────────────────────────────────────────────────────────────────

def load_servers_from_env() -> list[str]:
    """
    Load LM Studio server URLs from environment variables.

    Looks for variables matching pattern: LM_STUDIO_SERVER_1, LM_STUDIO_SERVER_2, etc.
    Returns list of non-empty server URLs.
    """
    servers = []
    i = 1
    while True:
        key = f"LM_STUDIO_SERVER_{i}"
        value = os.environ.get(key)
        if value is None:
            # No more servers defined
            break
        if value.strip():
            servers.append(value.strip())
        i += 1
    return servers


def get_default_servers() -> list[str]:
    """
    Get default server list from environment or fallback.

    Returns servers from environment variables if set,
    otherwise returns placeholder defaults.
    """
    servers = load_servers_from_env()
    if servers:
        return servers
    # Fallback to placeholder defaults
    return [
        "http://192.168.1.10:1234",
        "http://192.168.1.11:1234",
    ]


def get_gradio_port() -> int:
    """
    Get Gradio server port from environment or default.

    Returns port from GRADIO_PORT env var, or 7860 as default.
    """
    port_str = os.environ.get("GRADIO_PORT", "7860")
    try:
        return int(port_str)
    except ValueError:
        return 7860


# ─────────────────────────────────────────────────────────────────────
# HUGGINGFACE MODE
# ─────────────────────────────────────────────────────────────────────

# Vetted models for HF Spaces deployment
# These are known to work with the HF Inference API
HF_DEFAULT_MODELS: list[str] = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]


def is_huggingface_mode() -> bool:
    """
    Check if running in HuggingFace mode.

    HF mode is active when HF_TOKEN is set and no LM_STUDIO_SERVER_* vars are set.
    This allows explicit opt-in to HF mode on Spaces while allowing local
    development with LM Studio.
    """
    has_hf_token = bool(os.environ.get("HF_TOKEN"))
    has_lm_studio = bool(load_servers_from_env())
    return has_hf_token and not has_lm_studio


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN")


def get_hf_models() -> list[str]:
    """
    Get HuggingFace models to use.

    Returns HF_DEFAULT_MODELS for now. Could be extended to read from
    environment or config file.
    """
    return list(HF_DEFAULT_MODELS)


def get_beyond_compare_path() -> str:
    """
    Get Beyond Compare executable path from environment or default.

    Returns path from BEYOND_COMPARE_PATH env var, or platform default.
    """
    path = os.environ.get("BEYOND_COMPARE_PATH")
    if path:
        return path
    # Platform defaults
    if os.name == 'nt':  # Windows
        return r"C:\Program Files\Beyond Compare 4\BComp.exe"
    else:  # Linux/Mac
        return "bcompare"


# ─────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────

# ServerConfig is defined in local-inference-pool; re-exported here for backwards compatibility
from local_inference_pool import ServerConfig  # noqa: F401


class ModelConfig(BaseModel):
    """Configuration for a model to be tested."""
    model_id: str  # As it appears in LM Studio manifest
    display_name: Optional[str] = None  # Friendly name for UI tabs

    @property
    def tab_name(self) -> str:
        return self.display_name or self.model_id


class Message(BaseModel):
    """A single message in a conversation.

    Content can be:
    - str: Plain text message
    - list: Multimodal content (text + images) in OpenAI format
    """
    role: str  # "user", "assistant", or "system"
    content: Union[str, list]  # str or list of content parts

    def get_text(self) -> str:
        """Extract text content from message (for display)."""
        if isinstance(self.content, str):
            return self.content
        # Multimodal: find text parts
        for part in self.content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
        return ""

    def has_image(self) -> bool:
        """Check if message contains an image."""
        if isinstance(self.content, str):
            return False
        return any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in self.content
        )


def encode_image_to_data_url(image_path: str) -> str:
    """Encode an image file to a data URL for OpenAI vision API."""
    import base64
    import mimetypes

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_data}"


def build_multimodal_content(text: str, image_path: Optional[str] = None) -> Union[str, list]:
    """Build message content, optionally with an image.

    Returns str if no image, list of content parts if image present.
    """
    if not image_path:
        return text

    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}}
    ]


class ModelContext(BaseModel):
    """Complete conversation context for a single model."""
    model_id: str
    messages: list[Message] = []
    error: Optional[str] = None  # Set if model encountered an error

    def add_user_message(self, content: str, image_path: Optional[str] = None) -> None:
        """Add a user message, optionally with an image attachment."""
        msg_content = build_multimodal_content(content, image_path)
        self.messages.append(Message(role="user", content=msg_content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))

    def to_openai_messages(self, system_prompt: str) -> list[dict]:
        """Convert to OpenAI API message format."""
        result = [{"role": "system", "content": system_prompt}]
        result.extend([{"role": m.role, "content": m.content} for m in self.messages])
        return result

    def to_display_format(self) -> str:
        """Convert to human-readable format for UI display (markdown)."""
        lines = [f"### {self.model_id}"]
        for msg in self.messages:
            text = msg.get_text()
            if msg.role == "user":
                prefix = "**User:**"
                if msg.has_image():
                    prefix = "**User:** 🖼️"
                lines.append(f"{prefix} {text}")
            elif msg.role == "assistant":
                lines.append(f"**Assistant:** {text}")
        if self.error:
            lines.append(f"\n⚠️ **ERROR:** {self.error}")
        return "\n\n".join(lines)


class SessionState(BaseModel):
    """Complete state for a comparison session."""
    models: list[str]
    contexts: dict[str, ModelContext] = {}
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = DEFAULT_TEMPERATURE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    halted: bool = False
    halt_reason: Optional[str] = None
