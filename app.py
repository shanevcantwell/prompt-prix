"""HuggingFace Spaces entry point for prompt-prix."""

import os
print(f"[startup] TOGETHER_API_KEY set: {bool(os.environ.get('TOGETHER_API_KEY'))}")
print(f"[startup] HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
print(f"[startup] LM_STUDIO_SERVER_1: {os.environ.get('LM_STUDIO_SERVER_1', 'not set')}")

from prompt_prix.config import is_together_mode, is_huggingface_mode
print(f"[startup] is_together_mode(): {is_together_mode()}")
print(f"[startup] is_huggingface_mode(): {is_huggingface_mode()}")

from prompt_prix.main import _register_default_adapter
from prompt_prix.ui import create_app

# Register adapter (Together/HF mode auto-detected via env vars)
_register_default_adapter()

demo = create_app()

if __name__ == "__main__":
    demo.launch()
