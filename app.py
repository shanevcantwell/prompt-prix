"""HuggingFace Spaces entry point for prompt-prix."""

from prompt_prix.main import _register_default_adapter
from prompt_prix.ui import create_app

# Register adapter (HF mode auto-detected via HF_TOKEN env var)
_register_default_adapter()

demo = create_app()

if __name__ == "__main__":
    demo.launch(show_api=False)
