"""HuggingFace Spaces entry point for prompt-prix."""

from prompt_prix.ui import create_app

demo = create_app()

if __name__ == "__main__":
    demo.launch()
