"""Tests for prompt_prix.config module."""

import os
import pytest
from unittest.mock import patch


class TestMessage:
    """Tests for Message model."""

    def test_message_creation_user(self):
        """Test creating a user message."""
        from prompt_prix.config import Message

        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_message_creation_assistant(self):
        """Test creating an assistant message."""
        from prompt_prix.config import Message

        msg = Message(role="assistant", content="Hello! How can I help?")
        assert msg.role == "assistant"
        assert msg.content == "Hello! How can I help?"

    def test_message_creation_system(self):
        """Test creating a system message."""
        from prompt_prix.config import Message

        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."

    def test_message_multimodal_content(self):
        """Test creating a message with multimodal content (text + image)."""
        from prompt_prix.config import Message

        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
        ]
        msg = Message(role="user", content=content)
        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_message_get_text_from_string(self):
        """Test get_text returns content for string messages."""
        from prompt_prix.config import Message

        msg = Message(role="user", content="Hello!")
        assert msg.get_text() == "Hello!"

    def test_message_get_text_from_multimodal(self):
        """Test get_text extracts text from multimodal content."""
        from prompt_prix.config import Message

        content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}
        ]
        msg = Message(role="user", content=content)
        assert msg.get_text() == "Describe this image"

    def test_message_has_image_false_for_string(self):
        """Test has_image returns False for string content."""
        from prompt_prix.config import Message

        msg = Message(role="user", content="No image here")
        assert msg.has_image() is False

    def test_message_has_image_true_for_multimodal(self):
        """Test has_image returns True when image_url present."""
        from prompt_prix.config import Message

        content = [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        ]
        msg = Message(role="user", content=content)
        assert msg.has_image() is True

    def test_message_has_image_false_for_text_only_list(self):
        """Test has_image returns False for list without image."""
        from prompt_prix.config import Message

        content = [{"type": "text", "text": "Just text"}]
        msg = Message(role="user", content=content)
        assert msg.has_image() is False


class TestServerConfig:
    """Tests for ServerConfig model."""

    def test_server_config_defaults(self):
        """Test ServerConfig with default values."""
        from prompt_prix.config import ServerConfig

        config = ServerConfig(url="http://localhost:1234")
        assert config.url == "http://localhost:1234"
        assert config.available_models == []
        assert config.is_busy is False

    def test_server_config_with_models(self):
        """Test ServerConfig with models list."""
        from prompt_prix.config import ServerConfig

        config = ServerConfig(
            url="http://localhost:1234",
            available_models=["model-a", "model-b"]
        )
        assert config.available_models == ["model-a", "model-b"]

    def test_server_config_busy_state(self):
        """Test ServerConfig busy state."""
        from prompt_prix.config import ServerConfig

        config = ServerConfig(url="http://localhost:1234", is_busy=True)
        assert config.is_busy is True


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_model_config_basic(self):
        """Test ModelConfig with just model_id."""
        from prompt_prix.config import ModelConfig

        config = ModelConfig(model_id="llama-3.2-3b-instruct")
        assert config.model_id == "llama-3.2-3b-instruct"
        assert config.display_name is None

    def test_model_config_with_display_name(self):
        """Test ModelConfig with display name."""
        from prompt_prix.config import ModelConfig

        config = ModelConfig(
            model_id="llama-3.2-3b-instruct",
            display_name="Llama 3.2 3B"
        )
        assert config.display_name == "Llama 3.2 3B"

    def test_model_config_tab_name_uses_display_name(self):
        """Test tab_name returns display_name when set."""
        from prompt_prix.config import ModelConfig

        config = ModelConfig(
            model_id="llama-3.2-3b-instruct",
            display_name="Llama 3.2 3B"
        )
        assert config.tab_name == "Llama 3.2 3B"

    def test_model_config_tab_name_fallback_to_model_id(self):
        """Test tab_name returns model_id when display_name not set."""
        from prompt_prix.config import ModelConfig

        config = ModelConfig(model_id="llama-3.2-3b-instruct")
        assert config.tab_name == "llama-3.2-3b-instruct"


class TestModelContext:
    """Tests for ModelContext model."""

    def test_model_context_creation(self):
        """Test creating a ModelContext."""
        from prompt_prix.config import ModelContext

        context = ModelContext(model_id="test-model")
        assert context.model_id == "test-model"
        assert context.messages == []
        assert context.error is None

    def test_model_context_add_user_message(self):
        """Test adding user message to context."""
        from prompt_prix.config import ModelContext

        context = ModelContext(model_id="test-model")
        context.add_user_message("Hello!")

        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert context.messages[0].content == "Hello!"

    def test_model_context_add_user_message_with_image(self, tmp_path):
        """Test adding user message with image attachment."""
        from prompt_prix.config import ModelContext

        # Create a minimal valid PNG file
        png_file = tmp_path / "test.png"
        # Minimal 1x1 red PNG
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        png_file.write_bytes(png_bytes)

        context = ModelContext(model_id="test-model")
        context.add_user_message("What's in this image?", image_path=str(png_file))

        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert isinstance(context.messages[0].content, list)
        assert context.messages[0].has_image() is True
        assert context.messages[0].get_text() == "What's in this image?"

    def test_model_context_add_assistant_message(self):
        """Test adding assistant message to context."""
        from prompt_prix.config import ModelContext

        context = ModelContext(model_id="test-model")
        context.add_assistant_message("Hi there!")

        assert len(context.messages) == 1
        assert context.messages[0].role == "assistant"
        assert context.messages[0].content == "Hi there!"

    def test_model_context_to_openai_messages(self):
        """Test converting context to OpenAI message format."""
        from prompt_prix.config import ModelContext

        context = ModelContext(model_id="test-model")
        context.add_user_message("Question?")
        context.add_assistant_message("Answer.")

        messages = context.to_openai_messages("You are helpful.")

        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Question?"}
        assert messages[2] == {"role": "assistant", "content": "Answer."}

    def test_model_context_to_display_format(self):
        """Test converting context to display format."""
        from prompt_prix.config import ModelContext

        context = ModelContext(model_id="test-model")
        context.add_user_message("What is 2+2?")
        context.add_assistant_message("2+2 equals 4.")

        display = context.to_display_format()

        assert "**User:** What is 2+2?" in display
        # Assistant responses are wrapped in code blocks for readability
        assert "**Assistant:**" in display
        assert "2+2 equals 4." in display

    def test_model_context_to_display_format_with_error(self):
        """Test display format includes error when present."""
        from prompt_prix.config import ModelContext

        context = ModelContext(model_id="test-model")
        context.add_user_message("Hello")
        context.error = "Connection timeout"

        display = context.to_display_format()

        assert "**ERROR:** Connection timeout" in display

    def test_model_context_to_display_format_with_image(self, tmp_path):
        """Test display format shows image indicator."""
        from prompt_prix.config import ModelContext

        # Create a minimal PNG file
        png_file = tmp_path / "test.png"
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        png_file.write_bytes(png_bytes)

        context = ModelContext(model_id="test-model")
        context.add_user_message("Describe this", image_path=str(png_file))
        context.add_assistant_message("I see an image.")

        display = context.to_display_format()

        assert "🖼️" in display
        assert "Describe this" in display


class TestMultimodalHelpers:
    """Tests for multimodal content helper functions."""

    def test_build_multimodal_content_text_only(self):
        """Test build_multimodal_content returns string when no image."""
        from prompt_prix.config import build_multimodal_content

        result = build_multimodal_content("Hello world")
        assert result == "Hello world"

    def test_build_multimodal_content_with_image(self, tmp_path):
        """Test build_multimodal_content returns list with image."""
        from prompt_prix.config import build_multimodal_content

        # Create a minimal PNG file
        png_file = tmp_path / "test.png"
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        png_file.write_bytes(png_bytes)

        result = build_multimodal_content("What is this?", str(png_file))

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "What is this?"
        assert result[1]["type"] == "image_url"
        assert "data:image/png;base64," in result[1]["image_url"]["url"]

    def test_encode_image_to_data_url(self, tmp_path):
        """Test encode_image_to_data_url creates valid data URL."""
        from prompt_prix.config import encode_image_to_data_url

        # Create a minimal PNG file
        png_file = tmp_path / "test.png"
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        png_file.write_bytes(png_bytes)

        result = encode_image_to_data_url(str(png_file))

        assert result.startswith("data:image/png;base64,")
        # Verify it's valid base64
        import base64
        b64_part = result.split(",")[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == png_bytes

    def test_encode_image_to_data_url_jpeg(self, tmp_path):
        """Test encode_image_to_data_url handles JPEG files."""
        from prompt_prix.config import encode_image_to_data_url

        # Create a minimal JPEG file (just header for mime detection)
        jpg_file = tmp_path / "test.jpg"
        jpg_bytes = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46])
        jpg_file.write_bytes(jpg_bytes)

        result = encode_image_to_data_url(str(jpg_file))

        assert result.startswith("data:image/jpeg;base64,")


class TestSessionState:
    """Tests for SessionState model."""

    def test_session_state_creation(self):
        """Test creating a SessionState."""
        from prompt_prix.config import SessionState

        state = SessionState(models=["model-a", "model-b"])

        assert state.models == ["model-a", "model-b"]
        assert state.contexts == {}
        assert state.halted is False
        assert state.halt_reason is None

    def test_session_state_with_custom_settings(self):
        """Test SessionState with custom settings."""
        from prompt_prix.config import SessionState

        state = SessionState(
            models=["model-a"],
            system_prompt="Custom prompt",
            temperature=0.5,
            timeout_seconds=120,
            max_tokens=1024
        )

        assert state.system_prompt == "Custom prompt"
        assert state.temperature == 0.5
        assert state.timeout_seconds == 120
        assert state.max_tokens == 1024

    def test_session_state_defaults(self):
        """Test SessionState uses default values.

        Note: temperature defaults to None (use per-model defaults in LM Studio).
        """
        from prompt_prix.config import (
            SessionState,
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_TIMEOUT_SECONDS,
            DEFAULT_MAX_TOKENS
        )

        state = SessionState(models=["model-a"])

        assert state.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert state.temperature is None  # Use per-model defaults
        assert state.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
        assert state.max_tokens == DEFAULT_MAX_TOKENS


class TestLoadServersFromEnv:
    """Tests for load_servers_from_env function."""

    def test_load_servers_from_env_with_servers(self):
        """Test loading servers from environment variables."""
        from prompt_prix.config import load_servers_from_env

        with patch.dict(os.environ, {
            "LM_STUDIO_SERVER_1": "http://server1:1234",
            "LM_STUDIO_SERVER_2": "http://server2:1234",
        }, clear=False):
            servers = load_servers_from_env()

        assert "http://server1:1234" in servers
        assert "http://server2:1234" in servers

    def test_load_servers_from_env_empty(self):
        """Test loading servers when no env vars set."""
        from prompt_prix.config import load_servers_from_env

        # Clear any existing LM_STUDIO_SERVER vars
        env_copy = {k: v for k, v in os.environ.items() if not k.startswith("LM_STUDIO_SERVER")}
        with patch.dict(os.environ, env_copy, clear=True):
            servers = load_servers_from_env()

        assert servers == []

    def test_load_servers_from_env_skips_empty_values(self):
        """Test that empty env var values are skipped."""
        from prompt_prix.config import load_servers_from_env

        with patch.dict(os.environ, {
            "LM_STUDIO_SERVER_1": "http://server1:1234",
            "LM_STUDIO_SERVER_2": "",
        }, clear=False):
            servers = load_servers_from_env()

        assert "http://server1:1234" in servers
        assert "" not in servers


class TestConstants:
    """Tests for configuration constants."""

    def test_default_temperature_range(self):
        """Test default temperature is in valid range."""
        from prompt_prix.config import DEFAULT_TEMPERATURE

        assert 0.0 <= DEFAULT_TEMPERATURE <= 2.0

    def test_default_timeout_positive(self):
        """Test default timeout is positive."""
        from prompt_prix.config import DEFAULT_TIMEOUT_SECONDS

        assert DEFAULT_TIMEOUT_SECONDS > 0

    def test_default_max_tokens_positive(self):
        """Test default max tokens is positive."""
        from prompt_prix.config import DEFAULT_MAX_TOKENS

        assert DEFAULT_MAX_TOKENS > 0

    def test_manifest_refresh_interval_positive(self):
        """Test manifest refresh interval is positive."""
        from prompt_prix.config import MANIFEST_REFRESH_INTERVAL_SECONDS

        assert MANIFEST_REFRESH_INTERVAL_SECONDS > 0


class TestDockerComposeConfig:
    """Tests for docker-compose.yml configuration.

    These tests validate that Docker config respects .env settings
    rather than hardcoding values.
    """

    def test_gradio_port_not_hardcoded_in_environment(self):
        """Test that GRADIO_PORT is not explicitly set in docker-compose.yml environment.

        Bug #14: docker-compose.yml had `- GRADIO_PORT=7860` which overrides
        whatever the user sets in .env, making that setting useless.
        """
        from pathlib import Path
        import yaml

        compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found"

        with open(compose_file) as f:
            config = yaml.safe_load(f)

        # Check environment section doesn't hardcode GRADIO_PORT
        services = config.get("services", {})
        for service_name, service_config in services.items():
            env_list = service_config.get("environment", [])
            for env_item in env_list:
                if isinstance(env_item, str) and env_item.startswith("GRADIO_PORT="):
                    # Allow variable substitution like ${GRADIO_PORT:-7860}
                    # but not hardcoded values like GRADIO_PORT=7860
                    if "GRADIO_PORT=7860" in env_item and "${" not in env_item:
                        pytest.fail(
                            f"Service '{service_name}' has hardcoded GRADIO_PORT=7860. "
                            f"This overrides .env settings. Use variable substitution instead."
                        )

    def test_port_mapping_uses_variable_substitution(self):
        """Test that port mapping uses ${GRADIO_PORT} variable.

        Bug #14: Port mapping was hardcoded as "7860:7860" instead of
        using "${GRADIO_PORT:-7860}:${GRADIO_PORT:-7860}"
        """
        from pathlib import Path
        import yaml

        compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        with open(compose_file) as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})
        for service_name, service_config in services.items():
            ports = service_config.get("ports", [])
            for port in ports:
                if isinstance(port, str) and ":7860" in port:
                    if "${GRADIO_PORT" not in port:
                        pytest.fail(
                            f"Service '{service_name}' has hardcoded port 7860. "
                            f"Use '${{GRADIO_PORT:-7860}}' for variable substitution."
                        )
