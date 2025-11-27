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

        assert "[User]: What is 2+2?" in display
        assert "[Assistant]: 2+2 equals 4." in display

    def test_model_context_to_display_format_with_error(self):
        """Test display format includes error when present."""
        from prompt_prix.config import ModelContext

        context = ModelContext(model_id="test-model")
        context.add_user_message("Hello")
        context.error = "Connection timeout"

        display = context.to_display_format()

        assert "[ERROR]: Connection timeout" in display


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
        """Test SessionState uses default values."""
        from prompt_prix.config import (
            SessionState,
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_TEMPERATURE,
            DEFAULT_TIMEOUT_SECONDS,
            DEFAULT_MAX_TOKENS
        )

        state = SessionState(models=["model-a"])

        assert state.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert state.temperature == DEFAULT_TEMPERATURE
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
