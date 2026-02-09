"""Shared test fixtures for prompt-prix tests."""

import pytest
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock


# ─────────────────────────────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────────────────────────────

MOCK_SERVER_1 = "http://192.168.1.10:1234"
MOCK_SERVER_2 = "http://192.168.1.11:1234"
MOCK_SERVERS = [MOCK_SERVER_1, MOCK_SERVER_2]

MOCK_MODEL_1 = "llama-3.2-3b-instruct"
MOCK_MODEL_2 = "qwen2.5-7b-instruct"
MOCK_MODELS = [MOCK_MODEL_1, MOCK_MODEL_2]

MOCK_MANIFEST_RESPONSE = {
    "data": [
        {"id": MOCK_MODEL_1, "object": "model"},
        {"id": MOCK_MODEL_2, "object": "model"},
    ]
}

MOCK_COMPLETION_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1699000000,
    "model": MOCK_MODEL_1,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}

MOCK_STREAMING_CHUNKS = [
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":" France"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{"content":" Paris."},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"llama-3.2-3b-instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    'data: [DONE]',
]


# ─────────────────────────────────────────────────────────────────────
# FIXTURES - Data Models
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_servers():
    """Return list of mock server URLs."""
    return MOCK_SERVERS.copy()


@pytest.fixture
def mock_models():
    """Return list of mock model IDs."""
    return MOCK_MODELS.copy()


@pytest.fixture
def sample_messages():
    """Return sample conversation messages."""
    from prompt_prix.config import Message
    return [
        Message(role="user", content="What is the capital of France?"),
        Message(role="assistant", content="The capital of France is Paris."),
        Message(role="user", content="What about Germany?"),
        Message(role="assistant", content="The capital of Germany is Berlin."),
    ]


@pytest.fixture
def sample_model_context(mock_models, sample_messages):
    """Return a ModelContext with sample messages."""
    from prompt_prix.config import ModelContext
    context = ModelContext(model_id=mock_models[0])
    for msg in sample_messages:
        context.messages.append(msg)
    return context


@pytest.fixture
def sample_session_state(mock_models):
    """Return a SessionState with multiple model contexts."""
    from prompt_prix.config import SessionState, ModelContext, Message

    state = SessionState(
        models=mock_models,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        timeout_seconds=300,
        max_tokens=2048
    )

    # Initialize contexts for each model
    for model_id in mock_models:
        context = ModelContext(model_id=model_id)
        context.messages = [
            Message(role="user", content="What is the capital of France?"),
            Message(role="assistant", content="The capital of France is Paris."),
        ]
        state.contexts[model_id] = context

    return state


@pytest.fixture
def halted_session_state(sample_session_state):
    """Return a SessionState that has been halted due to error."""
    sample_session_state.halted = True
    sample_session_state.halt_reason = "Model llama-3.2-3b-instruct failed: Connection timeout"
    sample_session_state.contexts[MOCK_MODEL_1].error = "Connection timeout"
    return sample_session_state


# ─────────────────────────────────────────────────────────────────────
# FIXTURES - HTTP Mocking
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_manifest_response():
    """Return mock /v1/models response."""
    return MOCK_MANIFEST_RESPONSE.copy()


@pytest.fixture
def mock_completion_response():
    """Return mock /v1/chat/completions response."""
    return MOCK_COMPLETION_RESPONSE.copy()


@pytest.fixture
def mock_streaming_chunks():
    """Return mock streaming response chunks."""
    return MOCK_STREAMING_CHUNKS.copy()



# ─────────────────────────────────────────────────────────────────────
# FIXTURES - Temporary Files
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_system_prompt(tmp_path):
    """Create a temporary system prompt file."""
    prompt_file = tmp_path / "system_prompt.txt"
    prompt_file.write_text("You are a test assistant.")
    return prompt_file


@pytest.fixture
def tmp_batch_prompts(tmp_path):
    """Create a temporary batch prompts file."""
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(
        "What is the capital of France?\n"
        "What is the capital of Germany?\n"
        "What is the capital of Italy?\n"
    )
    return prompts_file
