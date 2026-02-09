"""
BenchmarkCase model - represents a single benchmark test case.

Explicit state management per CLAUDE.md: Pydantic model, not dict.
"""

from pydantic import BaseModel, field_validator
from typing import Optional, Any


class BenchmarkCase(BaseModel):
    """
    Single benchmark test case.

    Required fields:
        id: Unique identifier for the test
        user: User message content

    Optional fields provide context for tool-calling tests.
    """

    id: str
    name: str = ""
    category: str = ""
    severity: str = "warning"
    system: str = "You are a helpful assistant."
    user: str
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str] = None  # "required", "auto", "none"
    expected: Optional[dict] = None  # For future grading
    pass_criteria: Optional[str] = None
    fail_criteria: Optional[str] = None
    expected_response: Optional[str] = None  # Exemplar text for drift comparison
    messages: Optional[list[dict]] = None  # Pre-defined multi-turn conversation history
    mode: Optional[str] = None  # None (single-shot) or "react"
    mock_tools: Optional[dict[str, dict]] = None  # Mock tool responses for mode="react"
    max_iterations: int = 15  # Max tool call iterations for mode="react"

    @field_validator('id')
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        """Fail-fast: id cannot be empty."""
        if not v or not v.strip():
            raise ValueError("Test 'id' cannot be empty")
        return v.strip()

    @field_validator('user')
    @classmethod
    def user_not_empty(cls, v: str) -> str:
        """Fail-fast: user message cannot be empty."""
        if not v or not v.strip():
            raise ValueError("Test 'user' message cannot be empty")
        return v.strip()

    def to_messages(self) -> list[dict]:
        """
        Convert to OpenAI messages format.

        If pre-defined multi-turn history is set, returns a copy of it.
        Otherwise builds from system + user (single-turn, backward compatible).
        """
        if self.messages:
            return list(self.messages)
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user}
        ]

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        return self.name if self.name else self.id
