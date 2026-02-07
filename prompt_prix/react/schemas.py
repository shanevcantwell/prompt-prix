"""
ReAct loop trace schemas.

Adapted from langgraph-agentic-scaffold (react_mixin.py).
Pydantic models for structured iteration tracking.

Used by react_execute() MCP tool to record each step of
a model's ReAct loop execution.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool call made by the model."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ReActIteration(BaseModel):
    """One iteration of a ReAct loop.

    Records the model's tool call, the mock tool response,
    and whether the call was valid (parseable + matched a mock).
    """

    iteration: int
    tool_call: ToolCall
    observation: str  # Mock tool response or error message
    success: bool  # True if tool call parsed and matched a mock
    thought: Optional[str] = None  # Model's reasoning before tool call
    latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# Exception hierarchy for loop termination
# ─────────────────────────────────────────────────────────────────────


class ReActLoopTerminated(Exception):
    """Base exception for ReAct loop termination."""
    pass


class MaxIterationsExceeded(ReActLoopTerminated):
    """Model hit the iteration limit without completing."""
    pass


class StagnationDetected(ReActLoopTerminated):
    """Cycle detection found repeating tool call pattern."""
    pass
