"""
ReAct loop evaluation support for prompt-prix.

Provides cycle detection, trace schemas, and (future) loop execution
for evaluating model behavior under iterative ReAct execution.

Ported from langgraph-agentic-scaffold (LAS) with zero-dependency
cycle detection and Pydantic trace schemas.
"""

from prompt_prix.react.cycle_detection import detect_cycle, detect_cycle_with_pattern
from prompt_prix.react.schemas import (
    ToolCall,
    ReActIteration,
    ReActLoopTerminated,
    MaxIterationsExceeded,
    StagnationDetected,
)

__all__ = [
    "detect_cycle",
    "detect_cycle_with_pattern",
    "ToolCall",
    "ReActIteration",
    "ReActLoopTerminated",
    "MaxIterationsExceeded",
    "StagnationDetected",
]
