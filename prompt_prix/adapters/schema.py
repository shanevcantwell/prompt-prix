from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class InferenceTask(BaseModel):
    """
    Standardized request object for model inference across all adapters.
    Ensures that core Prompt Prix logic talks in a unified language
    regardless of whether the backend is LM Studio, Ollama, vLLM, etc.
    """
    model_id: str
    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = -1
    timeout_seconds: float = 60.0
    tools: Optional[List[Dict[str, Any]]] = None
    seed: Optional[int] = None
    repeat_penalty: Optional[float] = None
    # Server affinity override (0:model -> server 0)
    preferred_server_idx: Optional[int] = None 
    # The actual stripped model ID to send to the API
    api_model_id: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.api_model_id is None:
             self.api_model_id = self.model_id

class InferenceResult(BaseModel):
    """
    Standardized response object from an adapter.
    Abstracts away backend-specific response schemas (OpenAI-style, raw text, etc.)
    into a common format consumed by the Runner/Battery.
    """
    content: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
