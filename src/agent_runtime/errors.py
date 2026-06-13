# src/agent_runtime/errors.py
from pydantic import BaseModel
from typing import Any

class AgentError(BaseModel):
    """Base class for agent errors."""
    error_type: str
    message: str
    retryable: bool
    details: Any
