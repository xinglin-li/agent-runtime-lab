# src/agent_runtime/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    # call_id is the stable join key between the model-requested call and the tool result.
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]

class ToolResult(BaseModel):
    # ToolResult is intentionally shaped as an observation message, not an exception.
    # The runtime can feed both success and failure results back into the model loop.
    call_id: str
    tool_name: str
    ok: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class AgentMessage(BaseModel):
    # A single message shape covers user text, assistant text/tool calls, and tool observations.
    # Role-specific invariants are enforced by the runtime path that creates each message.
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None
