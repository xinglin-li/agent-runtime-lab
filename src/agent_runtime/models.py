# src/agent_runtime/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]

class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    ok: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class AgentMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None