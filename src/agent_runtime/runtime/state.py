# src/agent_runtime/runtime/state.py
from pydantic import BaseModel, Field
from typing import List, Optional
from agent_runtime.models import AgentMessage
from agent_runtime.tracing import TraceEvent

class AgentState(BaseModel):
    run_id: str
    messages: List[AgentMessage] = Field(default_factory=list)
    step_count: int = 0
    status: str = "running" # running, completed, max_steps_exceeded, failed
    trace_events: List[TraceEvent] = Field(default_factory=list)
    final_answer: Optional[str] = None
