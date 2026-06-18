# src/agent_runtime/runtime/state.py
from pydantic import BaseModel, Field
from typing import List, Optional
from agent_runtime.models import AgentMessage, AgentStep
from agent_runtime.tracing import TraceEvent


class AgentState(BaseModel):
    run_id: str
    messages: List[AgentMessage] = Field(default_factory=list)
    step_count: int = 0
    # Status: running, completed, max_steps_exceeded, failed, cancelled
    status: str = "running"
    # V2: structured ReAct trajectory (one AgentStep per loop iteration)
    steps: List[AgentStep] = Field(default_factory=list)
    trace_events: List[TraceEvent] = Field(default_factory=list)
    final_answer: Optional[str] = None
    # V2: explicit StopReason classifier
    stop_reason: Optional[str] = None