# src/agent_runtime/tracing.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from datetime import datetime, timezone

class TraceEvent(BaseModel):
    run_id: str
    event_type: str # run_started, model_requested, tool_started, tool_failed, etc.
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    step: int
    # payload stays schemaless so each event type can carry its own debug context.
    payload: Dict[str, Any] = Field(default_factory=dict)

class TraceRecorder:
    def __init__(self):
        self.events: List[TraceEvent] = []
        
    def record(self, run_id: str, event_type: str, step: int, payload: Dict[str, Any] = None):
        # Keep recording side-effect free: callers can persist or filter events later.
        event = TraceEvent(
            run_id=run_id,
            event_type=event_type,
            step=step,
            payload=payload or {}
        )
        self.events.append(event)
