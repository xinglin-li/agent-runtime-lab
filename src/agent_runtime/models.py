# src/agent_runtime/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ──────────────────────────────────────────────────────────────
# V1 models (kept backward-compatible)
# ──────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """Raw structured action draft returned by the provider.

    The runtime must first convert this into a ToolAction before execution.
    """
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    """Raw executor return value.

    The runtime wraps this into a ToolObservation before writing to state and trace.
    """
    call_id: str
    tool_name: str
    ok: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class AgentMessage(BaseModel):
    """Unified message shape for user, assistant, and tool observation roles."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_result: Optional[ToolResult] = None
    # V2: structured rationale carried alongside assistant messages.
    rationale: Optional["PlannerRationale"] = None


# ──────────────────────────────────────────────────────────────
# V2 Structured ReAct models
# ──────────────────────────────────────────────────────────────

class PlannerRationale(BaseModel):
    """A compact, auditable summary of why the model chose this action.

    MUST contain only a short natural-language summary.
    MUST NOT contain raw hidden chain-of-thought reasoning traces.
    """
    summary: str
    confidence: Optional[float] = None
    cited_state_keys: List[str] = Field(default_factory=list)


class ToolAction(BaseModel):
    """A validated, runtime-approved action bound to a rationale.

    ToolCall is the provider draft; ToolAction is what the runtime actually authorizes.
    """
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    rationale: Optional[PlannerRationale] = None


class ToolObservation(BaseModel):
    """Structured observation produced after tool execution.

    ToolResult is the raw executor output; ToolObservation is the runtime-wrapped
    observation that feeds into the agent's state and trace.
    """
    call_id: str
    tool_name: str
    ok: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class StopReason(BaseModel):
    """Explicit termination classifier written by the runtime.

    Allowed reasons:
      - final_answer
      - max_steps_exceeded
      - non_retryable_error
      - cancelled
      - provider_failed
    """
    reason: str
    message: Optional[str] = None
    step: int


class AgentStep(BaseModel):
    """The smallest unit of trajectory eval (Week 4, 8, 10).

    Each step records: why the model acted (rationale), what it chose (action),
    what happened (observation), and why it stopped (stop_reason, on the final step).
    """
    run_id: str
    step: int
    rationale: Optional[PlannerRationale] = None
    action: Optional[ToolAction] = None
    observation: Optional[ToolObservation] = None
    stop_reason: Optional[StopReason] = None