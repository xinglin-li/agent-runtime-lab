# src/agent_runtime/runtime/steps.py
"""AgentStep builder for the V2 Structured ReAct runtime.

AgentStepBuilder owns the mechanics of constructing a structured AgentStep
from the pieces produced by the runtime loop: rationale, action, observation,
and stop reason.

This keeps the AgentRuntime.loop clean and makes trajectory construction
separately testable.
"""
from agent_runtime.models import (
    AgentStep,
    PlannerRationale,
    ToolAction,
    ToolObservation,
    StopReason,
)
from typing import Optional


class AgentStepBuilder:
    """Constructs AgentStep instances incrementally.

    Typical usage inside the runtime loop:

        builder = AgentStepBuilder(run_id="abc", step=1)
        builder.set_rationale(PlannerRationale(summary="..."))
        builder.set_action(ToolAction(call_id="c1", ...))
        # tool executes ...
        builder.set_observation(ToolObservation(call_id="c1", ok=True, ...))
        step = builder.build()

    On the final step, set_stop_reason() is called instead of set_action/observation.
    """

    def __init__(self, run_id: str, step: int) -> None:
        self._run_id = run_id
        self._step = step
        self._rationale: Optional[PlannerRationale] = None
        self._action: Optional[ToolAction] = None
        self._observation: Optional[ToolObservation] = None
        self._stop_reason: Optional[StopReason] = None

    def set_rationale(self, rationale: PlannerRationale) -> None:
        self._rationale = rationale

    def set_action(self, action: ToolAction) -> None:
        self._action = action

    def set_observation(self, observation: ToolObservation) -> None:
        self._observation = observation

    def set_stop_reason(self, stop_reason: StopReason) -> None:
        self._stop_reason = stop_reason

    def build(self) -> AgentStep:
        return AgentStep(
            run_id=self._run_id,
            step=self._step,
            rationale=self._rationale,
            action=self._action,
            observation=self._observation,
            stop_reason=self._stop_reason,
        )

    @staticmethod
    def build_direct_answer_step(
        run_id: str,
        step: int,
        rationale: Optional[PlannerRationale] = None,
    ) -> AgentStep:
        """Convenience: build a step for a direct-answer (no tool call) response."""
        builder = AgentStepBuilder(run_id, step)
        if rationale:
            builder.set_rationale(rationale)
        builder.set_stop_reason(
            StopReason(reason="final_answer", step=step)
        )
        return builder.build()

    @staticmethod
    def build_stop_step(
        run_id: str,
        step: int,
        reason: str,
        message: Optional[str] = None,
    ) -> AgentStep:
        """Convenience: build a step that terminates without a tool action."""
        builder = AgentStepBuilder(run_id, step)
        builder.set_stop_reason(
            StopReason(reason=reason, message=message, step=step)
        )
        return builder.build()