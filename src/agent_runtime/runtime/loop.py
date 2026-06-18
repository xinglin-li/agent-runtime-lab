# src/agent_runtime/runtime/loop.py
"""V2 Structured ReAct Agent Runtime.

The runtime enforces a deterministic, traceable execution loop around a
probabilistic LLM provider. Every step produces a structured AgentStep
containing rationale, action, observation, and (on termination) a StopReason.
"""
import uuid
from pydantic import ValidationError

from agent_runtime.runtime.state import AgentState
from agent_runtime.runtime.steps import AgentStepBuilder
from agent_runtime.providers.base import BaseProvider
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.models import (
    AgentMessage,
    ToolResult,
    ToolAction,
    ToolObservation,
    PlannerRationale,
)
from agent_runtime.errors import AgentError
from agent_runtime.tracing import TraceRecorder


class AgentRuntime:
    def __init__(
        self,
        provider: BaseProvider,
        tool_registry: ToolRegistry,
        max_steps: int = 5,
        max_tool_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.max_tool_retries = max_tool_retries

    # -- helpers -------------------------------------------------

    @staticmethod
    def _extract_rationale(msg: AgentMessage) -> PlannerRationale | None:
        return msg.rationale

    @staticmethod
    def _tool_call_to_action(tc, rationale):
        return ToolAction(
            call_id=tc.call_id,
            tool_name=tc.tool_name,
            arguments=tc.arguments,
            rationale=rationale,
        )

    @staticmethod
    def _result_to_observation(result):
        return ToolObservation(
            call_id=result.call_id,
            tool_name=result.tool_name,
            ok=result.ok,
            output=result.output,
            error=result.error,
        )

    # -- tool execution ------------------------------------------

    def _execute_tool(self, action, state, recorder):
        step = state.step_count
        recorder.record(state.run_id, "tool_started", step, {
            "tool_name": action.tool_name,
            "call_id": action.call_id,
        })

        tool = self.tool_registry.get_tool(action.tool_name)

        attempt = 0
        tool_success = False
        last_exception_msg = ""

        while attempt < self.max_tool_retries:
            attempt += 1
            try:
                output = tool.execute(action.arguments)
                tool_success = True
                break
            except (ConnectionError, TimeoutError) as e:
                last_exception_msg = str(e)
                recorder.record(state.run_id, "tool_transient_error", step, {
                    "tool_name": action.tool_name,
                    "attempt": attempt,
                    "error": last_exception_msg,
                })
            except Exception:
                raise

        if tool_success:
            result = ToolResult(
                call_id=action.call_id,
                tool_name=action.tool_name,
                ok=True,
                output=output,
            )
            recorder.record(state.run_id, "tool_succeeded", step, {
                "tool_name": action.tool_name,
                "call_id": action.call_id,
            })
        else:
            err = AgentError(
                error_type="tool_execution_failed",
                message=f"Failed after {attempt} retries: {last_exception_msg}",
                retryable=False,
                details={},
            )
            result = ToolResult(
                call_id=action.call_id,
                tool_name=action.tool_name,
                ok=False,
                error=err.model_dump(),
            )
            recorder.record(state.run_id, "tool_failed", step, {
                "tool_name": action.tool_name,
                "call_id": action.call_id,
                "reason": "retries_exhausted",
            })
            state.status = "failed"
            state.stop_reason = "non_retryable_error"

        observation = self._result_to_observation(result)
        recorder.record(state.run_id, "observation_recorded", step, {
            "call_id": action.call_id,
            "ok": observation.ok,
        })
        return observation

    # -- main loop -----------------------------------------------

    def run(self, user_input: str) -> AgentState:
        run_id = str(uuid.uuid4())
        recorder = TraceRecorder()

        recorder.record(run_id, "run_started", 0, {"user_input": user_input})

        state = AgentState(
            run_id=run_id,
            messages=[AgentMessage(role="user", content=user_input)],
            step_count=0,
            status="running",
        )

        while state.status == "running":
            # -- hard guardrail --
            if state.step_count >= self.max_steps:
                state.status = "max_steps_exceeded"
                state.stop_reason = "max_steps_exceeded"
                recorder.record(run_id, "max_steps_exceeded", state.step_count, {})
                stop_step = AgentStepBuilder.build_stop_step(
                    run_id, state.step_count,
                    reason="max_steps_exceeded",
                    message=f"Reached max_steps={self.max_steps}",
                )
                state.steps.append(stop_step)
                recorder.record(run_id, "stop_reason_recorded", state.step_count, {
                    "reason": "max_steps_exceeded",
                })
                break

            # -- ask provider --
            recorder.record(run_id, "model_requested", state.step_count, {})
            try:
                assistant_msg = self.provider.generate(state.messages)
            except Exception as e:
                state.status = "failed"
                state.stop_reason = "provider_failed"
                recorder.record(run_id, "model_provider_failed", state.step_count, {
                    "error": str(e),
                })
                stop_step = AgentStepBuilder.build_stop_step(
                    run_id, state.step_count,
                    reason="provider_failed",
                    message=str(e),
                )
                state.steps.append(stop_step)
                recorder.record(run_id, "stop_reason_recorded", state.step_count, {
                    "reason": "provider_failed",
                })
                break

            state.messages.append(assistant_msg)
            state.step_count += 1
            recorder.record(run_id, "model_responded", state.step_count, {
                "message": assistant_msg.model_dump(),
            })

            # -- extract rationale --
            rationale = self._extract_rationale(assistant_msg)
            if rationale:
                recorder.record(run_id, "rationale_recorded", state.step_count, {
                    "summary": rationale.summary,
                })

            # -- tool-call path --
            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    recorder.record(run_id, "tool_call_received", state.step_count, {
                        "tool_call": tc.model_dump(),
                    })

                    action = self._tool_call_to_action(tc, rationale)
                    recorder.record(run_id, "tool_action_created", state.step_count, {
                        "tool_name": action.tool_name,
                        "call_id": action.call_id,
                    })

                    try:
                        observation = self._execute_tool(action, state, recorder)
                        # success: build result from observation
                        result = ToolResult(
                            call_id=observation.call_id,
                            tool_name=observation.tool_name,
                            ok=observation.ok,
                            output=observation.output,
                            error=observation.error,
                        )
                    except KeyError as e:
                        err = AgentError(
                            error_type="unknown_tool", message=str(e),
                            retryable=False, details={},
                        )
                        result = ToolResult(
                            call_id=tc.call_id, tool_name=tc.tool_name,
                            ok=False, error=err.model_dump(),
                        )
                        observation = self._result_to_observation(result)
                        recorder.record(run_id, "tool_validation_failed", state.step_count, {
                            "error_type": "unknown_tool",
                            "call_id": tc.call_id,
                        })
                        state.status = "failed"
                        state.stop_reason = "non_retryable_error"
                    except ValidationError as e:
                        err = AgentError(
                            error_type="invalid_arguments",
                            message="Args failed validation.",
                            retryable=True,
                            details=e.errors(include_url=False),
                        )
                        result = ToolResult(
                            call_id=tc.call_id, tool_name=tc.tool_name,
                            ok=False, error=err.model_dump(),
                        )
                        observation = self._result_to_observation(result)
                        recorder.record(run_id, "tool_validation_failed", state.step_count, {
                            "error_type": "invalid_arguments",
                            "call_id": tc.call_id,
                        })
                    except Exception as e:
                        err = AgentError(
                            error_type="tool_execution_failed", message=str(e),
                            retryable=False, details={},
                        )
                        result = ToolResult(
                            call_id=tc.call_id, tool_name=tc.tool_name,
                            ok=False, error=err.model_dump(),
                        )
                        observation = self._result_to_observation(result)
                        recorder.record(run_id, "tool_failed", state.step_count, {
                            "tool_name": tc.tool_name,
                            "call_id": tc.call_id,
                            "error": str(e),
                        })
                        state.status = "failed"
                        state.stop_reason = "non_retryable_error"

                    state.messages.append(AgentMessage(role="tool", tool_result=result))

                    # -- build AgentStep --
                    step_builder = AgentStepBuilder(run_id, state.step_count)
                    if rationale:
                        step_builder.set_rationale(rationale)
                    step_builder.set_action(action)
                    step_builder.set_observation(observation)
                    step = step_builder.build()
                    state.steps.append(step)

                    recorder.record(run_id, "step_completed", state.step_count, {
                        "call_id": tc.call_id,
                        "ok": observation.ok,
                    })

                    if state.status in ("failed", "max_steps_exceeded"):
                        recorder.record(run_id, "stop_reason_recorded", state.step_count, {
                            "reason": state.stop_reason,
                        })
                        break

                continue

            # -- direct-answer path (no tool calls) --
            state.status = "completed"
            state.final_answer = assistant_msg.content
            state.stop_reason = "final_answer"

            stop_step = AgentStepBuilder.build_direct_answer_step(
                run_id, state.step_count, rationale=rationale,
            )
            state.steps.append(stop_step)
            recorder.record(run_id, "stop_reason_recorded", state.step_count, {
                "reason": "final_answer",
            })
            recorder.record(run_id, "run_completed", state.step_count, {
                "final_answer": state.final_answer,
            })

        state.trace_events = recorder.events
        return state