import uuid
from pydantic import ValidationError
from agent_runtime.runtime.state import AgentState
from agent_runtime.providers.base import BaseProvider
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.models import AgentMessage, ToolResult
from agent_runtime.errors import AgentError
from agent_runtime.tracing import TraceRecorder

class AgentRuntime:
    def __init__(self, provider: BaseProvider, tool_registry: ToolRegistry, max_steps: int = 5, max_tool_retries: int = 3):
        self.provider = provider
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.max_tool_retries = max_tool_retries  # Lower-level automatic tool retry limit.

    def run(self, user_input: str) -> AgentState:
        run_id = str(uuid.uuid4())
        recorder = TraceRecorder()
        
        # Record the run start event.
        recorder.record(run_id, "run_started", 0, {"user_input": user_input})

        state = AgentState(
            run_id=run_id,
            messages=[AgentMessage(role="user", content=user_input)],
            step_count=0,
            status="running"
        )

        while state.status == "running":
            # max_steps is the hard guardrail against model-driven infinite loops.
            if state.step_count >= self.max_steps:
                state.status = "max_steps_exceeded"
                recorder.record(run_id, "max_steps_exceeded", state.step_count)
                break

            # Ask the provider for the next assistant message.
            recorder.record(run_id, "model_requested", state.step_count)
            try:
                assistant_msg = self.provider.generate(state.messages)
            except Exception as e:
                # Fatal boundary for provider failures.
                state.status = "failed"
                recorder.record(run_id, "model_provider_failed", state.step_count, {"error": str(e)})
                break
                
            state.messages.append(assistant_msg)
            state.step_count += 1
            recorder.record(run_id, "model_responded", state.step_count, {"message": assistant_msg.model_dump()})

            if assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    # Trace the requested action before execution so failed routing is still observable.
                    recorder.record(run_id, "tool_call_received", state.step_count, {"tool_call": tool_call.model_dump()})
                    
                    try:
                        # 1. Route through the tool registry.
                        tool = self.tool_registry.get_tool(tool_call.tool_name)
                        
                        # 2. Execute with lower-level retry handling.
                        attempt = 0
                        output = None
                        tool_success = False
                        last_exception_msg = ""
                        
                        while attempt < self.max_tool_retries:
                            attempt += 1
                            recorder.record(run_id, "tool_started", state.step_count, {"tool_name": tool.name, "attempt": attempt})
                            try:
                                # execute includes Pydantic validation.
                                output = tool.execute(tool_call.arguments)
                                tool_success = True
                                break
                            except (ConnectionError, TimeoutError) as e:
                                # Retry transient network or environment failures.
                                last_exception_msg = str(e)
                                recorder.record(run_id, "tool_transient_error", state.step_count, {"attempt": attempt, "error": last_exception_msg})
                                continue  # Trigger the next retry attempt.
                            except Exception as e:
                                # Non-transient errors should be classified by the outer boundary.
                                raise e
                        
                        if tool_success:
                            result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=True, output=output)
                            recorder.record(run_id, "tool_succeeded", state.step_count, {"tool_name": tool.name})
                        else:
                            # Retry attempts were exhausted.
                            err = AgentError(error_type="tool_execution_failed", message=f"Failed after {attempt} retries: {last_exception_msg}", retryable=False, details={})
                            result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
                            recorder.record(run_id, "tool_failed", state.step_count, {"tool_name": tool.name, "reason": "retries_exhausted"})
                            state.status = "failed" # Non-recoverable error; stop the state machine.

                    except KeyError as e:
                        err = AgentError(error_type="unknown_tool", message=str(e), retryable=False, details={})
                        result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
                        recorder.record(run_id, "tool_validation_failed", state.step_count, {"error_type": "unknown_tool", "message": str(e)})
                        state.status = "failed"  # Unknown tools are fatal and stop the state machine.
                        
                    except ValidationError as e:
                        # Bad arguments are retryable at the model level: the next prompt can include
                        # this structured observation and let the model repair its tool call.
                        err = AgentError(error_type="invalid_arguments", message="Args failed validation.", retryable=True, details=e.errors(include_url=False))
                        result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
                        recorder.record(run_id, "tool_validation_failed", state.step_count, {"error_type": "invalid_arguments"})
                        # Allow the model to correct invalid arguments in the next loop.

                    except Exception as e:
                        err = AgentError(error_type="tool_execution_failed", message=str(e), retryable=False, details={})
                        result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
                        recorder.record(run_id, "tool_failed", state.step_count, {"tool_name": tool_call.tool_name, "error": str(e)})
                        state.status = "failed"

                    state.messages.append(AgentMessage(role="tool", tool_result=result))
                    
            else:
                # No tool calls means the assistant message is treated as the final answer.
                state.status = "completed"
                state.final_answer = assistant_msg.content
                recorder.record(run_id, "run_completed", state.step_count, {"final_answer": state.final_answer})
                break
                
            recorder.record(run_id, "step_completed", state.step_count)

        state.trace_events = recorder.events
        return state
