# tests/test_runtime_loop.py
import pytest
from agent_runtime.models import AgentMessage, ToolCall
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.runtime.loop import AgentRuntime

@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(AddNumbersTool())
    return reg

def test_runtime_loop(registry):
    """The model can directly answer without calling tools."""
    fake_responses = [
        AgentMessage(role="assistant", content="Hello! How can I assist you today?")
    ]
    provider = FakeProvider(fake_responses)
    runtime = AgentRuntime(provider=provider, tool_registry=registry)
    
    state = runtime.run("Hi there!")
    assert state.status == "completed"
    assert state.stop_reason == "final_answer"
    assert state.final_answer == "Hello! How can I assist you today?"
    assert state.step_count == 1
    # V2: direct answer produces one AgentStep with StopReason.
    assert len(state.steps) == 1
    assert state.steps[0].stop_reason is not None
    assert state.steps[0].stop_reason.reason == "final_answer"

def test_single_tool_call_loop(registry):
    """The runtime should execute one tool call and then return the final answer."""
    fake_responses = [
        # First turn: the model requests a tool call.
        AgentMessage(
            role="assistant",
            tool_calls=[ToolCall(call_id="call_001", tool_name="add_numbers", arguments={"a": 2, "b": 3})]
        ),
        # Second turn: the model sees the tool result and gives the final answer.
        AgentMessage(role="assistant", content="The result of 2 + 3 is 5.")
    ]
    provider = FakeProvider(fake_responses)
    runtime = AgentRuntime(provider=provider, tool_registry=registry)
    
    state = runtime.run("What is 2 + 3?")
    
    assert state.status == "completed"
    assert state.step_count == 2
    # Verify the tool result was written into message history.
    tool_messages = [m for m in state.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_result.ok is True
    assert tool_messages[0].tool_result.output == {"result": 5}
    assert state.final_answer == "The result of 2 + 3 is 5."
    # V2: two AgentSteps — one tool-call step + one direct-answer step.
    assert len(state.steps) == 2
    tool_step = state.steps[0]
    assert tool_step.action is not None
    assert tool_step.action.call_id == "call_001"
    assert tool_step.observation is not None
    assert tool_step.observation.ok is True
    answer_step = state.steps[1]
    assert answer_step.stop_reason is not None
    assert answer_step.stop_reason.reason == "final_answer"

def test_max_steps_exceeded(registry):
    """Repeated tool calls should stop when max_steps is exceeded."""
    # Make the model request tool calls repeatedly.
    fake_responses = [
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c1", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c2", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c3", tool_name="add_numbers", arguments={"a": 1, "b": 1})]),
    ]
    provider = FakeProvider(fake_responses)
    # Set max_steps to 2.
    runtime = AgentRuntime(provider=provider, tool_registry=registry, max_steps=2)
    
    state = runtime.run("Loop me!")
    assert state.status == "max_steps_exceeded"
    assert state.stop_reason == "max_steps_exceeded"
    assert state.step_count == 2
    # V2: final AgentStep must carry a max_steps_exceeded StopReason.
    assert len(state.steps) > 0
    assert state.steps[-1].stop_reason is not None
    assert state.steps[-1].stop_reason.reason == "max_steps_exceeded"
    
    
