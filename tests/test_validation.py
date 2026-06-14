# tests/test_validation.py
import pytest
from agent_runtime.models import AgentMessage, ToolCall
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.runtime.loop import AgentRuntime

@pytest.fixture
def runtime_env():
    reg = ToolRegistry()
    reg.register(AddNumbersTool())
    return reg

def test_invalid_argument_types(runtime_env):
    """Invalid argument types should be returned as invalid_arguments errors."""
    fake_responses = [
        AgentMessage(role="assistant", tool_calls=[
            ToolCall(call_id="c1", tool_name="add_numbers", arguments={"a": "not_a_number", "b": 3})
        ]),
        AgentMessage(role="assistant", content="Sorry, I gave wrong args.")
    ]
    runtime = AgentRuntime(FakeProvider(fake_responses), runtime_env)
    state = runtime.run("calc")
    
    tool_msg = [m for m in state.messages if m.role == "tool"][0]
    res = tool_msg.tool_result
    assert res.ok is False
    assert res.error["error_type"] == "invalid_arguments"

def test_missing_arguments(runtime_env):
    """Missing required arguments should be rejected."""
    fake_responses = [
        AgentMessage(role="assistant", tool_calls=[
            ToolCall(call_id="c2", tool_name="add_numbers", arguments={"a": 10})
        ]),
    ]
    runtime = AgentRuntime(FakeProvider(fake_responses), runtime_env)
    state = runtime.run("calc")
    res = [m for m in state.messages if m.role == "tool"][0].tool_result
    assert res.ok is False
    assert "b" in str(res.error["details"])

def test_business_validation_boundary(runtime_env):
    """Domain validation should reject values beyond the configured boundary."""
    fake_responses = [
        AgentMessage(role="assistant", tool_calls=[
            ToolCall(call_id="c3", tool_name="add_numbers", arguments={"a": 999999, "b": 1})
        ]),
    ]
    runtime = AgentRuntime(FakeProvider(fake_responses), runtime_env)
    state = runtime.run("calc")
    res = [m for m in state.messages if m.role == "tool"][0].tool_result
    assert res.ok is False
    assert res.error["error_type"] == "invalid_arguments"

def test_unknown_tool_error(runtime_env):
    """Calls to unregistered tools should return unknown_tool errors."""
    fake_responses = [
        AgentMessage(role="assistant", tool_calls=[
            ToolCall(call_id="c4", tool_name="sub_numbers", arguments={"a": 1, "b": 1})
        ]),
    ]
    runtime = AgentRuntime(FakeProvider(fake_responses), runtime_env)
    state = runtime.run("calc")
    res = [m for m in state.messages if m.role == "tool"][0].tool_result
    assert res.ok is False
    assert res.error["error_type"] == "unknown_tool"
