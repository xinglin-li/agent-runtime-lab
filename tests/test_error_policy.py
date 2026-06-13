# tests/test_error_policy.py
import pytest
from agent_runtime.models import AgentMessage, ToolCall
from agent_runtime.tools.base import BaseTool
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.runtime.loop import AgentRuntime
from pydantic import BaseModel, Field
from typing import Type

# 构造一个会触发网络波动的假工具
class MockNetInput(BaseModel):
    pass
class MockNetOutput(BaseModel):
    data: str

class FlakyNetworkTool(BaseTool[MockNetInput, MockNetOutput]):
    def __init__(self):
        self.calls = 0
    @property
    def name(self) -> str: return "fetch_data"
    @property
    def description(self) -> str: return "Fetch from flaky server."
    @property
    def input_model(self) -> Type[MockNetInput]: return MockNetInput
    @property
    def output_model(self) -> Type[MockNetOutput]: return MockNetOutput
    
    def run(self, args: MockNetInput) -> MockNetOutput:
        self.calls += 1
        # 前2次调用故意抛出连接错误，第3次成功
        if self.calls < 3:
            raise ConnectionError("Timeout connection drop.")
        return MockNetOutput(data="Success Data")

def test_transient_error_self_healing():
    """验证底层重试机制：网络波动 2 次后第 3 次成功，系统对大模型无感愈合"""
    reg = ToolRegistry()
    flaky_tool = FlakyNetworkTool()
    reg.register(flaky_tool)
    
    fake_responses = [
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c_net", tool_name="fetch_data", arguments={})]),
        AgentMessage(role="assistant", content="Got data successfully.")
    ]
    
    runtime = AgentRuntime(FakeProvider(fake_responses), reg, max_tool_retries=3)
    state = runtime.run("Get server data.")
    
    assert state.status == "completed"
    assert flaky_tool.calls == 3  # 确实调用了 3 次
    
    # 验证 Trace 中是否完整记录了底层重试的轨迹
    transient_events = [e for e in state.trace_events if e.event_type == "tool_transient_error"]
    assert len(transient_events) == 2
    assert state.trace_events[-1].event_type == "run_completed"

def test_fatal_error_stops_runtime():
    """验证未知工具属于非 retryable 致命错误，必须立刻阻断状态机，防止无限循环"""
    reg = ToolRegistry() # 空注册表
    fake_responses = [
        AgentMessage(role="assistant", tool_calls=[ToolCall(call_id="c_fatal", tool_name="unknown_tool", arguments={})]),
    ]
    runtime = AgentRuntime(FakeProvider(fake_responses), reg)
    state = runtime.run("Do something crazy.")
    
    assert state.status == "failed"
    validation_failed_events = [e for e in state.trace_events if e.event_type == "tool_validation_failed"]
    assert len(validation_failed_events) == 1
    assert validation_failed_events[0].payload["error_type"] == "unknown_tool"