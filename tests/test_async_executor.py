# tests/test_async_executor.py
import pytest
import asyncio
import time
from agent_runtime.models import ToolCall
from agent_runtime.tools.base import BaseTool
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.idempotency_writer import IdempotencyRunMarkerTool
from agent_runtime.runtime.async_executor import AsyncToolExecutor
from pydantic import BaseModel
from typing import Type

class SlowInput(BaseModel):
    delay: float
class SlowOutput(BaseModel):
    slept: float

class MockSlowTool(BaseTool[SlowInput, SlowOutput]):
    @property
    def name(self) -> str: return "slow_tool"
    @property
    def description(self) -> str: return "Simulate block."
    @property
    def input_model(self) -> Type[SlowInput]: return SlowInput
    @property
    def output_model(self) -> Type[SlowOutput]: return SlowOutput
    def run(self, args: SlowInput) -> SlowOutput:
        time.sleep(args.delay) # 模拟同步阻塞
        return SlowOutput(slept=args.delay)

@pytest.fixture
def setup_env():
    reg = ToolRegistry()
    reg.register(MockSlowTool())
    reg.register(IdempotencyRunMarkerTool())
    return reg

@pytest.mark.asyncio
async def test_async_batch_speedup(setup_env):
    """测试 1：并发加速。3个相互独立、各阻塞0.5秒的任务，并发执行的总时间应显著小于单线程串行的1.5秒"""
    executor = AsyncToolExecutor(setup_env, max_concurrency=3)
    calls = [
        ToolCall(call_id="t1", tool_name="slow_tool", arguments={"delay": 0.5}),
        ToolCall(call_id="t2", tool_name="slow_tool", arguments={"delay": 0.5}),
        ToolCall(call_id="t3", tool_name="slow_tool", arguments={"delay": 0.5})
    ]
    
    start = time.perf_counter()
    results = await executor.execute_batch(calls, batch_timeout=2.0)
    duration = time.perf_counter() - start
    
    assert len(results) == 3
    assert all(r.ok for r in results)
    # 并发执行，整体耗时应该在 0.5 ~ 0.8 秒之间，绝对小于 1.5 秒
    assert duration < 1.0

@pytest.mark.asyncio
async def test_semaphore_limit(setup_env):
    """测试 2：信号量限流。最大并发度设为 1。此时 2 个 0.4 秒的任务被迫排队，总耗时应恢复到 0.8 秒以上"""
    executor = AsyncToolExecutor(setup_env, max_concurrency=1)
    calls = [
        ToolCall(call_id="t1", tool_name="slow_tool", arguments={"delay": 0.4}),
        ToolCall(call_id="t2", tool_name="slow_tool", arguments={"delay": 0.4})
    ]
    
    start = time.perf_counter()
    await executor.execute_batch(calls, batch_timeout=2.0)
    duration = time.perf_counter() - start
    
    # 证明排队限流生效了
    assert duration >= 0.8

@pytest.mark.asyncio
async def test_batch_timeout_and_cancellation(setup_env):
    """测试 3：整体超时控制。设置 0.2 秒的致命超时期，阻断 0.6 秒的长任务，检查系统是否能强行熔断"""
    executor = AsyncToolExecutor(setup_env, max_concurrency=2)
    calls = [
        ToolCall(call_id="tx", tool_name="slow_tool", arguments={"delay": 0.6})
    ]
    
    results = await executor.execute_batch(calls, batch_timeout=0.2)
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error["error_type"] == "timeout"

@pytest.mark.asyncio
async def test_idempotency_protection(setup_env):
    """测试 4：幂等性拦截。重复发起相同 operation_id 的写操作，第二次必须被拦截为 skipped"""
    executor = AsyncToolExecutor(setup_env, max_concurrency=2)
    
    call_first = ToolCall(call_id="w1", tool_name="write_run_marker", arguments={"operation_id": "tx_999", "content": "buy BTC"})
    call_second = ToolCall(call_id="w2", tool_name="write_run_marker", arguments={"operation_id": "tx_999", "content": "buy BTC"})
    
    # 第一步：首次提交
    res1 = await executor.execute_batch([call_first])
    assert res1[0].output["status"] == "committed"
    
    # 第二步：二次重复提交，触发幂等性保护
    res2 = await executor.execute_batch([call_second])
    assert res2[0].output["status"] == "skipped"
    assert "Idempotency hit" in res2[0].output["message"]