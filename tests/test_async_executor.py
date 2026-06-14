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
        time.sleep(args.delay) # Simulate synchronous blocking work.
        return SlowOutput(slept=args.delay)

@pytest.fixture
def setup_env():
    reg = ToolRegistry()
    reg.register(MockSlowTool())
    reg.register(IdempotencyRunMarkerTool())
    return reg

@pytest.mark.asyncio
async def test_async_batch_speedup(setup_env):
    """Three independent blocking tasks should complete faster when run concurrently."""
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
    # Concurrent execution should be clearly faster than the 1.5s serial baseline.
    assert duration < 1.0

@pytest.mark.asyncio
async def test_semaphore_limit(setup_env):
    """A max concurrency of 1 should force two tasks to run serially."""
    executor = AsyncToolExecutor(setup_env, max_concurrency=1)
    calls = [
        ToolCall(call_id="t1", tool_name="slow_tool", arguments={"delay": 0.4}),
        ToolCall(call_id="t2", tool_name="slow_tool", arguments={"delay": 0.4})
    ]
    
    start = time.perf_counter()
    await executor.execute_batch(calls, batch_timeout=2.0)
    duration = time.perf_counter() - start
    
    # Proves the semaphore queueing limit is active.
    assert duration >= 0.8

@pytest.mark.asyncio
async def test_batch_timeout_and_cancellation(setup_env):
    """A batch timeout should cancel a long-running task and return timeout results."""
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
    """Repeated write operations with the same operation_id should be skipped."""
    executor = AsyncToolExecutor(setup_env, max_concurrency=2)
    
    call_first = ToolCall(call_id="w1", tool_name="write_run_marker", arguments={"operation_id": "tx_999", "content": "buy BTC"})
    call_second = ToolCall(call_id="w2", tool_name="write_run_marker", arguments={"operation_id": "tx_999", "content": "buy BTC"})
    
    # First submission commits successfully.
    res1 = await executor.execute_batch([call_first])
    assert res1[0].output["status"] == "committed"
    
    # Second duplicate submission triggers idempotency protection.
    res2 = await executor.execute_batch([call_second])
    assert res2[0].output["status"] == "skipped"
    assert "Idempotency hit" in res2[0].output["message"]
