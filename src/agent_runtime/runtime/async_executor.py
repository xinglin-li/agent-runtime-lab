# src/agent_runtime/runtime/async_executor.py
import asyncio
import time
from typing import List, Dict, Any
from agent_runtime.models import ToolCall, ToolResult
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.errors import AgentError
from pydantic import ValidationError

class AsyncToolExecutor:
    def __init__(self, tool_registry: ToolRegistry, max_concurrency: int = 2):
        self.tool_registry = tool_registry
        self.semaphore = asyncio.Semaphore(max_concurrency) # 核心并发闸门
    
    async def execute_single_tool_with_sem(self, tool_call: ToolCall) -> ToolResult:
        # 使用信号量卡住并发上限，超出限制的任务会在这一步异步挂起等待
        async with self.semaphore:
            try:
                tool = self.tool_registry.get_tool(tool_call.tool_name)
                # 为了兼容我们之前写的同步工具 execute，我们使用 asyncio.to_thread 
                # 将同步阻塞的 I/O 或进程调度无缝转换为异步非阻塞协程
                # 如果未来工具本身就是 async def，则可以直接 await
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(None, tool.execute, tool_call.arguments)
                
                return ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=True, output=output)

            except KeyError as e:
                err = AgentError(error_type="unknown_tool", message=str(e), retryable=False, details={})
                return ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
            except ValidationError as e:
                err = AgentError(error_type="invalid_arguments", message="Validation failed", retryable=False, details=e.errors(include_url=False))
                return ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
            except Exception as e:
                err = AgentError(error_type="tool_execution_failed", message=str(e), retryable=False, details={})
                return ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
        
    async def execute_batch(self, tool_calls: List[ToolCall], batch_timeout: float = 5.0) -> List[ToolResult]:
        """
        并发执行一组工具，支持整体超时。
        一旦触发超时，所有尚未完成的异步任务会被瞬间 Cancel 释放，不引发资源泄露。
        """
        tasks = [asyncio.create_task(self.execute_single_tool_with_sem(call)) for call in tool_calls]
        
        try:
            # 开启高维限时全家桶
            return await asyncio.wait_for(asyncio.gather(*tasks), timeout=batch_timeout)
        except asyncio.TimeoutError:
            # 捕获整体超时：果断把所有子任务全部强行掐断
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # 等待它们安全退出，收拾残局
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 为每一个任务组装结构化的超时 Observation
            results = []
            for call in tool_calls:
                err = AgentError(
                    error_type="timeout",
                    message=f"Batch execution timed out after {batch_timeout} seconds.",
                    retryable=True,
                    details={}
                )
                results.append(ToolResult(call_id=call.call_id, tool_name=call.tool_name, ok=False, error=err.model_dump()))
            return results
