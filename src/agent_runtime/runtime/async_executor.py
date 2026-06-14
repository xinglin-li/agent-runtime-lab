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
        self.semaphore = asyncio.Semaphore(max_concurrency) # Core concurrency gate.
    
    async def execute_single_tool_with_sem(self, tool_call: ToolCall) -> ToolResult:
        # Use a semaphore to enforce the concurrency limit; excess tasks wait here.
        async with self.semaphore:
            try:
                tool = self.tool_registry.get_tool(tool_call.tool_name)
                # Run synchronous tool execution in an executor to avoid blocking the event loop.
                # If future tools are native async functions, they can be awaited directly.
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
        Execute a batch of tool calls concurrently with a batch-level timeout.
        On timeout, pending tasks are cancelled and gathered to avoid leaks.
        """
        tasks = [asyncio.create_task(self.execute_single_tool_with_sem(call)) for call in tool_calls]
        
        try:
            # Enforce the batch-level timeout.
            return await asyncio.wait_for(asyncio.gather(*tasks), timeout=batch_timeout)
        except asyncio.TimeoutError:
            # Batch timeout: cancel all unfinished child tasks.
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to settle.
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Return one timeout observation per requested call so callers can preserve call_id mapping.
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
