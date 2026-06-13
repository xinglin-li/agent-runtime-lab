import uuid
from pydantic import ValidationError
from agent_runtime.runtime.state import AgentState
from agent_runtime.providers.base import BaseProvider
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.models import AgentMessage, ToolResult
from agent_runtime.errors import AgentError

class AgentRuntime:
    def __init__(self, provider: BaseProvider, tool_registry: ToolRegistry, max_steps: int = 10):
        self.provider = provider
        self.tool_registry = tool_registry
        self.max_steps = max_steps
    
    def run(self, user_input: str) -> AgentState:
        # Initialize the agent state with the user's input
        state = AgentState(
            run_id=str(uuid.uuid4()),
            messages=[AgentMessage(role="user", content=user_input)],
            step_count=0,
            status="running"
        )
    
        while state.status == "running":
            # Check if we've exceeded the maximum number of steps to prevent infinite loops
            if state.step_count >= self.max_steps:
                state.status = "max_steps_exceeded"
                break
            
            # Generate a response from the provider based on the current conversation history
            assistant_msg = self.provider.generate(state.messages)
            
            # Append the assistant's message to the conversation history and increment the step count
            state.messages.append(assistant_msg)
            state.step_count += 1
            
            # If the assistant message includes tool calls, execute them and append the results to the conversation history
            if assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    try:
                        # 1. 尝试获取工具（未知工具边界）
                        tool = self.tool_registry.get_tool(tool_call.tool_name)
                        
                        # 2. 执行（内部包含 Pydantic 的输入/输出校验）
                        output = tool.execute(tool_call.arguments)
                        
                        result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=True, output=output)
                        
                    except KeyError as e:
                        # 未知工具错误
                        err = AgentError(error_type="unknown_tool", message=str(e), retryable=False, details={})
                        result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
                        
                    except ValidationError as e:
                        # 参数校验失败（缺失、类型错误、业务规则越界）
                        err = AgentError(
                            error_type="invalid_arguments",
                            message="Arguments failed schema validation.",
                            retryable=True,  # 允许模型看到错误后修正参数重试
                            details=e.errors(include_url=False)
                        )
                        result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())
                        
                    except Exception as e:
                        # 工具运行时内部不可预知的崩溃
                        err = AgentError(error_type="tool_execution_failed", message=str(e), retryable=False, details={})
                        result = ToolResult(call_id=tool_call.call_id, tool_name=tool_call.tool_name, ok=False, error=err.model_dump())

                    state.messages.append(AgentMessage(role="tool", tool_result=result))
            else:
                state.status = "completed"
                state.final_answer = assistant_msg.content
                break
        
        return state
