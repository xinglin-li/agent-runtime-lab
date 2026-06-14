# src/agent_runtime/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uuid
import asyncio

from agent_runtime.runtime.state import AgentState
from agent_runtime.runtime.loop import AgentRuntime
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.models import AgentMessage

app = FastAPI(title="NYC AI Agent Runtime Service", version="1.0.0")

# 充当今天生产环境中的物理数据库/状态缓存器 (In-Memory DB)
RUNS_DATABASE: Dict[str, AgentState] = {}
# 用于存放后台异步运行的任务句柄，以便随时进行主动Cancellation
ACTIVE_TASKS: Dict[str, asyncio.Task] = {}

# 在 API 层进行初始化和工具装配
def get_configured_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(AddNumbersTool()) # 默认注册加法工具
    return reg

# 定义标准的 HTTP 接口契约模型 (DTO)
class CreateRunRequest(BaseModel):
    user_input: str = Field(..., description="The query prompt for the agent to execute.")
    max_steps: Optional[int] = Field(5, description="Upper bound of agent loops.")

class RunSummaryResponse(BaseModel):
    run_id: str
    status: str
    step_count: int
    final_answer: Optional[str] = None

# ==================== RESTful 路由实现 ====================

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "agent-runtime-core"}

async def background_agent_worker(run_id: str, user_input: str, max_steps: int):
    """底层真实的异步后台 Worker，负责转动本地领域层的控制循环"""
    # 提前编排假模型的响应序列，以便于后续在 Web 集成测试中跑通确定性断言
    fake_responses = [
        AgentMessage(role="assistant", content="Processed via FastAPI Gateway seamlessly.")
    ]
    provider = FakeProvider(fake_responses)
    runtime = AgentRuntime(provider=provider, tool_registry=get_configured_registry(), max_steps=max_steps)
    
    try:
        # 真正开始在后台执行阻塞的控制循环
        # 注意：这里我们调用原本的同步 runtime，在后续完整架构中我们会将其整体重构为 async run
        # 为了演示，我们模拟一个轻微的后台计算耗时
        await asyncio.sleep(0.01)
        
        # 将运行逻辑交给领域层，获取最终状态
        final_state = runtime.run(user_input)
        # 将 run_id 纠正为 API 层分配的确定性 ID
        final_state.run_id = run_id
        
        # 物理同步回状态数据库
        RUNS_DATABASE[run_id] = final_state
    except asyncio.CancelledError:
        # 如果任务在运行途中被用户发送取消请求强行掐断，捕获取消异常，做安全收尾
        if run_id in RUNS_DATABASE:
            RUNS_DATABASE[run_id].status = "cancelled"
    except Exception as e:
        if run_id in RUNS_DATABASE:
            RUNS_DATABASE[run_id].status = "failed"

@app.post("/runs", response_model=RunSummaryResponse, status_code=status.HTTP_201_CREATED)
async def create_run(payload: CreateRunRequest, background_tasks: BackgroundTasks):
    """
    非阻塞端点：接收到请求后，初始化状态，并立刻将任务丢进线程池/协程后台，瞬间返回给客户端
    """
    run_id = str(uuid.uuid4())
    
    # 初始化一个最初始的空状态占位，写入数据库
    initial_state = AgentState(
        run_id=run_id,
        messages=[],
        step_count=0,
        status="running"
    )
    RUNS_DATABASE[run_id] = initial_state
    
    # 1. 经典方案：利用 FastAPI 自带的 BackgroundTasks 执行后台转动
    # background_tasks.add_task(background_agent_worker, run_id, payload.user_input, payload.max_steps)
    
    # 2. 为了支持主动取消（Cancellation），今天我们采用拉起独立 asyncio.Task 的高级方案：
    task = asyncio.create_task(background_agent_worker(run_id, payload.user_input, payload.max_steps))
    ACTIVE_TASKS[run_id] = task
    
    return RunSummaryResponse(
        run_id=run_id,
        status="running",
        step_count=0
    )

@app.get("/runs/{run_id}", response_model=RunSummaryResponse)
def get_run_status(run_id: str):
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail=f"Run database records for ID '{run_id}' not found.")
    state = RUNS_DATABASE[run_id]
    return RunSummaryResponse(
        run_id=state.run_id,
        status=state.status,
        step_count=state.step_count,
        final_answer=state.final_answer
    )

@app.get("/runs/{run_id}/trace")
def get_run_trace(run_id: str):
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"run_id": run_id, "trace_events": RUNS_DATABASE[run_id].trace_events}

@app.post("/runs/{run_id}/cancel")
def cancel_active_run(run_id: str):
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail="Run not found.")
    
    # 检查当前任务是否真的在后台活跃运行中
    if run_id in ACTIVE_TASKS and not ACTIVE_TASKS[run_id].done():
        # 发动最高级别的异步掐断轰炸：强行 cancel 协程
        ACTIVE_TASKS[run_id].cancel()
        RUNS_DATABASE[run_id].status = "cancelled"
        return {"run_id": run_id, "message": "Cancellation single emitted safely."}
    
    return {"run_id": run_id, "message": "Task was already stopped or not in active table."}
