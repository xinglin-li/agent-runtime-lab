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

# In-memory stand-in for a production run database or state cache.
RUNS_DATABASE: Dict[str, AgentState] = {}
# Active background task handles used for explicit cancellation.
ACTIVE_TASKS: Dict[str, asyncio.Task] = {}

# Initialize and assemble tools at the API layer.
def get_configured_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(AddNumbersTool()) # Register the default arithmetic tool.
    return reg

# HTTP contract models (DTOs).
class CreateRunRequest(BaseModel):
    user_input: str = Field(..., description="The query prompt for the agent to execute.")
    max_steps: Optional[int] = Field(5, description="Upper bound of agent loops.")

class RunSummaryResponse(BaseModel):
    # API responses deliberately expose a compact DTO instead of the full AgentState.
    # Internal messages, tool payloads, and trace details can contain sensitive data.
    run_id: str
    status: str
    step_count: int
    final_answer: Optional[str] = None

# ==================== REST Routes ====================

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "agent-runtime-core"}

async def background_agent_worker(run_id: str, user_input: str, max_steps: int):
    """Async background worker that runs the local domain control loop."""
    # Predefine fake model responses so web integration tests remain deterministic.
    fake_responses = [
        AgentMessage(role="assistant", content="Processed via FastAPI Gateway seamlessly.")
    ]
    provider = FakeProvider(fake_responses)
    runtime = AgentRuntime(provider=provider, tool_registry=get_configured_registry(), max_steps=max_steps)
    
    try:
        # Start the blocking control loop in the background.
        # The runtime is still synchronous; a production version should move to async run.
        # Simulate a small amount of background work for the API lifecycle test.
        await asyncio.sleep(0.01)
        
        # Delegate execution to the domain layer and retrieve the final state.
        final_state = runtime.run(user_input)
        # Preserve the API-assigned run_id so clients can poll by the ticket they received.
        final_state.run_id = run_id
        
        # Persist the final state into the in-memory run store.
        RUNS_DATABASE[run_id] = final_state
    except asyncio.CancelledError:
        # Mark the run as cancelled if a user cancellation interrupts the task.
        if run_id in RUNS_DATABASE:
            RUNS_DATABASE[run_id].status = "cancelled"
    except Exception as e:
        if run_id in RUNS_DATABASE:
            RUNS_DATABASE[run_id].status = "failed"

@app.post("/runs", response_model=RunSummaryResponse, status_code=status.HTTP_201_CREATED)
async def create_run(payload: CreateRunRequest, background_tasks: BackgroundTasks):
    """
    Non-blocking endpoint: initialize state, schedule background execution,
    and immediately return a run ticket to the client.
    """
    run_id = str(uuid.uuid4())
    
    # Create the initial running state and write it to the run store.
    initial_state = AgentState(
        run_id=run_id,
        messages=[],
        step_count=0,
        status="running"
    )
    RUNS_DATABASE[run_id] = initial_state
    
    # 1. Classic option: use FastAPI BackgroundTasks for background execution.
    # background_tasks.add_task(background_agent_worker, run_id, payload.user_input, payload.max_steps)
    
    # 2. Use a standalone asyncio.Task so the API can actively cancel it later.
    # BackgroundTasks does not expose a task handle after scheduling.
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
    # Demo endpoint returns raw trace events; production should redact sensitive payloads.
    return {"run_id": run_id, "trace_events": RUNS_DATABASE[run_id].trace_events}

@app.post("/runs/{run_id}/cancel")
def cancel_active_run(run_id: str):
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail="Run not found.")
    
    # Check whether the task is still active in the background.
    if run_id in ACTIVE_TASKS and not ACTIVE_TASKS[run_id].done():
        # Request cooperative cancellation of the coroutine.
        ACTIVE_TASKS[run_id].cancel()
        RUNS_DATABASE[run_id].status = "cancelled"
        return {"run_id": run_id, "message": "Cancellation single emitted safely."}
    
    return {"run_id": run_id, "message": "Task was already stopped or not in active table."}
