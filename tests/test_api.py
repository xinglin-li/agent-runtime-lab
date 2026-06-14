# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from agent_runtime.api import app, RUNS_DATABASE

client = TestClient(app)

def test_health_endpoint():
    """The health probe should return HTTP 200 with the expected payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "agent-runtime-core"}

def test_create_and_poll_run_lifecycle():
    """Create a non-blocking run, poll its result, and fetch its trace."""
    # 1. Submit a run request.
    post_res = client.post("/runs", json={"user_input": "Hello agent", "max_steps": 3})
    assert post_res.status_code == 201
    data = post_res.json()
    run_id = data["run_id"]
    assert data["status"] == "running"

    # 2. Wait briefly so the background coroutine can update the in-memory store.
    import time
    time.sleep(0.05)

    # 3. Poll the run status endpoint.
    get_res = client.get(f"/runs/{run_id}")
    assert get_res.status_code == 200
    summary = get_res.json()
    assert summary["status"] == "completed"
    assert "FastAPI Gateway" in summary["final_answer"]

    # 4. Fetch the structured trace endpoint.
    trace_res = client.get(f"/runs/{run_id}/trace")
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    assert trace_data["run_id"] == run_id
    assert isinstance(trace_data["trace_events"], list)

def test_query_non_existent_run():
    """Querying a missing run_id should return HTTP 404."""
    res = client.get("/runs/non-existent-uuid-666")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]
