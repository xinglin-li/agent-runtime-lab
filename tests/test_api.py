# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from agent_runtime.api import app, RUNS_DATABASE

client = TestClient(app)

def test_health_endpoint():
    """测试情况 1：最基础的健康检查探针必须返回 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "agent-runtime-core"}

def test_create_and_poll_run_lifecycle():
    """测试情况 2：完整的非阻塞创建任务、轮询结果、并拉取 Trace 的生命周期契约"""
    # 1. 提交一条运行指令
    post_res = client.post("/runs", json={"user_input": "Hello agent", "max_steps": 3})
    assert post_res.status_code == 201
    data = post_res.json()
    run_id = data["run_id"]
    assert data["status"] == "running"

    # 2. 模拟轻微等待，确保后台协程已经把任务跑完并物理更新了数据库
    import time
    time.sleep(0.05)

    # 3. 轮询状态接口
    get_res = client.get(f"/runs/{run_id}")
    assert get_res.status_code == 200
    summary = get_res.json()
    assert summary["status"] == "completed"
    assert "FastAPI Gateway" in summary["final_answer"]

    # 4. 拉取结构化 Trace 端点
    trace_res = client.get(f"/runs/{run_id}/trace")
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    assert trace_data["run_id"] == run_id
    assert isinstance(trace_data["trace_events"], list)

def test_query_non_existent_run():
    """测试情况 3：查询不存在的非法边界 run_id，接口必须返回 404 错误"""
    res = client.get("/runs/non-existent-uuid-666")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]