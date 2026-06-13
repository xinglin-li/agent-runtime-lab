# tests/test_script_runner.py
import pytest
from agent_runtime.tools.script_runner import ControlledScriptRunnerTool, ScriptRunnerInput
from pydantic import ValidationError

@pytest.fixture
def runner_tool():
    return ControlledScriptRunnerTool()

def test_script_execution_success(runner_tool):
    """测试情况 1：合法参数正常调用被批准的两个脚本"""
    # 1. 测试 safe_describe_csv
    args_desc = ScriptRunnerInput(script_name="safe_describe_csv", file_path="series/monthly_sales.csv")
    out_desc = runner_tool.run(args_desc)
    assert out_desc.exit_code == 0
    assert "Headers:" in out_desc.stdout
    assert out_desc.timed_out is False

    # 2. 测试 safe_series_summary
    args_sum = ScriptRunnerInput(script_name="safe_series_summary", file_path="series/monthly_sales.csv")
    out_sum = runner_tool.run(args_sum)
    assert out_sum.exit_code == 0
    assert "Metrics: Mean=123.33" in out_sum.stdout

def test_unapproved_script_rejected(runner_tool):
    """测试情况 2：如果模型试图运行未批准的脚本（如恶意调用未知名称），应直接在 Pydantic 层被卡死"""
    with pytest.raises(ValidationError, match="not approved for execution"):
        ScriptRunnerInput(script_name="malicious_clean_db", file_path="series/monthly_sales.csv")

def test_path_traversal_attack_blocked(runner_tool):
    """测试情况 3：路径穿越注入攻击防御（试图利用 ../ 偷看系统敏感文件）"""
    with pytest.raises(ValidationError, match="Security violation"):
        ScriptRunnerInput(script_name="safe_describe_csv", file_path="../../../../etc/passwd")

def test_script_timeout(runner_tool):
    """测试情况 4：脚本执行发生挂起或超时（设置极其苛刻的超时阈值），应被系统强行熔断"""
    # 传入 timeout_seconds=0 逼迫其立刻超时
    args = ScriptRunnerInput(script_name="safe_describe_csv", file_path="series/monthly_sales.csv", timeout_seconds=0)
    out = runner_tool.run(args)
    assert out.timed_out is True
    assert out.exit_code == -1
    assert "timed out" in out.stderr