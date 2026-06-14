# tests/test_script_runner.py
import pytest
from agent_runtime.tools.script_runner import ControlledScriptRunnerTool, ScriptRunnerInput
from pydantic import ValidationError

@pytest.fixture
def runner_tool():
    return ControlledScriptRunnerTool()

def test_script_execution_success(runner_tool):
    """Approved scripts should execute successfully with valid arguments."""
    # 1. Test safe_describe_csv.
    args_desc = ScriptRunnerInput(script_name="safe_describe_csv", file_path="series/monthly_sales.csv")
    out_desc = runner_tool.run(args_desc)
    assert out_desc.exit_code == 0
    assert "Headers:" in out_desc.stdout
    assert out_desc.timed_out is False

    # 2. Test safe_series_summary.
    args_sum = ScriptRunnerInput(script_name="safe_series_summary", file_path="series/monthly_sales.csv")
    out_sum = runner_tool.run(args_sum)
    assert out_sum.exit_code == 0
    assert "Metrics: Mean=123.33" in out_sum.stdout

def test_unapproved_script_rejected(runner_tool):
    """Unapproved script names should be rejected at the Pydantic boundary."""
    with pytest.raises(ValidationError, match="not approved for execution"):
        ScriptRunnerInput(script_name="malicious_clean_db", file_path="series/monthly_sales.csv")

def test_path_traversal_attack_blocked(runner_tool):
    """Path traversal attempts should be blocked before subprocess execution."""
    with pytest.raises(ValidationError, match="Security violation"):
        ScriptRunnerInput(script_name="safe_describe_csv", file_path="../../../../etc/passwd")

def test_script_timeout(runner_tool):
    """Hung or slow scripts should be terminated by the hard timeout."""
    # Force an immediate timeout.
    args = ScriptRunnerInput(script_name="safe_describe_csv", file_path="series/monthly_sales.csv", timeout_seconds=0)
    out = runner_tool.run(args)
    assert out.timed_out is True
    assert out.exit_code == -1
    assert "timed out" in out.stderr
