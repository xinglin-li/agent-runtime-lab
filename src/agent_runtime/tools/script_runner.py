# src/agent_runtime/tools/script_runner.py
import subprocess
from pathlib import Path
from typing import Type, Dict, Any
from pydantic import BaseModel, Field, field_validator
from agent_runtime.tools.base import BaseTool

# Allowlist of scripts that may be executed.
# This is an application-level control, not a full OS sandbox.
APPROVED_SCRIPTS = {
    "safe_describe_csv": "scripts/safe_describe_csv.py",
    "safe_series_summary": "scripts/safe_series_summary.py"
}

class ScriptRunnerInput(BaseModel):
    script_name: str = Field(..., description="The name of the approved script to run.")
    file_path: str = Field(..., description="The relative path to the data file inside sample_data/.")
    timeout_seconds: int = Field(default=5, description="Hard timeout limit for the subprocess.")

    @field_validator("script_name")
    @classmethod
    def validate_script(cls, v: str) -> str:
        if v not in APPROVED_SCRIPTS:
            raise ValueError(f"Script '{v}' is not approved for execution.")
        return v

    @field_validator("file_path")
    @classmethod
    def validate_path_boundary(cls, v: str) -> str:
        # Boundary guard: never allow ../ traversal outside sample_data.
        base_dir = Path("sample_data").resolve()
        target_path = (base_dir / v).resolve()
        
        if not target_path.is_relative_to(base_dir):
            raise ValueError("Security violation: Access path is outside allowed directory constraint.")
        return v
    
class ScriptRunnerOutput(BaseModel):
    script_name: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool
    
class ControlledScriptRunnerTool(BaseTool[ScriptRunnerInput, ScriptRunnerOutput]):
    @property
    def name(self) -> str:
        return "run_approved_script"

    @property
    def description(self) -> str:
        return "Execute a pre-approved Python script on safe sample data under strict constraints."

    @property
    def input_model(self) -> Type[ScriptRunnerInput]:
        return ScriptRunnerInput

    @property
    def output_model(self) -> Type[ScriptRunnerOutput]:
        return ScriptRunnerOutput

    def run(self, args: ScriptRunnerInput) -> ScriptRunnerOutput:
        import time
        
        script_path = APPROVED_SCRIPTS[args.script_name]
        # file_path has already been normalized and boundary-checked by ScriptRunnerInput.
        full_data_path = str(Path("sample_data").resolve() / args.file_path)
        
        # Core guardrail 1: never use shell=True or string-concatenated commands.
        # Pass an explicit argument array to the OS process creation API.
        cmd = ["python", script_path, full_data_path]
        
        start_time = time.perf_counter()
        try:
            # Core guardrail 2: always enforce a bounded timeout.
            res = subprocess.run(
                cmd,
                capture_output=True,  # Capture stdout and stderr.
                text=True,            # Decode output as strings.
                timeout=args.timeout_seconds,
                shell=False           # Do not invoke a system shell.
            )
            duration = (time.perf_counter() - start_time) * 1000
            
            return ScriptRunnerOutput(
                script_name=args.script_name,
                exit_code=res.returncode,
                stdout=res.stdout.strip(),
                stderr=res.stderr.strip(),
                duration_ms=duration,
                timed_out=False
            )
            
        except subprocess.TimeoutExpired as e:
            duration = (time.perf_counter() - start_time) * 1000
            # Convert timeout into a typed result instead of crashing the parent process.
            return ScriptRunnerOutput(
                script_name=args.script_name,
                exit_code=-1,
                stdout="",
                stderr=f"Process timed out after {args.timeout_seconds} seconds.",
                duration_ms=duration,
                timed_out=True
            )
