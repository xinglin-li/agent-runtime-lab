# src/agent_runtime/tools/arithmetic.py
from agent_runtime.tools.base import BaseTool
from typing import Dict, Any, Type
from pydantic import BaseModel, Field, field_validator

class AddInput(BaseModel):
    a: float = Field(..., description="The first number")
    b: float = Field(..., description="The second number")

    # 顺手加一个业务校验（Domain Validation）作为演示：假设我们系统不允许计算超过 10000 的加f法
    @field_validator("a", "b")
    @classmethod
    def limit_max_value(cls, v: float) -> float:
        if abs(v) > 10000:
            raise ValueError("Number magnitude cannot exceed 10000.")
        return v

class AddOutput(BaseModel):
    result: float = Field(..., description="The sum of a and b")

class AddNumbersTool(BaseTool[AddInput, AddOutput]):
    @property
    def name(self) -> str:
        return "add_numbers"

    @property
    def description(self) -> str:
        return "Add two numbers together safely with validation."

    @property
    def input_model(self) -> Type[AddInput]:
        return AddInput

    @property
    def output_model(self) -> Type[AddOutput]:
        return AddOutput

    def run(self, args: AddInput) -> AddOutput:
        return AddOutput(result=args.a + args.b)