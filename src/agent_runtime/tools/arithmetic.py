# src/agent_runtime/tools/arithmetic.py
from agent_runtime.tools.base import BaseTool
from typing import Dict, Any, Type
from pydantic import BaseModel, Field, field_validator

class AddInput(BaseModel):
    a: float = Field(..., description="The first number")
    b: float = Field(..., description="The second number")

    # Domain validation demo: reject calculations with values whose magnitude exceeds 10000.
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
        # Business logic receives already-validated input; no raw dict parsing belongs here.
        return AddOutput(result=args.a + args.b)
