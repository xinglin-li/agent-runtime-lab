# src/agent_runtime/tools/idempotency_writer.py
from pydantic import BaseModel, Field
from agent_runtime.tools.base import BaseTool
from typing import Type, Set

class IdempotentInput(BaseModel):
    operation_id: str = Field(..., description="Unique client-generated token to ensure idempotency.")
    content: str = Field(..., description="The context data to commit.")

class IdempotentOutput(BaseModel):
    status: str
    message: str

class IdempotencyRunMarkerTool(BaseTool[IdempotentInput, IdempotentOutput]):
    def __init__(self):
        # Use an in-memory set as a stand-in for a production Redis idempotency table.
        # This is process-local and only suitable for tests or single-process demos.
        self._seen_operations: Set[str] = set()
    
    @property
    def name(self) -> str:
        return "write_run_marker"
    
    @property
    def description(self) -> str:
        return "Safely commit data with unique operation_id validation to prevent duplicate execution."
    
    @property
    def input_model(self) -> Type[IdempotentInput]:
        return IdempotentInput

    @property
    def output_model(self) -> Type[IdempotentOutput]:
        return IdempotentOutput
    
    def run(self, args: IdempotentInput) -> IdempotentOutput:
        # Core guardrail: reject duplicate submissions.
        if args.operation_id in self._seen_operations:
            return IdempotentOutput(
                status="skipped",
                message=f"Idempotency hit: Operation '{args.operation_id}' has already been processed. Guarded against double action."
            )
        
        # First-time operation: simulate a write and record the token.
        self._seen_operations.add(args.operation_id)
        return IdempotentOutput(
            status="committed",
            message=f"Successfully committed data for operation: {args.operation_id}"
        )
