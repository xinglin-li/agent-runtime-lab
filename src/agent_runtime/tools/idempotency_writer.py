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
        # 使用一个封闭的集合（Set）充当生产环境中的 Redis 去重键表
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
        # 核心防线：检查是否是重复提交的任务
        if args.operation_id in self._seen_operations:
            return IdempotentOutput(
                status="skipped",
                message=f"Idempotency hit: Operation '{args.operation_id}' has already been processed. Guarded against double action."
            )
        
        # 如果是首次看到，模拟耗时的写入动作并记录令牌
        self._seen_operations.add(args.operation_id)
        return IdempotentOutput(
            status="committed",
            message=f"Successfully committed data for operation: {args.operation_id}"
        )