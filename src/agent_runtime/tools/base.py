# src/agent_runtime/tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Generic, TypeVar
from pydantic import BaseModel

# Generic bounds for tool input and output models.
InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT', bound=BaseModel)

class BaseTool(ABC, Generic[InputT, OutputT]):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of what the tool does."""
        pass
    
    @property
    @abstractmethod
    def input_model(self) -> Type[InputT]:
        """Return the Pydantic model class that defines the input schema for the tool."""
        pass
    
    @property
    @abstractmethod
    def output_model(self) -> Type[OutputT]:
        """Return the Pydantic model class that defines the output schema for the tool."""
        pass
    
    @abstractmethod
    def run(self, args: InputT) -> OutputT:
        """Run the tool with the given input arguments and return the output."""
        pass

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified runtime entry point.
        Handles deserialization, input validation, execution, and output validation.
        """
        # Tool implementations should override run(), not execute(); this keeps every tool
        # behind the same validation and serialization boundary.
        # 1. Coerce and validate input arguments.
        validated_input = self.input_model.model_validate(arguments)
        
        # 2. Execute tool logic.
        validated_output = self.run(validated_input)
        
        # 3. Return the validated output as a dictionary.
        return validated_output.model_dump()

    def get_json_schema(self) -> Dict[str, Any]:
        """Generate an OpenAI-style tool schema for LLM consumption."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema()
        }
