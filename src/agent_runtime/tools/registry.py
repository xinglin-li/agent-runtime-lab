# src/agent_runtime/tools/registry.py
from typing import Dict, List, Any
from agent_runtime.tools.base import BaseTool

class ToolRegistry:
    def __init__(self):
        # The registry is the code-level allowlist for model-requested tool names.
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        # Defensive programming: never silently overwrite a tool with the same name.
        if tool.name in self._tools:
            raise KeyError(f"Tool collision detected: '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> BaseTool:
        """Retrieve a tool by its name."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}'")
        return self._tools[name]
    
    def list_schemas(self) -> List[Dict[str, Any]]:
        """List all registered tools in OpenAI schema format."""
        # The model should see schemas, not raw Python callable objects.
        return [tool.get_json_schema() for tool in self._tools.values()]
