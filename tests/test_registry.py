# tests/test_registry.py
import pytest
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool

def test_tool_collision():
    reg = ToolRegistry()
    reg.register(AddNumbersTool())
    
    # 重复注册同名工具必须直接炸开，不能允许静默覆盖
    with pytest.raises(KeyError, match="Tool collision detected"):
        reg.register(AddNumbersTool())

def test_list_schemas():
    reg = ToolRegistry()
    reg.register(AddNumbersTool())
    schemas = reg.list_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "add_numbers"
    assert "parameters" in schemas[0]
