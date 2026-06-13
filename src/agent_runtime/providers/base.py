# src/agent_runtime/providers/base.py
from abc import ABC, abstractmethod
from typing import List
from agent_runtime.models import AgentMessage

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[AgentMessage]) -> AgentMessage:
        """Generate a response based on the input messages."""
        pass
