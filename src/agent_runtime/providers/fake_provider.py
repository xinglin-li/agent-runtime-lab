# src/agent_runtime/providers/fake_provider.py
from agent_runtime.providers.base import BaseProvider
from agent_runtime.models import AgentMessage
from typing import List

class FakeProvider(BaseProvider):
    def __init__(self, responses: List[AgentMessage]):
        self.responses = list(reversed(responses))  # Reverse to pop from the end
        
    def generate(self, messages: List[AgentMessage]) -> AgentMessage:
        if not self.responses:
            return AgentMessage(role="assistant", content="No more responses available.")
        return self.responses.pop()
