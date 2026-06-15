from abc import ABC, abstractmethod

from autoflow.graph.state import AutoFlowState


class BaseAgent(ABC):
    """所有工作流 Agent 的统一异步接口。"""

    name: str

    @abstractmethod
    async def run(self, state: AutoFlowState) -> AutoFlowState:
        """Run one agent step and return updated state."""
