from __future__ import annotations

from autoflow.agents.base import BaseAgent
from autoflow.agents.discovery_reasoner import DiscoveryReasonerAgent
from autoflow.agents.recon import ReconAgent
from autoflow.graph.state import AutoFlowState


class DiscoveryAgent(BaseAgent):
    """合并 recon、attack surface 抽象和发现阶段策略生成。"""

    name = "discovery"

    def __init__(
        self,
        recon: ReconAgent | None = None,
        reasoner: DiscoveryReasonerAgent | None = None,
    ) -> None:
        self.recon = recon or ReconAgent()
        self.reasoner = reasoner or DiscoveryReasonerAgent()

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "discovery"
        state = await self.recon.run(state)
        state = await self.reasoner.run(state)
        state["current_phase"] = "discovery"
        return state
