from __future__ import annotations

from autoflow.agents.executor import ExecutorAgent, ValidationExecutorAgent
from autoflow.agents.discovery import DiscoveryAgent
from autoflow.agents.discovery_reasoner import DiscoveryReasonerAgent
from autoflow.agents.planner import PlannerAgent
from autoflow.agents.recon import ReconAgent
from autoflow.agents.researcher import ResearcherAgent
from autoflow.agents.reporter import ReporterAgent
from autoflow.agents.strategist import StrategistAgent
from autoflow.agents.validation import ValidationAgent
from autoflow.agents.verifier import VerifierAgent
from autoflow.graph.state import AutoFlowState
from autoflow.memory.redis_store import RedisMemoryStore


class AutoFlowNodes:
    def __init__(
        self,
        planner: PlannerAgent | None = None,
        discovery: DiscoveryAgent | None = None,
        discovery_reasoner: DiscoveryReasonerAgent | None = None,
        recon: ReconAgent | None = None,
        researcher: ResearcherAgent | None = None,
        executor: ExecutorAgent | None = None,
        validation_executor: ValidationExecutorAgent | None = None,
        verifier: VerifierAgent | None = None,
        validation: ValidationAgent | None = None,
        strategist: StrategistAgent | None = None,
        reporter: ReporterAgent | None = None,
        redis_memory_store: RedisMemoryStore | None = None,
    ) -> None:
        self.planner = planner or PlannerAgent()
        self.discovery_reasoner = discovery_reasoner or DiscoveryReasonerAgent()
        self.discovery = discovery or DiscoveryAgent(recon=recon, reasoner=self.discovery_reasoner)
        self.recon = recon or ReconAgent()
        self.researcher = researcher or ResearcherAgent()
        self.executor = executor or ExecutorAgent()
        self.validation_executor = validation_executor or ValidationExecutorAgent()
        self.verifier = verifier or VerifierAgent()
        self.validation = validation or ValidationAgent()
        self.strategist = strategist or StrategistAgent()
        self.reporter = reporter or ReporterAgent()
        self.redis_memory_store = redis_memory_store or RedisMemoryStore.from_settings()

    async def planner_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("planner", await self.planner.run(self._hydrate("planner", state)))

    async def discovery_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("discovery", await self.discovery.run(self._hydrate("discovery", state)))

    async def recon_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("recon", await self.recon.run(self._hydrate("recon", state)))

    async def researcher_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("research", await self.researcher.run(self._hydrate("research", state)))

    async def executor_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("execute", await self.executor.run(self._hydrate("execute", state)))

    async def verifier_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("verify", await self.verifier.run(self._hydrate("verify", state)))

    async def validation_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("validation", await self.validation.run(self._hydrate("validation", state)))

    async def validation_executor_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("validation_execute", await self.validation_executor.run(self._hydrate("validation_execute", state)))

    async def strategist_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("strategy", await self.discovery_reasoner.run(self._hydrate("strategy", state)))

    async def reporter_node(self, state: AutoFlowState) -> AutoFlowState:
        return self._record("report", await self.reporter.run(self._hydrate("report", state)))

    def _hydrate(self, node: str, state: AutoFlowState) -> AutoFlowState:
        error = self.redis_memory_store.hydrate_state_memory(state)
        if error:
            state["redis_memory_error"] = f"{node}: {error}"
        return state

    def _record(self, node: str, state: AutoFlowState) -> AutoFlowState:
        error = self.redis_memory_store.record_node_state(node, state)
        if error:
            state["redis_memory_error"] = error
        return state
