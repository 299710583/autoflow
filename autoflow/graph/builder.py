from __future__ import annotations
from langgraph.graph import END, StateGraph

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
from autoflow.graph.edges import (
    route_after_executor,
    route_after_planner,
    route_after_recon,
    route_after_researcher,
    route_after_strategist,
    route_after_validation,
    route_after_verifier,
)
from autoflow.graph.nodes import AutoFlowNodes
from autoflow.graph.state import AutoFlowState


def build_assessment_graph(
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
    checkpointer=None,
):
    nodes = AutoFlowNodes(
        planner=planner,
        discovery=discovery,
        discovery_reasoner=discovery_reasoner,
        recon=recon,
        researcher=researcher,
        executor=executor,
        validation_executor=validation_executor,
        verifier=verifier,
        validation=validation,
        strategist=strategist,
        reporter=reporter,
    )

    graph = StateGraph(AutoFlowState)
    graph.add_node("planner", nodes.planner_node)
    graph.add_node("discovery", nodes.discovery_node)
    graph.add_node("recon", nodes.recon_node)
    graph.add_node("research", nodes.researcher_node)
    graph.add_node("execute", nodes.executor_node)
    graph.add_node("validation_execute", nodes.validation_executor_node)
    graph.add_node("verify", nodes.verifier_node)
    graph.add_node("validation", nodes.validation_node)
    graph.add_node("strategy", nodes.strategist_node)
    graph.add_node("report", nodes.reporter_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "recon": "discovery",
            "discovery": "discovery",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "discovery",
        route_after_strategist,
        {
            "execute": "execute",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "recon",
        route_after_recon,
        {
            "research": "research",
            "verify": "verify",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "research",
        route_after_researcher,
        {
            "strategy": "strategy",
            "execute": "execute",
            "verify": "verify",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "execute",
        route_after_executor,
        {
            "verify": "verify",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "verify",
        route_after_verifier,
        {
            "validation": "validation",
            "strategy": "strategy",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "strategy": "strategy",
            "validation_execute": "validation_execute",
            "execute": "execute",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "validation_execute",
        route_after_executor,
        {
            "strategy": "strategy",
            "verify": "verify",
            "report": "report",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "strategy",
        route_after_strategist,
        {
            "execute": "execute",
            "report": "report",
            "end": END,
        },
    )
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
