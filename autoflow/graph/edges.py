from __future__ import annotations

from autoflow.graph.state import AutoFlowState


def route_after_planner(state: AutoFlowState) -> str:
    return state.get("next_action", "recon")


def route_after_recon(state: AutoFlowState) -> str:
    return state.get("next_action", "research")


def route_after_researcher(state: AutoFlowState) -> str:
    return state.get("next_action", "verify")


def route_after_executor(state: AutoFlowState) -> str:
    return state.get("next_action", "verify")


def route_after_verifier(state: AutoFlowState) -> str:
    return state.get("next_action", "validation")


def route_after_validation(state: AutoFlowState) -> str:
    return state.get("next_action", "strategy")


def route_after_strategist(state: AutoFlowState) -> str:
    return state.get("next_action", "report")
