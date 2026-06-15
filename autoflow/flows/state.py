from __future__ import annotations

from typing import TypedDict

from autoflow.flows.models import AssessmentFlow


class FlowRuntimeState(TypedDict, total=False):
    flow: AssessmentFlow
    active_task_id: str | None
    active_subtask_id: str | None
    last_action_id: str | None
    web_recon: list[dict]
    tool_observations: list[dict]
    executed_action_fingerprints: list[str]
    strategy_round: int
    max_rounds: int
    approvals_required: list[dict]
    approved_actions: list[dict]
    rejected_actions: list[dict]
    next_agent: str
    next_action: str
    errors: list[str]
