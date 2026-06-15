from typing import TypedDict

from autoflow.flows.models import AssessmentFlow


class AutoFlowState(TypedDict, total=False):
    flow: AssessmentFlow
    flow_id: str
    project_id: str
    target_scope: list[str]
    rules_of_engagement: dict
    current_phase: str
    active_task_id: str | None
    active_subtask_id: str | None
    last_action_id: str | None
    task_queue: list[dict]
    running_tasks: list[dict]
    completed_tasks: list[dict]
    failed_tasks: list[dict]
    assets: list[dict]
    web_recon: list[dict]
    agent_memory: dict
    memory_context: dict
    attack_surfaces: list[dict]
    follow_up_tasks: list[dict]
    executed_tasks: list[dict]
    tool_observations: list[dict]
    executed_action_fingerprints: list[str]
    strategy_round: int
    max_rounds: int
    findings: list[dict]
    validation_plans: list[dict]
    test_plans: list[dict]
    evidences: list[dict]
    verification: dict
    report_markdown: str
    approvals_required: list[dict]
    approved_actions: list[dict]
    rejected_actions: list[dict]
    risk_level: str
    next_action: str
    redis_memory_error: str
