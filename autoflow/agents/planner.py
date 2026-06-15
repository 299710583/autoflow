from __future__ import annotations

import json
from typing import Any

from autoflow.agents.base import BaseAgent
from autoflow.flows.models import AssessmentFlow, AssessmentTask, FlowStatus, RiskLevel
from autoflow.graph.state import AutoFlowState
from autoflow.llm.client import LLMClient
from autoflow.llm.client import parse_json_object
from autoflow.settings import settings


ALLOWED_PLANNER_TASK_TYPES = {"recon"}
ALLOWED_PLANNER_RISKS = {"low"}


PLANNER_SYSTEM_PROMPT = """You are the planning agent for AutoFlow, an authorized security assessment framework.
Return only a compact JSON object. Do not include markdown.
You may only plan low-risk reconnaissance tasks.
Do not plan exploitation, brute force, privilege escalation, lateral movement, persistence, destructive writes, or evasion.
"""


class PlannerAgent(BaseAgent):
    """为一次评估流程创建初始低风险任务计划。"""

    name = "planner"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_llm: bool | None = None,
        json_repair_attempts: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.json_repair_attempts = json_repair_attempts

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "planning"
        flow = state.get("flow")

        # 当图从原始目标启动时，按需创建 Flow。
        if flow is None:
            target_scope = state.get("target_scope", [])
            if not target_scope:
                raise ValueError("PlannerAgent requires state['target_scope'] when no flow exists")
            flow = AssessmentFlow(
                name=state.get("project_id", "autoflow-assessment"),
                target_scope=target_scope,
                rules_of_engagement=state.get("rules_of_engagement", {}),
                status=FlowStatus.PLANNING,
            )
            state["flow"] = flow
            state["flow_id"] = flow.id

        # LLM 输出只作为候选计划，进入 Flow 前必须经过过滤。
        planned_tasks = self._plan_tasks(flow)
        if planned_tasks:
            added_count = self._add_planned_tasks(flow, planned_tasks)
            if added_count == 0:
                self._ensure_recon_tasks(flow)
        else:
            self._ensure_recon_tasks(flow)

        flow.status = FlowStatus.RUNNING
        state["task_queue"] = [
            {
                "id": task.id,
                "type": task.type,
                "target": task.target,
                "risk_level": task.risk_level.value,
                "status": task.status.value,
            }
            for task in flow.tasks
        ]
        state["next_action"] = "recon"
        return state

    def _plan_tasks(self, flow: AssessmentFlow) -> list[dict[str, Any]]:
        if not self._should_use_llm():
            return []

        # 收窄提示词输出范围：模型只能提出低风险 recon 任务。
        client = self.llm_client or LLMClient()
        prompt = {
            "target_scope": flow.target_scope,
            "rules_of_engagement": flow.rules_of_engagement,
            "allowed_task_types": sorted(ALLOWED_PLANNER_TASK_TYPES),
            "allowed_risk_levels": sorted(ALLOWED_PLANNER_RISKS),
            "output_schema": {
                "tasks": [
                    {
                        "type": "recon",
                        "target": "target from target_scope",
                        "objective": "short objective",
                        "risk_level": "low",
                        "priority": 10,
                    }
                ]
            },
        }
        response = self._complete_json_with_repair(
            client=client,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            max_tokens=1024,
        )

        tasks = response.get("tasks", [])
        return tasks if isinstance(tasks, list) else []

    def _complete_json_with_repair(
        self,
        *,
        client: LLMClient,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        max_attempts = max(1, self.json_repair_attempts + 1)
        for attempt in range(max_attempts):
            content = client.complete_messages(messages=messages, max_tokens=max_tokens)
            try:
                return parse_json_object(content)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt >= max_attempts - 1:
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed as JSON. "
                            f"Parser error: {exc}. Return exactly one JSON object with a tasks array. "
                            "Do not include markdown fences, comments, analysis text, apologies, or explanations."
                        ),
                    }
                )
        raise ValueError(f"PlannerAgent failed to obtain valid JSON after {max_attempts} attempts: {last_error}")

    def _should_use_llm(self) -> bool:
        if self.use_llm is not None:
            return self.use_llm
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required for PlannerAgent. Set use_llm=False for offline tests.")
        return True

    def _add_planned_tasks(self, flow: AssessmentFlow, planned_tasks: list[dict[str, Any]]) -> int:
        existing = {(task.type, task.target) for task in flow.tasks}
        added_count = 0
        for item in planned_tasks:
            task_type = item.get("type")
            target = item.get("target")
            risk_level = item.get("risk_level", "low")
            if task_type not in ALLOWED_PLANNER_TASK_TYPES:
                continue
            if risk_level not in ALLOWED_PLANNER_RISKS:
                continue
            if target not in flow.target_scope:
                continue
            if (task_type, target) in existing:
                continue

            # 只有过滤通过、在授权范围内的低风险任务才会进入可执行 Flow。
            flow.add_task(
                AssessmentTask(
                    type=task_type,
                    target=target,
                    objective=item.get("objective") or f"Discover open services for {target}",
                    risk_level=RiskLevel.LOW,
                    priority=int(item.get("priority", 10)),
                )
            )
            existing.add((task_type, target))
            added_count += 1
        return added_count
    
    def _ensure_recon_tasks(self, flow: AssessmentFlow) -> None:
        existing = {(task.type, task.target) for task in flow.tasks}
        for target in flow.target_scope:
            if ("recon", target) in existing:
                continue
            flow.add_task(
                AssessmentTask(
                    type="recon",
                    target=target,
                    objective=f"Discover open services for {target}",
                    risk_level=RiskLevel.LOW,
                    priority=10,
                )
            )
