from __future__ import annotations

from autoflow.flows.models import AssessmentFlow, AssessmentTask, MemoryItem


class FlowService:
    def create_flow(
        self,
        name: str,
        target_scope: list[str],
        rules_of_engagement: dict | None = None,
    ) -> AssessmentFlow:
        return AssessmentFlow(
            name=name,
            target_scope=target_scope,
            rules_of_engagement=rules_of_engagement or {},
        )

    def add_task(self, flow: AssessmentFlow, task: AssessmentTask) -> AssessmentTask:
        return flow.add_task(task)

    def add_memory(self, flow: AssessmentFlow, memory: MemoryItem) -> MemoryItem:
        return flow.add_memory(memory)

