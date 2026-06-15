from __future__ import annotations

from dataclasses import dataclass

from autoflow.flows.models import ActionStatus, SubTask
from autoflow.supervision.limits import SupervisionLimits


@dataclass(frozen=True)
class SupervisionDecision:
    allowed: bool
    reason: str = ""
    require_reflection: bool = False


class SupervisionMonitor:
    def __init__(self, limits: SupervisionLimits | None = None) -> None:
        self.limits = limits or SupervisionLimits()

    def evaluate_subtask(self, subtask: SubTask) -> SupervisionDecision:
        if len(subtask.actions) >= self.limits.total_action_limit:
            return SupervisionDecision(
                allowed=False,
                reason="total_action_limit_reached",
                require_reflection=self.limits.require_reflection,
            )

        same_tool_count = self._latest_same_tool_count(subtask)
        if same_tool_count >= self.limits.same_tool_limit:
            return SupervisionDecision(
                allowed=False,
                reason="same_tool_limit_reached",
                require_reflection=self.limits.require_reflection,
            )

        no_progress_count = self._latest_no_progress_count(subtask)
        if no_progress_count >= self.limits.no_progress_limit:
            return SupervisionDecision(
                allowed=False,
                reason="no_progress_limit_reached",
                require_reflection=self.limits.require_reflection,
            )

        return SupervisionDecision(allowed=True)

    def _latest_same_tool_count(self, subtask: SubTask) -> int:
        if not subtask.actions:
            return 0

        latest_tool = subtask.actions[-1].tool
        count = 0
        for action in reversed(subtask.actions):
            if action.tool != latest_tool:
                break
            count += 1
        return count

    def _latest_no_progress_count(self, subtask: SubTask) -> int:
        count = 0
        for action in reversed(subtask.actions):
            has_progress = bool(action.artifacts or action.result_summary)
            if action.status == ActionStatus.SUCCEEDED and has_progress:
                break
            count += 1
        return count

