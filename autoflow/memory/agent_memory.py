from __future__ import annotations

from autoflow.graph.state import AutoFlowState


class AgentMemoryBuilder:
    """Builds compact working memory for reasoning agents."""

    def build(self, state: AutoFlowState) -> dict:
        flow = state.get("flow")
        executed_tasks = state.get("executed_tasks", [])
        return {
            "project_id": state.get("project_id"),
            "flow_id": flow.id if flow else state.get("flow_id"),
            "target_scope": state.get("target_scope") or (flow.target_scope if flow else []),
            "user_prompt": state.get("user_prompt") or (flow.metadata.get("user_prompt") if flow else ""),
            "rules_of_engagement": state.get("rules_of_engagement") or (flow.rules_of_engagement if flow else {}),
            "known_assets": state.get("assets", []),
            "web_context": state.get("web_recon", []),
            "attack_surfaces": state.get("attack_surfaces", []),
            "candidate_findings": [
                finding for finding in state.get("findings", []) if finding.get("status", "candidate") == "candidate"
            ],
            "validation_plans": state.get("validation_plans", []),
            "tool_observations": [
                {
                    "tool": item.get("tool"),
                    "profile": item.get("profile"),
                    "target": item.get("target"),
                    "status": item.get("status"),
                    "summary": item.get("summary", ""),
                    "signals": item.get("signals", [])[:20],
                }
                for item in state.get("tool_observations", [])
            ],
            "executed_actions": [
                self._task_summary(item) for item in executed_tasks if item.get("status") == "completed"
            ],
            "failed_actions": [
                self._task_summary(item) for item in executed_tasks if item.get("status") == "failed"
            ],
            "approvals_required": state.get("approvals_required", []),
            "do_not_repeat": state.get("executed_action_fingerprints", []),
            "flow_memories": [
                memory.model_dump(mode="json")
                for memory in flow.memories
            ]
            if flow
            else [],
        }

    def _task_summary(self, item: dict) -> dict:
        task = item.get("task", {})
        return {
            "action_id": item.get("action_id"),
            "status": item.get("status"),
            "tool": task.get("tool"),
            "profile": task.get("profile"),
            "target": task.get("target"),
            "summary": item.get("summary", ""),
            "error": item.get("error", ""),
        }
