from __future__ import annotations

from autoflow.graph.state import AutoFlowState


class MemoryContextBuilder:
    """Build the shared context passed between planning, strategy, and execution stages."""

    def build(self, state: AutoFlowState) -> dict:
        flow = state.get("flow")
        return {
            "project_id": state.get("project_id"),
            "flow_id": flow.id if flow else state.get("flow_id"),
            "target_scope": state.get("target_scope") or (flow.target_scope if flow else []),
            "rules_of_engagement": state.get("rules_of_engagement") or (flow.rules_of_engagement if flow else {}),
            "user_prompt": state.get("user_prompt") or (flow.metadata.get("user_prompt") if flow else ""),
            "assets": state.get("assets", []),
            "web_recon": state.get("web_recon", []),
            "attack_surfaces": state.get("attack_surfaces", []),
            "findings": state.get("findings", []),
            "test_plans": state.get("test_plans", []),
            "executed_tasks": state.get("executed_tasks", []),
            "tool_observations": state.get("tool_observations", []),
            "approvals_required": state.get("approvals_required", []),
            "approved_actions": state.get("approved_actions", []),
            "rejected_actions": state.get("rejected_actions", []),
            "flow_memories": [
                memory.model_dump(mode="json")
                for memory in flow.memories
            ]
            if flow
            else [],
        }
