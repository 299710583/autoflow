from __future__ import annotations

from autoflow.graph.state import AutoFlowState
from autoflow.memory.agent_memory import AgentMemoryBuilder


class MemoryContextBuilder:
    """Build the shared context passed between planning, strategy, and execution stages."""

    def __init__(self, memory_builder: AgentMemoryBuilder | None = None) -> None:
        self.memory_builder = memory_builder or AgentMemoryBuilder()

    def build(self, state: AutoFlowState, persisted_memory: dict | None = None) -> dict:
        flow = state.get("flow")
        agent_memory = self.memory_builder.build(state, persisted_memory=persisted_memory or state.get("agent_memory"))
        return {
            "project_id": state.get("project_id"),
            "flow_id": flow.id if flow else state.get("flow_id"),
            "target_scope": state.get("target_scope") or (flow.target_scope if flow else []),
            "rules_of_engagement": state.get("rules_of_engagement") or (flow.rules_of_engagement if flow else {}),
            "user_prompt": state.get("user_prompt") or (flow.metadata.get("user_prompt") if flow else ""),
            "agent_memory": agent_memory,
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
