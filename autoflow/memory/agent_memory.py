from __future__ import annotations

from typing import Any

from autoflow.graph.state import AutoFlowState


COLLECTION_LIMIT = 80
OBSERVATION_LIMIT = 120


class AgentMemoryBuilder:
    """Build compact working memory for reasoning agents.

    The builder can merge the current in-process state with a previously
    persisted Redis memory pack. This lets every reasoning agent start from the
    same compact context instead of only seeing data produced in the current
    Python process.
    """

    def build(self, state: AutoFlowState, persisted_memory: dict[str, Any] | None = None) -> dict:
        flow = state.get("flow")
        executed_tasks = state.get("executed_tasks", [])
        current = {
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
        if not persisted_memory:
            return self._with_counts(current)
        return self._with_counts(self._merge_memory(persisted_memory, current))

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

    def _merge_memory(self, persisted: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        result = {
            "project_id": current.get("project_id") or persisted.get("project_id"),
            "flow_id": current.get("flow_id") or persisted.get("flow_id"),
            "target_scope": self._merge_scalars(persisted.get("target_scope", []), current.get("target_scope", [])),
            "user_prompt": current.get("user_prompt") or persisted.get("user_prompt", ""),
            "rules_of_engagement": {
                **self._dict_or_empty(persisted.get("rules_of_engagement")),
                **self._dict_or_empty(current.get("rules_of_engagement")),
            },
            "known_assets": self._merge_dicts(
                persisted.get("known_assets", []),
                current.get("known_assets", []),
                key_fields=("ip", "hostname"),
            ),
            "web_context": self._merge_dicts(
                persisted.get("web_context", []),
                current.get("web_context", []),
                key_fields=("target",),
            ),
            "attack_surfaces": self._merge_dicts(
                persisted.get("attack_surfaces", []),
                current.get("attack_surfaces", []),
                key_fields=("id", "target", "surface_type"),
            ),
            "candidate_findings": self._merge_dicts(
                persisted.get("candidate_findings", []),
                current.get("candidate_findings", []),
                key_fields=("id", "title", "target"),
            ),
            "validation_plans": self._merge_dicts(
                persisted.get("validation_plans", []),
                current.get("validation_plans", []),
                key_fields=("id", "finding_id", "target"),
            ),
            "tool_observations": self._merge_dicts(
                persisted.get("tool_observations", []),
                current.get("tool_observations", []),
                key_fields=("id", "tool", "profile", "target", "summary"),
                limit=OBSERVATION_LIMIT,
            ),
            "executed_actions": self._merge_dicts(
                persisted.get("executed_actions", []),
                current.get("executed_actions", []),
                key_fields=("action_id", "tool", "profile", "target"),
            ),
            "failed_actions": self._merge_dicts(
                persisted.get("failed_actions", []),
                current.get("failed_actions", []),
                key_fields=("action_id", "tool", "profile", "target", "error"),
            ),
            "approvals_required": self._merge_dicts(
                persisted.get("approvals_required", []),
                current.get("approvals_required", []),
                key_fields=("action_id", "tool", "profile", "target"),
            ),
            "do_not_repeat": self._merge_scalars(persisted.get("do_not_repeat", []), current.get("do_not_repeat", [])),
            "flow_memories": self._merge_dicts(
                persisted.get("flow_memories", []),
                current.get("flow_memories", []),
                key_fields=("id", "source", "content"),
            ),
        }
        return result

    def _with_counts(self, memory: dict[str, Any]) -> dict[str, Any]:
        counts = {
            "known_assets": len(memory.get("known_assets", [])),
            "web_context": len(memory.get("web_context", [])),
            "attack_surfaces": len(memory.get("attack_surfaces", [])),
            "candidate_findings": len(memory.get("candidate_findings", [])),
            "validation_plans": len(memory.get("validation_plans", [])),
            "tool_observations": len(memory.get("tool_observations", [])),
            "executed_actions": len(memory.get("executed_actions", [])),
            "failed_actions": len(memory.get("failed_actions", [])),
            "approvals_required": len(memory.get("approvals_required", [])),
        }
        return {**memory, "memory_counts": counts}

    def _merge_scalars(self, previous: Any, current: Any, limit: int = COLLECTION_LIMIT) -> list[Any]:
        result = []
        for item in [*self._list_or_empty(previous), *self._list_or_empty(current)]:
            if item in result:
                continue
            result.append(item)
        return result[-limit:]

    def _merge_dicts(
        self,
        previous: Any,
        current: Any,
        *,
        key_fields: tuple[str, ...],
        limit: int = COLLECTION_LIMIT,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        order: list[tuple[Any, ...]] = []
        for item in [*self._dict_list_or_empty(previous), *self._dict_list_or_empty(current)]:
            key = self._dict_key(item, key_fields)
            if key not in merged:
                order.append(key)
                merged[key] = item
                continue
            merged[key] = self._merge_dict_values(merged[key], item)
        return [merged[key] for key in order[-limit:]]

    def _dict_key(self, item: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
        values = tuple(item.get(field) for field in fields if item.get(field) not in (None, ""))
        if values:
            return values
        return (str(item),)

    def _merge_dict_values(self, previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        merged = dict(previous)
        for key, value in current.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = self._merge_scalars(merged[key], value)
            elif isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    def _dict_or_empty(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _list_or_empty(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def _dict_list_or_empty(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]
