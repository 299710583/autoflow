from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from autoflow.graph.state import AutoFlowState
from autoflow.memory.agent_memory import AgentMemoryBuilder
from autoflow.memory.context import MemoryContextBuilder
from autoflow.settings import settings


class RedisMemoryStore:
    """Small Redis-backed runtime memory index for AutoFlow assessments."""

    def __init__(
        self,
        url: str | None = None,
        *,
        enabled: bool | None = None,
        key_prefix: str | None = None,
        ttl_seconds: int | None = None,
        client: Redis | None = None,
    ) -> None:
        self.url = url or settings.redis_url
        self.enabled = settings.redis_enabled if enabled is None else enabled
        self.key_prefix = key_prefix or settings.redis_key_prefix
        self.ttl_seconds = settings.redis_ttl_seconds if ttl_seconds is None else ttl_seconds
        self._client = client
        self.memory_builder = AgentMemoryBuilder()
        self.context_builder = MemoryContextBuilder(memory_builder=self.memory_builder)

    @classmethod
    def from_settings(cls) -> "RedisMemoryStore":
        return cls()

    def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            return bool(self.client.ping())
        except RedisError:
            return False

    def record_node_state(self, node: str, state: AutoFlowState) -> str | None:
        if not self.enabled:
            self.refresh_state_memory(state)
            return None
        flow_id = self._flow_id(state)
        if not flow_id:
            self.refresh_state_memory(state)
            return "missing flow_id"
        try:
            summary = self._state_summary(node, state)
            memory_pack = self.refresh_state_memory(state)
            self.set_json(self._key(flow_id, "latest_state"), summary)
            self.set_json(self._key(flow_id, "memory_pack"), memory_pack)
            self.append_event(
                flow_id,
                {
                    "event_type": "graph_node_completed",
                    "node": node,
                    "current_phase": state.get("current_phase", ""),
                    "next_action": state.get("next_action", ""),
                    "summary": summary,
                },
            )
            self._store_collection(flow_id, "observations", state.get("tool_observations", []), "id")
            self._store_collection(flow_id, "findings", state.get("findings", []), "id")
            self._store_collection(flow_id, "validation_plans", state.get("validation_plans", []), "id")
            return None
        except RedisError as exc:
            return str(exc)
        except TypeError as exc:
            return f"serialization error: {exc}"

    def hydrate_state_memory(self, state: AutoFlowState) -> str | None:
        """Load persisted Redis memory into state before an agent runs."""

        if not self.enabled:
            self.refresh_state_memory(state)
            return None
        flow_id = self._flow_id(state)
        if not flow_id:
            self.refresh_state_memory(state)
            return None
        try:
            persisted = self.get_memory_pack(flow_id)
            self.refresh_state_memory(state, persisted_memory=persisted)
            return None
        except RedisError as exc:
            self.refresh_state_memory(state)
            return str(exc)
        except TypeError as exc:
            self.refresh_state_memory(state)
            return f"serialization error: {exc}"

    def refresh_state_memory(
        self,
        state: AutoFlowState,
        persisted_memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the canonical memory pack and place it on the shared state."""

        base_memory = persisted_memory or state.get("agent_memory")
        memory_pack = self.memory_builder.build(state, persisted_memory=base_memory)
        state["agent_memory"] = memory_pack
        state["memory_context"] = self.context_builder.build(state, persisted_memory=memory_pack)
        return memory_pack

    def append_event(self, flow_id: str, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": self._now(),
            **event,
        }
        key = self._key(flow_id, "events")
        self.client.xadd(key, {"payload": self._json(payload)}, maxlen=1000, approximate=True)
        self._expire(key)

    def set_json(self, key: str, value: Any) -> None:
        self.client.set(key, self._json(value))
        self._expire(key)

    def get_json(self, key: str) -> Any:
        raw = self.client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(str(raw))

    def get_memory_pack(self, flow_id: str) -> dict[str, Any]:
        value = self.get_json(self._key(flow_id, "memory_pack"))
        return value if isinstance(value, dict) else {}

    def get_latest_state_summary(self, flow_id: str) -> dict[str, Any]:
        value = self.get_json(self._key(flow_id, "latest_state"))
        return value if isinstance(value, dict) else {}

    @property
    def client(self) -> Redis:
        if self._client is None:
            self._client = Redis.from_url(self.url, decode_responses=True)
        return self._client

    def _store_collection(self, flow_id: str, name: str, items: list[dict[str, Any]], id_key: str) -> None:
        key = self._key(flow_id, name)
        pipe = self.client.pipeline()
        pipe.delete(key)
        for index, item in enumerate(items[-200:]):
            item_id = str(item.get(id_key) or item.get("action_id") or f"{name}:{index}")
            pipe.hset(key, item_id, self._json(item))
        if self.ttl_seconds > 0:
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def _state_summary(self, node: str, state: AutoFlowState) -> dict[str, Any]:
        flow_id = self._flow_id(state)
        flow = state.get("flow")
        return {
            "flow_id": flow_id,
            "node": node,
            "timestamp": self._now(),
            "current_phase": state.get("current_phase", ""),
            "next_action": state.get("next_action", ""),
            "strategy_round": state.get("strategy_round", 0),
            "max_rounds": state.get("max_rounds", 0),
            "target_scope": state.get("target_scope") or (flow.target_scope if flow else []),
            "counts": {
                "assets": len(state.get("assets", [])),
                "web_recon": len(state.get("web_recon", [])),
                "attack_surfaces": len(state.get("attack_surfaces", [])),
                "test_plans": len(state.get("test_plans", [])),
                "executed_tasks": len(state.get("executed_tasks", [])),
                "tool_observations": len(state.get("tool_observations", [])),
                "findings": len(state.get("findings", [])),
                "validation_plans": len(state.get("validation_plans", [])),
            },
            "latest_findings": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "severity": item.get("severity"),
                    "status": item.get("status"),
                    "target": item.get("target"),
                }
                for item in state.get("findings", [])[-10:]
            ],
        }

    def _flow_id(self, state: AutoFlowState) -> str:
        flow = state.get("flow")
        return str(flow.id if flow else state.get("flow_id", ""))

    def _key(self, flow_id: str, suffix: str) -> str:
        return f"{self.key_prefix}:flow:{flow_id}:{suffix}"

    def _expire(self, key: str) -> None:
        if self.ttl_seconds > 0:
            self.client.expire(key, self.ttl_seconds)

    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()
