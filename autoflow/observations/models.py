from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from autoflow.flows.models import new_id, utc_now


class ToolObservation(BaseModel):
    """一次工具执行后的结构化观察结果，对应 ReAct 里的 Observation。"""

    id: str = Field(default_factory=lambda: new_id("observation"))
    action_id: str = ""
    plan_id: str = ""
    tool: str
    profile: str = ""
    target: str = ""
    status: str = "completed"
    risk_level: str = "low"
    summary: str = ""
    raw_result: str = ""
    stderr: str = ""
    artifact_id: str | None = None
    signals: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Any = Field(default_factory=utc_now)
