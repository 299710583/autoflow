from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApprovalDecisionRequest(BaseModel):
    decided_by: str = "user"
    reason: str = ""


class ApprovalRequestRead(BaseModel):
    action_id: str
    plan_id: str = ""
    target: str = ""
    risk_level: str = "medium"
    action_kind: str = "tool"
    tool: str = ""
    profile: str = ""
    rationale: str = ""
    status: str
    requested_at: str
    decided_at: str | None = None
    decided_by: str = ""
    decision_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
