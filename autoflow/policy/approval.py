from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRequest:
    action_id: str
    plan_id: str = ""
    target: str = ""
    risk_level: str = "medium"
    action_kind: str = "tool"
    tool: str = ""
    profile: str = ""
    rationale: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(default_factory=utc_now)
    decided_at: datetime | None = None
    decided_by: str = ""
    decision_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_action(cls, action: dict[str, Any]) -> "ApprovalRequest":
        return cls(
            action_id=action.get("action_id", ""),
            plan_id=action.get("plan_id", ""),
            target=action.get("target", ""),
            risk_level=action.get("risk_level", "medium"),
            action_kind=action.get("action_kind", "tool"),
            tool=action.get("tool", ""),
            profile=action.get("profile", ""),
            rationale=action.get("rationale", ""),
            metadata={"action": action},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "plan_id": self.plan_id,
            "target": self.target,
            "risk_level": self.risk_level,
            "action_kind": self.action_kind,
            "tool": self.tool,
            "profile": self.profile,
            "rationale": self.rationale,
            "status": self.status.value,
            "requested_at": self.requested_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decided_by": self.decided_by,
            "decision_reason": self.decision_reason,
            "metadata": self.metadata,
        }

    def approve(self, decided_by: str = "", reason: str = "") -> None:
        self.status = ApprovalStatus.APPROVED
        self.decided_at = utc_now()
        self.decided_by = decided_by
        self.decision_reason = reason

    def reject(self, decided_by: str = "", reason: str = "") -> None:
        self.status = ApprovalStatus.REJECTED
        self.decided_at = utc_now()
        self.decided_by = decided_by
        self.decision_reason = reason


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, ApprovalRequest] = {}

    def upsert_from_action(self, action: dict[str, Any]) -> ApprovalRequest:
        action_id = action.get("action_id", "")
        if not action_id:
            raise ValueError("Approval action requires action_id")
        if action_id not in self._items:
            self._items[action_id] = ApprovalRequest.from_action(action)
        return self._items[action_id]

    def list(self, status: ApprovalStatus | None = None) -> list[ApprovalRequest]:
        items = list(self._items.values())
        if status is None:
            return items
        return [item for item in items if item.status == status]

    def get(self, action_id: str) -> ApprovalRequest:
        try:
            return self._items[action_id]
        except KeyError as exc:
            raise KeyError(f"Unknown approval action '{action_id}'") from exc

    def approve(self, action_id: str, decided_by: str = "", reason: str = "") -> ApprovalRequest:
        item = self.get(action_id)
        item.approve(decided_by=decided_by, reason=reason)
        return item

    def reject(self, action_id: str, decided_by: str = "", reason: str = "") -> ApprovalRequest:
        item = self.get(action_id)
        item.reject(decided_by=decided_by, reason=reason)
        return item

    def clear(self) -> None:
        self._items.clear()


approval_store = InMemoryApprovalStore()
