from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""
    requires_approval: bool = False


class PolicyEngine:
    def __init__(
        self,
        default_allowed_risk: set[str] | None = None,
        require_approval: set[str] | None = None,
        forbidden_actions: set[str] | None = None,
    ) -> None:
        self.default_allowed_risk = default_allowed_risk or {"low"}
        self.require_approval = require_approval or {"medium", "high", "critical"}
        self.forbidden_actions = forbidden_actions or set()

    @classmethod
    def from_file(cls, path: str | Path = "configs/policy.yaml") -> "PolicyEngine":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}
        return cls.from_dict(raw_config)

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> "PolicyEngine":
        raw_policy = raw_config.get("policy", raw_config)
        return cls(
            default_allowed_risk=set(raw_policy.get("default_allowed_risk", ["low"])),
            require_approval=set(raw_policy.get("require_approval", ["medium", "high", "critical"])),
            forbidden_actions=set(raw_policy.get("forbidden_actions", [])),
        )

    def evaluate_tool_intent(
        self,
        tool_name: str,
        profile_name: str,
        risk_level: str,
        action: str | None = None,
        approval_granted: bool = False,
    ) -> PolicyDecision:
        if action and action in self.forbidden_actions:
            return PolicyDecision(
                allowed=False,
                reason=f"Action '{action}' is forbidden by policy",
            )

        risk = risk_level.lower()
        if risk in self.default_allowed_risk:
            return PolicyDecision(allowed=True)

        if risk in self.require_approval:
            if approval_granted:
                return PolicyDecision(
                    allowed=True,
                    reason=(
                        f"Tool '{tool_name}' profile '{profile_name}' has risk '{risk}' "
                        "and was approved"
                    ),
                )
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' profile '{profile_name}' has risk '{risk}' "
                    "and requires approval"
                ),
                requires_approval=True,
            )

        return PolicyDecision(
            allowed=False,
            reason=f"Tool '{tool_name}' profile '{profile_name}' has unsupported risk '{risk}'",
        )
