from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from string import Formatter
from typing import Any

from autoflow.executor.tool_registry import ToolRegistry
from autoflow.policy.engine import PolicyDecision, PolicyEngine


SHELL_CONTROL_CHARS = re.compile(r"[;&|`$<>\n\r]")


@dataclass(frozen=True)
class CommandSpec:
    tool: str
    profile: str
    command: list[str]
    executor: str
    risk_level: str
    timeout: int
    image: str | None = None
    parser: str | None = None
    policy: PolicyDecision | None = None


class CommandBuilder:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry.from_file()
        self.policy_engine = policy_engine or PolicyEngine.from_file()

    def build(self, intent: dict[str, Any]) -> CommandSpec:
        tool_name = self._require_str(intent, "tool")
        profile_name = self._require_str(intent, "profile")
        args = intent.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Tool intent 'args' must be a dictionary")

        tool, profile = self.registry.get_profile(tool_name, profile_name)
        self._validate_template_args(profile.template, profile.allowed_args, args)
        safe_args = self._sanitize_args(args)

        policy = self.policy_engine.evaluate_tool_intent(
            tool_name=tool.name,
            profile_name=profile.name,
            risk_level=profile.risk or tool.risk,
            action=intent.get("action"),
            approval_granted=bool(intent.get("approval_granted", False)),
        )
        if not policy.allowed:
            raise PermissionError(policy.reason)

        rendered = profile.template.format(**safe_args)
        command = shlex.split(rendered, posix=True)
        if not command:
            raise ValueError("Rendered command is empty")

        return CommandSpec(
            tool=tool.name,
            profile=profile.name,
            command=command,
            executor=tool.executor,
            risk_level=profile.risk or tool.risk,
            timeout=profile.timeout,
            image=tool.image,
            parser=profile.parser,
            policy=policy,
        )

    def _validate_template_args(
        self,
        template: str,
        allowed_args: list[str],
        args: dict[str, Any],
    ) -> None:
        placeholders = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name is not None and field_name != ""
        }
        allowed = set(allowed_args)
        provided = set(args)

        unknown_args = provided - allowed
        if unknown_args:
            raise ValueError(f"Unsupported tool args: {sorted(unknown_args)}")

        undeclared_placeholders = placeholders - allowed
        if undeclared_placeholders:
            raise ValueError(f"Template placeholders are not allowed: {sorted(undeclared_placeholders)}")

        missing_args = placeholders - provided
        if missing_args:
            raise ValueError(f"Missing required tool args: {sorted(missing_args)}")

    def _sanitize_args(self, args: dict[str, Any]) -> dict[str, str]:
        safe_args: dict[str, str] = {}
        for key, value in args.items():
            if not isinstance(value, str):
                raise ValueError(f"Tool arg '{key}' must be a string")
            if value == "":
                raise ValueError(f"Tool arg '{key}' cannot be empty")
            if SHELL_CONTROL_CHARS.search(value):
                raise ValueError(f"Tool arg '{key}' contains unsafe shell control characters")
            safe_args[key] = value
        return safe_args

    def _require_str(self, intent: dict[str, Any], key: str) -> str:
        value = intent.get(key)
        if not isinstance(value, str) or value == "":
            raise ValueError(f"Tool intent requires non-empty string field '{key}'")
        return value
