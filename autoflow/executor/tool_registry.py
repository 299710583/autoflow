from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ToolProfile(BaseModel):
    name: str
    template: str
    allowed_args: list[str] = Field(default_factory=list)
    timeout: int = 300
    parser: str | None = None
    risk: str | None = None
    description: str | None = None
    target_required: bool = True


class ToolDefinition(BaseModel):
    name: str
    enabled: bool = True
    risk: str = "low"
    executor: str = "ssh"
    image: str | None = None
    profiles: dict[str, ToolProfile] = Field(default_factory=dict)

    def get_profile(self, profile_name: str) -> ToolProfile:
        try:
            profile = self.profiles[profile_name]
        except KeyError as exc:
            raise KeyError(f"Unknown profile '{profile_name}' for tool '{self.name}'") from exc

        if profile.risk is None:
            profile.risk = self.risk
        return profile


class ToolRegistry:
    def __init__(self, tools: dict[str, ToolDefinition]) -> None:
        self.tools = tools

    @classmethod
    def from_file(cls, path: str | Path = "configs/tools.yaml") -> "ToolRegistry":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}
        return cls.from_dict(raw_config)

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> "ToolRegistry":
        raw_tools = raw_config.get("tools", raw_config)
        tools: dict[str, ToolDefinition] = {}

        for tool_name, raw_tool in raw_tools.items():
            raw_profiles = raw_tool.get("profiles", {})
            profiles = {
                profile_name: ToolProfile(name=profile_name, **raw_profile)
                for profile_name, raw_profile in raw_profiles.items()
            }
            tools[tool_name] = ToolDefinition(
                name=tool_name,
                enabled=raw_tool.get("enabled", True),
                risk=raw_tool.get("risk", "low"),
                executor=raw_tool.get("executor", "ssh"),
                image=raw_tool.get("image"),
                profiles=profiles,
            )

        return cls(tools)

    def get_tool(self, tool_name: str) -> ToolDefinition:
        try:
            tool = self.tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool '{tool_name}'") from exc

        if not tool.enabled:
            raise ValueError(f"Tool '{tool_name}' is disabled")
        return tool

    def get_profile(self, tool_name: str, profile_name: str) -> tuple[ToolDefinition, ToolProfile]:
        tool = self.get_tool(tool_name)
        return tool, tool.get_profile(profile_name)
