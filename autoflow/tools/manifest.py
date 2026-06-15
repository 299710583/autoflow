from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_MANIFEST_PATH = Path("configs/tool_manifest.yaml")


class ToolManifestRegistry:
    """Loads prompt-facing tool descriptions for LLM reasoning."""

    def __init__(self, path: str | Path = DEFAULT_MANIFEST_PATH) -> None:
        self.path = Path(path)
        self._tools: list[dict[str, Any]] | None = None

    def all(self) -> list[dict[str, Any]]:
        if self._tools is None:
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            tools = payload.get("tools", [])
            self._tools = tools if isinstance(tools, list) else []
        return list(self._tools)

    def for_phase(self, phase: str) -> list[dict[str, Any]]:
        return [tool for tool in self.all() if tool.get("phase") == phase]

    def for_phases(self, phases: set[str] | None = None) -> list[dict[str, Any]]:
        if phases is None:
            return self.all()
        return [tool for tool in self.all() if str(tool.get("phase", "")) in phases]

    def by_profile(self, tool_name: str, profile_name: str) -> list[dict[str, Any]]:
        return [
            tool
            for tool in self.all()
            if str(tool.get("tool", "")) == tool_name and str(tool.get("profile", "")) == profile_name
        ]

    def allowed_profiles(self, phase: str) -> set[tuple[str, str, str]]:
        result = set()
        for tool in self.for_phase(phase):
            result.add(
                (
                    str(tool.get("action_kind", "tool")),
                    str(tool.get("tool", "")),
                    str(tool.get("profile", "")),
                )
            )
        return result

    def prompt_manifest(self, phase: str | set[str] | None) -> list[dict[str, Any]]:
        keys = [
            "phase",
            "tool",
            "profile",
            "action_kind",
            "purpose",
            "input_schema",
            "risk_level",
            "approval_required",
            "best_for",
            "avoid_when",
            "output_summary",
        ]
        if isinstance(phase, str):
            tools = self.for_phase(phase)
        else:
            tools = self.for_phases(phase)
        return [{key: tool.get(key) for key in keys if key in tool} for tool in tools]
