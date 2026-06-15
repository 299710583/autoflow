from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ToolInstallSpec:
    tool: str
    method: str
    package: str
    risk: str


class ToolInstallRegistry:
    def __init__(self, specs: dict[str, ToolInstallSpec]) -> None:
        self.specs = specs

    @classmethod
    def from_file(cls, path: str | Path = "configs/tool_installs.yaml") -> "ToolInstallRegistry":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}
        return cls.from_dict(raw_config)

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> "ToolInstallRegistry":
        specs: dict[str, ToolInstallSpec] = {}
        for tool, raw_spec in raw_config.get("tools", raw_config).items():
            specs[tool] = ToolInstallSpec(
                tool=tool,
                method=raw_spec.get("method", "apt"),
                package=raw_spec["package"],
                risk=raw_spec.get("risk", "medium"),
            )
        return cls(specs)

    def get(self, tool: str) -> ToolInstallSpec | None:
        return self.specs.get(tool)
