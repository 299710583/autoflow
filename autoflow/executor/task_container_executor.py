from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from autoflow.executor.command_builder import CommandSpec
from autoflow.executor.container_manager import CONTAINER_ARTIFACT_DIR, TaskContainerManager
from autoflow.executor.ssh_executor import CommandResult
from autoflow.executor.tool_installer import ToolInstallRegistry


class TaskContainerExecutor:
    """Run a command inside a disposable container and install missing whitelisted tools."""

    def __init__(self, install_registry: ToolInstallRegistry | None = None) -> None:
        self.install_registry = install_registry or ToolInstallRegistry.from_file()

    def execute(self, spec: CommandSpec) -> CommandResult:
        if not spec.image:
            raise ValueError(f"Task container executor requires an image for tool '{spec.tool}'")

        translated_command, mounted_host_dir = self._translate_artifact_paths(spec.command)
        with TaskContainerManager(spec.image, artifact_dir=mounted_host_dir) as container:
            tool_name = translated_command[0]
            if not self._tool_exists(container, tool_name):
                install_result = self._install_tool(container, tool_name, timeout=max(spec.timeout, 600))
                if not install_result.succeeded:
                    return CommandResult(
                        command=spec.command,
                        command_text=install_result.command_text,
                        exit_code=install_result.exit_code,
                        stdout=install_result.stdout,
                        stderr=install_result.stderr or f"Failed to install missing tool '{tool_name}'",
                    )
            return container.exec(translated_command, timeout=spec.timeout)

    def _tool_exists(self, container: TaskContainerManager, tool: str) -> bool:
        result = container.exec(["bash", "-lc", f"command -v {tool}"], timeout=30)
        return result.succeeded

    def _install_tool(self, container: TaskContainerManager, tool: str, timeout: int) -> CommandResult:
        spec = self.install_registry.get(tool)
        if spec is None:
            return CommandResult(
                command=[tool],
                command_text=f"install {tool}",
                exit_code=127,
                stdout="",
                stderr=f"Tool '{tool}' is missing and is not in the install allowlist",
            )
        if spec.method != "apt":
            return CommandResult(
                command=[tool],
                command_text=f"install {tool}",
                exit_code=126,
                stdout="",
                stderr=f"Unsupported install method '{spec.method}' for tool '{tool}'",
            )

        return container.exec(
            [
                "bash",
                "-lc",
                f"apt-get update && apt-get install -y --no-install-recommends {spec.package}",
            ],
            timeout=timeout,
        )

    def _translate_artifact_paths(self, command: list[str]) -> tuple[list[str], Path | None]:
        translated: list[str] = []
        mounted_host_dir: Path | None = None

        for arg in command:
            if self._looks_like_local_path(arg):
                local_path = Path(arg).resolve()
                if mounted_host_dir is None:
                    mounted_host_dir = local_path.parent
                if local_path.parent == mounted_host_dir:
                    translated.append(f"{CONTAINER_ARTIFACT_DIR}/{local_path.name}")
                    continue
            translated.append(arg)

        return translated, mounted_host_dir

    def _looks_like_local_path(self, value: str) -> bool:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return False
        if value.startswith("/") and not Path(value).drive:
            return False
        if "/" in value or "\\" in value:
            return True
        return bool(Path(value).drive)
