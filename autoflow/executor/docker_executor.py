from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from autoflow.executor.command_builder import CommandSpec
from autoflow.executor.ssh_executor import CommandResult


CONTAINER_ARTIFACT_DIR = "/work/artifacts"


@dataclass(frozen=True)
class DockerRunSpec:
    docker_command: list[str]
    translated_tool_command: list[str]
    mounted_host_dir: Path | None = None


class DockerExecutor:
    def execute(self, spec: CommandSpec) -> CommandResult:
        run_spec = self.build_docker_run_spec(spec)

        try:
            completed = subprocess.run(
                run_spec.docker_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=spec.timeout,
                check=False,
            )
            return CommandResult(
                command=spec.command,
                command_text=" ".join(run_spec.docker_command),
                exit_code=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=spec.command,
                command_text=" ".join(run_spec.docker_command),
                exit_code=124,
                stdout=self._decode_timeout_output(exc.stdout),
                stderr=self._decode_timeout_output(exc.stderr) or "Docker command timed out",
                timed_out=True,
            )

    def build_docker_run_spec(self, spec: CommandSpec) -> DockerRunSpec:
        if not spec.image:
            raise ValueError(f"Docker executor requires an image for tool '{spec.tool}'")

        translated_command, mounted_host_dir = self._translate_artifact_paths(spec.command)
        docker_command = ["docker", "run", "--rm"]
        if mounted_host_dir is not None:
            docker_command.extend(
                [
                    "-v",
                    f"{mounted_host_dir}:{CONTAINER_ARTIFACT_DIR}",
                ]
            )
        docker_command.append(spec.image)
        docker_command.extend(translated_command)

        return DockerRunSpec(
            docker_command=docker_command,
            translated_tool_command=translated_command,
            mounted_host_dir=mounted_host_dir,
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

    def _decode_timeout_output(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
