from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from autoflow.executor.ssh_executor import CommandResult


CONTAINER_WORKDIR = "/work"
CONTAINER_ARTIFACT_DIR = "/work/artifacts"


@dataclass(frozen=True)
class ContainerExecResult:
    result: CommandResult
    container_id: str


class TaskContainerManager:
    """Create short-lived Docker containers that behave like a clean task snapshot."""

    def __init__(self, image: str, artifact_dir: Path | None = None) -> None:
        self.image = image
        self.artifact_dir = artifact_dir
        self.container_id: str | None = None

    def __enter__(self) -> "TaskContainerManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def start(self) -> str:
        if self.container_id:
            return self.container_id

        command = ["docker", "create", "-w", CONTAINER_WORKDIR]
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            command.extend(["-v", f"{self.artifact_dir.resolve()}:{CONTAINER_ARTIFACT_DIR}"])
        command.extend([self.image, "sleep", "infinity"])
        created = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if created.returncode != 0:
            raise RuntimeError(created.stderr.strip() or "Failed to create task container")

        self.container_id = created.stdout.strip()
        started = subprocess.run(
            ["docker", "start", self.container_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if started.returncode != 0:
            container_id = self.container_id
            self.container_id = None
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            raise RuntimeError(started.stderr.strip() or "Failed to start task container")
        return self.container_id

    def exec(self, command: list[str], timeout: int = 300) -> CommandResult:
        container_id = self.start()
        docker_command = ["docker", "exec", container_id, *command]
        try:
            completed = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                command=command,
                command_text=" ".join(docker_command),
                exit_code=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                command_text=" ".join(docker_command),
                exit_code=124,
                stdout=self._decode_timeout_output(exc.stdout),
                stderr=self._decode_timeout_output(exc.stderr) or "Docker exec timed out",
                timed_out=True,
            )

    def cleanup(self) -> None:
        if not self.container_id:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.container_id = None

    def _decode_timeout_output(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
