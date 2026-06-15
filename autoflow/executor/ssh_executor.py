from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from autoflow.executor.command_builder import CommandSpec


@dataclass(frozen=True)
class SSHConnectionConfig:
    host: str
    port: int = 22
    username: str = "kali"
    password: str = ""
    key_path: str = ""
    timeout: int = 10


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    command_text: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class SSHExecutor:
    def __init__(self, config: SSHConnectionConfig) -> None:
        self.config = config

    def execute(self, spec: CommandSpec) -> CommandResult:
        command_text = self.format_command(spec.command)
        return self.execute_command(command=spec.command, command_text=command_text, timeout=spec.timeout)

    def execute_command(
        self,
        command: list[str],
        command_text: str | None = None,
        timeout: int = 300,
    ) -> CommandResult:
        command_text = command_text or self.format_command(command)

        try:
            import paramiko
        except ImportError as exc:
            raise RuntimeError(
                "Paramiko is required for SSH execution. Install project dependencies in the "
                "qwen-skills environment before checking Kali connectivity."
            ) from exc

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "timeout": self.config.timeout,
            "banner_timeout": self.config.timeout,
            "auth_timeout": self.config.timeout,
        }
        if self.config.key_path:
            connect_kwargs["key_filename"] = str(Path(self.config.key_path))
        elif self.config.password:
            connect_kwargs["password"] = self.config.password

        try:
            client.connect(**connect_kwargs)
            _, stdout, stderr = client.exec_command(command_text, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode("utf-8", errors="replace")
            stderr_text = stderr.read().decode("utf-8", errors="replace")
            return CommandResult(
                command=command,
                command_text=command_text,
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
            )
        finally:
            client.close()

    def check_connection(self) -> CommandResult:
        return self.execute_command(["whoami"], timeout=30)

    def format_command(self, command: list[str]) -> str:
        if not command:
            raise ValueError("Command cannot be empty")
        return shlex.join(command)
