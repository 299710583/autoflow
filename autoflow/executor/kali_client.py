from __future__ import annotations

from autoflow.executor.command_builder import CommandBuilder, CommandSpec
from autoflow.executor.ssh_executor import CommandResult, SSHConnectionConfig, SSHExecutor
from autoflow.settings import settings


class KaliClient:
    def __init__(
        self,
        ssh_executor: SSHExecutor | None = None,
        command_builder: CommandBuilder | None = None,
    ) -> None:
        self.ssh_executor = ssh_executor or SSHExecutor(
            SSHConnectionConfig(
                host=settings.kali_host,
                port=settings.kali_port,
                username=settings.kali_username,
                password=settings.kali_password,
                key_path=settings.kali_key_path,
            )
        )
        self.command_builder = command_builder or CommandBuilder()

    def build_command(self, intent: dict) -> CommandSpec:
        return self.command_builder.build(intent)

    def execute(self, intent: dict) -> CommandResult:
        spec = self.build_command(intent)
        if spec.executor != "ssh":
            raise ValueError(f"KaliClient only supports ssh specs, got executor '{spec.executor}'")
        return self.ssh_executor.execute(spec)

    def execute_spec(self, spec: CommandSpec) -> CommandResult:
        if spec.executor != "ssh":
            raise ValueError(f"KaliClient only supports ssh specs, got executor '{spec.executor}'")
        return self.ssh_executor.execute(spec)

    def check_connection(self) -> CommandResult:
        return self.ssh_executor.check_connection()
