from __future__ import annotations

from autoflow.executor.command_builder import CommandBuilder, CommandSpec
from autoflow.executor.docker_executor import DockerExecutor
from autoflow.executor.ssh_executor import CommandResult, SSHConnectionConfig, SSHExecutor
from autoflow.executor.task_container_executor import TaskContainerExecutor
from autoflow.settings import settings


class ExecutionClient:
    def __init__(
        self,
        command_builder: CommandBuilder | None = None,
        docker_executor: DockerExecutor | None = None,
        task_container_executor: TaskContainerExecutor | None = None,
        ssh_executor: SSHExecutor | None = None,
    ) -> None:
        self.command_builder = command_builder or CommandBuilder()
        self.docker_executor = docker_executor or DockerExecutor()
        self.task_container_executor = task_container_executor or TaskContainerExecutor()
        self.ssh_executor = ssh_executor or SSHExecutor(
            SSHConnectionConfig(
                host=settings.kali_host,
                port=settings.kali_port,
                username=settings.kali_username,
                password=settings.kali_password,
                key_path=settings.kali_key_path,
            )
        )

    def build_command(self, intent: dict) -> CommandSpec:
        return self.command_builder.build(intent)

    def execute(self, intent: dict) -> CommandResult:
        return self.execute_spec(self.build_command(intent))

    def execute_spec(self, spec: CommandSpec) -> CommandResult:
        if spec.executor == "docker":
            return self.task_container_executor.execute(spec)
        if spec.executor == "docker_run":
            return self.docker_executor.execute(spec)
        if spec.executor == "ssh":
            return self.ssh_executor.execute(spec)
        raise ValueError(f"Unsupported executor '{spec.executor}'")
