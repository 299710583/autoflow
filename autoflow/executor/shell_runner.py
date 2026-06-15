from __future__ import annotations

import shlex
from pathlib import Path

from autoflow.executor.container_manager import CONTAINER_ARTIFACT_DIR, TaskContainerManager
from autoflow.executor.script_runner import SCRIPT_IMAGE
from autoflow.executor.shell_policy import ShellPolicy
from autoflow.executor.ssh_executor import CommandResult


class ShellRunner:
    """Run bounded bash snippets inside a disposable Docker container."""

    def __init__(self, policy: ShellPolicy | None = None, image: str = SCRIPT_IMAGE) -> None:
        self.policy = policy or ShellPolicy()
        self.image = image

    def run_command(
        self,
        command: str,
        target: str,
        target_scope: list[str],
        artifact_dir: Path,
        timeout: int = 120,
        policy_profile: str = "container_lab_shell",
    ) -> CommandResult:
        decision = self.policy.evaluate(command, target, target_scope, profile_name=policy_profile)
        if not decision.allowed:
            return CommandResult(
                command=["bash", "shell.sh"],
                command_text="shell_policy_check",
                exit_code=126,
                stdout="",
                stderr=decision.reason,
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        script_path = artifact_dir / "shell.sh"
        script_path.write_text(self._render_script(command, target), encoding="utf-8", newline="\n")

        with TaskContainerManager(self.image, artifact_dir=artifact_dir) as container:
            return container.exec(["bash", f"{CONTAINER_ARTIFACT_DIR}/shell.sh"], timeout=timeout)

    def _render_script(self, command: str, target: str) -> str:
        return "\n".join(
            [
                "set -eu",
                f"TARGET={shlex.quote(target)}",
                f"ARTIFACT_DIR={shlex.quote(CONTAINER_ARTIFACT_DIR)}",
                command.strip(),
                "",
            ]
        )
