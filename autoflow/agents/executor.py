from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from autoflow.agents.base import BaseAgent
from autoflow.artifacts.store import ArtifactStore
from autoflow.executor.execution_client import ExecutionClient
from autoflow.executor.script_author import ScriptAuthor
from autoflow.executor.script_runner import ScriptRunner
from autoflow.executor.shell_runner import ShellRunner
from autoflow.executor.web_recon import WebReconClient
from autoflow.flows.models import (
    Action,
    ArtifactType,
    AssessmentTask,
    FindingConfidence,
    FindingStatus,
    MemoryItem,
    MemoryKind,
    RiskLevel,
    SubTask,
    TaskStatus,
    ValidationResult,
    ValidationResultStatus,
)
from autoflow.graph.state import AutoFlowState
from autoflow.observations.parser import ToolObservationParser
from autoflow.policy.approval import ApprovalStatus, approval_store
from autoflow.runtime.actions import action_fingerprint


TASK_TOOL_PROFILES = {
    ("web_fingerprint", "whatweb"): "web_fingerprint",
}


class ExecutorAgent(BaseAgent):
    """通过工具执行层运行已允许的低风险后续任务。"""

    name = "executor"
    current_phase = "execution"
    next_action_after_run = "verify"

    def __init__(
        self,
        execution_client: ExecutionClient | None = None,
        artifact_store: ArtifactStore | None = None,
        script_runner: ScriptRunner | None = None,
        script_author: ScriptAuthor | None = None,
        shell_runner: ShellRunner | None = None,
        observation_parser: ToolObservationParser | None = None,
        web_recon_client: WebReconClient | None = None,
    ) -> None:
        self.execution_client = execution_client or ExecutionClient()
        self.artifact_store = artifact_store or ArtifactStore()
        self.script_runner = script_runner or ScriptRunner()
        self.script_author = script_author or ScriptAuthor()
        self.shell_runner = shell_runner or ShellRunner()
        self.observation_parser = observation_parser or ToolObservationParser()
        self.web_recon_client = web_recon_client or WebReconClient()

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = self.current_phase
        flow = state.get("flow")
        if flow is None:
            raise ValueError("ExecutorAgent requires state['flow']")

        executed_tasks = list(state.get("executed_tasks", []))
        tool_observations = list(state.get("tool_observations", []))
        executed_fingerprints = set(state.get("executed_action_fingerprints", []))
        approvals_required = list(state.get("approvals_required", []))
        approved_ids = self._approval_ids(state.get("approved_actions", []), ApprovalStatus.APPROVED)
        rejected_ids = self._approval_ids(state.get("rejected_actions", []), ApprovalStatus.REJECTED)
        executed_ids = {
            item.get("action_id")
            for item in executed_tasks
            if item.get("action_id")
        }

        candidate_actions = self._candidate_actions(state)

        for candidate in candidate_actions:
            self._sync_candidate_tool_risk(candidate)
            fingerprint = action_fingerprint(candidate)
            candidate["fingerprint"] = fingerprint
            if candidate["action_id"] in executed_ids:
                continue
            if fingerprint in executed_fingerprints:
                continue
            if candidate["action_id"] in rejected_ids:
                executed_tasks.append(
                    {
                        "action_id": candidate["action_id"],
                        "task": candidate,
                        "status": "skipped",
                        "reason": "approval_rejected",
                    }
                )
                continue

            approval_required = self._requires_approval(candidate, state)
            approval_granted = candidate["action_id"] in approved_ids
            if approval_required and not approval_granted:
                if candidate not in approvals_required:
                    approvals_required.append(candidate)
                approval_store.upsert_from_action(candidate)
                continue

            if candidate.get("action_kind") == "web_recon":
                executed_task = self._execute_web_recon_action(
                    flow=flow,
                    candidate=candidate,
                    state=state,
                )
                executed_tasks.append(executed_task)
                self._append_observation(tool_observations, executed_task)
                if executed_task.get("status") == "completed":
                    executed_fingerprints.add(fingerprint)
                continue

            if candidate.get("action_kind") == "shell" or candidate.get("tool") == "bash_runner":
                executed_task = self._execute_shell_action(
                    flow=flow,
                    candidate=candidate,
                    target_scope=flow.target_scope,
                )
                executed_tasks.append(executed_task)
                self._append_observation(tool_observations, executed_task)
                if executed_task.get("status") == "completed":
                    executed_fingerprints.add(fingerprint)
                continue

            if candidate.get("action_kind", "tool") == "script" or candidate.get("tool") == "script_runner":
                executed_task = self._execute_script_action(
                    flow=flow,
                    candidate=candidate,
                    target_scope=flow.target_scope,
                )
                executed_tasks.append(executed_task)
                self._append_observation(tool_observations, executed_task)
                if executed_task.get("status") == "completed":
                    executed_fingerprints.add(fingerprint)
                continue

            if candidate.get("action_kind", "tool") != "tool":
                executed_tasks.append(
                    {
                        "action_id": candidate["action_id"],
                        "task": candidate,
                        "status": "skipped",
                        "reason": "unsupported_action_kind",
                    }
                )
                continue

            tool = candidate.get("tool", "")
            task_type = candidate.get("type", "")
            profile = candidate.get("profile") or TASK_TOOL_PROFILES.get((task_type, tool))
            if profile is None:
                executed_tasks.append(
                    {
                        "action_id": candidate["action_id"],
                        "task": candidate,
                        "status": "skipped",
                        "reason": "unsupported_follow_up_task",
                    }
                )
                continue

            # 将每次后续执行同步记录到 Flow 层级中，便于审计。
            task = flow.add_task(
                AssessmentTask(
                    type=task_type,
                    target=candidate["target"],
                    objective=candidate.get("rationale", f"Execute {task_type}"),
                    risk_level=RiskLevel.LOW,
                    priority=20,
                )
            )
            subtask = task.add_subtask(
                SubTask(agent=self.name, objective=task.objective, risk_level=RiskLevel.LOW)
            )
            action = subtask.add_action(
                Action(
                    tool=tool,
                    intent={
                        "tool": tool,
                        "profile": profile,
                        "args": {
                            "target": candidate["target"],
                            **candidate.get("args", {}),
                        },
                    },
                    risk_level=RiskLevel.LOW,
                )
            )

            output_path = self.artifact_store.reserve_action_path(flow.id, action.id, f"{tool}.txt")
            tool_output_path = self._tool_output_path(flow.id, action.id, tool, profile)
            action.intent["args"] = self._prepare_tool_args(
                tool=tool,
                profile=profile,
                target=candidate["target"],
                raw_args=candidate.get("args", {}),
                output_path=tool_output_path,
            )
            if approval_granted:
                action.intent["approval_granted"] = True
            try:
                spec = self.execution_client.build_command(action.intent)
            except PermissionError:
                candidate["risk_level"] = str(candidate.get("risk_level") or "medium")
                candidate["requires_approval"] = True
                if candidate not in approvals_required:
                    approvals_required.append(candidate)
                approval_store.upsert_from_action(candidate)
                task.status = TaskStatus.PENDING
                subtask.status = TaskStatus.PENDING
                continue
            action.command_preview = " ".join(spec.command)
            action.mark_started()
            task.status = TaskStatus.RUNNING
            subtask.status = TaskStatus.RUNNING

            result = self.execution_client.execute_spec(spec)
            action.metadata["exit_code"] = result.exit_code
            action.metadata["stdout"] = result.stdout
            action.metadata["stderr"] = result.stderr
            self._write_text_artifact(output_path, result.stdout, result.stderr)
            # 即使工具执行失败，也把 stdout/stderr 保存为 Artifact。
            raw_artifact = self.artifact_store.register(
                path=output_path,
                artifact_type=ArtifactType.RAW_OUTPUT,
                action_id=action.id,
                summary=f"{tool} output",
            )
            action.artifacts.append(raw_artifact)
            tool_output_artifact = self._register_tool_output_artifact(
                action_id=action.id,
                tool=tool,
                path=tool_output_path,
            )
            if tool_output_artifact is not None:
                action.artifacts.append(tool_output_artifact)

            if result.succeeded:
                summary = self._summarize_output(tool, result.stdout)
                action.mark_succeeded(summary)
                task.status = TaskStatus.COMPLETED
                subtask.status = TaskStatus.COMPLETED
                flow.add_memory(
                    MemoryItem(
                        kind=MemoryKind.OBSERVATION,
                        content=summary,
                        source=action.id,
                        references=[raw_artifact.id],
                    )
                )
                executed_task = {
                    "action_id": candidate["action_id"],
                    "task": candidate,
                    "status": "completed",
                    "artifact_id": raw_artifact.id,
                    "tool_output_artifact_id": tool_output_artifact.id if tool_output_artifact else None,
                    "summary": summary,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                executed_tasks.append(executed_task)
                self._append_observation(tool_observations, executed_task)
                executed_fingerprints.add(fingerprint)
            else:
                error = result.stderr or f"Command exited with code {result.exit_code}"
                action.mark_failed(error)
                task.status = TaskStatus.FAILED
                subtask.status = TaskStatus.FAILED
                executed_task = {
                    "action_id": candidate["action_id"],
                    "task": candidate,
                    "status": "failed",
                    "artifact_id": raw_artifact.id,
                    "tool_output_artifact_id": tool_output_artifact.id if tool_output_artifact else None,
                    "error": error,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                executed_tasks.append(executed_task)
                self._append_observation(tool_observations, executed_task)

        state["executed_tasks"] = executed_tasks
        state["tool_observations"] = tool_observations
        state["executed_action_fingerprints"] = sorted(executed_fingerprints)
        state["approvals_required"] = approvals_required
        state["next_action"] = self.next_action_after_run
        return state

    def _sync_candidate_tool_risk(self, candidate: dict) -> None:
        if candidate.get("action_kind", "tool") != "tool":
            return
        tool = candidate.get("tool", "")
        profile = candidate.get("profile") or TASK_TOOL_PROFILES.get((candidate.get("type", ""), tool))
        if not tool or not profile:
            return
        command_builder = getattr(self.execution_client, "command_builder", None) or getattr(
            self.execution_client, "builder", None
        )
        if command_builder is None:
            return
        try:
            tool_def, tool_profile = command_builder.registry.get_profile(tool, profile)
        except Exception:
            return
        real_risk = tool_profile.risk or tool_def.risk
        if self._risk_rank(real_risk) > self._risk_rank(str(candidate.get("risk_level", "low"))):
            candidate["risk_level"] = real_risk
        if real_risk != "low":
            candidate["requires_approval"] = True

    def _risk_rank(self, risk: str) -> int:
        order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        return order.get(risk, 1)

    def _candidate_actions(self, state: AutoFlowState) -> list[dict]:
        return [
            *self._actions_from_test_plans(state.get("test_plans", [])),
            *self._actions_from_legacy_follow_ups(state.get("follow_up_tasks", [])),
        ]

    def _requires_approval(self, candidate: dict, state: AutoFlowState) -> bool:
        return bool(candidate.get("requires_approval") or candidate.get("risk_level") != "low")

    def _actions_from_test_plans(self, test_plans: list[dict]) -> list[dict]:
        actions: list[dict] = []
        for plan in test_plans:
            for action in plan.get("actions", []):
                actions.append(
                    {
                        "action_id": action.get("id", ""),
                        "plan_id": plan.get("id", ""),
                        "type": plan.get("strategy", "test_plan_action"),
                        "target": action.get("target") or plan.get("target"),
                        "tool": action.get("tool", ""),
                        "profile": action.get("profile", ""),
                        "risk_level": action.get("risk_level", "low"),
                        "requires_approval": action.get("requires_approval", False),
                        "action_kind": action.get("action_kind", "tool"),
                        "args": action.get("args", {}),
                        "expected_impact": action.get("expected_impact", ""),
                        "rationale": action.get("rationale", ""),
                        "name": action.get("name", ""),
                        "script_template": action.get("script_template"),
                        "script_source": action.get("metadata", {}).get("script_source"),
                        "shell_command": action.get("metadata", {}).get("shell_command"),
                        "metadata": action.get("metadata", {}),
                    }
                )
        return actions

    def _execute_web_recon_action(self, flow, candidate: dict, state: AutoFlowState) -> dict:
        task = flow.add_task(
            AssessmentTask(
                type=candidate.get("type", "web_recon_refresh"),
                target=candidate["target"],
                objective=candidate.get("rationale", "Refresh web recon for discovered path"),
                risk_level=RiskLevel.LOW,
                priority=15,
            )
        )
        subtask = task.add_subtask(
            SubTask(agent=self.name, objective=task.objective, risk_level=RiskLevel.LOW)
        )
        action = subtask.add_action(
            Action(
                tool="web_recon",
                intent={
                    "action_kind": "web_recon",
                    "target": candidate["target"],
                },
                risk_level=RiskLevel.LOW,
            )
        )

        action.mark_started()
        task.status = TaskStatus.RUNNING
        subtask.status = TaskStatus.RUNNING
        result = self.web_recon_client.recon(candidate["target"])
        output = json.dumps(result, ensure_ascii=False, indent=2)
        output_path = self.artifact_store.reserve_action_path(flow.id, action.id, "web_recon.json")
        self._write_text_artifact(output_path, output, "")
        raw_artifact = self.artifact_store.register(
            path=output_path,
            artifact_type=ArtifactType.STRUCTURED_RESULT,
            action_id=action.id,
            summary="web_recon output",
        )
        action.artifacts.append(raw_artifact)

        web_recon = list(state.get("web_recon", []))
        web_recon = [item for item in web_recon if item.get("target") != result.get("target")]
        web_recon.append(result)
        state["web_recon"] = web_recon

        if result.get("error"):
            action.mark_failed(result["error"])
            task.status = TaskStatus.FAILED
            subtask.status = TaskStatus.FAILED
            return {
                "action_id": candidate["action_id"],
                "task": candidate,
                "status": "failed",
                "artifact_id": raw_artifact.id,
                "error": result["error"],
                "stdout": output,
                "stderr": result["error"],
            }

        summary = (
            f"web_recon completed: {result.get('target')} "
            f"status={result.get('status_code')} title={result.get('title', '')!r}"
        )
        action.mark_succeeded(summary)
        task.status = TaskStatus.COMPLETED
        subtask.status = TaskStatus.COMPLETED
        flow.add_memory(
            MemoryItem(
                kind=MemoryKind.OBSERVATION,
                content=summary,
                source=action.id,
                references=[raw_artifact.id],
            )
        )
        return {
            "action_id": candidate["action_id"],
            "task": candidate,
            "status": "completed",
            "artifact_id": raw_artifact.id,
            "summary": summary,
            "stdout": output,
            "stderr": "",
        }

    def _actions_from_legacy_follow_ups(self, follow_up_tasks: list[dict]) -> list[dict]:
        actions: list[dict] = []
        for index, task in enumerate(follow_up_tasks):
            actions.append(
                {
                    "action_id": task.get("id", f"legacy_follow_up_{index}"),
                    "type": task.get("type", ""),
                    "target": task.get("target", ""),
                    "tool": task.get("tool", ""),
                    "profile": TASK_TOOL_PROFILES.get((task.get("type", ""), task.get("tool", "")), ""),
                    "risk_level": task.get("risk_level", "low"),
                    "requires_approval": task.get("requires_approval", False),
                    "action_kind": "tool",
                    "args": task.get("args", {}),
                    "expected_impact": "",
                    "rationale": task.get("reason", ""),
                    "name": task.get("type", "follow_up_task"),
                    "script_template": None,
                    "script_source": None,
                    "metadata": {},
                }
            )
        return actions

    def _actions_from_validation_plans(self, validation_plans: list[dict]) -> list[dict]:
        actions: list[dict] = []
        for plan in validation_plans:
            if plan.get("status") in {"executed", "completed", "failed"}:
                continue
            plan_metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
            action_category = self._metadata_category(plan_metadata)
            for action in plan.get("actions", []):
                action_metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
                actions.append(
                    {
                        "action_id": action.get("id", ""),
                        "plan_id": plan.get("id", ""),
                        "validation_plan_id": plan.get("id", ""),
                        "finding_id": plan.get("finding_id", ""),
                        "type": "validation",
                        "target": action.get("target") or plan.get("target"),
                        "tool": action.get("tool", ""),
                        "profile": action.get("profile", ""),
                        "risk_level": action.get("risk_level", plan.get("risk_level", "medium")),
                        "requires_approval": action.get("requires_approval", False),
                        "action_kind": action.get("action_kind", "tool"),
                        "args": action.get("args", {}),
                        "expected_impact": action.get("expected_impact", ""),
                        "rationale": action.get("rationale") or plan.get("rationale", ""),
                        "name": action.get("name", ""),
                        "script_template": action.get("script_template"),
                        "script_source": action_metadata.get("script_source"),
                        "shell_command": action_metadata.get("shell_command"),
                        "metadata": {
                            "category": action_metadata.get("category") or action_category,
                            "finding": plan_metadata.get("finding"),
                            **action_metadata,
                            "validation_plan_id": plan.get("id", ""),
                            "finding_id": plan.get("finding_id", ""),
                            "validation_objective": plan.get("objective", ""),
                        },
                    }
                )
        return actions

    def _metadata_category(self, metadata: dict) -> str:
        if metadata.get("category"):
            return str(metadata["category"])
        finding = metadata.get("finding")
        if isinstance(finding, dict):
            finding_metadata = finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}
            if finding_metadata.get("category"):
                return str(finding_metadata["category"])
        return ""

    def _execute_shell_action(self, flow, candidate: dict, target_scope: list[str]) -> dict:
        task = flow.add_task(
            AssessmentTask(
                type=candidate.get("type", "shell_action"),
                target=candidate["target"],
                objective=candidate.get("rationale", "Execute bounded shell validation action"),
                risk_level=RiskLevel.LOW,
                priority=20,
            )
        )
        subtask = task.add_subtask(
            SubTask(agent=self.name, objective=task.objective, risk_level=RiskLevel.LOW)
        )
        command = candidate.get("shell_command") or candidate.get("metadata", {}).get("shell_command")
        action = subtask.add_action(
            Action(
                tool="bash_runner",
                intent={
                    "action_kind": "shell",
                    "target": candidate["target"],
                    "shell_command": command,
                },
                risk_level=RiskLevel.LOW,
            )
        )

        artifact_dir = self.artifact_store.reserve_action_path(flow.id, action.id, "shell-output.txt").parent
        output_path = artifact_dir / "shell-output.txt"
        action.command_preview = f"bash_runner {candidate.get('profile', 'bounded_bash')}"
        if candidate.get("risk_level") != "low":
            action.intent["approval_granted"] = True
        action.mark_started()
        task.status = TaskStatus.RUNNING
        subtask.status = TaskStatus.RUNNING

        if not command:
            result = None
            error = "Shell action requires metadata.shell_command"
        else:
            policy_profile = candidate.get("metadata", {}).get("shell_policy_profile")
            if not policy_profile:
                policy_profile = "container_lab_shell"
            result = self.shell_runner.run_command(
                command=command,
                target=candidate["target"],
                target_scope=target_scope,
                artifact_dir=artifact_dir,
                policy_profile=policy_profile,
            )
            error = result.stderr or f"Shell command exited with code {result.exit_code}"

        if result is not None:
            action.metadata["exit_code"] = result.exit_code
            action.metadata["stdout"] = result.stdout
            action.metadata["stderr"] = result.stderr
            action.metadata["shell_command"] = command
            self._write_text_artifact(output_path, result.stdout, result.stderr)
        else:
            action.metadata["shell_command"] = command
            self._write_text_artifact(output_path, "", error)

        raw_artifact = self.artifact_store.register(
            path=output_path,
            artifact_type=ArtifactType.RAW_OUTPUT,
            action_id=action.id,
            summary="bash_runner output",
        )
        action.artifacts.append(raw_artifact)

        if result is not None and result.succeeded:
            summary = self._summarize_output("bash_runner", result.stdout)
            action.mark_succeeded(summary)
            task.status = TaskStatus.COMPLETED
            subtask.status = TaskStatus.COMPLETED
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.OBSERVATION,
                    content=summary,
                    source=action.id,
                    references=[raw_artifact.id],
                )
            )
            return {
                "action_id": candidate["action_id"],
                "task": {**candidate, "tool": "bash_runner", "profile": candidate.get("profile", "bounded_bash")},
                "status": "completed",
                "artifact_id": raw_artifact.id,
                "summary": summary,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        action.mark_failed(error)
        task.status = TaskStatus.FAILED
        subtask.status = TaskStatus.FAILED
        return {
            "action_id": candidate["action_id"],
            "task": {**candidate, "tool": "bash_runner", "profile": candidate.get("profile", "bounded_bash")},
            "status": "failed",
            "artifact_id": raw_artifact.id,
            "error": error,
            "stdout": result.stdout if result is not None else "",
            "stderr": result.stderr if result is not None else error,
        }

    def _execute_script_action(self, flow, candidate: dict, target_scope: list[str]) -> dict:
        task = flow.add_task(
            AssessmentTask(
                type=candidate.get("type", "script_action"),
                target=candidate["target"],
                objective=candidate.get("rationale", "Execute constrained script action"),
                risk_level=RiskLevel.LOW,
                priority=20,
            )
        )
        subtask = task.add_subtask(
            SubTask(agent=self.name, objective=task.objective, risk_level=RiskLevel.LOW)
        )
        action = subtask.add_action(
            Action(
                tool=candidate.get("tool", "script_runner"),
                intent={
                    "action_kind": "script",
                    "script_template": candidate.get("script_template"),
                    "target": candidate["target"],
                },
                risk_level=RiskLevel.LOW,
            )
        )

        artifact_dir = self.artifact_store.reserve_action_path(flow.id, action.id, "script-output.txt").parent
        output_path = artifact_dir / "script-output.txt"
        action.command_preview = f"script_runner {candidate.get('script_template') or 'generated_script'}"
        if candidate.get("risk_level") != "low":
            action.intent["approval_granted"] = True
        action.mark_started()
        task.status = TaskStatus.RUNNING
        subtask.status = TaskStatus.RUNNING

        script_context = {**candidate, "target_scope": target_scope}
        policy_profile = self._script_policy_profile(candidate)
        max_attempts = self._max_script_attempts(candidate)
        attempts: list[dict] = []
        script_source = candidate.get("script_source")
        result = None
        error = ""

        for attempt_index in range(1, max_attempts + 1):
            try:
                if script_source:
                    result = self.script_runner.run_script(
                        script=script_source,
                        target=candidate["target"],
                        target_scope=target_scope,
                        artifact_dir=artifact_dir,
                        policy_profile=policy_profile,
                    )
                elif candidate.get("script_template"):
                    result = self.script_runner.run_template(
                        template=candidate["script_template"],
                        target=candidate["target"],
                        target_scope=target_scope,
                        artifact_dir=artifact_dir,
                        policy_profile=policy_profile,
                    )
                elif candidate.get("metadata", {}).get("script_goal"):
                    script_source = self.script_author.author(script_context)
                    action.intent["script_source"] = script_source
                    result = self.script_runner.run_script(
                        script=script_source,
                        target=candidate["target"],
                        target_scope=target_scope,
                        artifact_dir=artifact_dir,
                        policy_profile=policy_profile,
                    )
                else:
                    raise ValueError(
                        "Script action requires script_template, script_source, or metadata.script_goal"
                    )
            except ValueError as exc:
                result = None
                error = str(exc)
            else:
                error = result.stderr or f"Script exited with code {result.exit_code}"

            attempt = {
                "attempt": attempt_index,
                "policy_profile": policy_profile,
                "status": "completed" if result is not None and result.succeeded else "failed",
                "exit_code": result.exit_code if result is not None else None,
                "stdout": result.stdout if result is not None else "",
                "stderr": result.stderr if result is not None else error,
                "script_source": script_source or f"template:{candidate.get('script_template')}",
            }
            attempts.append(attempt)
            if result is not None and result.succeeded:
                break
            if not candidate.get("metadata", {}).get("script_goal"):
                break
            if attempt_index >= max_attempts:
                break
            script_source = self.script_author.repair(
                script_context,
                failed_script=script_source or "",
                failure=attempt,
            )
            action.intent["script_source"] = script_source

        if result is not None:
            action.metadata["exit_code"] = result.exit_code
            action.metadata["stdout"] = result.stdout
            action.metadata["stderr"] = result.stderr
            action.metadata["script_attempts"] = attempts
            action.metadata["script_policy_profile"] = policy_profile
            self._write_text_artifact(output_path, result.stdout, result.stderr)
        else:
            action.metadata["script_attempts"] = attempts
            action.metadata["script_policy_profile"] = policy_profile
            self._write_text_artifact(output_path, "", error)

        raw_artifact = self.artifact_store.register(
            path=output_path,
            artifact_type=ArtifactType.RAW_OUTPUT,
            action_id=action.id,
            summary="script_runner output",
        )
        action.artifacts.append(raw_artifact)

        if result is not None and result.succeeded:
            summary = self._summarize_output("script_runner", result.stdout)
            action.mark_succeeded(summary)
            task.status = TaskStatus.COMPLETED
            subtask.status = TaskStatus.COMPLETED
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.OBSERVATION,
                    content=summary,
                    source=action.id,
                    references=[raw_artifact.id],
                )
            )
            return {
                "action_id": candidate["action_id"],
                "task": candidate,
                "status": "completed",
                "artifact_id": raw_artifact.id,
                "summary": summary,
                "script_attempts": attempts,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        action.mark_failed(error)
        task.status = TaskStatus.FAILED
        subtask.status = TaskStatus.FAILED
        return {
            "action_id": candidate["action_id"],
            "task": candidate,
            "status": "failed",
            "artifact_id": raw_artifact.id,
            "error": error,
            "script_attempts": attempts,
            "stdout": result.stdout if result is not None else "",
            "stderr": result.stderr if result is not None else error,
        }

    def _append_observation(self, observations: list[dict], executed_task: dict) -> None:
        task = executed_task.get("task", {})
        if not task.get("tool"):
            return
        observation = self.observation_parser.parse(
            executed_task=executed_task,
            stdout=executed_task.get("stdout", ""),
            stderr=executed_task.get("stderr", ""),
        )
        observations.append(observation.model_dump(mode="json"))

    def _prepare_tool_args(
        self,
        *,
        tool: str,
        profile: str,
        target: str,
        raw_args: dict,
        output_path: Path | None,
    ) -> dict[str, str]:
        args = {key: str(value) for key, value in raw_args.items() if value is not None and str(value) != ""}
        _, tool_profile = self._tool_profile(tool, profile)
        if "target" in tool_profile.allowed_args and "target" not in args:
            args["target"] = target
        if tool == "nmap":
            host, port = self._split_host_port(args.get("target", target))
            args["target"] = host
            if "port" in tool_profile.allowed_args and "port" not in args and port:
                args["port"] = port
        if "output" in tool_profile.allowed_args and "output" not in args:
            if output_path is None:
                raise ValueError(f"Tool {tool}/{profile} requires output path")
            args["output"] = str(output_path)
        return args

    def _tool_output_path(self, flow_id: str, action_id: str, tool: str, profile: str) -> Path | None:
        _, tool_profile = self._tool_profile(tool, profile)
        if "output" not in tool_profile.allowed_args:
            return None
        suffix = "xml" if tool == "nmap" else "txt"
        return self.artifact_store.reserve_action_path(flow_id, action_id, f"{tool}-tool-output.{suffix}")

    def _register_tool_output_artifact(self, *, action_id: str, tool: str, path: Path | None):
        if path is None or not path.exists():
            return None
        return self.artifact_store.register(
            path=path,
            artifact_type=ArtifactType.RAW_OUTPUT,
            action_id=action_id,
            summary=f"{tool} structured output",
        )

    def _tool_profile(self, tool: str, profile: str):
        builder = getattr(self.execution_client, "command_builder", None)
        if builder is None:
            builder = getattr(self.execution_client, "builder", None)
        if builder is None:
            raise AttributeError("Execution client must expose command_builder or builder")
        return builder.registry.get_profile(tool, profile)

    def _split_host_port(self, target: str) -> tuple[str, str | None]:
        parsed = urlparse(target if "://" in target else f"//{target}")
        if parsed.hostname and parsed.port:
            return parsed.hostname, str(parsed.port)
        return target, None

    def _script_policy_profile(self, candidate: dict) -> str:
        metadata = candidate.get("metadata", {})
        if metadata.get("script_policy_profile"):
            return metadata["script_policy_profile"]
        risk_level = candidate.get("risk_level", "low")
        if risk_level == "high":
            return "high_lab_poc"
        if risk_level == "medium":
            return "medium_artifact_script"
        return "low_readonly_http"

    def _max_script_attempts(self, candidate: dict) -> int:
        raw_value = candidate.get("metadata", {}).get("max_script_attempts", 2)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return 2
        return max(1, min(value, 5))

    def _approval_ids(self, state_items: list[dict], status: ApprovalStatus) -> set[str]:
        ids = {
            item.get("action_id")
            for item in state_items
            if item.get("action_id") and item.get("status", status.value) == status.value
        }
        for item in approval_store.list(status):
            ids.add(item.action_id)
        return ids

    def _write_text_artifact(self, path: Path, stdout: str, stderr: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{stdout or ''}\n{stderr or ''}".strip() + "\n", encoding="utf-8")

    def _summarize_output(self, tool: str, stdout: str) -> str:
        stdout = stdout or ""
        first_line = next((line.strip() for line in stdout.splitlines() if line.strip()), "")
        if first_line:
            return f"{tool} completed: {first_line[:300]}"
        return f"{tool} completed"


class ValidationExecutorAgent(ExecutorAgent):
    """Execute ValidationPlan actions in lab mode without blocking on approval."""

    name = "validation_executor"
    current_phase = "validation_execution"
    next_action_after_run = "strategy"

    def _candidate_actions(self, state: AutoFlowState) -> list[dict]:
        actions = self._actions_from_validation_plans(state.get("validation_plans", []))
        executed_ids = {
            item.get("action_id")
            for item in state.get("executed_tasks", [])
            if item.get("action_id")
        }
        covered_fingerprints = set(state.get("executed_action_fingerprints", []))
        pending_actions = [
            action
            for action in actions
            if action.get("action_id") not in executed_ids
            and action_fingerprint(action) not in covered_fingerprints
        ]
        ordered = sorted(
            pending_actions,
            key=self._validation_action_priority,
            reverse=True,
        )
        budget = self._validation_action_budget(state)
        if budget > 0:
            return ordered[:budget]
        return ordered

    def _validation_action_budget(self, state: AutoFlowState) -> int:
        raw_value = state.get("validation_action_budget", 0)
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return 0

    def _validation_action_priority(self, candidate: dict) -> tuple[int, int, int, int, str]:
        category = self._candidate_category(candidate)
        risk = str(candidate.get("risk_level") or "medium")
        action_kind = str(candidate.get("action_kind") or "tool")
        tool = str(candidate.get("tool") or "")
        return (
            self._category_priority(category),
            self._risk_priority(risk),
            self._action_kind_priority(action_kind, tool),
            self._validation_role_priority(candidate),
            str(candidate.get("action_id") or ""),
        )

    def _candidate_category(self, candidate: dict) -> str:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        if metadata.get("category"):
            return str(metadata["category"])
        finding = metadata.get("finding")
        if isinstance(finding, dict):
            finding_metadata = finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}
            if finding_metadata.get("category"):
                return str(finding_metadata["category"])
        objective = str(metadata.get("validation_objective") or candidate.get("rationale") or "").lower()
        target = str(candidate.get("target") or "").lower()
        combined = f"{objective} {target}"
        if "api" in combined:
            return "api_exposure"
        if "directory listing" in combined or "/ftp" in combined:
            return "directory_listing"
        if "debug" in combined or "metrics" in combined:
            return "debug_endpoint_exposed"
        if "config" in combined or "package" in combined:
            return "public_config_exposure"
        if "cors" in combined:
            return "cors_wildcard"
        if "header" in combined:
            return "missing_security_header:unknown"
        return ""

    def _category_priority(self, category: str) -> int:
        if category == "api_exposure":
            return 100
        if category == "directory_listing":
            return 95
        if category == "debug_endpoint_exposed":
            return 90
        if category == "public_config_exposure":
            return 85
        if category in {"web_risk_observation"}:
            return 80
        if category == "cors_wildcard":
            return 70
        if category.startswith("missing_security_header:"):
            return 60
        if category == "weak_cache_control":
            return 55
        if category in {"sensitive_path_exposed", "robots_txt_exposure"}:
            return 45
        if category.startswith("informational_header:"):
            return 25
        if category == "tech_stack_fingerprint":
            return 10
        return 30

    def _risk_priority(self, risk: str) -> int:
        order = {
            "critical": 50,
            "high": 40,
            "medium": 30,
            "low": 20,
            "info": 10,
        }
        return order.get(risk, 0)

    def _action_kind_priority(self, action_kind: str, tool: str) -> int:
        if action_kind == "script" or tool == "script_runner":
            return 40
        if action_kind == "shell" or tool == "bash_runner":
            return 35
        if action_kind == "tool" and tool in {"curl", "nuclei", "nikto"}:
            return 30
        if action_kind == "web_recon":
            return 20
        return 10

    def _validation_role_priority(self, candidate: dict) -> int:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        role = str(metadata.get("validation_role") or "")
        order = {
            "active_validation": 40,
            "reproduction_evidence": 35,
            "raw_http_evidence": 30,
            "confirmatory": 25,
            "context_refresh": 10,
        }
        return order.get(role, 0)

    def _requires_approval(self, candidate: dict, state: AutoFlowState) -> bool:
        if state.get("rules_of_engagement", {}).get("validation_auto_approve", True):
            return False
        return super()._requires_approval(candidate, state)

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        candidates = self._candidate_actions(state)
        before_ids = {
            item.get("action_id")
            for item in state.get("executed_tasks", [])
            if item.get("action_id")
        }
        state = await super().run(state)
        new_results = {
            item.get("action_id"): item
            for item in state.get("executed_tasks", [])
            if item.get("action_id") and item.get("action_id") not in before_ids
        }
        covered_fingerprints = set(state.get("executed_action_fingerprints", []))
        covered_actions = {
            candidate.get("action_id")
            for candidate in candidates
            if candidate.get("action_id") and action_fingerprint(candidate) in covered_fingerprints
        }
        state["validation_plans"] = self._mark_validation_plan_statuses(
            state.get("validation_plans", []),
            new_results,
            covered_actions,
        )
        validation_results = self._build_validation_results(state.get("validation_plans", []), new_results)
        if validation_results:
            existing_results = list(state.get("validation_results", []))
            state["validation_results"] = [*existing_results, *[item.model_dump(mode="json") for item in validation_results]]
            self._apply_validation_results_to_findings(state, validation_results)
            flow = state.get("flow")
            if flow is not None:
                for result in validation_results:
                    flow.add_validation_result(result)
                    flow.add_memory(
                        MemoryItem(
                            kind=MemoryKind.FINDING,
                            content=f"Validation result for {result.finding_id}: {result.status.value}",
                            source=self.name,
                            references=result.executed_action_ids,
                            metadata=result.model_dump(mode="json"),
                        )
                    )
        return state

    def _mark_validation_plan_statuses(
        self,
        validation_plans: list[dict],
        new_results: dict[str, dict],
        precovered_actions: set[str],
    ) -> list[dict]:
        updated: list[dict] = []
        for plan in validation_plans:
            actions = plan.get("actions", [])
            action_ids = [action.get("id") for action in actions if action.get("id")]
            if not action_ids:
                updated.append(plan)
                continue
            plan_results = [new_results[action_id] for action_id in action_ids if action_id in new_results]
            for action_id in action_ids:
                if action_id in new_results or action_id not in precovered_actions:
                    continue
                plan_results.append(
                    {
                        "action_id": action_id,
                        "status": "completed",
                        "summary": "Equivalent action was already executed earlier in the flow.",
                        "error": "",
                        "artifact_id": None,
                    }
                )
            if not plan_results:
                updated.append(plan)
                continue
            status = "completed" if all(item.get("status") == "completed" for item in plan_results) else "failed"
            updated.append(
                {
                    **plan,
                    "status": status,
                    "execution_results": [
                        {
                            "action_id": item.get("action_id"),
                            "status": item.get("status"),
                            "summary": item.get("summary", ""),
                            "error": item.get("error", ""),
                            "artifact_id": item.get("artifact_id"),
                        }
                        for item in plan_results
                    ],
                }
            )
        return updated

    def _build_validation_results(
        self,
        validation_plans: list[dict],
        new_results: dict[str, dict],
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for plan in validation_plans:
            execution_results = plan.get("execution_results", [])
            if not execution_results:
                continue
            plan_action_ids = [
                action.get("id")
                for action in plan.get("actions", [])
                if action.get("id")
            ]
            full_results = [
                new_results[action_id]
                for action_id in plan_action_ids
                if action_id in new_results
            ]
            for item in execution_results:
                action_id = item.get("action_id")
                if action_id and action_id not in new_results:
                    full_results.append(item)
            results.append(self._evaluate_validation_plan(plan, full_results))
        return results

    def _evaluate_validation_plan(self, plan: dict, action_results: list[dict]) -> ValidationResult:
        finding = self._plan_finding(plan)
        category = self._finding_category(finding, plan)
        combined_text = self._combined_result_text(action_results)
        completed = [item for item in action_results if item.get("status") == "completed"]
        failed = [item for item in action_results if item.get("status") == "failed"]
        evidence = self._validation_evidence(action_results)
        status, reasoning = self._classify_validation(
            category=category,
            combined_text=combined_text,
            completed_count=len(completed),
            failed_count=len(failed),
        )
        confidence = FindingConfidence.HIGH if status == ValidationResultStatus.VALIDATED and completed else FindingConfidence.MEDIUM
        reproduction_steps = self._reproduction_steps(plan, action_results)
        return ValidationResult(
            finding_id=str(plan.get("finding_id", "")),
            validation_plan_id=str(plan.get("id", "")),
            status=status,
            confidence=confidence,
            impact=self._impact_summary(status, category, plan),
            reproduction_steps=reproduction_steps,
            evidence=evidence,
            executed_action_ids=[
                str(item.get("action_id"))
                for item in action_results
                if item.get("action_id")
            ],
            reasoning=reasoning,
            metadata={
                "category": category,
                "validation_plan_status": plan.get("status"),
                "success_criteria": plan.get("success_criteria", []),
                "failure_criteria": plan.get("failure_criteria", []),
            },
        )

    def _classify_validation(
        self,
        *,
        category: str,
        combined_text: str,
        completed_count: int,
        failed_count: int,
    ) -> tuple[ValidationResultStatus, str]:
        text = combined_text.lower()
        positive_patterns = {
            "cors_wildcard": [
                "access_control_allow_origin\": \"*\"",
                "access-control-allow-origin\": \"*\"",
                "allow_origin=*",
                "access-control-allow-origin: *",
            ],
            "weak_cache_control": ["weak-cache-control", "public, max-age", "\"cache-control\": \"public"],
            "api_exposure": ["\"status\": 200", "status=200", "application/json", "\"json_keys\"", "\"sensitivity_hints\""],
            "debug_endpoint_exposed": ["matched_keywords", "prometheus", "nodejs", "heap", "process", "runtime"],
            "directory_listing": ["interesting_entries", "directory_listing_validation_probe", "listing directory", "entry_count"],
            "public_config_exposure": ["sensitivity_hints", "secret", "token", "password", "dependency", "package.json"],
            "sensitive_path_exposed": ["status=200", "\"status\": 200", "http/1.1 200", "listing directory"],
            "robots_txt_exposure": ["robots.txt", "disallow", "status=200", "http/1.1 200"],
            "informational_header:x-recruiting": ["x-recruiting", "/#/jobs"],
        }
        if category.startswith("missing_security_header:"):
            header = category.split(":", 1)[1]
            if header in text and ("missing" in text or "absent" in text):
                return ValidationResultStatus.VALIDATED, f"Validation output confirms missing header: {header}."
            if completed_count and header in text:
                return ValidationResultStatus.INCONCLUSIVE, f"Header {header} was mentioned, but missing/present state is ambiguous."
            if completed_count:
                return ValidationResultStatus.FALSE_POSITIVE, f"Validation completed without confirming missing header: {header}."

        patterns = positive_patterns.get(category, [])
        if any(pattern in text for pattern in patterns):
            return ValidationResultStatus.VALIDATED, f"Validation output matched positive indicators for {category or 'finding'}."
        if completed_count and not failed_count:
            if category in {"tech_stack_fingerprint"}:
                return ValidationResultStatus.VALIDATED, "Technology fingerprint was corroborated by completed validation actions."
            return ValidationResultStatus.FALSE_POSITIVE, "Validation actions completed but did not match the expected vulnerability indicators."
        if completed_count:
            return ValidationResultStatus.INCONCLUSIVE, "Some validation actions completed, but failures or weak evidence prevent confirmation."
        return ValidationResultStatus.INCONCLUSIVE, "Validation did not produce completed evidence."

    def _apply_validation_results_to_findings(
        self,
        state: AutoFlowState,
        validation_results: list[ValidationResult],
    ) -> None:
        result_by_finding = {result.finding_id: result for result in validation_results}
        findings = []
        for finding in state.get("findings", []):
            finding_id = finding.get("id")
            result = result_by_finding.get(finding_id)
            if result is None:
                findings.append(finding)
                continue
            updated = dict(finding)
            metadata = dict(updated.get("metadata") or {})
            validation_ids = list(metadata.get("validation_result_ids", []))
            validation_ids.append(result.id)
            metadata["validation_result_ids"] = validation_ids
            metadata["validation_status"] = result.status.value
            metadata["validation_reasoning"] = result.reasoning
            metadata["reproduction_steps"] = result.reproduction_steps
            updated["metadata"] = metadata
            evidence = list(updated.get("evidence", []))
            evidence.extend(result.evidence)
            updated["evidence"] = self._dedupe_text(evidence)
            if result.status == ValidationResultStatus.VALIDATED:
                updated["status"] = FindingStatus.VALIDATED.value
                updated["confidence"] = result.confidence.value
            elif result.status == ValidationResultStatus.FALSE_POSITIVE:
                updated["status"] = FindingStatus.FALSE_POSITIVE.value
            findings.append(updated)
        state["findings"] = findings

    def _plan_finding(self, plan: dict) -> dict:
        metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
        finding = metadata.get("finding")
        return finding if isinstance(finding, dict) else {}

    def _finding_category(self, finding: dict, plan: dict) -> str:
        metadata = finding.get("metadata") if isinstance(finding.get("metadata"), dict) else {}
        if metadata.get("category"):
            return str(metadata["category"])
        plan_metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
        return str(plan_metadata.get("category", ""))

    def _combined_result_text(self, action_results: list[dict]) -> str:
        parts: list[str] = []
        for item in action_results:
            for key in ("summary", "stdout", "stderr", "error"):
                value = item.get(key)
                if value:
                    parts.append(str(value))
        return "\n".join(parts)

    def _validation_evidence(self, action_results: list[dict]) -> list[str]:
        evidence: list[str] = []
        for item in action_results:
            action_id = item.get("action_id", "")
            status = item.get("status", "")
            summary = item.get("summary") or item.get("error") or ""
            if summary:
                evidence.append(f"{action_id}: {status}: {str(summary)[:500]}")
            stdout = self._trim_evidence_output(item.get("stdout", ""))
            if stdout:
                evidence.append(f"{action_id}: stdout: {stdout}")
            stderr = self._trim_evidence_output(item.get("stderr", ""))
            if stderr and status != "completed":
                evidence.append(f"{action_id}: stderr: {stderr}")
        return self._dedupe_text(evidence)

    def _trim_evidence_output(self, value: object, max_chars: int = 1200) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        compact = "\n".join(lines[:20])
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    def _reproduction_steps(self, plan: dict, action_results: list[dict]) -> list[str]:
        steps = [f"Target: {plan.get('target', '')}", f"Objective: {plan.get('objective', '')}"]
        for action in plan.get("actions", []):
            tool = action.get("tool", "")
            profile = action.get("profile", "")
            target = action.get("target") or plan.get("target", "")
            if action.get("action_kind") == "shell":
                command = action.get("metadata", {}).get("shell_command", "")
                steps.append(f"Run container shell validation against {target}: {command}")
            elif action.get("action_kind") == "script":
                template = action.get("script_template") or action.get("metadata", {}).get("script_goal", "")
                steps.append(f"Run {tool}/{profile} against {target}: {template}")
            else:
                steps.append(f"Run {tool}/{profile} against {target}")
        return [step for step in steps if step.strip()]

    def _impact_summary(self, status: ValidationResultStatus, category: str, plan: dict) -> str:
        if status == ValidationResultStatus.VALIDATED:
            return f"Candidate finding '{category or plan.get('objective', 'validation')}' was confirmed by validation evidence."
        if status == ValidationResultStatus.FALSE_POSITIVE:
            return "Validation did not reproduce the candidate vulnerability indicators."
        return "Validation evidence is insufficient for confirmation."

    def _dedupe_text(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result
