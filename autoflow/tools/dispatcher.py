from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from autoflow.artifacts.store import ArtifactStore
from autoflow.executor.execution_client import ExecutionClient
from autoflow.executor.script_runner import ScriptRunner
from autoflow.executor.shell_runner import ShellRunner
from autoflow.executor.web_recon import WebReconClient
from autoflow.flows.models import (
    Action,
    ArtifactType,
    AssessmentTask,
    MemoryItem,
    MemoryKind,
    RiskLevel,
    SubTask,
    TaskStatus,
)
from autoflow.graph.state import AutoFlowState
from autoflow.memory.agent_memory import AgentMemoryBuilder
from autoflow.observations.parser import ToolObservationParser
from autoflow.runtime.actions import canonical_target
from autoflow.tools.catalog import ToolCatalog


class ToolDispatcher:
    """Execute LLM tool calls and return compact, LLM-readable results."""

    def __init__(
        self,
        catalog: ToolCatalog | None = None,
        execution_client: ExecutionClient | None = None,
        artifact_store: ArtifactStore | None = None,
        script_runner: ScriptRunner | None = None,
        shell_runner: ShellRunner | None = None,
        web_recon_client: WebReconClient | None = None,
        observation_parser: ToolObservationParser | None = None,
        memory_builder: AgentMemoryBuilder | None = None,
        result_limit: int = 4000,
    ) -> None:
        self.catalog = catalog or ToolCatalog()
        self.execution_client = execution_client or ExecutionClient()
        self.artifact_store = artifact_store or ArtifactStore()
        self.script_runner = script_runner or ScriptRunner()
        self.shell_runner = shell_runner or ShellRunner()
        self.web_recon_client = web_recon_client or WebReconClient()
        self.observation_parser = observation_parser or ToolObservationParser()
        self.memory_builder = memory_builder or AgentMemoryBuilder()
        self.result_limit = result_limit

    def dispatch(self, name: str, arguments: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        try:
            if name == "read_agent_memory":
                memory = self.memory_builder.build(state, persisted_memory=state.get("agent_memory"))
                state["agent_memory"] = memory
                return self._ok(name, "", memory, "Returned agent memory pack.")
            if name == "list_known_targets":
                targets = sorted(self._known_targets(state))
                return self._ok(name, "", {"targets": targets}, f"Returned {len(targets)} known targets.")
            if name == "search_observations":
                return self._search_observations(name, arguments, state)
            if name == "web_recon_fetch_page":
                return self._dispatch_web_recon(name, arguments, state)
            if name == "run_shell__bounded_bash":
                return self._dispatch_shell(name, arguments, state)
            if name.startswith("run_script__"):
                return self._dispatch_script(name, arguments, state)
            if name.startswith("run_") and "__" in name:
                return self._dispatch_container_tool(name, arguments, state)
            return self._error(name, "", f"Unknown tool function '{name}'")
        except Exception as exc:
            target = str(arguments.get("target", "")) if isinstance(arguments, dict) else ""
            return self._error(name, target, str(exc))

    def tool_message_content(self, result: dict[str, Any]) -> str:
        return self._truncate(json.dumps(result, ensure_ascii=False, sort_keys=True))

    def _dispatch_web_recon(self, name: str, arguments: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        target = canonical_target(str(arguments.get("target", "")))
        self._require_authorized_target(target, state)
        result = self.web_recon_client.recon(target)
        output = json.dumps(result, ensure_ascii=False, indent=2)
        executed_task = self._record_artifact_action(
            state=state,
            tool="web_recon",
            profile="fetch_page",
            action_kind="web_recon",
            target=target,
            risk_level="low",
            stdout=output,
            stderr=result.get("error", ""),
            succeeded=not bool(result.get("error")),
            summary=(
                f"web_recon completed: {result.get('target')} "
                f"status={result.get('status_code')} title={result.get('title', '')!r}"
            ),
            artifact_name="web_recon.json",
            artifact_type=ArtifactType.STRUCTURED_RESULT,
        )
        web_recon = [item for item in state.get("web_recon", []) if item.get("target") != result.get("target")]
        web_recon.append(result)
        state["web_recon"] = web_recon
        self._append_observation(state, executed_task)
        return self._ok(name, target, self._compact_web_recon(result), executed_task.get("summary", "web_recon completed"))

    def _dispatch_script(self, name: str, arguments: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        template = name.split("__", 1)[1]
        target = canonical_target(str(arguments.get("target", "")))
        self._require_authorized_target(target, state)
        flow = state.get("flow")
        if flow is None:
            raise ValueError("ToolDispatcher requires state['flow'] for script execution")
        artifact_dir = self.artifact_store.reserve_action_path(flow.id, f"toolcall_{template}", "script-output.txt").parent
        risk = "medium" if template != "security_headers_check" else "low"
        result = self.script_runner.run_template(
            template=template,
            target=target,
            target_scope=state.get("target_scope", flow.target_scope),
            artifact_dir=artifact_dir,
            policy_profile="medium_artifact_script" if risk == "medium" else "low_readonly_http",
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        succeeded = result.succeeded
        summary = self._summary("script_runner", stdout)
        executed_task = self._record_artifact_action(
            state=state,
            tool="script_runner",
            profile=template,
            action_kind="script",
            target=target,
            risk_level=risk,
            stdout=stdout,
            stderr=stderr,
            succeeded=succeeded,
            summary=summary,
            artifact_name="script-output.txt",
            artifact_type=ArtifactType.RAW_OUTPUT,
        )
        self._append_observation(state, executed_task)
        return self._ok(name, target, self._json_or_text(stdout, stderr), summary) if succeeded else self._error(name, target, stderr)

    def _dispatch_shell(self, name: str, arguments: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        target = canonical_target(str(arguments.get("target", "")))
        self._require_authorized_target(target, state)
        flow = state.get("flow")
        if flow is None:
            raise ValueError("ToolDispatcher requires state['flow'] for shell execution")
        command = str(arguments.get("command", ""))
        policy_profile = str(arguments.get("policy_profile") or "container_lab_shell")
        artifact_dir = self.artifact_store.reserve_action_path(flow.id, "toolcall_bash_runner", "shell-output.txt").parent
        result = self.shell_runner.run_command(
            command=command,
            target=target,
            target_scope=state.get("target_scope", flow.target_scope),
            artifact_dir=artifact_dir,
            policy_profile=policy_profile,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        summary = self._summary("bash_runner", stdout)
        executed_task = self._record_artifact_action(
            state=state,
            tool="bash_runner",
            profile="bounded_bash",
            action_kind="shell",
            target=target,
            risk_level="medium" if policy_profile.startswith("medium") else "low",
            stdout=stdout,
            stderr=stderr,
            succeeded=result.succeeded,
            summary=summary,
            artifact_name="shell-output.txt",
            artifact_type=ArtifactType.RAW_OUTPUT,
        )
        self._append_observation(state, executed_task)
        payload = {"stdout_excerpt": self._truncate(stdout), "stderr_excerpt": self._truncate(stderr, 1200)}
        return self._ok(name, target, payload, summary) if result.succeeded else self._error(name, target, stderr or summary)

    def _dispatch_container_tool(self, name: str, arguments: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        raw = name.removeprefix("run_")
        tool_name, profile_name = raw.split("__", 1)
        args = {key: str(value) for key, value in arguments.items() if value is not None}
        profile = self.execution_client.command_builder.registry.get_profile(tool_name, profile_name)[1]
        target = ""
        if profile.target_required or "target" in profile.allowed_args:
            target = canonical_target(str(args.get("target", "")))
            self._require_authorized_target(target, state)
        elif "path" in profile.allowed_args:
            args["path"] = self._safe_container_scan_path(str(args.get("path", "")))
            target = args["path"]
        tool_output_path = self._tool_output_path(state, tool_name, profile_name)
        args = self._prepare_container_tool_args(
            tool=tool_name,
            profile=profile_name,
            target=target,
            raw_args=args,
            output_path=tool_output_path,
        )
        intent = {"tool": tool_name, "profile": profile_name, "args": args, "approval_granted": True}
        spec = self.execution_client.build_command(intent)
        result = self.execution_client.execute_spec(spec)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        summary = self._summary(tool_name, stdout)
        executed_task = self._record_artifact_action(
            state=state,
            tool=tool_name,
            profile=profile_name,
            action_kind="tool",
            target=target,
            risk_level=spec.risk_level,
            stdout=stdout,
            stderr=stderr,
            succeeded=result.succeeded,
            summary=summary,
            artifact_name=f"{tool_name}.txt",
            artifact_type=ArtifactType.RAW_OUTPUT,
        )
        tool_output_artifact = self._register_tool_output_artifact(
            action_id=executed_task["action_id"],
            tool=tool_name,
            path=tool_output_path,
        )
        if tool_output_artifact is not None:
            executed_task["tool_output_artifact_id"] = tool_output_artifact.id
        self._append_observation(state, executed_task)
        payload = {
            "stdout_excerpt": self._truncate(stdout),
            "stderr_excerpt": self._truncate(stderr, 1200),
            "signals": state.get("tool_observations", [])[-1].get("signals", []) if state.get("tool_observations") else [],
        }
        return self._ok(name, target, payload, summary) if result.succeeded else self._error(name, target, stderr or summary)

    def _prepare_container_tool_args(
        self,
        *,
        tool: str,
        profile: str,
        target: str,
        raw_args: dict[str, str],
        output_path: Path | None,
    ) -> dict[str, str]:
        args = {key: str(value) for key, value in raw_args.items() if value is not None and str(value) != ""}
        _, tool_profile = self.execution_client.command_builder.registry.get_profile(tool, profile)
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

    def _safe_container_scan_path(self, raw_path: str) -> str:
        if not raw_path:
            raise ValueError("Source/artifact scan tool requires non-empty path")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
        allowed_roots = [
            (Path.cwd() / "data" / "artifacts").resolve(),
            (Path.cwd() / "data" / "source").resolve(),
            (Path.cwd() / "data" / "source_audit").resolve(),
        ]
        if not resolved.exists():
            raise ValueError(f"Source/artifact scan path does not exist: {raw_path}")
        if not any(self._is_relative_to(resolved, root) for root in allowed_roots):
            allowed = ", ".join(str(root) for root in allowed_roots)
            raise ValueError(f"Source/artifact scan path must be under one of: {allowed}")
        return resolved.as_posix()

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _tool_output_path(self, state: AutoFlowState, tool: str, profile: str) -> Path | None:
        flow = state.get("flow")
        if flow is None:
            raise ValueError("ToolDispatcher requires state['flow'] for container tool execution")
        _, tool_profile = self.execution_client.command_builder.registry.get_profile(tool, profile)
        if "output" not in tool_profile.allowed_args:
            return None
        suffix = "xml" if tool == "nmap" else "txt"
        action_id = f"toolcall_{tool}_{profile}_{uuid4().hex[:10]}"
        return self.artifact_store.reserve_action_path(flow.id, action_id, f"{tool}-tool-output.{suffix}")

    def _register_tool_output_artifact(self, *, action_id: str, tool: str, path: Path | None):
        if path is None or not path.exists():
            return None
        return self.artifact_store.register(
            path=path,
            artifact_type=ArtifactType.RAW_OUTPUT,
            action_id=action_id,
            summary=f"{tool} structured output",
        )

    def _search_observations(self, name: str, arguments: dict[str, Any], state: AutoFlowState) -> dict[str, Any]:
        query = str(arguments.get("query", "")).lower()
        matches = []
        for observation in state.get("tool_observations", []):
            text = json.dumps(observation, ensure_ascii=False).lower()
            if query in text:
                matches.append(observation)
        return self._ok(name, "", {"matches": matches[:20]}, f"Found {len(matches)} matching observations.")

    def _record_artifact_action(
        self,
        *,
        state: AutoFlowState,
        tool: str,
        profile: str,
        action_kind: str,
        target: str,
        risk_level: str,
        stdout: str,
        stderr: str,
        succeeded: bool,
        summary: str,
        artifact_name: str,
        artifact_type: ArtifactType,
    ) -> dict[str, Any]:
        flow = state.get("flow")
        if flow is None:
            raise ValueError("ToolDispatcher requires state['flow']")
        risk = RiskLevel(risk_level) if risk_level in RiskLevel._value2member_map_ else RiskLevel.LOW
        task = flow.add_task(
            AssessmentTask(
                type=f"tool_call:{tool}/{profile}",
                target=target,
                objective=f"Function call {tool}/{profile}",
                risk_level=risk,
                priority=30,
            )
        )
        subtask = task.add_subtask(SubTask(agent="tool_loop", objective=task.objective, risk_level=risk))
        action = subtask.add_action(
            Action(
                tool=tool,
                intent={
                    "action_kind": action_kind,
                    "tool": tool,
                    "profile": profile,
                    "target": target,
                },
                risk_level=risk,
            )
        )
        action.mark_started()
        task.status = TaskStatus.RUNNING
        subtask.status = TaskStatus.RUNNING
        output_path = self.artifact_store.reserve_action_path(flow.id, action.id, artifact_name)
        self._write_text_artifact(output_path, stdout, stderr)
        artifact = self.artifact_store.register(
            path=output_path,
            artifact_type=artifact_type,
            action_id=action.id,
            summary=f"{tool}/{profile} output",
        )
        action.artifacts.append(artifact)
        if succeeded:
            action.mark_succeeded(summary)
            task.status = TaskStatus.COMPLETED
            subtask.status = TaskStatus.COMPLETED
            flow.add_memory(MemoryItem(kind=MemoryKind.OBSERVATION, content=summary, source=action.id, references=[artifact.id]))
        else:
            action.mark_failed(stderr or "Tool call failed")
            task.status = TaskStatus.FAILED
            subtask.status = TaskStatus.FAILED
        executed_task = {
            "action_id": action.id,
            "task": {
                "action_id": action.id,
                "type": task.type,
                "target": target,
                "tool": tool,
                "profile": profile,
                "risk_level": risk.value,
                "requires_approval": False,
                "action_kind": action_kind,
                "metadata": {"source": "tool_loop"},
            },
            "status": "completed" if succeeded else "failed",
            "artifact_id": artifact.id,
            "summary": summary if succeeded else "",
            "error": "" if succeeded else stderr,
            "stdout": stdout,
            "stderr": stderr,
        }
        executed_tasks = list(state.get("executed_tasks", []))
        executed_tasks.append(executed_task)
        state["executed_tasks"] = executed_tasks
        return executed_task

    def _append_observation(self, state: AutoFlowState, executed_task: dict[str, Any]) -> None:
        observation = self.observation_parser.parse(
            executed_task=executed_task,
            stdout=executed_task.get("stdout", ""),
            stderr=executed_task.get("stderr", ""),
        )
        observations = list(state.get("tool_observations", []))
        observations.append(observation.model_dump(mode="json"))
        state["tool_observations"] = observations

    def _require_authorized_target(self, target: str, state: AutoFlowState) -> None:
        if not target:
            raise ValueError("Tool call requires non-empty target")
        known = self._known_targets(state)
        if target in known:
            return
        target_host = self._host(target)
        known_hosts = {self._host(value) for value in known}
        if target_host and target_host in known_hosts:
            return
        raise ValueError(f"Target '{target}' is outside authorized or discovered scope")

    def _known_targets(self, state: AutoFlowState) -> set[str]:
        targets = {canonical_target(str(value)) for value in state.get("target_scope", [])}
        for asset in state.get("assets", []):
            host = str(asset.get("ip", ""))
            if host:
                targets.add(host)
            for port in asset.get("ports", []):
                port_number = port.get("port")
                if port_number:
                    targets.add(f"{host}:{port_number}")
                    scheme = "https" if int(port_number) in {443, 8443} else "http"
                    targets.add(f"{scheme}://{host}:{port_number}")
        for item in state.get("web_recon", []):
            for key in ("target",):
                if item.get(key):
                    targets.add(canonical_target(str(item[key])))
            for key in ("links", "interesting_paths"):
                for value in item.get(key, [])[:200]:
                    targets.add(canonical_target(str(value)))
            robots = item.get("robots") or {}
            for value in robots.get("interesting_paths", [])[:200]:
                targets.add(canonical_target(str(value)))
        for finding in state.get("findings", []):
            if finding.get("target"):
                targets.add(canonical_target(str(finding["target"])))
        for surface in state.get("attack_surfaces", []):
            if surface.get("target"):
                targets.add(canonical_target(str(surface["target"])))
            for value in surface.get("entrypoints", []):
                targets.add(canonical_target(str(value)))
        return {target for target in targets if target}

    def _ok(self, name: str, target: str, result: Any, summary: str) -> dict[str, Any]:
        return {"ok": True, "tool_call": name, "target": target, "summary": summary, "result": result}

    def _error(self, name: str, target: str, error: str) -> dict[str, Any]:
        return {"ok": False, "tool_call": name, "target": target, "error": self._truncate(error, 1200)}

    def _compact_web_recon(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "target": result.get("target"),
            "status_code": result.get("status_code"),
            "title": result.get("title"),
            "links": result.get("links", [])[:30],
            "forms": result.get("forms", [])[:10],
            "scripts": result.get("scripts", [])[:20],
            "interesting_paths": result.get("interesting_paths", [])[:30],
            "robots": result.get("robots", {}),
            "error": result.get("error", ""),
        }

    def _json_or_text(self, stdout: str, stderr: str) -> Any:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"stdout_excerpt": self._truncate(stdout), "stderr_excerpt": self._truncate(stderr, 1200)}

    def _summary(self, tool: str, stdout: str) -> str:
        stdout = stdout or ""
        first_line = next((line.strip() for line in stdout.splitlines() if line.strip()), "")
        return f"{tool} completed: {first_line[:300]}" if first_line else f"{tool} completed"

    def _write_text_artifact(self, path: Path, stdout: str, stderr: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{stdout or ''}\n{stderr or ''}".strip() + "\n", encoding="utf-8")

    def _truncate(self, value: str, limit: int | None = None) -> str:
        limit = limit or self.result_limit
        return value if len(value) <= limit else value[: limit - 20] + "\n...[truncated]"

    def _host(self, value: str) -> str:
        parsed = urlparse(value)
        if parsed.hostname:
            return parsed.hostname
        if ":" in value:
            return value.split(":", 1)[0]
        return value

    def _split_host_port(self, target: str) -> tuple[str, str | None]:
        parsed = urlparse(target if "://" in target else f"//{target}")
        if parsed.hostname and parsed.port:
            return parsed.hostname, str(parsed.port)
        return target, None
