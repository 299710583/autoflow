from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from autoflow.agents.base import BaseAgent
from autoflow.artifacts.store import ArtifactStore
from autoflow.executor.execution_client import ExecutionClient
from autoflow.executor.kali_client import KaliClient
from autoflow.executor.parsers.nmap import parse_nmap_xml
from autoflow.executor.web_recon import WebReconClient
from autoflow.flows.models import (
    Action,
    ArtifactType,
    AssessmentTask,
    MemoryItem,
    MemoryKind,
    SubTask,
    TaskStatus,
)
from autoflow.graph.state import AutoFlowState


class ReconAgent(BaseAgent):
    """执行低风险服务探测，并将 nmap XML 转成结构化资产。"""

    name = "recon"

    def __init__(
        self,
        kali_client: KaliClient | None = None,
        execution_client: ExecutionClient | None = None,
        artifact_store: ArtifactStore | None = None,
        web_recon_client: WebReconClient | None = None,
    ) -> None:
        self.execution_client = execution_client or kali_client or ExecutionClient()
        self.artifact_store = artifact_store or ArtifactStore()
        self.web_recon_client = web_recon_client or WebReconClient()

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "recon"

        flow = state.get("flow")
        if flow is None:
            raise ValueError("ReconAgent requires state['flow']")

        task = self._select_task(flow.tasks)
        if task is None:
            state["next_action"] = "research"
            return state

        # 每次工具执行都记录为 SubTask -> Action -> Artifact。
        subtask = task.add_subtask(
            SubTask(
                agent=self.name,
                objective=f"Run safe service discovery against {task.target}",
                risk_level=task.risk_level,
            )
        )
        action = subtask.add_action(
            Action(
                tool="nmap",
                intent={
                    "tool": "nmap",
                    "profile": "safe_service_scan",
                    "args": {},
                },
                risk_level=task.risk_level,
            )
        )

        output_path = self.artifact_store.reserve_action_path(flow.id, action.id, "nmap.xml")
        # 将可选的 host:port 目标转换为安全的单端口 nmap profile。
        profile, args = self._nmap_profile_and_args(task.target, output_path)
        action.intent["profile"] = profile
        action.intent["args"] = args

        spec = self.execution_client.build_command(action.intent)
        action.command_preview = " ".join(spec.command)
        action.mark_started()
        task.status = TaskStatus.RUNNING
        subtask.status = TaskStatus.RUNNING

        try:
            result = self.execution_client.execute_spec(spec)
            action.metadata["exit_code"] = result.exit_code
            action.metadata["stdout"] = result.stdout
            action.metadata["stderr"] = result.stderr

            if not result.succeeded:
                action.mark_failed(result.stderr or f"Command exited with code {result.exit_code}")
                task.status = TaskStatus.FAILED
                subtask.status = TaskStatus.FAILED
                return state

            # 先登记原始 nmap XML，再解析为结构化资产。
            raw_artifact = self.artifact_store.register(
                path=output_path,
                artifact_type=ArtifactType.RAW_OUTPUT,
                action_id=action.id,
                summary="nmap XML output",
            )
            action.artifacts.append(raw_artifact)

            assets = parse_nmap_xml(output_path) if Path(output_path).exists() else []
            state["assets"] = [*state.get("assets", []), *assets]
            action.metadata["assets"] = assets
            web_recon_results = self._run_web_recon(task.target, assets)
            if web_recon_results:
                state["web_recon"] = [*state.get("web_recon", []), *web_recon_results]
                web_recon_path = self.artifact_store.reserve_action_path(flow.id, action.id, "web_recon.json")
                self._write_json_artifact(web_recon_path, web_recon_results)
                web_artifact = self.artifact_store.register(
                    path=web_recon_path,
                    artifact_type=ArtifactType.STRUCTURED_RESULT,
                    action_id=action.id,
                    summary="web recon structure",
                )
                action.artifacts.append(web_artifact)
                flow.add_memory(
                    MemoryItem(
                        kind=MemoryKind.OBSERVATION,
                        content=self._summarize_web_recon(web_recon_results),
                        source=action.id,
                        references=[web_artifact.id],
                        metadata={"web_recon": web_recon_results},
                    )
                )
            action.mark_succeeded(f"Discovered {self._open_port_count(assets)} open ports")
            task.status = TaskStatus.COMPLETED
            subtask.status = TaskStatus.COMPLETED

            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.OBSERVATION,
                    content=action.result_summary,
                    source=action.id,
                    references=[raw_artifact.id],
                )
            )
        finally:
            state["active_task_id"] = task.id
            state["active_subtask_id"] = subtask.id
            state["last_action_id"] = action.id
            state["next_action"] = "research"

        return state

    def _select_task(self, tasks: list[AssessmentTask]) -> AssessmentTask | None:
        for task in tasks:
            if task.type == "recon" and task.status == TaskStatus.PENDING:
                return task
        return None

    def _remote_output_path(self, local_path: Path) -> str:
        return str(local_path).replace("\\", "/")

    def _nmap_profile_and_args(self, target: str, output_path: Path) -> tuple[str, dict[str, str]]:
        host, port = self._split_host_port(target)
        args = {
            "target": host,
            "output": self._remote_output_path(output_path),
        }
        if port:
            args["port"] = port
            return "safe_service_scan_port", args
        return "safe_service_scan", args

    def _split_host_port(self, target: str) -> tuple[str, str | None]:
        # 加上 // 后，urlparse 可同时处理普通 host:port 和完整 URL。
        parsed = urlparse(target if "://" in target else f"//{target}")
        if parsed.hostname and parsed.port:
            return parsed.hostname, str(parsed.port)
        return target, None

    def _open_port_count(self, assets: list[dict]) -> int:
        return sum(len(asset.get("ports", [])) for asset in assets)

    def _run_web_recon(self, original_target: str, assets: list[dict]) -> list[dict]:
        targets = self._web_recon_targets(original_target, assets)
        results: list[dict] = []
        for target in targets:
            results.append(self.web_recon_client.recon(target))
        return results

    def _web_recon_targets(self, original_target: str, assets: list[dict]) -> list[str]:
        parsed = urlparse(original_target if "://" in original_target else f"//{original_target}")
        explicit_port = parsed.port
        explicit_scheme = "http" if parsed.scheme in {"", "http"} else parsed.scheme
        targets: list[str] = []

        if not explicit_port and parsed.scheme not in {"http", "https"}:
            return []

        if parsed.hostname and (explicit_port or parsed.scheme in {"http", "https"}):
            scheme = explicit_scheme if explicit_scheme in {"http", "https"} else "http"
            netloc = parsed.netloc or original_target
            targets.append(f"{scheme}://{netloc}")

        for asset in assets:
            host = asset.get("ip")
            if not host:
                continue
            for port in asset.get("ports", []):
                port_number = port.get("port")
                service = (port.get("service") or "").lower()
                if explicit_port and port_number != explicit_port:
                    continue
                if service not in {"http", "https", "http-alt", "nessus"} and port_number not in {
                    80,
                    443,
                    3000,
                    3001,
                    5000,
                    8000,
                    8080,
                    8443,
                    8834,
                }:
                    continue
                scheme = "https" if service == "https" or port_number in {443, 8443} else "http"
                targets.append(f"{scheme}://{host}:{port_number}")

        return list(dict.fromkeys(targets))

    def _write_json_artifact(self, path: Path, payload: list[dict]) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _summarize_web_recon(self, results: list[dict]) -> str:
        parts = []
        for item in results:
            parts.append(
                f"{item.get('target')} status={item.get('status_code')} "
                f"title={item.get('title')!r} links={len(item.get('links', []))} "
                f"forms={len(item.get('forms', []))} scripts={len(item.get('scripts', []))}"
            )
        return "Web recon: " + "; ".join(parts)
