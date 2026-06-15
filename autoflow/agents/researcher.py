from __future__ import annotations

import json
from typing import Any

from autoflow.agents.base import BaseAgent
from autoflow.flows.models import AttackSurface, MemoryItem, MemoryKind
from autoflow.graph.state import AutoFlowState
from autoflow.llm.client import LLMClient
from autoflow.memory.context import MemoryContextBuilder
from autoflow.settings import settings


WEB_LIKE_SERVICES = {"http", "https", "http-alt", "nessus"}
WEB_LIKE_PORTS = {80, 443, 3000, 3001, 5000, 8000, 8080, 8443, 8834}


RESEARCHER_SYSTEM_PROMPT = """You are AutoFlow's discovery analysis agent for an authorized security assessment.
Return only JSON. Do not include markdown.
Analyze recon results and decide attack surfaces from evidence.
Do not invent targets outside the authorized recon data.
Do not plan exploitation, brute force, privilege escalation, lateral movement, persistence, destructive writes, or evasion.
"""


class ResearcherAgent(BaseAgent):
    """分析资产并抽象出攻击面，不直接决定具体工具执行。"""

    name = "researcher"

    def __init__(self, use_llm: bool | None = None, llm_client: LLMClient | None = None) -> None:
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.context_builder = MemoryContextBuilder()

    async def run(self, state: AutoFlowState) -> AutoFlowState:
        state["current_phase"] = "research"
        assets = state.get("assets", [])
        web_recon = state.get("web_recon", [])
        if self._should_use_llm():
            attack_surfaces = self._build_attack_surfaces_with_llm(state, assets, web_recon)
        else:
            attack_surfaces = self._build_attack_surfaces(assets, web_recon)

        state["attack_surfaces"] = [surface.model_dump(mode="json") for surface in attack_surfaces]
        state["memory_context"] = self.context_builder.build(state)
        # 兼容旧状态字段：执行任务现在由 Strategist 生成 TestPlanAction。
        state["follow_up_tasks"] = []

        flow = state.get("flow")
        if flow is not None:
            for surface in attack_surfaces:
                flow.add_attack_surface(surface)
            flow.add_memory(
                MemoryItem(
                    kind=MemoryKind.DECISION,
                    content=f"Researcher identified {len(attack_surfaces)} attack surfaces",
                    source=self.name,
                    references=[surface.id for surface in attack_surfaces],
                )
            )

        state["next_action"] = "strategy"
        return state

    def _should_use_llm(self) -> bool:
        if self.use_llm is not None:
            return self.use_llm
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required for ResearcherAgent. Set use_llm=False for offline tests.")
        return True

    def _build_attack_surfaces_with_llm(
        self,
        state: AutoFlowState,
        assets: list[dict],
        web_recon: list[dict] | None = None,
    ) -> list[AttackSurface]:
        client = self.llm_client or LLMClient()
        prompt = {
            "user_prompt": state.get("user_prompt", ""),
            "target_scope": state.get("target_scope", []),
            "rules_of_engagement": state.get("rules_of_engagement", {}),
            "assets": assets,
            "web_recon": web_recon or [],
            "allowed_surface_types": ["web_application", "network_service", "api", "static_content", "unknown"],
            "output_schema": {
                "attack_surfaces": [
                    {
                        "target": "http://host:port or host:port from recon data",
                        "surface_type": "web_application",
                        "technology": "observed technology or empty string",
                        "entrypoints": ["authorized URL or service endpoint"],
                        "related_assets": ["host:port"],
                        "rationale": "why this is an attack surface based on recon evidence",
                        "metadata": {"source": "llm_researcher"},
                    }
                ]
            },
        }
        response = client.complete_json(
            prompt=json.dumps(prompt, ensure_ascii=False),
            system=RESEARCHER_SYSTEM_PROMPT,
            max_tokens=2048,
        )
        items = response.get("attack_surfaces", [])
        if not isinstance(items, list):
            return []
        return self._coerce_llm_surfaces(items, assets, web_recon or [])

    def _coerce_llm_surfaces(
        self,
        items: list[dict[str, Any]],
        assets: list[dict],
        web_recon: list[dict],
    ) -> list[AttackSurface]:
        allowed_targets = self._allowed_targets(assets, web_recon)
        surfaces: list[AttackSurface] = []
        for item in items:
            target = str(item.get("target", "")).rstrip("/")
            if not target or target not in allowed_targets:
                continue
            entrypoints = [
                str(value).rstrip("/")
                for value in item.get("entrypoints", [])
                if str(value).rstrip("/") in allowed_targets
            ]
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            surfaces.append(
                AttackSurface(
                    target=target,
                    surface_type=str(item.get("surface_type") or "unknown"),
                    technology=str(item.get("technology") or ""),
                    entrypoints=entrypoints or [target],
                    related_assets=[str(value) for value in item.get("related_assets", [])][:20],
                    rationale=str(item.get("rationale") or "LLM identified this surface from recon evidence."),
                    metadata={**metadata, "source": "llm_researcher"},
                )
            )
        return surfaces

    def _allowed_targets(self, assets: list[dict], web_recon: list[dict]) -> set[str]:
        targets: set[str] = set()
        for asset in assets:
            host = asset.get("ip")
            if not host:
                continue
            targets.add(str(host))
            for port in asset.get("ports", []):
                port_number = port.get("port")
                if not port_number:
                    continue
                port_number_int = int(port_number)
                service_target = f"{host}:{port_number_int}"
                targets.add(service_target)
                service = (port.get("service") or "").lower()
                if self._is_web_like(service, port_number_int):
                    scheme = "https" if service == "https" or port_number_int in {443, 8443} else "http"
                    targets.add(f"{scheme}://{service_target}")
        for item in web_recon:
            target = str(item.get("target", "")).rstrip("/")
            if target:
                targets.add(target)
            for key in ("links", "interesting_paths"):
                for value in item.get(key, [])[:100]:
                    if value:
                        targets.add(str(value).rstrip("/"))
            for form in item.get("forms", [])[:50]:
                action = str(form.get("action", "")).rstrip("/")
                if action:
                    targets.add(action)
        return targets

    def _build_attack_surfaces(self, assets: list[dict], web_recon: list[dict] | None = None) -> list[AttackSurface]:
        surfaces: list[AttackSurface] = []
        recon_by_target = {item.get("target"): item for item in web_recon or []}
        for asset in assets:
            host = asset.get("ip")
            if not host:
                continue
            for port in asset.get("ports", []):
                port_number = port.get("port")
                if not port_number:
                    continue
                service = (port.get("service") or "").lower()
                product = port.get("product") or ""
                target = f"{host}:{port_number}"

                if self._is_web_like(service, port_number):
                    scheme = "https" if service == "https" or port_number in {443, 8443} else "http"
                    entrypoint = f"{scheme}://{target}"
                    surfaces.append(
                        AttackSurface(
                            target=entrypoint,
                            surface_type="web_application",
                            technology=product or service,
                            entrypoints=self._web_entrypoints(entrypoint, recon_by_target.get(entrypoint)),
                            related_assets=[target],
                            rationale=f"{service or port_number} appears web-like and can be fingerprinted safely.",
                            metadata={
                                "asset": asset,
                                "port": port,
                                "web_recon": recon_by_target.get(entrypoint),
                            },
                        )
                    )
                else:
                    surfaces.append(
                        AttackSurface(
                            target=target,
                            surface_type="network_service",
                            technology=product or service,
                            entrypoints=[target],
                            related_assets=[target],
                            rationale="Open service should be reviewed before deeper validation.",
                            metadata={"asset": asset, "port": port},
                        )
                    )
        return surfaces

    def _is_web_like(self, service: str, port_number: int) -> bool:
        return service in WEB_LIKE_SERVICES or port_number in WEB_LIKE_PORTS

    def _web_entrypoints(self, entrypoint: str, web_recon: dict | None) -> list[str]:
        if not web_recon:
            return [entrypoint]
        values = [
            entrypoint,
            *web_recon.get("links", []),
            *[form.get("action", "") for form in web_recon.get("forms", [])],
            *web_recon.get("interesting_paths", []),
        ]
        return [value for value in dict.fromkeys(values) if value][:100]
